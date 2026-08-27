"""Backtest the model-free signals at FULL daily resolution. No GPU, no Kronos.

The study's one statistically significant finding is that short-horizon reversal carries real
cross-sectional information in this universe. That is only interesting if it survives costs, and
the Kronos signal grid cannot answer it: those signals exist only on the dates Kronos was run.
Every baseline here is computed from the price panel alone, so it can be evaluated on all 756
test sessions at any rebalance frequency.

Runs the identical engine, the identical execution lag, the identical cost model and the
identical random-signal control as the Kronos grid, so the numbers are directly comparable.
Also reports every arm against BOTH benchmarks (cap-weighted index and equal-weight universe)
and, optionally, on the point-in-time universe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import bench_series, load_market, metrics, run_backtest  # noqa: E402
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import (compute_ic, equal_weight_benchmark,  # noqa: E402
                                momentum_scores, random_scores, summarise_ic)

HORIZONS = [1, 5, 10, 20]
COSTS = [0.0, 15.0, 30.0, 50.0]
REBALANCES = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}


def zscore_window(market, dates, lookback=90):
    """(close_t - mean_lookback) / std_lookback -- the mechanical factor Kronos reproduces."""
    c = market.closes
    m = c.rolling(lookback).mean()
    s = c.rolling(lookback).std()
    return ((c - m) / s).reindex(dates)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price.parquet"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--membership", default=None)
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-vol-arms", action="store_true",
                    help="add trailing-volatility sorts. These were NOT in the pre-registered "
                         "family; run them under a separate tag so the family-wise bar computed "
                         "for the original 17 arms stays honest.")
    args = ap.parse_args()

    market = load_market(args.panel, args.calendar)
    dates = [d for d in market.calendar
             if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    print(f"{len(dates)} sessions {dates[0].date()} -> {dates[-1].date()}", flush=True)

    elig = market.tradable.reindex(dates)
    if args.membership:
        mem = pd.read_parquet(args.membership)
        mem["date"] = pd.to_datetime(mem["date"])
        pit = (mem.assign(v=True).pivot(index="date", columns="symbol", values="v")
               .reindex(index=elig.index, columns=elig.columns).fillna(False).astype(bool))
        elig = elig & pit
        print(f"point-in-time universe: mean {elig.sum(axis=1).mean():.1f} names/session", flush=True)
    # a name is only eligible once it has `lookback` bars of history, matching Kronos exactly
    have_hist = market.closes.notna().rolling(args.lookback).sum().reindex(dates) >= args.lookback
    elig = elig & have_hist.fillna(False)

    sig = {}
    for h in (1, 2, 5, 10, 20, 60, 120):
        sig[f"momentum_{h}d"] = momentum_scores(market, dates, h).where(elig)
        sig[f"reversal_{h}d"] = (-momentum_scores(market, dates, h)).where(elig)
    if args.with_vol_arms:
        for wdw in (20, 60):
            v = market.oo_ret.rolling(wdw).std().reindex(dates)
            sig[f"highvol_{wdw}d"] = v.where(elig)
            sig[f"lowvol_{wdw}d"] = (-v).where(elig)
    z = zscore_window(market, dates, args.lookback)
    sig["zscore_reversal"] = (-z).where(elig)
    sig["zscore_momentum"] = z.where(elig)
    # what Kronos's score would be if it reverted exactly 64% of the way (the fitted number)
    # the fitted reproduction of the zero-shot score: slope -0.701 of anchor_z, pooled
    # R2 0.70 over all 756 dates (bench/out/nse/mechanism.json). The coefficient is a
    # negative scalar, so every rank, IC and backtest number is identical for any
    # negative value -- it is written out only so the label matches the measurement.
    sig["kronos_emulator"] = (-0.701 * z).where(elig)

    bench = bench_series(market, "cnx200")
    bench_ew = equal_weight_benchmark(market, elig)

    print("information coefficients ...", flush=True)
    ic = {}
    for name, w in sig.items():
        per = {}
        for h in HORIZONS:
            per[f"open_h{h}"] = summarise_ic(compute_ic(w, market, h).to_numpy(), lag=h,
                                             seed=args.seed)
        ic[name] = per
        p = per["open_h10"]
        p1 = per["open_h1"]
        print(f"  {name:18s} IC1 {p1['mean_ic']:+.4f} (t {p1['t_newey_west']:+5.2f}"
              f"{'*' if p1['significant'] else ' '})   "
              f"IC10 {p['mean_ic']:+.4f} (t {p['t_newey_west']:+5.2f}"
              f"{'*' if p['significant'] else ' '})", flush=True)

    print("backtests ...", flush=True)
    rows = []
    for name, w in sig.items():
        for rname, r in REBALANCES.items():
            for c in COSTS:
                res = run_backtest(w, market, topk=args.topk, n_drop=args.n_drop,
                                   hold_thresh=args.hold_thresh, rebalance=r,
                                   cost_bps_round_trip=c)
                m = metrics(res, bench, seed=args.seed)
                m_ew = metrics(res, bench_ew, seed=args.seed)
                m["ann_excess_vs_ew200"] = m_ew["ann_excess_net"]
                m["ir_vs_ew200"] = m_ew["information_ratio"]
                m["excess_ci95_ann_vs_ew200"] = m_ew["excess_ci95_annualised"]
                rows.append({"arm": name, "rebalance": rname, "cost_bps": c, **m})
        print(f"  {name} done", flush=True)

    print(f"random control, {args.n_random} seeds x {len(REBALANCES)} frequencies ...", flush=True)
    rnd = []
    shell = sig["reversal_10d"]
    for rname, r in REBALANCES.items():
        for s in range(args.n_random):
            res = run_backtest(random_scores(shell, s), market, topk=args.topk,
                               n_drop=args.n_drop, hold_thresh=args.hold_thresh,
                               rebalance=r, cost_bps_round_trip=30.0)
            m = metrics(res, bench, seed=args.seed)
            m_ew = metrics(res, bench_ew, seed=args.seed)
            rnd.append({"rebalance": rname, "seed": s,
                        "ann_excess_net": m["ann_excess_net"],
                        "ann_excess_vs_ew200": m_ew["ann_excess_net"],
                        "turnover_ann": m["turnover_ann"]})
        print(f"  {rname} done", flush=True)

    g = pd.DataFrame(rows)
    g.to_parquet(os.path.join(OUT_NSE, f"grid_{args.tag}.parquet"), index=False)
    rd = pd.DataFrame(rnd)
    rd.to_parquet(os.path.join(OUT_NSE, f"random_dist_{args.tag}.parquet"), index=False)

    summary = {"n_sessions": len(dates), "range": [str(dates[0].date()), str(dates[-1].date())],
               "membership": args.membership, "ic": ic, "n_cells": len(g)}
    for rname in REBALANCES:
        sub = rd[rd.rebalance == rname]["ann_excess_vs_ew200"].to_numpy()
        summary[f"random_{rname}_vs_ew"] = {
            "mean": float(sub.mean()), "p05": float(np.percentile(sub, 5)),
            "p95": float(np.percentile(sub, 95))}
    best = g[(g.cost_bps == 30.0)].sort_values("ann_excess_vs_ew200", ascending=False)
    summary["top10_vs_equal_weight_at_30bps"] = best.head(10)[
        ["arm", "rebalance", "ann_return_net", "ann_excess_net", "ann_excess_vs_ew200",
         "ir_vs_ew200", "turnover_ann", "cost_drag_ann_pct"]].to_dict("records")
    with open(os.path.join(OUT_NSE, f"grid_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    sidecar(os.path.join(OUT_NSE, f"grid_{args.tag}_meta.json"), vars(args))

    print("\ntop arms at 30 bps, ranked by excess over the equal-weight universe:")
    print(best.head(14)[["arm", "rebalance", "ann_return_net", "ann_excess_net",
                         "ann_excess_vs_ew200", "ir_vs_ew200", "turnover_ann",
                         "cost_drag_ann_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
