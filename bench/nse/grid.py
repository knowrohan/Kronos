"""Full backtest grid: signal_def x scaling x rebalance x cost. Reported whole, never as a max.

4 signal definitions x 3 scalings x {daily, weekly} x {0, 15, 30, 50} bps = 96 cells per model.
A maximum over 96 arms is not an estimate, so grid.py exists to make the whole surface
visible and to let report.py show where the pre-registered cell sits inside it.

Baselines (momentum, reversal, anchor) and the random-signal distribution are run through the
identical engine and appear as extra rows, so every number in the writeup is comparable.
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
from bench.nse.common import OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import (anchor_scores, equal_weight_benchmark,  # noqa: E402
                                equal_weight_universe, momentum_scores, random_scores)
from bench.nse.costs import PRESETS  # noqa: E402

DEFS = ["last", "mean", "max", "min"]
SCALINGS = ["z", "ret", "recon"]
REBALANCES = {"daily": 1, "weekly": 5}
COSTS = [0.0, 15.0, 30.0, 50.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=os.path.join(OUT_NSE, "signals_zeroshot.parquet"))
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--benchmark", default="cnx200")
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--extra-ndrop", type=int, nargs="*", default=[1, 2, 5],
                    help="rebalance-frequency arm from plan 4.2")
    ap.add_argument("--panel", default=None,
                    help="price panel to score against; defaults to the NSE panel. MUST match "
                         "the panel the signals were generated from, or forward returns are "
                         "taken from the wrong market entirely.")
    ap.add_argument("--calendar", default=None)
    ap.add_argument("--bench-path", default=None, help="benchmark parquet for this market")
    ap.add_argument("--membership", default=None,
                    help="parquet of date,symbol point-in-time index membership; masks every "
                         "score matrix, control and equal-weight benchmark to actual members, "
                         "which removes the survivorship bias from the universe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    market = load_market(args.panel, args.calendar)
    bench = bench_series(market, args.benchmark, args.bench_path)
    sig = pd.concat([pd.read_parquet(p) for p in args.signals.split(",")], ignore_index=True)
    dates = sorted(sig["date"].unique())
    print(f"{len(dates)} signal dates {pd.Timestamp(dates[0]).date()} -> "
          f"{pd.Timestamp(dates[-1]).date()}", flush=True)

    wide = {c: sig.pivot(index="date", columns="symbol", values=c).sort_index()
            for c in sig.columns if c.split("_")[0] in DEFS}
    if args.membership:
        mem = pd.read_parquet(args.membership)
        mem["date"] = pd.to_datetime(mem["date"])
        pit = (mem.assign(v=True).pivot(index="date", columns="symbol", values="v")
               .reindex(index=wide["last_ret"].index,
                        columns=wide["last_ret"].columns).fillna(False).astype(bool))
        n_before = int(wide["last_ret"].notna().sum().sum())
        wide = {k: v.where(pit) for k, v in wide.items()}
        n_after = int(wide["last_ret"].notna().sum().sum())
        print(f"point-in-time mask: {n_before} -> {n_after} scored (name, date) cells "
              f"({n_after / max(n_before, 1) * 100:.1f}%)", flush=True)
    mask = wide["last_ret"].notna()
    # equal-weight benchmark over the SAME eligible universe the strategy picks from
    bench_ew = equal_weight_benchmark(market, mask)
    baselines = {
        # Short-horizon reversal is the one free signal that tested significant in the pilot,
        # so it is backtested properly rather than left as an IC footnote.
        "reversal_1d": -momentum_scores(market, mask.index, 1),
        "reversal_5d": -momentum_scores(market, mask.index, 5),
        "reversal_10d": -momentum_scores(market, mask.index, 10),
        "momentum_10d": momentum_scores(market, mask.index, 10),
        "momentum_20d": momentum_scores(market, mask.index, 20),
        "momentum_60d": momentum_scores(market, mask.index, 60),
        "anchor_reversal": anchor_scores(sig, negate=True),
        "anchor_momentum": anchor_scores(sig, negate=False),
    }
    # every baseline and control is confined to the SAME eligible cross-section as Kronos
    for k in baselines:
        baselines[k] = baselines[k].reindex(index=mask.index, columns=mask.columns).where(mask)

    rows = []

    def add(family, name, w, reb_name, reb, cost, **kw):
        res = run_backtest(w, market, topk=args.topk, n_drop=kw.pop("n_drop", args.n_drop),
                           hold_thresh=args.hold_thresh, rebalance=reb,
                           cost_bps_round_trip=cost, **kw)
        m = metrics(res, bench, seed=args.seed)
        m_ew = metrics(res, bench_ew, seed=args.seed)
        m["ann_excess_vs_ew200"] = m_ew["ann_excess_net"]
        m["ir_vs_ew200"] = m_ew["information_ratio"]
        m["excess_ci95_ann_vs_ew200"] = m_ew["excess_ci95_annualised"]
        rows.append({"family": family, "arm": name, "rebalance": reb_name,
                     "cost_bps": cost, **m})
        return m

    print("main grid ...", flush=True)
    for d in DEFS:
        for sc in SCALINGS:
            col = f"{d}_{sc}"
            for rname, r in REBALANCES.items():
                for c in COSTS:
                    add("kronos", col, wide[col], rname, r, c)
        print(f"  {d} done", flush=True)

    print("baselines ...", flush=True)
    for name, w in baselines.items():
        for rname, r in REBALANCES.items():
            for c in COSTS:
                add("baseline", name, w, rname, r, c)

    print("controls ...", flush=True)
    prim = wide["last_ret"]
    for rname, r in REBALANCES.items():
        for c in COSTS:
            add("control", "reversed_last_ret", -prim, rname, r, c)
            ew = equal_weight_universe(market, prim.index, r, mask=mask)
            m = metrics(ew, bench, seed=args.seed)
            m["ann_excess_vs_ew200"] = metrics(ew, bench_ew, seed=args.seed)["ann_excess_net"]
            rows.append({"family": "control", "arm": "equal_weight_universe",
                         "rebalance": rname, "cost_bps": c, **m})

    print("n_drop / weighting / dropout-mode sensitivity ...", flush=True)
    for nd in args.extra_ndrop:
        for rname, r in REBALANCES.items():
            add("sensitivity", f"last_ret_ndrop{nd}", prim, rname, r, 30.0, n_drop=nd)
    for wmode in ("drift", "rebalance"):
        add("sensitivity", f"last_ret_w_{wmode}", prim, "weekly", 5, 30.0, weighting=wmode)
    for dmode in ("qlib", "naive"):
        add("sensitivity", f"last_ret_dropout_{dmode}", prim, "weekly", 5, 30.0, dropout_mode=dmode)
    for preset in ("zerodha", "flat20", "plan_draft"):
        m = metrics(run_backtest(prim, market, topk=args.topk, n_drop=args.n_drop,
                                 hold_thresh=args.hold_thresh, rebalance=5,
                                 cost_model=PRESETS[preset], capital=1e7), bench, seed=args.seed)
        rows.append({"family": "sensitivity", "arm": f"last_ret_cost_{preset}",
                     "rebalance": "weekly", "cost_bps": m["one_way_cost_bps"] * 2, **m})

    print(f"random-signal distribution, {args.n_random} seeds x2 rebalance ...", flush=True)
    rnd_rows = []
    for rname, r in REBALANCES.items():
        for s in range(args.n_random):
            _r = run_backtest(random_scores(prim, s), market, topk=args.topk,
                              n_drop=args.n_drop, hold_thresh=args.hold_thresh,
                              rebalance=r, cost_bps_round_trip=30.0)
            m = metrics(_r, bench, seed=args.seed)
            m2 = metrics(_r, bench_ew, seed=args.seed)
            rnd_rows.append({"rebalance": rname, "seed": s,
                             "ann_excess_net": m["ann_excess_net"],
                             "ann_excess_vs_ew200": m2["ann_excess_net"],
                             "ann_return_net": m["ann_return_net"],
                             "information_ratio": m["information_ratio"],
                             "turnover_ann": m["turnover_ann"]})
        print(f"  {rname} done", flush=True)
    rnd = pd.DataFrame(rnd_rows)
    rnd.to_parquet(os.path.join(OUT_NSE, f"random_dist_{args.tag}.parquet"), index=False)

    grid = pd.DataFrame(rows)
    grid.to_parquet(os.path.join(OUT_NSE, f"grid_{args.tag}.parquet"), index=False)

    # daily curves for the figures: primary arm, the two free baselines, and the benchmarks
    curves = {"primary": prim, "momentum": baselines["momentum_10d"],
              "anchor": baselines["anchor_reversal"], "reversed": -prim}
    for nm, w in curves.items():
        r = run_backtest(w, market, topk=args.topk, n_drop=args.n_drop,
                         hold_thresh=args.hold_thresh, rebalance=5, cost_bps_round_trip=30.0)
        r["daily"].to_parquet(os.path.join(OUT_NSE, f"bt_{nm}_{args.tag}.parquet"))
    ewr = equal_weight_universe(market, prim.index, 5, mask=mask)["daily"]
    ewr.to_parquet(os.path.join(OUT_NSE, f"bt_equalweight_{args.tag}.parquet"))
    idx = pd.read_parquet(os.path.join(OUT_NSE, f"bt_primary_{args.tag}.parquet")).index
    pd.DataFrame({"bench": bench.reindex(idx).fillna(0.0),
                  "bench_ew": bench_ew.reindex(idx).fillna(0.0)}).to_parquet(
        os.path.join(OUT_NSE, f"bt_bench_{args.tag}.parquet"))

    # where does the pre-registered cell sit?
    pre = grid[(grid.arm == "last_ret") & (grid.rebalance == "weekly") & (grid.cost_bps == 30.0)]
    ex = rnd[rnd.rebalance == "weekly"]["ann_excess_net"].to_numpy()
    # How much of any "excess vs the index" is just weighting + survivorship, before any
    # signal? This is the single largest term in the whole study, so it gets measured directly.
    dfirst, dlast = prim.index.min(), prim.index.max()
    win = (pd.Series(market.calendar) >= dfirst) & (pd.Series(market.calendar) <= dlast)
    nsess = int(win.sum())
    def _ann(sr):
        sr = sr[(sr.index >= dfirst) & (sr.index <= dlast)].dropna()
        return float((1 + sr).prod() ** (252 / max(len(sr), 1)) - 1)
    universe_effects = {
        "sessions": nsess,
        "cap_weighted_index_ann": _ann(bench),
        "equal_weight_eligible_universe_ann": _ann(bench_ew),
        "gap_weighting_plus_survivorship_ann": _ann(bench_ew) - _ann(bench),
        "note": ("equal-weighting a current-membership NIFTY 200 list beats the cap-weighted "
                 "index by this much with ZERO signal; it is weighting + survivorship, not alpha, "
                 "and it is why the random-signal distribution is the right comparison"),
    }
    summary = {
        "n_cells": len(grid),
        "universe_effects": universe_effects,
        "prereg_cell": pre.iloc[0].to_dict() if len(pre) else None,
        "kronos_grid_excess_range": [
            float(grid[grid.family == "kronos"]["ann_excess_net"].min()),
            float(grid[grid.family == "kronos"]["ann_excess_net"].max())],
        "kronos_grid_share_positive_excess": float(
            (grid[grid.family == "kronos"]["ann_excess_net"] > 0).mean()),
        "random_weekly": {
            "mean": float(ex.mean()), "std": float(ex.std(ddof=1)),
            "p05": float(np.percentile(ex, 5)), "p50": float(np.percentile(ex, 50)),
            "p95": float(np.percentile(ex, 95)),
            "mean_turnover_ann": float(rnd[rnd.rebalance == "weekly"]["turnover_ann"].mean()),
        },
    }
    if len(pre):
        p = float(pre.iloc[0]["ann_excess_net"])
        summary["prereg_percentile_in_random"] = float((ex < p).mean() * 100)
        summary["prereg_empirical_p_one_sided"] = float((ex >= p).mean())
    with open(os.path.join(OUT_NSE, f"grid_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    sidecar(os.path.join(OUT_NSE, f"grid_{args.tag}_meta.json"), vars(args))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
