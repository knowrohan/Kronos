"""Is the winning blend a plateau or a spike?

`robustness.py` reports that a 50/50 rank blend of 20-session momentum and 5-session reversal,
rebalanced monthly, beats both parents and is positive in every calendar year. That blend was
chosen by looking at the results, which is exactly how overfitting happens. The cheapest
diagnostic that distinguishes a real effect from a lucky cell is the shape of the surface AROUND
the winner: a genuine effect varies smoothly and its neighbours also work; a fitted artefact
sits on a spike surrounded by nothing.

So: sweep the momentum lookback, the reversal lookback, the blend weight and the rebalance
frequency, and report what fraction of the whole neighbourhood is positive, how the winner ranks
inside its own surface, and how steeply the surface falls away from it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import load_market, metrics, run_backtest  # noqa: E402
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import equal_weight_benchmark, momentum_scores  # noqa: E402
from bench.nse.robustness import build_universe, rank_pct  # noqa: E402

MOM_H = [10, 15, 20, 25, 30, 40, 60]
REV_H = [2, 3, 5, 8, 10]
WEIGHTS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]     # 0 = pure reversal, 1 = pure momentum
REBAL = {"weekly": 5, "biweekly": 10, "monthly": 21}
TRADING_DAYS = 252


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price_pit.parquet"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--membership", default=os.path.join(DATA_NSE, "membership_pit.parquet"))
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--tag", default="surface_pit")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild the JSON from the saved parquet, no backtests")
    args = ap.parse_args()
    if args.summary_only:
        summarise(pd.read_parquet(f"{OUT_NSE}/surface_{args.tag}.parquet"), args)
        return

    market = load_market(args.panel, args.calendar)
    dates = [d for d in market.calendar
             if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    elig = build_universe(market, dates, args.membership, args.lookback)
    bench_ew = equal_weight_benchmark(market, elig)
    print(f"{len(dates)} sessions | universe mean {elig.sum(axis=1).mean():.1f} names | "
          f"{len(MOM_H) * len(REV_H) * len(WEIGHTS) * len(REBAL)} cells", flush=True)

    mom = {h: rank_pct(momentum_scores(market, dates, h).where(elig)) for h in MOM_H}
    rev = {h: rank_pct((-momentum_scores(market, dates, h)).where(elig)) for h in REV_H}

    rows = []
    for mh in MOM_H:
        for rh in REV_H:
            for w in WEIGHTS:
                sc = (w * mom[mh] + (1 - w) * rev[rh]).where(elig)
                for rname, r in REBAL.items():
                    res = run_backtest(sc, market, topk=args.topk, n_drop=args.n_drop,
                                       hold_thresh=args.hold_thresh, rebalance=r,
                                       cost_bps_round_trip=args.cost_bps)
                    m = metrics(res, bench_ew, seed=args.seed)
                    d = res["daily"]
                    b = bench_ew.reindex(d.index).fillna(0.0)
                    exc = d["net_return"] - b
                    yrs = {}
                    for yr, g in exc.groupby(exc.index.year):
                        yrs[int(yr)] = float((1 + d["net_return"].loc[g.index]).prod()
                                             - (1 + b.loc[g.index]).prod())
                    rows.append({"mom_h": mh, "rev_h": rh, "w_mom": w, "rebalance": rname,
                                 "ann_excess_vs_ew200": m["ann_excess_net"],
                                 "ir_vs_ew200": m["information_ratio"],
                                 "turnover_ann": m["turnover_ann"],
                                 "positive_years": sum(1 for v in yrs.values() if v > 0),
                                 "n_years": len(yrs)})
        print(f"  mom_h={mh} done ({len(rows)} cells)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(f"{OUT_NSE}/surface_{args.tag}.parquet", index=False)
    summarise(df, args)


def summarise(df: pd.DataFrame, args) -> dict:
    win = df[(df.mom_h == 20) & (df.rev_h == 5) & (df.w_mom == 0.5)
             & (df.rebalance == "monthly")].iloc[0]
    # the neighbourhood: one grid step away in any single dimension, same rebalance
    mi, ri, wi = MOM_H.index(20), REV_H.index(5), WEIGHTS.index(0.5)
    nb_keys = ([(MOM_H[i], 5, 0.5) for i in (mi - 1, mi + 1)]
               + [(20, REV_H[i], 0.5) for i in (ri - 1, ri + 1)]
               + [(20, 5, WEIGHTS[i]) for i in (wi - 1, wi + 1)])
    nb = df[df.rebalance == "monthly"].set_index(["mom_h", "rev_h", "w_mom"])
    neigh = [float(nb.loc[k, "ann_excess_vs_ew200"]) for k in nb_keys if k in nb.index]

    blends = df[(df.w_mom > 0) & (df.w_mom < 1)]
    summary = {
        "n_cells": int(len(df)),
        "winner": {k: (float(win[k]) if k != "rebalance" else win[k])
                   for k in ("mom_h", "rev_h", "w_mom", "rebalance", "ann_excess_vs_ew200",
                             "ir_vs_ew200", "positive_years", "n_years")},
        "winner_rank_of_all_cells": int((df.ann_excess_vs_ew200
                                         > win.ann_excess_vs_ew200).sum() + 1),
        "all_cells": {"share_positive": float((df.ann_excess_vs_ew200 > 0).mean()),
                      "median_excess": float(df.ann_excess_vs_ew200.median()),
                      "p25_excess": float(df.ann_excess_vs_ew200.quantile(.25)),
                      "p75_excess": float(df.ann_excess_vs_ew200.quantile(.75)),
                      "max_excess": float(df.ann_excess_vs_ew200.max())},
        "monthly_blends_only": {
            "n": int(len(blends[blends.rebalance == "monthly"])),
            "share_positive": float((blends[blends.rebalance == "monthly"]
                                     .ann_excess_vs_ew200 > 0).mean()),
            "median_excess": float(blends[blends.rebalance == "monthly"]
                                   .ann_excess_vs_ew200.median())},
        "immediate_neighbours": {
            "n": len(neigh), "values": neigh,
            "share_positive": float(np.mean([v > 0 for v in neigh])),
            "median": float(np.median(neigh)),
            "winner_minus_neighbour_median": float(win.ann_excess_vs_ew200 - np.median(neigh))},
        # is the blend better than its own parents at every (mom_h, rev_h, rebalance)?
        "blend_beats_both_parents_share": None,
    }
    beats = []
    for (mh, rh, rname), g in df.groupby(["mom_h", "rev_h", "rebalance"]):
        gi = g.set_index("w_mom")["ann_excess_vs_ew200"]
        if 1.0 in gi.index and 0.0 in gi.index:
            mid = gi.loc[[w for w in WEIGHTS if 0 < w < 1]]
            beats.append(bool(mid.max() > max(gi.loc[1.0], gi.loc[0.0])))
    summary["blend_beats_both_parents_share"] = float(np.mean(beats)) if beats else None
    summary["blend_beats_both_parents_n"] = len(beats)

    # ---- how smoothly does each pure arm vary with its own lookback? ----------------------
    # A real effect decays monotonically as the horizon moves away from where it lives; a fitted
    # cell is a spike whose neighbours at the SAME frequency do not work. Spearman rho of excess
    # against horizon separates the two, and the sign of the neighbours settles it.
    smooth = {}
    for rname in REBAL:
        g = df[df.rebalance == rname]
        blk = {}
        for label, w, horizons, col in (("momentum", 1.0, MOM_H, "mom_h"),
                                        ("reversal", 0.0, REV_H, "rev_h")):
            ser = g[g.w_mom == w].groupby(col).ann_excess_vs_ew200.first().reindex(horizons)
            rho = sps.spearmanr(np.asarray(ser.index, float), ser.to_numpy())
            best_h = ser.idxmax()
            others = ser.drop(best_h)
            blk[label] = {
                "by_horizon": {int(k): float(v) for k, v in ser.items()},
                "spearman_rho_vs_horizon": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
                "range": float(ser.max() - ser.min()),
                "share_positive": float((ser > 0).mean()),
                "best_horizon": int(best_h),
                "best_excess": float(ser.max()),
                "other_horizons_share_positive": float((others > 0).mean()),
                "other_horizons_median": float(others.median()),
                "is_isolated_spike": bool((others > 0).mean() == 0.0),
            }
        smooth[rname] = blk
    summary["horizon_smoothness"] = smooth
    print("\n-- pure-arm excess vs its own lookback (same frequency) --")
    for rname, blk in smooth.items():
        for label, b in blk.items():
            flag = "ISOLATED SPIKE" if b["is_isolated_spike"] else (
                "plateau" if b["share_positive"] >= 0.8 else "mixed")
            print(f"  {rname:9s} {label:8s} best h={b['best_horizon']:<3d} "
                  f"{b['best_excess']:+.4f} | others {b['other_horizons_share_positive']:5.0%} "
                  f"positive, median {b['other_horizons_median']:+.4f} | rho "
                  f"{b['spearman_rho_vs_horizon']:+.2f} -> {flag}")

    path = f"{OUT_NSE}/surface_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    sidecar(path.replace(".json", "_meta.json"), vars(args))

    print(f"\nwinner  mom20/rev5/w0.5 monthly: {win.ann_excess_vs_ew200:+.4f} "
          f"IR {win.ir_vs_ew200:+.2f}  ranked "
          f"{summary['winner_rank_of_all_cells']}/{len(df)} of all cells")
    print(f"all cells      : {summary['all_cells']['share_positive']:.1%} positive, "
          f"median {summary['all_cells']['median_excess']:+.4f}")
    print(f"monthly blends : {summary['monthly_blends_only']['share_positive']:.1%} positive, "
          f"median {summary['monthly_blends_only']['median_excess']:+.4f}")
    print(f"6 neighbours   : {summary['immediate_neighbours']['share_positive']:.0%} positive, "
          f"median {summary['immediate_neighbours']['median']:+.4f}, "
          f"winner is {summary['immediate_neighbours']['winner_minus_neighbour_median']:+.4f} "
          f"above them")
    print(f"blend > both parents in "
          f"{summary['blend_beats_both_parents_share']:.1%} of "
          f"{summary['blend_beats_both_parents_n']} (mom_h, rev_h, freq) triples")
    print(f"\nwrote {path}")
    return summary


if __name__ == "__main__":
    main()
