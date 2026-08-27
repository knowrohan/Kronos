"""Does Kronos forecast anything at all here? Try volatility instead of direction.

The directional result in this study is a null with a known mechanism: per-window z-scoring
makes the predicted close a distance-from-own-mean, so ranking on it ranks on trailing z-score.
That argument does not apply to the *spread* of the predicted path. `max_z - min_z` is the range
of the sampled future in units of the same trailing standard deviation that contaminated the
level, and the contamination cancels in the difference: both ends are measured from the same
anchor. Multiplying by sigma/close puts it in percent-of-price, which is directly comparable
across names.

So there are three untested quantities sitting in the signal file at zero extra GPU cost:

    pred_range_rel = max_z - min_z                      predicted range / trailing vol
    pred_range_pct = (max_z - min_z) * sigma / close    predicted range, % of price
    pred_skew      = max_z + min_z - 2 * last_z         path asymmetry around the endpoint

Volatility is far more predictable than direction, so the interesting question is not whether
these have an IC -- they will -- but whether they beat the trailing volatility that any
practitioner would use for free. That is an orthogonalised test, and it is the one run here.
Direction is tested too, because a low-volatility tilt is a real premium in its own right.
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
                                newey_west_se, summarise_ic)
from bench.nse.robustness import build_universe  # noqa: E402

HORIZONS = [1, 5, 10, 20]
REBALANCES = {"weekly": 5, "biweekly": 10, "monthly": 21}
TRADING_DAYS = 252


def realized_vol_forward(market, dates, horizon: int) -> pd.DataFrame:
    """Realized std of open-to-open returns over the `horizon` sessions AFTER the trade.

    Same execution convention as everything else: the signal is known at t's close, the position
    is taken at t+1's open, so the realized window starts at t+1.
    """
    r = market.oo_ret
    fwd = r.rolling(horizon).std().shift(-(horizon + 1))
    return fwd.reindex(dates)


def realized_abs_forward(market, dates, horizon: int) -> pd.DataFrame:
    r = market.oo_ret
    fwd = (r.rolling(horizon).sum().abs()).shift(-(horizon + 1))
    return fwd.reindex(dates)


def trailing_vol(market, dates, window: int = 20) -> pd.DataFrame:
    return market.oo_ret.rolling(window).std().reindex(dates)


def ic_against(scores: pd.DataFrame, target: pd.DataFrame, lag: int, seed=0) -> dict:
    """Per-date Spearman of score vs an arbitrary realized target (not just forward return)."""
    common = scores.index.intersection(target.index)
    ics = []
    for d in common:
        a, b = scores.loc[d], target.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 20:
            continue
        ics.append(float(a[m].rank().corr(b[m].rank())))
    return summarise_ic(np.asarray(ics, float), lag=lag, seed=seed)


def orthogonalise(scores: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Per-date OLS of rank(score) on rank(control); return the residual as the new score.

    If Kronos's volatility forecast is just trailing volatility wearing a hat, this residual has
    no information and the orthogonalised IC collapses.
    """
    out = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    for d in scores.index:
        if d not in control.index:
            continue
        a, b = scores.loc[d], control.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 20:
            continue
        x = b[m].rank().to_numpy(float)
        y = a[m].rank().to_numpy(float)
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = y - y.mean()
        out.loc[d, m[m].index] = y - float((x * y).mean()) * x
    return out


def conditional_ic(scores: pd.DataFrame, target: pd.DataFrame, control: pd.DataFrame,
                   n_bucket: int = 5, lag: int = 10, seed: int = 0) -> dict:
    """Spearman of score vs target WITHIN buckets of the control, pooled across buckets.

    This exists because rank-orthogonalisation misbehaves when the score is a PRODUCT of the
    control and something else -- exactly the case here, since pred_range_pct is (predicted
    range in sigma units) x (sigma / close). The residual of a product on one of its factors is
    a nonlinear function of that factor, so an orthogonalised IC can come out larger than the
    raw IC, which is not a thing real incremental information can do. Conditioning instead of
    residualising has no such failure mode: inside a narrow band of trailing volatility, either
    the score still sorts realized volatility or it does not.
    """
    ics = []
    for d in scores.index:
        if d not in target.index or d not in control.index:
            continue
        a, b, c = scores.loc[d], target.loc[d], control.loc[d]
        m = a.notna() & b.notna() & c.notna()
        if m.sum() < 5 * n_bucket:
            continue
        cr = c[m].rank(pct=True)
        for q in range(n_bucket):
            sel = cr[(cr > q / n_bucket) & (cr <= (q + 1) / n_bucket)].index
            if len(sel) < 10:
                continue
            x, y = a[sel], b[sel]
            ics.append(float(x.rank().corr(y.rank())))
    return summarise_ic(np.asarray(ics, float), lag=lag, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=os.path.join(OUT_NSE, "signals_zeroshot.parquet"),
                    help="one path, or several comma-separated and concatenated")
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price.parquet"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--membership", default=None)
    ap.add_argument("--tag", default="vol_zeroshot")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # comma-separated, matching controls.py/grid.py: the point-in-time universe needs the
    # base panel's signals plus the extra names that have since left the index
    sg = pd.concat([pd.read_parquet(p) for p in args.signals.split(",")], ignore_index=True)
    sg["date"] = pd.to_datetime(sg["date"])
    market = load_market(args.panel, args.calendar)
    dates = sorted(d for d in sg["date"].unique() if pd.Timestamp(d) in market.idx)
    dates = [pd.Timestamp(d) for d in dates]
    elig = build_universe(market, dates, args.membership, args.lookback)
    print(f"{len(dates)} signal dates {dates[0].date()} -> {dates[-1].date()}", flush=True)

    def wide(col):
        return sg.pivot(index="date", columns="symbol", values=col).reindex(dates)

    mx, mn, lz = wide("max_z"), wide("min_z"), wide("last_z")
    sig_px = wide("sigma_close") / wide("close_t")
    sig = {
        "pred_range_rel": (mx - mn),
        "pred_range_pct": (mx - mn) * sig_px,
        "pred_skew": (mx + mn - 2 * lz),
        "trailing_vol_20d": trailing_vol(market, dates, 20),
        "trailing_sigma_pct": sig_px,          # the 90-bar sigma the model itself normalised by
    }
    sig = {k: v.reindex(columns=elig.columns).where(elig) for k, v in sig.items()}

    out = {"config": vars(args), "n_dates": len(dates),
           "date_range": [str(dates[0].date()), str(dates[-1].date())]}

    # ---- 1. do these forecast realized volatility at all? -------------------------------
    print("\nIC vs REALIZED VOLATILITY (Spearman, Newey-West t):", flush=True)
    vol_ic = {}
    for name in ("pred_range_rel", "pred_range_pct", "trailing_vol_20d", "trailing_sigma_pct"):
        per = {}
        for h in (5, 10, 20):
            tgt = realized_vol_forward(market, dates, h).where(elig)
            per[f"h{h}"] = ic_against(sig[name], tgt, lag=h, seed=args.seed)
        vol_ic[name] = per
        p = per["h10"]
        print(f"  {name:20s} h10 IC {p['mean_ic']:+.4f} (t {p['t_newey_west']:+6.2f}"
              f"{'*' if p['significant'] else ' '})", flush=True)
    out["vol_ic"] = vol_ic

    # ---- 2. does Kronos beat free trailing volatility? -----------------------------------
    print("\nORTHOGONALISED vs trailing vol -- the test that matters:", flush=True)
    orth = {}
    for name in ("pred_range_rel", "pred_range_pct"):
        for ctrl in ("trailing_vol_20d", "trailing_sigma_pct"):
            resid = orthogonalise(sig[name], sig[ctrl])
            per = {}
            for h in (5, 10, 20):
                tgt = realized_vol_forward(market, dates, h).where(elig)
                per[f"h{h}"] = ic_against(resid, tgt, lag=h, seed=args.seed)
            orth[f"{name}|{ctrl}"] = per
            p = per["h10"]
            raw = vol_ic[name]["h10"]["mean_ic"]
            print(f"  {name:16s} - {ctrl:20s} h10 IC {raw:+.4f} -> {p['mean_ic']:+.4f} "
                  f"(t {p['t_newey_west']:+6.2f}{'*' if p['significant'] else ' '})"
                  f"   retained {100 * p['mean_ic'] / raw if raw else 0:5.1f}%", flush=True)
    out["orthogonalised_vol_ic"] = orth

    # the reverse direction: does trailing vol survive orthogonalising against Kronos?
    rev = {}
    for ctrl in ("pred_range_rel", "pred_range_pct"):
        resid = orthogonalise(sig["trailing_vol_20d"], sig[ctrl])
        tgt = realized_vol_forward(market, dates, 10).where(elig)
        rev[f"trailing_vol_20d|{ctrl}"] = ic_against(resid, tgt, lag=10, seed=args.seed)
        p = rev[f"trailing_vol_20d|{ctrl}"]
        print(f"  {'trailing_vol_20d':16s} - {ctrl:20s} h10 IC "
              f"{vol_ic['trailing_vol_20d']['h10']['mean_ic']:+.4f} -> {p['mean_ic']:+.4f} "
              f"(t {p['t_newey_west']:+6.2f}{'*' if p['significant'] else ' '})", flush=True)
    out["reverse_orthogonalised"] = rev

    # ---- 2b. the same question without the product artefact ------------------------------
    print("\nCONDITIONAL on trailing-vol quintile (immune to the product artefact):",
          flush=True)
    cond = {}
    tgt10 = realized_vol_forward(market, dates, 10).where(elig)
    for name in ("pred_range_rel", "pred_range_pct", "pred_skew"):
        for ctrl in ("trailing_vol_20d", "trailing_sigma_pct"):
            c = conditional_ic(sig[name], tgt10, sig[ctrl], lag=10, seed=args.seed)
            cond[f"{name}|{ctrl}"] = c
            print(f"  {name:16s} within {ctrl:20s} IC {c['mean_ic']:+.4f} "
                  f"(t {c['t_newey_west']:+6.2f}{'*' if c['significant'] else ' '})", flush=True)
    # and the control's own within-bucket IC, as the scale to read those against
    for ctrl in ("trailing_vol_20d",):
        c = conditional_ic(sig["trailing_sigma_pct"], tgt10, sig[ctrl], lag=10, seed=args.seed)
        cond[f"trailing_sigma_pct|{ctrl}"] = c
        print(f"  {'trailing_sigma_pct':16s} within {ctrl:20s} IC {c['mean_ic']:+.4f} "
              f"(t {c['t_newey_west']:+6.2f}{'*' if c['significant'] else ' '})"
              f"   <- a free control, for scale", flush=True)
    out["conditional_ic"] = cond

    # ---- 3. directional: does any of it predict RETURNS? ---------------------------------
    print("\nIC vs FORWARD RETURN (is there a low-vol premium to harvest?):", flush=True)
    ret_ic = {}
    for name, w in sig.items():
        per = {}
        for h in HORIZONS:
            per[f"h{h}"] = summarise_ic(compute_ic(w, market, h).to_numpy(), lag=h,
                                        seed=args.seed)
        ret_ic[name] = per
        p = per["h10"]
        print(f"  {name:20s} h10 IC {p['mean_ic']:+.4f} (t {p['t_newey_west']:+6.2f}"
              f"{'*' if p['significant'] else ' '})", flush=True)
    out["return_ic"] = ret_ic

    # ---- 4. backtest the low-vol tilt ----------------------------------------------------
    print("\nBACKTEST (long the lowest predicted volatility):", flush=True)
    bench_ew = equal_weight_benchmark(market, elig)
    bench_cap = bench_series(market, "cnx200")
    rows = []
    for name in ("pred_range_rel", "pred_range_pct", "trailing_vol_20d", "pred_skew"):
        for sgn, lab in ((-1.0, "low"), (1.0, "high")):
            for rname, r in REBALANCES.items():
                try:
                    res = run_backtest(sgn * sig[name], market, topk=args.topk,
                                       n_drop=args.n_drop, hold_thresh=args.hold_thresh,
                                       rebalance=r, cost_bps_round_trip=args.cost_bps)
                except ValueError:
                    continue
                m = metrics(res, bench_cap, seed=args.seed)
                m_ew = metrics(res, bench_ew, seed=args.seed)
                rows.append({"arm": f"{lab}_{name}", "rebalance": rname,
                             "ann_return_net": m["ann_return_net"],
                             "ann_excess_vs_cnx200": m["ann_excess_net"],
                             "ann_excess_vs_ew200": m_ew["ann_excess_net"],
                             "ir_vs_ew200": m_ew["information_ratio"],
                             "vol_ann": m["vol_ann"], "sharpe_net": m["sharpe_net"],
                             "max_drawdown": m["max_drawdown"],
                             "turnover_ann": m["turnover_ann"]})
    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt = bt.sort_values("ann_excess_vs_ew200", ascending=False)
        bt.to_parquet(f"{OUT_NSE}/vol_backtest_{args.tag}.parquet", index=False)
        print(bt.head(10).to_string(index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
        out["backtest"] = rows

    path = f"{OUT_NSE}/{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    sidecar(path.replace(".json", "_meta.json"), vars(args))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
