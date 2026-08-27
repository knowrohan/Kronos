"""Controls and information-coefficient analysis. All run off CACHED signals, so seconds each.

The repo has NO IC code anywhere -- verified, zero matches for
``rank_ic|information_coefficient|spearman|pearson`` across finetune/ and model/ -- so the
standard errors below are written from scratch and are the part most worth scrutinising.

Daily ICs computed against an h-session forward return OVERLAP, so they are autocorrelated by
construction and a naive t-test overstates significance by roughly sqrt(h). Two corrections
are reported and must agree before any claim is made:
  * Newey-West with lag = h
  * moving-block bootstrap, block = 21 sessions

CONTROLS
  1 random      scores shuffled within each date, N seeds -> a DISTRIBUTION
  2 equalweight buy-and-hold every eligible name
  3 reversed    negated scores
  4 ic_horizon  IC at 1 / 5 / 10 / 20 sessions
  5 momentum    trailing 10-session return -- costs nothing, Kronos costs ~4 GPU-hours
  6 anchor      -(close_t - mean_90)/std_90, the mechanical reversal factor that an
                autoregressive sampler regressing to the window mean reproduces for free
  7 sanity      perfect foresight (must win big) and constant signal (must be ~flat, ~0 turnover)
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
from bench.nse.backtest import (Market, TRADING_DAYS, annualise, bench_series,  # noqa: E402
                                block_bootstrap_ci, load_market, metrics, run_backtest)
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402

HORIZONS = [1, 5, 10, 20]


# ----------------------------------------------------------------- standard errors


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """HAC standard error of the mean of an autocorrelated series (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    d = x - x.mean()
    g0 = float((d * d).sum() / n)
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float((d[k:] * d[:-k]).sum() / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return float(np.sqrt(max(var, 0.0) / n))


def summarise_ic(ics: np.ndarray, lag: int, seed=0) -> dict:
    ics = np.asarray(ics, dtype=float)
    ics = ics[np.isfinite(ics)]
    if len(ics) == 0:
        return {"n_dates": 0}
    m = float(ics.mean())
    se_nw = newey_west_se(ics, lag)
    se_naive = float(ics.std(ddof=1) / np.sqrt(len(ics)))
    lo, hi = block_bootstrap_ci(ics, lambda a, axis: a.mean(axis=axis), seed=seed)
    return {
        "n_dates": int(len(ics)), "mean_ic": m, "std_ic": float(ics.std(ddof=1)),
        "se_newey_west": se_nw, "t_newey_west": m / se_nw if se_nw > 0 else float("nan"),
        "se_naive": se_naive, "t_naive": m / se_naive if se_naive > 0 else float("nan"),
        "bootstrap_ci95": [lo, hi], "share_dates_positive": float((ics > 0).mean()),
        "icir_ann": float(m / ics.std(ddof=1) * np.sqrt(TRADING_DAYS / max(lag, 1)))
        if ics.std(ddof=1) > 0 else float("nan"),
        "significant": bool(se_nw > 0 and abs(m / se_nw) > 2 and (lo > 0 or hi < 0)),
    }


# ----------------------------------------------------------------- IC


def compute_ic(scores_wide: pd.DataFrame, market: Market, horizon: int,
               price="open", lag=1, min_names=20) -> pd.Series:
    """Per-date Spearman rank IC of score vs realized forward return."""
    out = {}
    for d, row in scores_wide.iterrows():
        si = market.idx.get(d)
        if si is None:
            continue
        fwd = market.forward_return(si, horizon, price=price, lag=lag)
        if not len(fwd):
            continue
        s = row.dropna()
        common = s.index.intersection(fwd.dropna().index)
        if len(common) < min_names:
            continue
        out[d] = sps.spearmanr(s[common].to_numpy(), fwd[common].to_numpy()).statistic
    return pd.Series(out, dtype=float)


# ----------------------------------------------------------------- control signals


def random_scores(scores_wide: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle scores WITHIN each date -- preserves the eligible universe exactly."""
    rng = np.random.default_rng(seed)
    a = scores_wide.to_numpy(copy=True)
    for i in range(a.shape[0]):
        m = np.isfinite(a[i])
        v = a[i, m]
        rng.shuffle(v)
        a[i, m] = v
    return pd.DataFrame(a, index=scores_wide.index, columns=scores_wide.columns)


def momentum_scores(market: Market, dates, horizon=10) -> pd.DataFrame:
    """Trailing `horizon`-session close-to-close return, known at t's close."""
    c = market.closes
    mom = c / c.shift(horizon) - 1.0
    return mom.reindex(dates)


def anchor_scores(signals: pd.DataFrame, negate=True) -> pd.DataFrame:
    """-(close_t - mean_90)/std_90, taken straight from the cached anchor_z column."""
    w = signals.pivot(index="date", columns="symbol", values="anchor_z")
    return -w if negate else w


def perfect_foresight_scores(market: Market, dates, horizon=1) -> pd.DataFrame:
    """The return the strategy is about to earn. Sanity only -- it must win enormously."""
    o = market.opens
    fwd = o.shift(-(1 + horizon)) / o.shift(-1) - 1.0
    return fwd.reindex(dates)


def constant_scores(scores_wide: pd.DataFrame) -> pd.DataFrame:
    """Same score for every eligible name. Must give ~zero turnover after the first fill."""
    a = np.where(np.isfinite(scores_wide.to_numpy()), 1.0, np.nan)
    return pd.DataFrame(a, index=scores_wide.index, columns=scores_wide.columns)


def eligible_mask_daily(market: Market, mask: pd.DataFrame | None) -> pd.DataFrame:
    """Forward-fill a per-signal-date eligibility mask onto every session.

    The strategy's universe is the SCORED universe (>= lookback bars ending at t), which is a
    few names smaller than `market.tradable` because of mid-window IPOs. Any equal-weight
    comparison has to use the same set or it is not apples-to-apples.
    """
    if mask is None:
        return market.tradable
    m = (mask.astype(float)
         .reindex(index=market.calendar, columns=market.tradable.columns)
         .ffill().fillna(0.0) > 0.5)
    return m & market.tradable


def equal_weight_benchmark(market: Market, mask: pd.DataFrame | None = None) -> pd.Series:
    """Daily open-to-open return of an equal-weighted portfolio of the eligible universe."""
    el = eligible_mask_daily(market, mask)
    return market.oo_ret.where(el).mean(axis=1)


def equal_weight_universe(market: Market, dates, rebalance=5, mask=None) -> dict:
    """Buy-and-hold every eligible name, same execution lag and rebalance cadence."""
    el = eligible_mask_daily(market, mask)
    sig = [d for d in dates if d in market.idx][::rebalance]
    rows = []
    prev = None
    for k, d in enumerate(sig):
        si = market.idx[d]
        ei = si + 1
        if ei >= len(market.calendar) - 1:
            break
        end = market.idx[sig[k + 1]] + 1 if k + 1 < len(sig) else ei + 1
        for s in range(ei, min(end, len(market.calendar) - 1)):
            tr = el.iloc[s]
            names = tr[tr].index
            r = market.oo_ret.iloc[s].reindex(names).dropna()
            rows.append({"date": market.calendar[s], "gross_return": float(r.mean()),
                         "cost": 0.0, "net_return": float(r.mean()), "turnover": 0.0,
                         "n_held": len(r), "n_buy": 0, "n_sell": 0})
    df = pd.DataFrame(rows).drop_duplicates(subset="date").set_index("date")
    return {"daily": df, "rebalances": pd.DataFrame([{"turnover": 0.0}]), "one_way_cost": 0.0}


# ----------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default=os.path.join(OUT_NSE, "signals_zeroshot.parquet"))
    ap.add_argument("--column", default="last_ret")
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--benchmark", default="cnx200")
    ap.add_argument("--skip-backtests", action="store_true", help="IC only (fast G0 gate)")
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
    sig = pd.concat([pd.read_parquet(p) for p in args.signals.split(",")], ignore_index=True)
    dates = sorted(sig["date"].unique())
    scores = sig.pivot(index="date", columns="symbol", values=args.column).sort_index()
    pit = None
    if args.membership:
        mem = pd.read_parquet(args.membership)
        mem["date"] = pd.to_datetime(mem["date"])
        pit = (mem.assign(v=True).pivot(index="date", columns="symbol", values="v")
               .reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool))
        scores = scores.where(pit)
        print(f"point-in-time mask: {int(pit.sum().sum())} member cells", flush=True)
    bench = bench_series(market, args.benchmark, args.bench_path)
    out: dict = {"column": args.column, "n_dates": len(dates),
                 "date_range": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())]}

    score_cols = [c for c in sig.columns
                  if c.split("_")[0] in ("last", "mean", "max", "min")]

    # ---------------- 4 IC by horizon, for every arm --------------------------
    print("computing IC grid ...", flush=True)
    ic_grid = {}
    for col in score_cols:
        w = sig.pivot(index="date", columns="symbol", values=col).sort_index()
        if pit is not None:
            w = w.where(pit.reindex(index=w.index, columns=w.columns).fillna(False))
        per_col = {}
        for h in HORIZONS:
            ics = compute_ic(w, market, h, price="open", lag=1)
            per_col[f"open_h{h}"] = summarise_ic(ics.to_numpy(), lag=h, seed=args.seed)
            per_col[f"open_h{h}"]["ic_series_mean_by_year"] = {
                str(y): float(v) for y, v in ics.groupby(ics.index.year).mean().items()}
        ics_c = compute_ic(w, market, 10, price="close", lag=0)
        per_col["close_h10_model_target"] = summarise_ic(ics_c.to_numpy(), lag=10, seed=args.seed)
        ic_grid[col] = per_col
        p = per_col["open_h10"]
        print(f"  {col:14s} IC10 {p.get('mean_ic', float('nan')):+.4f} "
              f"t_NW {p.get('t_newey_west', float('nan')):+.2f} "
              f"CI [{p['bootstrap_ci95'][0]:+.4f},{p['bootstrap_ci95'][1]:+.4f}] "
              f"pos {p.get('share_dates_positive', float('nan')):.2f}", flush=True)
    out["ic_grid"] = ic_grid

    # ---------------- 5, 6 cost-free baselines --------------------------------
    print("baselines ...", flush=True)
    base_signals = {
        # Short-horizon reversal is the one free signal that tested significant in the pilot,
        # so it gets the same treatment as Kronos rather than a footnote.
        "reversal_1d": -momentum_scores(market, scores.index, 1),
        "reversal_5d": -momentum_scores(market, scores.index, 5),
        "reversal_10d": -momentum_scores(market, scores.index, 10),
        "momentum_10d": momentum_scores(market, scores.index, 10),
        "momentum_20d": momentum_scores(market, scores.index, 20),
        "momentum_60d": momentum_scores(market, scores.index, 60),
        "anchor_reversal": anchor_scores(sig, negate=True).reindex(scores.index),
        "anchor_momentum": anchor_scores(sig, negate=False).reindex(scores.index),
    }
    # baselines are only comparable on the exact same eligible cross-section
    mask = scores.notna()
    for k in base_signals:
        base_signals[k] = base_signals[k].reindex(columns=scores.columns).where(mask)

    out["baseline_ic"] = {}
    for name, w in base_signals.items():
        per = {}
        for h in HORIZONS:
            per[f"open_h{h}"] = summarise_ic(compute_ic(w, market, h).to_numpy(), lag=h, seed=args.seed)
        out["baseline_ic"][name] = per
        p = per["open_h10"]
        print(f"  {name:18s} IC10 {p.get('mean_ic', float('nan')):+.4f} "
              f"t_NW {p.get('t_newey_west', float('nan')):+.2f}", flush=True)

    # correlation of the Kronos score with the mechanical baselines
    kw = scores
    corr = {}
    for name, w in base_signals.items():
        per_date = []
        for d in kw.index:
            a, b = kw.loc[d].dropna(), w.loc[d].dropna() if d in w.index else pd.Series(dtype=float)
            c = a.index.intersection(b.index)
            if len(c) > 20:
                per_date.append(sps.spearmanr(a[c].to_numpy(), b[c].to_numpy()).statistic)
        corr[name] = float(np.nanmean(per_date)) if per_date else float("nan")
    out["cross_sectional_rank_corr_with_kronos"] = corr
    print("  rank-corr with Kronos:", {k: round(v, 3) for k, v in corr.items()}, flush=True)

    # ---------------- incremental content: IC of Kronos ORTHOGONAL to the free factors ----
    # Within each date, regress the Kronos rank-score on the baseline rank-score(s) and keep
    # the residual. If the residual carries no IC, Kronos adds nothing a one-line feature
    # does not already provide -- gate G2b.
    def _rank(s):
        return (s.rank() - (len(s) + 1) / 2) / max(len(s), 1)

    out["orthogonalised_ic"] = {}
    for basis_name, basis in (("anchor_reversal", ["anchor_reversal"]),
                              ("momentum_10d", ["momentum_10d"]),
                              ("anchor+momentum", ["anchor_reversal", "momentum_10d"])):
        resid = {}
        for d in kw.index:
            y = kw.loc[d].dropna()
            X = []
            for b in basis:
                w = base_signals[b]
                if d not in w.index:
                    X = None
                    break
                X.append(w.loc[d])
            if X is None:
                continue
            cols = [x.dropna() for x in X]
            common = y.index
            for c in cols:
                common = common.intersection(c.index)
            if len(common) < 30:
                continue
            yv = _rank(y[common]).to_numpy()
            Xv = np.column_stack([_rank(c[common]).to_numpy() for c in cols]
                                 + [np.ones(len(common))])
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            resid[d] = pd.Series(yv - Xv @ beta, index=common)
        rw = pd.DataFrame(resid).T.reindex(columns=kw.columns)
        per = {}
        for h in HORIZONS:
            per[f"open_h{h}"] = summarise_ic(compute_ic(rw, market, h).to_numpy(), lag=h, seed=args.seed)
        out["orthogonalised_ic"][f"kronos_resid_of_{basis_name}"] = per
        pp = per["open_h10"]
        print(f"  kronos residual of {basis_name:16s} IC10 {pp.get('mean_ic', float('nan')):+.4f} "
              f"t_NW {pp.get('t_newey_west', float('nan')):+.2f}", flush=True)

    if not args.skip_backtests:
        bt = dict(topk=args.topk, n_drop=args.n_drop, hold_thresh=args.hold_thresh,
                  rebalance=args.rebalance, cost_bps_round_trip=args.cost_bps)

        def run(w, **kw2):
            return metrics(run_backtest(w, market, **{**bt, **kw2}), bench, seed=args.seed)

        print("strategy + deterministic controls ...", flush=True)
        out["strategy"] = run(scores)
        out["reversed"] = run(-scores)
        out["constant"] = run(constant_scores(scores))
        out["perfect_foresight"] = run(perfect_foresight_scores(market, scores.index, 1))
        out["equal_weight_universe"] = metrics(
            equal_weight_universe(market, scores.index, args.rebalance, mask=mask),
            bench, seed=args.seed)
        for name, w in base_signals.items():
            out[f"baseline_{name}"] = run(w)
        for k in ("strategy", "reversed", "constant", "perfect_foresight",
                  "equal_weight_universe", "baseline_momentum_10d", "baseline_anchor_reversal"):
            m = out[k]
            print(f"  {k:26s} excess {m['ann_excess_net'] * 100:+7.2f}%/yr  "
                  f"IR {m['information_ratio']:+5.2f}  turnover {m['turnover_ann']:6.2f}x  "
                  f"cost {m['cost_drag_ann_pct']:5.2f}%/yr", flush=True)

        print(f"random-signal distribution, {args.n_random} seeds ...", flush=True)
        rnd = []
        for s in range(args.n_random):
            m = run(random_scores(scores, s))
            rnd.append({"seed": s, "ann_excess_net": m["ann_excess_net"],
                        "ann_return_net": m["ann_return_net"],
                        "information_ratio": m["information_ratio"],
                        "turnover_ann": m["turnover_ann"]})
            if (s + 1) % 50 == 0:
                print(f"    {s + 1}/{args.n_random}", flush=True)
        rd = pd.DataFrame(rnd)
        ex = rd["ann_excess_net"].to_numpy()
        strat_ex = out["strategy"]["ann_excess_net"]
        out["random_distribution"] = {
            "n_seeds": args.n_random,
            "mean": float(ex.mean()), "std": float(ex.std(ddof=1)),
            "pct": {str(q): float(np.percentile(ex, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "strategy_excess": strat_ex,
            "strategy_percentile_in_random": float((ex < strat_ex).mean() * 100),
            "empirical_p_one_sided": float((ex >= strat_ex).mean()),
            "mean_turnover_ann": float(rd["turnover_ann"].mean()),
        }
        rd.to_parquet(os.path.join(OUT_NSE, f"random_dist_{args.tag}.parquet"), index=False)
        r = out["random_distribution"]
        print(f"  strategy {strat_ex * 100:+.2f}%/yr sits at pct {r['strategy_percentile_in_random']:.1f} "
              f"of random (mean {r['mean'] * 100:+.2f}%, p95 {r['pct']['95'] * 100:+.2f}%), "
              f"one-sided p={r['empirical_p_one_sided']:.3f}", flush=True)

    path = os.path.join(OUT_NSE, f"controls_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    sidecar(path.replace(".json", "_meta.json"), vars(args))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
