"""Post-hoc robustness for the arms that survived the point-in-time universe.

`baseline_sweep.py` searched 17 arms x 4 rebalance frequencies = 68 cells and reported the best
ones. That is a search, and a search needs four things before its winners mean anything:

  A. a FAMILY-WISE null -- the distribution of the *maximum* statistic over 68 cells when none
     of them has any information. The per-cell p95 that `baseline_sweep` reports is the wrong
     bar for a number that was picked as the best of 68.
  B. sub-period stability -- the test window is three years. An edge that lives in one of them
     is not an edge.
  C. a capacity curve -- the sweep charges a flat 30 bps. The real Indian cost function is not
     scale-invariant (Rs 13 DP charge, Rs 20 brokerage floors), so the same strategy earns a
     different number at Rs 1 lakh and at Rs 10 crore. This is the number that decides whether
     the finding is worth acting on.
  D. blends -- the two survivors trade on near-orthogonal horizons. Do they add?
  E. a liquidity ceiling -- the statutory cost model has no impact term, so it says bigger is
     always cheaper, which is false. Sizing every fill against 20-session median traded value
     gives the participation rate, and a square-root impact charge on top of it gives the
     capital at which the edge is actually consumed.

Nothing here uses Kronos or a GPU; it runs on the price panel alone at full daily resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import (annualise, bench_series, load_market,  # noqa: E402
                                max_drawdown, metrics, run_backtest)
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import (equal_weight_benchmark, momentum_scores,  # noqa: E402
                                random_scores)
from bench.nse.costs import PRESETS  # noqa: E402

REBALANCES = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}
# Rs 1 lakh to Rs 10,000 crore. The top of the range is well past any retail book; it is there
# because the NIFTY 200 median 20-session traded value is ~Rs 127 crore/day, so the statutory
# cost curve alone never bites and only market impact can produce a ceiling.
CAPITALS = [1e5, 3e5, 1e6, 1e7, 1e8, 1e9, 3e9, 1e10, 3e10, 1e11]
REF_CAPITAL = 1e8          # Rs 10 crore -- the reference column printed to the console
TRADING_DAYS = 252

# the arms baseline_sweep.py searched over -- the family the winners were selected from
ARM_NAMES = ([f"momentum_{h}d" for h in (1, 2, 5, 10, 20, 60, 120)]
             + [f"reversal_{h}d" for h in (1, 2, 5, 10, 20, 60, 120)]
             + ["zscore_reversal", "zscore_momentum", "kronos_emulator"]
             # the three blends are a SECOND round of search over the same data, so they belong
             # in the family whose maximum defines the bar -- otherwise the bar is too low
             + ["blend_mom20_rev5", "blend_mom20_mom120", "blend_all3"])

_G: dict = {}          # per-worker globals for the null draws


def zscore_window(market, dates, lookback=90):
    c = market.closes
    return ((c - c.rolling(lookback).mean()) / c.rolling(lookback).std()).reindex(dates)


def build_universe(market, dates, membership, lookback):
    elig = market.tradable.reindex(dates)
    if membership:
        mem = pd.read_parquet(membership)
        mem["date"] = pd.to_datetime(mem["date"])
        pit = (mem.assign(v=True).pivot(index="date", columns="symbol", values="v")
               .reindex(index=elig.index, columns=elig.columns).fillna(False).astype(bool))
        elig = elig & pit
    have = market.closes.notna().rolling(lookback).sum().reindex(dates) >= lookback
    return elig & have.fillna(False)


def build_arms(market, dates, elig, lookback):
    sig = {}
    for h in (1, 2, 5, 10, 20, 60, 120):
        m = momentum_scores(market, dates, h).where(elig)
        sig[f"momentum_{h}d"] = m
        sig[f"reversal_{h}d"] = (-m).where(elig)
    z = zscore_window(market, dates, lookback)
    sig["zscore_reversal"] = (-z).where(elig)
    sig["zscore_momentum"] = z.where(elig)
    # the fitted reproduction of the zero-shot score: slope -0.701 of anchor_z, pooled
    # R2 0.70 over all 756 dates (bench/out/nse/mechanism.json). The coefficient is a
    # negative scalar, so every rank, IC and backtest number is identical for any
    # negative value -- it is written out only so the label matches the measurement.
    sig["kronos_emulator"] = (-0.701 * z).where(elig)
    return sig


def rank_pct(w: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date -- puts arms on a common scale before blending."""
    return w.rank(axis=1, pct=True)


# --------------------------------------------------------------- A. family-wise null

def _init_worker(panel, calendar, membership, lookback, start, end, topk, n_drop,
                 hold_thresh, cost_bps):
    market = load_market(panel, calendar)
    dates = [d for d in market.calendar
             if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    elig = build_universe(market, dates, membership, lookback)
    # the shell carries the eligibility PATTERN and distinct values for random_scores to
    # permute. An all-NaN frame would leave nothing to shuffle (empty book, excess = -bench);
    # a constant frame would tie every score and collapse TopkDropout to its degenerate case.
    ramp = np.tile(np.arange(len(elig.columns), dtype=float), (len(dates), 1))
    shell = pd.DataFrame(ramp, index=pd.DatetimeIndex(dates), columns=elig.columns).where(elig)
    assert shell.notna().sum().sum() > 0, "empty eligibility shell"
    _G.update(market=market, shell=shell, bench_ew=equal_weight_benchmark(market, elig),
              topk=topk, n_drop=n_drop, hold_thresh=hold_thresh, cost_bps=cost_bps)


def _one_draw(seed: int) -> dict:
    """One family-wise draw: len(ARM_NAMES) independent random signals x 4 frequencies.

    Independent panels overstate the spread of the maximum relative to the real family, whose
    arms are heavily correlated (momentum_20d and momentum_60d share most of their ranking).
    That makes the resulting threshold CONSERVATIVE, which is the direction we want.
    """
    g = _G
    exc: dict[str, list[float]] = {k: [] for k in REBALANCES}
    irs: list[float] = []
    for a, _ in enumerate(ARM_NAMES):
        w = random_scores(g["shell"], seed * 1_000_003 + a)
        for rname, r in REBALANCES.items():
            res = run_backtest(w, g["market"], topk=g["topk"], n_drop=g["n_drop"],
                               hold_thresh=g["hold_thresh"], rebalance=r,
                               cost_bps_round_trip=g["cost_bps"])
            m = metrics(res, g["bench_ew"], seed=seed)
            exc[rname].append(m["ann_excess_net"])
            irs.append(m["information_ratio"])
    flat = np.array([v for vs in exc.values() for v in vs])
    out = {"seed": seed, "max_excess": float(flat.max()), "max_ir": float(np.max(irs)),
           # the max is the right bar for a CHERRY-PICKED cell. These two are the right bar for
           # a claim about a whole family -- "every momentum lookback works" is a statement about
           # the median and the share positive, and those have a much tighter null.
           "median_excess": float(np.median(flat)),
           "share_positive": float((flat > 0).mean())}
    for rname, vs in exc.items():
        out[f"max_excess_{rname}"] = float(np.max(vs))
        out[f"median_excess_{rname}"] = float(np.median(vs))
    return out


# --------------------------------------------------------------- B. sub-period stability

def yearly_breakdown(res, bench_ew) -> list[dict]:
    d = res["daily"]
    b = bench_ew.reindex(d.index).fillna(0.0)
    out = []
    for yr, idx in d.groupby(d.index.year).groups.items():
        sub, bb = d.loc[idx], b.loc[idx]
        exc = sub["net_return"] - bb
        out.append({
            "year": int(yr), "n_days": int(len(sub)),
            "net_return": float((1 + sub["net_return"]).prod() - 1),
            "bench_ew_return": float((1 + bb).prod() - 1),
            "excess": float((1 + sub["net_return"]).prod() - (1 + bb).prod()),
            "excess_ann": annualise(exc),
            "ir": float(exc.mean() / exc.std() * np.sqrt(TRADING_DAYS)) if exc.std() > 0 else 0.0,
            "max_drawdown": max_drawdown(sub["net_return"]),
        })
    return sorted(out, key=lambda r: r["year"])



# --------------------------------------------------------------- E. liquidity / impact

def adv_rupees(panel_path: str, calendar, window: int = 20) -> pd.DataFrame:
    """20-session rolling MEDIAN traded value in rupees, per symbol.

    Median rather than mean: Indian mid-caps have occasional block days that would flatter a
    mean-based capacity number by an order of magnitude.
    """
    p = pd.read_parquet(panel_path)
    p["date"] = pd.to_datetime(p["date"])
    p["turnover_rs"] = (p[["open", "high", "low", "close"]].mean(axis=1) * p["volume"])
    tv = p.pivot(index="date", columns="symbol", values="turnover_rs").sort_index()
    return tv.rolling(window, min_periods=max(window // 2, 1)).median().reindex(calendar)


def daily_vol(market, window: int = 20) -> pd.DataFrame:
    """Rolling close-to-close volatility, the scale in the square-root impact law."""
    r = market.closes / market.closes.shift(1) - 1.0
    return r.rolling(window, min_periods=max(window // 2, 1)).std()


def liquidity_profile(res, adv, vol, capital: float, impact_coef: float = 1.0) -> dict:
    """Participation rates and square-root impact for one backtest at one capital level.

    impact_bps = impact_coef * sigma_daily * sqrt(participation) * 1e4  -- the Almgren-Chriss
    shape, with impact_coef=1 the usual starting calibration. It is a MODEL, not a measurement:
    it is here to show where the edge breaks, not to price a fill.
    """
    tr = res["trades"]
    if not len(tr):
        return {}
    d = tr["exec_date"].to_numpy()
    syms = tr["symbol"].to_numpy()
    notion = tr["weight_delta"].abs().to_numpy() * capital
    a = adv.reindex(index=pd.DatetimeIndex(d)).to_numpy()
    v = vol.reindex(index=pd.DatetimeIndex(d)).to_numpy()
    col = {c: i for i, c in enumerate(adv.columns)}
    ci = np.array([col.get(s, -1) for s in syms])
    ok = ci >= 0
    rows = np.arange(len(d))
    advv = np.full(len(d), np.nan)
    volv = np.full(len(d), np.nan)
    advv[ok] = a[rows[ok], ci[ok]]
    volv[ok] = v[rows[ok], ci[ok]]
    part = notion / advv
    good = np.isfinite(part) & (advv > 0)
    w_all = np.abs(tr["weight_delta"].to_numpy())
    part_g = part[good]
    imp_bps = impact_coef * np.nan_to_num(volv[good]) * np.sqrt(part_g) * 1e4
    # annualised drag over EVERY fill, dust included -- dust has ~zero impact by construction
    w = w_all[good]
    n_days = len(res["daily"])
    drag = float((imp_bps / 1e4 * w).sum()) * TRADING_DAYS / max(n_days, 1)
    # ...but the participation PERCENTILES must exclude it. Under "drift" weighting every
    # rebalance renormalises the book, so all 50 held names get a ~1e-6 weight delta that is
    # not a real order; those would otherwise own the median. Material = >1 bp of the book.
    mat = w > 1e-4
    pm = part_g[mat] if mat.any() else part_g
    return {
        "capital": capital, "n_fills": int(good.sum()), "n_material_fills": int(mat.sum()),
        "participation_median": float(np.median(pm)),
        "participation_p95": float(np.percentile(pm, 95)),
        "participation_p99": float(np.percentile(pm, 99)),
        "participation_max": float(pm.max()),
        "share_fills_over_5pct_adv": float((pm > 0.05).mean()),
        "share_fills_over_10pct_adv": float((pm > 0.10).mean()),
        "impact_bps_median": float(np.median(imp_bps[mat])) if mat.any() else 0.0,
        "impact_bps_notional_weighted": float((imp_bps * w).sum() / max(w.sum(), 1e-12)),
        "impact_drag_ann": drag,
    }


# --------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price_pit.parquet"),
                    help="must be the _pit panel whenever --membership is point-in-time: it "
                         "carries the names that have since left the index")
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--membership", default=os.path.join(DATA_NSE, "membership_pit.parquet"))
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--tag", default="robust_pit")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=5)
    ap.add_argument("--hold-thresh", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=30.0)
    ap.add_argument("--n-family", type=int, default=200, help="family-wise null draws")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cost-preset", default="zerodha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--skip-family", action="store_true")
    ap.add_argument("--family-parquet", default=None,
                    help="reuse the draws from a previous family-null run instead of "
                         "recomputing them (they are the expensive part)")
    ap.add_argument("--skip-liquidity", action="store_true")
    ap.add_argument("--impact-coef", type=float, default=1.0)
    ap.add_argument("--adv-window", type=int, default=20)
    args = ap.parse_args()

    market = load_market(args.panel, args.calendar)
    dates = [d for d in market.calendar
             if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    elig = build_universe(market, dates, args.membership, args.lookback)
    print(f"{len(dates)} sessions {dates[0].date()} -> {dates[-1].date()} | "
          f"universe mean {elig.sum(axis=1).mean():.1f} names", flush=True)

    sig = build_arms(market, dates, elig, args.lookback)
    bench_ew = equal_weight_benchmark(market, elig)
    bench_cap = bench_series(market, "cnx200")
    out: dict = {"config": vars(args)}

    # ---- D. blends (built first so they can be carried through B and C) ----------------
    print("blends ...", flush=True)
    r20, rev5, r120 = (rank_pct(sig["momentum_20d"]), rank_pct(sig["reversal_5d"]),
                       rank_pct(sig["momentum_120d"]))
    sig["blend_mom20_rev5"] = ((r20 + rev5) / 2).where(elig)
    sig["blend_mom20_mom120"] = ((r20 + r120) / 2).where(elig)
    sig["blend_all3"] = ((r20 + rev5 + r120) / 3).where(elig)

    # ---- the shortlist carried into every downstream test ------------------------------
    shortlist = [("momentum_20d", "monthly"), ("reversal_5d", "weekly"),
                 ("momentum_120d", "monthly"), ("momentum_20d", "weekly"),
                 ("reversal_1d", "monthly"), ("kronos_emulator", "monthly"),
                 ("blend_mom20_rev5", "weekly"), ("blend_mom20_rev5", "monthly"),
                 ("blend_mom20_mom120", "monthly"), ("blend_all3", "monthly")]

    # ---- headline table at the flat cost, both benchmarks -------------------------------
    print("headline + yearly stability ...", flush=True)
    head, yearly = [], {}
    for arm, rname in shortlist:
        res = run_backtest(sig[arm], market, topk=args.topk, n_drop=args.n_drop,
                           hold_thresh=args.hold_thresh, rebalance=REBALANCES[rname],
                           cost_bps_round_trip=args.cost_bps)
        m_cap = metrics(res, bench_cap, seed=args.seed)
        m_ew = metrics(res, bench_ew, seed=args.seed)
        head.append({"arm": arm, "rebalance": rname, "ann_return_net": m_cap["ann_return_net"],
                     "ann_excess_vs_cnx200": m_cap["ann_excess_net"],
                     "ann_excess_vs_ew200": m_ew["ann_excess_net"],
                     "ir_vs_ew200": m_ew["information_ratio"],
                     "excess_ci95_ann_vs_ew200": m_ew["excess_ci95_annualised"],
                     "turnover_ann": m_cap["turnover_ann"],
                     "max_drawdown": m_cap["max_drawdown"]})
        yearly[f"{arm}|{rname}"] = yearly_breakdown(res, bench_ew)
        y = yearly[f"{arm}|{rname}"]
        pos = sum(1 for r in y if r["excess_ann"] > 0)
        print(f"  {arm:20s} {rname:8s} exc_ew {m_ew['ann_excess_net']:+.4f} "
              f"IR {m_ew['information_ratio']:+.2f}  positive years {pos}/{len(y)}", flush=True)
    out["headline"] = head
    out["yearly"] = yearly

    # ---- C. capacity curve --------------------------------------------------------------
    print(f"capacity curve, preset={args.cost_preset} ...", flush=True)
    cm = PRESETS[args.cost_preset]
    cap_rows = []
    for arm, rname in shortlist:
        for cap in CAPITALS:
            res = run_backtest(sig[arm], market, topk=args.topk, n_drop=args.n_drop,
                               hold_thresh=args.hold_thresh, rebalance=REBALANCES[rname],
                               cost_model=cm, capital=cap)
            m_ew = metrics(res, bench_ew, seed=args.seed)
            cap_rows.append({"arm": arm, "rebalance": rname, "capital": cap,
                             "per_leg_notional": cap / args.topk,
                             "one_way_bps": cm.one_way_bps(cap / args.topk),
                             "round_trip_bps": cm.round_trip_bps(cap / args.topk),
                             "ann_return_net": m_ew["ann_return_net"],
                             "ann_excess_vs_ew200": m_ew["ann_excess_net"],
                             "ir_vs_ew200": m_ew["information_ratio"],
                             "turnover_ann": m_ew["turnover_ann"],
                             "cost_drag_ann_pct": m_ew["cost_drag_ann_pct"]})
        best = [r for r in cap_rows if r["arm"] == arm and r["rebalance"] == rname]
        brk = next((r["capital"] for r in best if r["ann_excess_vs_ew200"] > 0), None)
        ref = next(r for r in best if r["capital"] == REF_CAPITAL)
        print(f"  {arm:20s} {rname:8s} breakeven capital "
              f"{'Rs %.0e' % brk if brk else 'never positive'}"
              f"  @Rs10cr {ref['ann_excess_vs_ew200']:+.4f}", flush=True)
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_parquet(f"{OUT_NSE}/capacity_{args.tag}.parquet", index=False)
    out["capacity"] = cap_rows

    # ---- E. liquidity ceiling ------------------------------------------------------------
    if not args.skip_liquidity:
        print(f"liquidity ceiling (impact_coef={args.impact_coef}) ...", flush=True)
        adv = adv_rupees(args.panel, market.calendar, args.adv_window)
        vol = daily_vol(market, args.adv_window)
        liq_rows = []
        for arm, rname in shortlist:
            res = run_backtest(sig[arm], market, topk=args.topk, n_drop=args.n_drop,
                               hold_thresh=args.hold_thresh, rebalance=REBALANCES[rname],
                               cost_model=cm, capital=1e7, log_trades=True)
            base_excess = {r["capital"]: r["ann_excess_vs_ew200"]
                           for r in cap_rows if r["arm"] == arm and r["rebalance"] == rname}
            for capital in CAPITALS:
                prof = liquidity_profile(res, adv, vol, capital, args.impact_coef)
                if not prof:
                    continue
                prof.update(arm=arm, rebalance=rname)
                prof["ann_excess_before_impact"] = base_excess.get(capital, float("nan"))
                prof["ann_excess_after_impact"] = (prof["ann_excess_before_impact"]
                                                   - prof["impact_drag_ann"])
                liq_rows.append(prof)
            mine = [r for r in liq_rows if r["arm"] == arm and r["rebalance"] == rname]
            ref = next(r for r in mine if r["capital"] == REF_CAPITAL)
            ceiling = None
            for r in mine:
                if r["ann_excess_after_impact"] > 0:
                    ceiling = r["capital"]
            print(f"  {arm:20s} {rname:8s} impact-adjusted ceiling "
                  f"{('Rs %.0e' % ceiling) if ceiling else 'none'}"
                  f" | @Rs10cr part_med {ref['participation_median']:.3%}"
                  f" excess {ref['ann_excess_after_impact']:+.4f}", flush=True)
        liq_df = pd.DataFrame(liq_rows)
        liq_df.to_parquet(f"{OUT_NSE}/liquidity_{args.tag}.parquet", index=False)
        out["liquidity"] = liq_rows

    # ---- A. family-wise null -------------------------------------------------------------
    if args.family_parquet:
        fam = pd.read_parquet(args.family_parquet)
        print(f"family-wise null: reusing {len(fam)} draws from {args.family_parquet}",
              flush=True)
        _summarise_family(fam, out, head)
    elif not args.skip_family:
        print(f"family-wise null: {args.n_family} draws x {len(ARM_NAMES)} arms x "
              f"{len(REBALANCES)} freqs = {args.n_family * len(ARM_NAMES) * len(REBALANCES)} "
              f"backtests, {args.workers} workers ...", flush=True)
        init_args = (args.panel, args.calendar, args.membership, args.lookback, args.start,
                     args.end, args.topk, args.n_drop, args.hold_thresh, args.cost_bps)
        draws = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=init_args) as ex:
            for i, d in enumerate(ex.map(_one_draw, range(args.n_family), chunksize=1)):
                draws.append(d)
                if (i + 1) % 10 == 0:
                    e = np.array([x["max_excess"] for x in draws])
                    print(f"  [{i+1}/{args.n_family}] max_excess mean {e.mean():+.4f} "
                          f"p95 {np.percentile(e, 95):+.4f}", flush=True)
        fam = pd.DataFrame(draws)
        fam.to_parquet(f"{OUT_NSE}/family_null_{args.tag}.parquet", index=False)
        _summarise_family(fam, out, head)

    path = f"{OUT_NSE}/robustness_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    sidecar(path.replace(".json", "_meta.json"), vars(args))
    print(f"\nwrote {path}", flush=True)


def _summarise_family(fam: pd.DataFrame, out: dict, head: list[dict]) -> None:
    """Fold a family-null draw table into `out` and stamp family-wise p-values on `head`."""
    exc, ir = fam["max_excess"].to_numpy(), fam["max_ir"].to_numpy()
    out["family_null"] = {
        "n_draws": int(len(fam)), "n_cells_per_draw": len(ARM_NAMES) * len(REBALANCES),
        "max_excess": {"mean": float(exc.mean()), "p50": float(np.percentile(exc, 50)),
                       "p90": float(np.percentile(exc, 90)),
                       "p95": float(np.percentile(exc, 95)),
                       "p99": float(np.percentile(exc, 99))},
        "max_ir": {"mean": float(ir.mean()), "p50": float(np.percentile(ir, 50)),
                   "p90": float(np.percentile(ir, 90)), "p95": float(np.percentile(ir, 95)),
                   "p99": float(np.percentile(ir, 99))},
    }
    for col in fam.columns:
        if col in ("seed", "max_excess", "max_ir"):
            continue
        v = fam[col].to_numpy()
        out["family_null"][col] = {"mean": float(v.mean()),
                                   "p50": float(np.percentile(v, 50)),
                                   "p90": float(np.percentile(v, 90)),
                                   "p95": float(np.percentile(v, 95)),
                                   "p99": float(np.percentile(v, 99))}
    # family-wise p-value for each shortlist arm: P(null max >= observed)
    for row in head:
        row["p_fwer_excess"] = float((exc >= row["ann_excess_vs_ew200"]).mean())
        row["p_fwer_ir"] = float((ir >= row["ir_vs_ew200"]).mean())
        row["clears_fwer_p05"] = bool(row["p_fwer_excess"] < 0.05)
    out["headline"] = head
    print("\nfamily-wise corrected:")
    for row in head:
        print(f"  {row['arm']:20s} {row['rebalance']:8s} exc {row['ann_excess_vs_ew200']:+.4f}"
              f"  p_fwer {row['p_fwer_excess']:.3f}"
              f"  {'CLEARS' if row['clears_fwer_p05'] else 'fails'}", flush=True)
    print(f"  [null] max-excess p95 over {out['family_null']['n_cells_per_draw']} cells: "
          f"{out['family_null']['max_excess']['p95']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
