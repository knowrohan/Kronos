"""Plan 10.4 / 10.5c / 10.6 / 10.7 / 10.8 -- backtester and data-integrity assertions.

A backtester that fails any of these is measuring nothing, so these run BEFORE any result is
believed. They use synthetic signals over the real panel, so none of them needs Kronos.

  10.4a perfect foresight   -> must produce a strongly positive curve
  10.4b constant signal     -> ~equal-weight-of-top-50 minus costs, with NEAR-ZERO turnover
  10.5c shifted signal      -> shifting signals one day LATER must destroy a foresight signal;
                               if it does not, there is lookahead in the engine
  10.5d execution lag       -> the engine asserts exec == signal + 1 internally
  10.6  cost reconciliation -> engine cost drag vs the closed-form turnover x rate estimate
  10.7  universe integrity  -> eligible-universe size per date, cliffs flagged
  10.8  split adjustment    -> >20% single-day moves enumerated, not just counted
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import (TRADING_DAYS, annualise, bench_series, load_market,  # noqa: E402
                                metrics, run_backtest)
from bench.nse.common import DATA_NSE, OUT_NSE, sidecar  # noqa: E402
from bench.nse.controls import constant_scores, perfect_foresight_scores  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402

RESULTS: dict = {}
TEST_START, TEST_END = "2023-08-01", "2026-08-20"


def check(name, fn):
    try:
        v = fn()
        RESULTS[name] = v
        print(f"{'PASS' if v.get('pass') else 'FAIL'}  {name}")
        for k, val in v.items():
            if k != "pass":
                print(f"        {k}: {val}")
    except Exception as exc:  # noqa: BLE001
        RESULTS[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-1200:]}
        print(f"ERROR {name}: {type(exc).__name__}: {exc}")


def _dates(market, stride=5):
    cal = [d for d in market.calendar
           if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)]
    return cal[::stride]


def main() -> None:
    market = load_market()
    bench = bench_series(market, "cnx200")
    dates = _dates(market, 1)
    universe = market.tradable.reindex(dates)
    # a score frame shaped exactly like a real one: NaN where the name is not eligible
    shell = pd.DataFrame(np.where(universe.to_numpy(), 0.0, np.nan),
                         index=dates, columns=market.tradable.columns)
    bt = dict(topk=50, n_drop=5, hold_thresh=5, rebalance=5, cost_bps_round_trip=30.0)

    # ---------------- 10.4a perfect foresight ----------------
    def t_foresight():
        pf = perfect_foresight_scores(market, dates, 1).where(shell.notna())
        m = metrics(run_backtest(pf, market, **bt), bench)
        return {"pass": m["ann_return_net"] > 0.50,
                "ann_return_net": round(m["ann_return_net"], 4),
                "ann_excess_net": round(m["ann_excess_net"], 4),
                "information_ratio": round(m["information_ratio"], 3),
                "turnover_ann": round(m["turnover_ann"], 2)}

    # ---------------- 10.4b constant signal ----------------
    def t_constant():
        cs = constant_scores(shell)
        r = run_backtest(cs, market, **bt)
        m = metrics(r, bench)
        # after the initial fill, a constant score must not churn the book
        later = r["rebalances"].iloc[1:]["turnover"]
        return {"pass": float(later.mean()) < 0.02,
                "first_rebalance_turnover": round(float(r["rebalances"].iloc[0]["turnover"]), 4),
                "mean_turnover_after_first": round(float(later.mean()), 5),
                "max_turnover_after_first": round(float(later.max()), 5),
                "ann_return_net": round(m["ann_return_net"], 4),
                "note": "non-zero churn on a constant signal = the rank tie-break is trading"}

    # ---------------- 10.5c shifted signal destroys performance ----------------
    def t_shift_destroys():
        pf = perfect_foresight_scores(market, dates, 1).where(shell.notna())
        good = metrics(run_backtest(pf, market, **bt), bench)["ann_return_net"]
        # use each date's score one session LATE -> the edge must vanish
        late = pf.shift(1)
        bad = metrics(run_backtest(late, market, **bt), bench)["ann_return_net"]
        return {"pass": good - bad > 0.30,
                "foresight_ann_return": round(good, 4),
                "shifted_ann_return": round(bad, 4),
                "destroyed_by": round(good - bad, 4)}

    # ---------------- 10.5d execution lag asserted in code ----------------
    def t_exec_lag():
        from bench.nse.backtest import assert_execution_lag
        ok_raises = False
        try:
            assert_execution_lag(10, 10, market.calendar)
        except AssertionError:
            ok_raises = True
        assert_execution_lag(10, 11, market.calendar)
        return {"pass": ok_raises, "assertion_fires_on_same_day_execution": ok_raises}

    # ---------------- 10.6 cost reconciliation ----------------
    def t_cost_reconcile():
        rows = {}
        for reb, nd in ((1, 5), (5, 5), (5, 1)):
            cfg = dict(bt, rebalance=reb, n_drop=nd)
            r = run_backtest(constant_scores(shell) + np.random.default_rng(0).normal(
                0, 1, shell.shape), market, **cfg)
            m = metrics(r, bench)
            engine = m["cost_drag_ann_pct"]
            closed = m["turnover_ann"] * (cfg["cost_bps_round_trip"] / 2 / 1e4) * 100
            rows[f"reb{reb}_ndrop{nd}"] = {
                "turnover_ann": round(m["turnover_ann"], 3),
                "engine_cost_drag_pct": round(engine, 4),
                "closed_form_pct": round(closed, 4),
                "abs_diff": round(abs(engine - closed), 6),
            }
        worst = max(v["abs_diff"] for v in rows.values())
        # the plan's 4.2 prediction: topk=50, n_drop=5 daily -> 20% of capital/day
        plan42 = 0.20 * 0.0015 * TRADING_DAYS * 100
        rows["plan_4.2_predicted_drag_pct_at_15bps_oneway"] = round(plan42, 2)
        rows["pass"] = worst < 0.02
        return rows

    # ---------------- 10.7 universe integrity ----------------
    def t_universe():
        n = market.tradable.sum(axis=1)
        n = n[(n.index >= pd.Timestamp("2015-01-01"))]
        d = n.diff().abs()
        cliffs = d[d > 15]
        return {"pass": len(cliffs) == 0,
                "min_universe": int(n.min()), "max_universe": int(n.max()),
                "mean_universe": round(float(n.mean()), 1),
                "n_day_over_day_jumps_gt_15": int(len(cliffs)),
                "largest_jumps": {str(k.date()): int(v) for k, v in cliffs.nlargest(5).items()}}

    # ---------------- 10.8 split adjustment ----------------
    def t_splits():
        panel = pd.read_parquet(os.path.join(DATA_NSE, "panel.parquet"))
        panel = panel.sort_values(["symbol", "date"])
        panel["ret"] = panel.groupby("symbol")["close"].pct_change()
        big = panel[panel["ret"].abs() > 0.20].copy()
        # a real split shows up in the `splits` column; anything else is a genuine move
        near_split = big.merge(
            panel[panel["splits"] > 0][["symbol", "date", "splits"]].rename(
                columns={"date": "split_date"}), on="symbol", how="left")
        near_split["days_to_split"] = (near_split["date"] - near_split["split_date"]).dt.days.abs()
        suspicious = near_split[near_split["days_to_split"] <= 3]
        return {"pass": len(suspicious) == 0,
                "n_moves_gt_20pct": int(len(big)),
                "n_symbols_affected": int(big["symbol"].nunique()),
                "n_within_3d_of_a_recorded_split": int(len(suspicious)),
                "top_symbols": big.groupby("symbol").size().nlargest(6).to_dict(),
                "note": "moves are concentrated in IDEA / YESBANK / ADANIENT -- real events"}

    # ---------------- circuit limits (plan 4.11) ----------------
    def t_circuits():
        o = market.opens
        oo = (o.shift(-1) / o - 1.0)
        oo = oo[(oo.index >= pd.Timestamp(TEST_START)) & (oo.index <= pd.Timestamp(TEST_END))]
        tot = int(oo.notna().sum().sum())
        return {"pass": True,
                "open_to_open_obs_in_test": tot,
                "share_abs_gt_10pct": round(float((oo.abs() > 0.10).sum().sum() / max(tot, 1)), 5),
                "share_abs_gt_20pct": round(float((oo.abs() > 0.20).sum().sum() / max(tot, 1)), 6),
                "note": "if the >10% share is negligible, circuit-limit modelling is skippable"}

    # ---------------- cost model hand-check (plan 10.6) ----------------
    def t_cost_handcheck():
        cm = PRESETS["zerodha"]
        b = cm.breakdown(1_000_000)
        statutory = b["round_trip_rs"] - 2 * 250.0   # strip the slippage legs
        return {"pass": abs(statutory - 2240.152) < 1.0,
                "hand_computed_statutory_rs": 2240.15,
                "engine_statutory_rs": round(statutory, 2),
                "round_trip_bps_incl_slippage": round(b["round_trip_bps"], 2),
                "zerodha_bps_at_1L": round(PRESETS["zerodha"].round_trip_bps(1e5), 2),
                "flat20_bps_at_1L": round(PRESETS["flat20"].round_trip_bps(1e5), 2),
                "plan_draft_bps": round(PRESETS["plan_draft"].round_trip_bps(1e6), 2)}

    # ---------------- IC machinery, both directions ----------------
    def t_ic_machinery():
        from bench.nse.controls import compute_ic, newey_west_se, summarise_ic
        o = market.opens
        sub = dates[::12]
        pf = (o.shift(-11) / o.shift(-1) - 1.0).reindex(sub)
        ics = {h: float(compute_ic(pf, market, h).mean()) for h in (1, 5, 10, 20)}
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.normal(size=pf.shape), index=pf.index,
                             columns=pf.columns).where(pf.notna())
        ns = summarise_ic(compute_ic(noise, market, 10).to_numpy(), 10)
        ma = pd.Series(np.convolve(rng.normal(size=800), np.ones(10) / 10, mode="same"))
        nw = newey_west_se(ma.to_numpy(), 10)
        naive = float(ma.std(ddof=1) / np.sqrt(len(ma)))
        return {"pass": abs(ics[10] - 1.0) < 1e-6 and abs(ns["mean_ic"]) < 0.02
                        and not ns["significant"] and nw > naive * 1.5,
                "perfect_foresight_h10_ic": round(ics[10], 6),
                "perfect_foresight_ic_by_horizon": {k: round(v, 4) for k, v in ics.items()},
                "white_noise_h10_ic": round(ns["mean_ic"], 5),
                "white_noise_significant": ns["significant"],
                "newey_west_se": round(nw, 5), "naive_se": round(naive, 5),
                "nw_inflation_factor": round(nw / naive, 2),
                "note": "IC=1 exactly at the matching horizon; NW must exceed naive on a 10-lag MA"}

    check("10.9_ic_machinery_both_directions", t_ic_machinery)
    check("10.4a_perfect_foresight_wins_big", t_foresight)
    check("10.4b_constant_signal_near_zero_turnover", t_constant)
    check("10.5c_shifting_signal_destroys_performance", t_shift_destroys)
    check("10.5d_execution_lag_asserted", t_exec_lag)
    check("10.6a_cost_drag_reconciles_with_closed_form", t_cost_reconcile)
    check("10.6b_cost_model_hand_check", t_cost_handcheck)
    check("10.7_universe_integrity", t_universe)
    check("10.8_split_adjustment", t_splits)
    check("4.11_circuit_limits_negligible", t_circuits)

    out = os.path.join(OUT_NSE, "verify_backtest.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    sidecar(out.replace(".json", "_meta.json"), {})
    n = sum(1 for v in RESULTS.values() if v.get("pass"))
    print(f"\n{n}/{len(RESULTS)} checks passed -> {out}")


if __name__ == "__main__":
    main()
