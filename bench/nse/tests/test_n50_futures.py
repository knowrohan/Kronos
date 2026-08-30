"""Tests for the single-stock futures venue (NIFTY50_PLAN sec.6.2, `bench/nse/n50_futures.py`).

Two things in that module can be wrong in ways no reader would catch from the output: the sign
and cost arithmetic of a levered short position, and the expiry-crossing detector, whose answer
("how many of these holds must be rolled or physically settled") is stated as a fact about the
NSE calendar. Both are pinned here on synthetic data with hand-computable answers.

Everything is synthetic except `test_research_lots_match_costs`, which reads the on-disk
research JSON and is skipped when it is absent, so the suite runs on a machine that has never
built the panel.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse.backtest import Market  # noqa: E402
from bench.nse.costs import FUTURES_PRESETS, LOT_SIZES, PRESETS  # noqa: E402
from bench.nse.n50_futures import (  # noqa: E402
    CASH_NOTIONAL,
    HOLD_SESSIONS,
    PROXY_CONTRACT_RS,
    arm_weekly,
    block_bootstrap,
    build_legs,
    count_crossings,
    crosses_expiry,
    expiry_dates,
    last_weekday_of_month,
    load_scores,
    lot_for,
    net_pnl,
    research_lots,
    rule_power,
    run,
    verdict_for,
    weekly_cycles,
    wilson_ci,
)

FUT = FUTURES_PRESETS["zerodha_fut"]
CASH = PRESETS["zerodha"]


# ------------------------------------------------------------------ fixtures


def make_market(n_dates: int = 60, symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
                start: str = "2024-01-01", drift=None, base: float = 1000.0):
    """A synthetic exchange calendar + price panel where every name's open path is known.

    `drift[sym]` is the per-session open-to-open return of that name, so a 5-session hold's
    gross move is exactly ``base * ((1+d)**(k+5) - (1+d)**(k+1))`` and any P&L assertion can be
    written down by hand. close == open so `Market.tradable` is True everywhere.
    """
    cal = list(pd.bdate_range(start, periods=n_dates))
    drift = {} if drift is None else drift
    rows = []
    for s in symbols:
        d = drift.get(s, 0.0)
        px = base * (1.0 + d) ** np.arange(n_dates)
        for i, day in enumerate(cal):
            rows.append({"date": day, "symbol": s, "open": px[i], "high": px[i],
                         "low": px[i], "close": px[i], "volume": 1e6})
    return Market(pd.DataFrame(rows), cal), cal, list(symbols)


def make_scores(cal, symbols, order=None, n_dates=None):
    """Constant cross-sectional ranking: `order[0]` is always top-1, `order[-1]` bottom-1."""
    order = list(symbols) if order is None else list(order)
    days = cal if n_dates is None else cal[:n_dates]
    rows = []
    for day in days:
        for j, s in enumerate(order):
            rows.append({"date": day, "symbol": s, "score": float(len(order) - j),
                         "is_n50": True})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Wilson interval


def test_wilson_matches_closed_form():
    p, lo, hi = wilson_ci(60, 100)
    assert p == pytest.approx(0.60)
    z, n, ph = 1.96, 100, 0.6
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z / d * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    assert lo == pytest.approx(c - h) and hi == pytest.approx(c + h)
    # the textbook property Wald lacks: bounded in [0, 1] and non-degenerate at k = 0
    p0, lo0, hi0 = wilson_ci(0, 10)
    assert p0 == 0.0 and lo0 == 0.0 and 0.0 < hi0 < 1.0
    # wider sample -> tighter interval at the same rate
    _, lo_small, _ = wilson_ci(6, 10)
    _, lo_big, _ = wilson_ci(600, 1000)
    assert lo_big > lo_small


def test_wilson_lower_bound_drives_the_kill_rule():
    """The plan's bar is on the LOWER bound: 60% over 113 weeks does not clear 55%."""
    _, lo, _ = wilson_ci(68, 113)          # 60.2% hit rate, the sample size this venue has
    assert lo < 0.55
    assert verdict_for(lo) == "KILL"
    assert verdict_for(0.5501) == "PROCEED"
    assert verdict_for(0.55) == "KILL"     # the rule is strict ">"


def test_rule_power_names_the_smallest_passing_hit_count():
    """`min_hits_to_pass` must be the smallest k whose Wilson LOWER bound clears the bar, and
    k-1 must fail -- otherwise the reported "the rule needed 73 of 113" is not a fact."""
    for n in (40, 113, 250):
        rp = rule_power(n, bar=0.55)
        k = rp["min_hits_to_pass"]
        assert k is not None
        assert wilson_ci(k, n)[1] > 0.55
        assert wilson_ci(k - 1, n)[1] <= 0.55
        assert rp["min_hit_rate_to_pass"] == pytest.approx(k / n)


def test_rule_power_is_low_at_the_bar_and_rises_with_the_truth():
    """The kill rule is a lower-bound rule, so a strategy sitting exactly ON the bar almost
    never passes it at n = 113; power must still be monotone in the true hit rate."""
    rp = rule_power(113, bar=0.55)
    pw = rp["power_vs_true_hit_rate"]
    assert pw["0.55"] < 0.10
    assert pw["0.50"] < pw["0.55"] < pw["0.60"] < pw["0.65"]
    # exact binomial, not a simulation: check one value against a direct sum
    k, n, q = rp["min_hits_to_pass"], 113, 0.60
    direct = sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(k, n + 1))
    assert pw["0.60"] == pytest.approx(direct)


def test_rule_power_is_unreachable_when_the_bar_is_one():
    assert rule_power(20, bar=1.0)["min_hits_to_pass"] is None


# ------------------------------------------------------------------ expiry calendar


def test_last_thursday_known_months():
    assert last_weekday_of_month(2026, 8) == pd.Timestamp("2026-08-27")
    assert last_weekday_of_month(2026, 1) == pd.Timestamp("2026-01-29")
    assert last_weekday_of_month(2024, 2) == pd.Timestamp("2024-02-29")   # leap-year Thursday
    assert last_weekday_of_month(2025, 7) == pd.Timestamp("2025-07-31")   # month ends Thursday
    assert last_weekday_of_month(2026, 8, weekday=1) == pd.Timestamp("2026-08-25")  # Tuesday


def test_expiry_backs_up_over_a_holiday():
    """Nominal expiry not a session -> the PREVIOUS session, including a multi-day block."""
    cal = list(pd.bdate_range("2026-01-01", "2026-03-31"))
    got = expiry_dates(cal, weekday=3)
    assert pd.Timestamp("2026-01-29") in got
    assert pd.Timestamp("2026-02-26") in got
    assert pd.Timestamp("2026-03-26") in got

    holed = [d for d in cal if d not in (pd.Timestamp("2026-02-26"),
                                         pd.Timestamp("2026-02-25"))]
    got2 = expiry_dates(holed, weekday=3)
    assert pd.Timestamp("2026-02-24") in got2          # backed up two sessions
    assert pd.Timestamp("2026-02-26") not in got2
    assert len(got2) == 3                              # still one expiry per month


def test_crossing_detector_strict_and_inclusive():
    exp = [pd.Timestamp("2026-02-26")]
    # hold spanning the expiry
    assert crosses_expiry(pd.Timestamp("2026-02-24"), pd.Timestamp("2026-03-03"), exp)
    # hold that EXITS on the expiry session: flat at the open, so no roll (strict reading)
    assert not crosses_expiry(pd.Timestamp("2026-02-19"), pd.Timestamp("2026-02-26"), exp)
    # hold that ENTERS on the expiry session: it does cross (the contract settles that evening)
    assert crosses_expiry(pd.Timestamp("2026-02-26"), pd.Timestamp("2026-03-05"), exp)
    # clear of it
    assert not crosses_expiry(pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-09"), exp)

    cycles = [{"entry_date": pd.Timestamp("2026-02-24"), "exit_date": pd.Timestamp("2026-03-03")},
              {"entry_date": pd.Timestamp("2026-02-19"), "exit_date": pd.Timestamp("2026-02-26")},
              {"entry_date": pd.Timestamp("2026-03-02"), "exit_date": pd.Timestamp("2026-03-09")}]
    c = count_crossings(cycles, exp)
    assert c == {"n_weeks": 3, "n_cross_strict": 1, "n_cross_inclusive": 2,
                 "frac_cross_strict": pytest.approx(1 / 3),
                 "frac_cross_inclusive": pytest.approx(2 / 3)}


def test_a_five_session_hold_crosses_roughly_a_quarter_of_the_time():
    """Sanity bound on the headline caveat: ~5 of every 21 sessions' holds must cross."""
    cal = list(pd.bdate_range("2024-01-01", "2026-08-25"))
    exp = expiry_dates(cal, weekday=3)
    cycles = [{"entry_date": cal[i], "exit_date": cal[i + HOLD_SESSIONS]}
              for i in range(0, len(cal) - HOLD_SESSIONS, 5)]
    frac = count_crossings(cycles, exp)["frac_cross_strict"]
    assert 0.10 < frac < 0.40


# ------------------------------------------------------------------ lots


def test_lot_for_uses_the_exchange_lot_then_falls_back_to_a_75_lakh_contract():
    lot, proxy = lot_for("RELIANCE", 1330.0)
    assert (lot, proxy) == (LOT_SIZES["RELIANCE"], False)
    lot, proxy = lot_for("NOTALISTEDNAME", 1500.0)
    assert proxy is True and lot == round(PROXY_CONTRACT_RS / 1500.0)
    assert abs(lot * 1500.0 - PROXY_CONTRACT_RS) <= 1500.0    # within one share of Rs 7.5 lakh
    assert lot_for("NOTALISTEDNAME", float("nan")) == (0, True)
    assert lot_for("NOTALISTEDNAME", 0.0) == (0, True)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "nse", "research",
                                                    "futures_facts.json")),
                    reason="research JSON not on disk")
def test_research_lots_match_costs():
    assert research_lots() == LOT_SIZES


# ------------------------------------------------------------------ P&L sign and arithmetic


def test_long_and_short_pnl_signs_on_a_rising_name():
    """AAA rises 1%/session and is always top-1; FFF is always bottom-1 and also rises."""
    drift = {s: 0.01 for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")}
    market, cal, syms = make_market(drift=drift)
    scores = make_scores(cal, syms)
    cycles = weekly_cycles(scores["date"], market)
    legs = build_legs(scores, market, cycles, FUT, CASH)

    longs = legs.loc[legs["side"] > 0]
    shorts = legs.loc[legs["side"] < 0]
    assert (longs["gross_pnl_rs"] > 0).all()      # everything rose, so every long wins gross
    assert (shorts["gross_pnl_rs"] < 0).all()     # and every short loses gross
    assert set(longs["slot"]) == {"top1", "top2", "top3"}
    assert set(shorts["slot"]) == {"bottom1"}
    assert (shorts["symbol"] == "FFF").all() and (legs.loc[legs["slot"] == "top1",
                                                           "symbol"] == "AAA").all()

    row = legs.iloc[0]
    assert row["gross_pnl_rs"] == pytest.approx(
        row["lot"] * (row["exit_px"] - row["entry_px"]) * row["side"])
    assert row["notional_rs"] == pytest.approx(row["lot"] * row["entry_px"])
    # a 5-session hold at +1%/session is exactly (1.01^5 - 1) on the entry price
    assert row["ret"] == pytest.approx(1.01 ** HOLD_SESSIONS - 1.0, rel=1e-9)


def test_falling_short_makes_money_and_the_pair_is_the_difference():
    drift = {"AAA": 0.01, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0, "EEE": 0.0, "FFF": -0.01}
    market, cal, syms = make_market(drift=drift)
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market), FUT, CASH)
    top1 = arm_weekly(legs, "top1", 55.0)
    bot1 = arm_weekly(legs, "bottom1", 55.0)
    pair = arm_weekly(legs, "pair", 55.0)
    assert (top1["gross_rs"] > 0).all()
    assert (bot1["gross_rs"] > 0).all()           # short a falling name -> positive gross
    assert pair["pnl_rs"].to_numpy() == pytest.approx(
        (top1["pnl_rs"] + bot1["pnl_rs"]).to_numpy())
    assert (pair["margin_rs"] > top1["margin_rs"]).all()   # two lots, two margins


def test_cost_arithmetic_matches_the_cost_model_line_by_line():
    market, cal, syms = make_market(drift={"AAA": 0.002})
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market), FUT, CASH)
    row = legs.iloc[0]
    n = row["notional_rs"]
    assert row["txn_rs"] == pytest.approx(FUT.leg(n, "buy") + FUT.leg(n, "sell"))
    assert row["txn_rs"] == pytest.approx(FUT.round_trip_bps(n) * n / 1e4)
    # carry per 1 bps/month, so 55 bps/month over 5 of 21 sessions:
    assert row["carry_rs_per_bpm"] == pytest.approx(n * HOLD_SESSIONS / 21.0 / 1e4)
    assert row["carry_rs_per_bpm"] * 55.0 == pytest.approx(
        n * FUT.holding_cost_bps(HOLD_SESSIONS) / 1e4)
    net55 = net_pnl(legs, 55.0)
    assert net55.iloc[0] == pytest.approx(row["gross_pnl_rs"] - row["txn_rs"]
                                          - 55.0 * row["carry_rs_per_bpm"])
    # a higher roll strictly reduces P&L, by exactly the carry difference
    assert (net_pnl(legs, 65.0) - net55).to_numpy() == pytest.approx(
        (-10.0 * legs["carry_rs_per_bpm"]).to_numpy())


def test_zero_cost_preset_leaves_gross_untouched():
    market, cal, syms = make_market(drift={"AAA": 0.01})
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market),
                      FUTURES_PRESETS["zero"], PRESETS["zero"])
    assert (legs["txn_rs"] == 0).all()
    assert net_pnl(legs, 0.0).to_numpy() == pytest.approx(legs["gross_pnl_rs"].to_numpy())
    assert legs["cash_net_rs"].to_numpy() == pytest.approx(legs["cash_gross_rs"].to_numpy())


def test_carry_sign_ambiguity_on_the_short_leg():
    """Crediting the short's carry moves it by exactly 2x the carry; longs are untouched."""
    market, cal, syms = make_market(drift={"FFF": -0.005})
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market), FUT, CASH)
    charged = net_pnl(legs, 55.0)
    credited = net_pnl(legs, 55.0, credit_short_carry=True)
    is_short = legs["side"].to_numpy() < 0
    assert (credited - charged).to_numpy()[~is_short] == pytest.approx(0.0)
    assert (credited - charged).to_numpy()[is_short] == pytest.approx(
        (2 * 55.0 * legs["carry_rs_per_bpm"]).to_numpy()[is_short])


def test_cash_comparison_leg_uses_the_delivery_cost_model():
    market, cal, syms = make_market(drift={"AAA": 0.01})
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market), FUT, CASH)
    row = legs.loc[legs["slot"] == "top1"].iloc[0]
    r = row["ret"]
    assert row["cash_gross_rs"] == pytest.approx(CASH_NOTIONAL * r)
    assert row["cash_txn_rs"] == pytest.approx(
        CASH.leg(CASH_NOTIONAL, "buy") + CASH.leg(CASH_NOTIONAL * (1 + r), "sell"))
    # the futures lot is many times the cash leg, which is the whole point of the venue
    assert row["notional_rs"] > 5 * CASH_NOTIONAL


# ------------------------------------------------------------------ aggregation


def test_arm_weekly_sums_legs_and_drops_incomplete_weeks():
    market, cal, syms = make_market(drift={s: 0.003 for s in ("AAA", "BBB", "CCC")})
    scores = make_scores(cal, syms)
    legs = build_legs(scores, market, weekly_cycles(scores["date"], market), FUT, CASH)
    t1 = arm_weekly(legs, "top1", 55.0)
    t3 = arm_weekly(legs, "top3", 55.0)
    assert len(t1) == len(t3)
    assert (t3["n_legs"] == 3).all() and (t1["n_legs"] == 1).all()
    assert (t3["pnl_rs"].to_numpy() > t1["pnl_rs"].to_numpy()).all()   # three rising lots > one
    assert t3["notional_rs"].to_numpy() == pytest.approx(3 * t1["notional_rs"].to_numpy())

    broken = legs.copy()
    victim = broken.index[(broken["slot"] == "top2")][0]
    broken.loc[victim, ["ok", "gross_pnl_rs"]] = [False, np.nan]
    t3b = arm_weekly(broken, "top3", 55.0)
    assert len(t3b) == len(t3) - 1                                # that week is dropped whole
    assert len(arm_weekly(broken, "top1", 55.0)) == len(t1)       # top1 unaffected


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_on_a_constant_winner_and_a_constant_loser():
    n, weeks = 40, 156
    win = np.full(n, 5_000.0)
    margin = np.full(n, 150_000.0)
    b = block_bootstrap(win, margin, draws=200, weeks=weeks, seed=1)
    assert b["p_final_gt_start"] == 1.0
    assert b["median_final_rs"] == pytest.approx(1_000_000.0 + weeks * 5_000.0)
    assert b["p_margin_call"] == 0.0 and b["p_loss_gt_30pct_anytime"] == 0.0
    assert b["median_max_drawdown_rs"] == 0.0

    lose = np.full(n, -5_000.0)
    b2 = block_bootstrap(lose, np.full(n, 400_000.0), draws=200, weeks=weeks, seed=1)
    assert b2["p_final_gt_start"] == 0.0
    assert b2["p_loss_gt_30pct_anytime"] == 1.0        # -Rs 7.8 lakh over 156 weeks
    assert b2["p_margin_call"] == 1.0                  # equity crosses 1.5 x Rs 4 lakh
    assert b2["median_max_drawdown_rs"] == pytest.approx(weeks * 5_000.0)
    assert b2["p_ruin_equity_le_0"] == 0.0             # Rs 2.2 lakh left: broke, not wiped out
    b3 = block_bootstrap(np.full(n, -8_000.0), margin, draws=200, weeks=weeks, seed=1)
    assert b3["p_ruin_equity_le_0"] == 1.0             # -Rs 12.5 lakh on a Rs 10 lakh book


def test_bootstrap_margin_call_is_a_statement_about_margin_not_loss():
    """Flat P&L, but a margin bigger than 2/3 of the book -> called on the first week."""
    n = 40
    flat = np.zeros(n)
    b = block_bootstrap(flat, np.full(n, 700_000.0), draws=100, weeks=20, seed=2)
    assert b["p_margin_call"] == 1.0
    assert b["p_loss_gt_30pct_anytime"] == 0.0 and b["median_final_rs"] == 1_000_000.0
    b2 = block_bootstrap(flat, np.full(n, 200_000.0), draws=100, weeks=20, seed=2)
    assert b2["p_margin_call"] == 0.0


def test_bootstrap_refuses_a_sample_shorter_than_its_block():
    assert "error" in block_bootstrap(np.zeros(3), np.zeros(3), block=4, draws=10)


# ------------------------------------------------------------------ end to end


def test_run_end_to_end_on_a_rigged_panel_says_proceed():
    """Top-1 always rises 1.5%/session, bottom-1 always falls: the venue should pass its bar."""
    drift = {"AAA": 0.015, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0, "EEE": 0.0, "FFF": -0.015}
    market, cal, syms = make_market(n_dates=200, drift=drift)
    scores = make_scores(cal, syms)
    res, legs = run(scores, market, draws=200, weeks=52, seed=3)
    for r in (45, 55, 65):
        k = f"roll{r}"
        assert res["arms"]["top1"][k]["futures"]["hit_rate"] == 1.0
        assert res["arms"]["bottom1"][k]["futures"]["hit_rate"] == 1.0
        assert res["arms"]["top1"][k]["verdict_top1_rule"] == "PROCEED"
    assert res["verdict"]["overall"] == "PROCEED"
    assert res["sample"]["n_legs_dropped_no_price"] == 0
    assert res["sample"]["n_weekly_cycles"] == len(arm_weekly(legs, "top1", 55.0))
    b = res["arms"]["top1"]["roll55"]["bootstrap"]
    assert b["p_final_gt_start"] == 1.0 and b["weeks"] == 52


def test_run_end_to_end_on_a_dead_signal_says_kill():
    """Every name is flat, so the score carries nothing and cost alone decides: with a strictly
    positive P&L required, the hit rate is 0 and the venue is killed at every roll setting."""
    drift = {s: 0.0 for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")}
    market, cal, syms = make_market(n_dates=200, drift=drift)
    scores = make_scores(cal, syms)
    res, _ = run(scores, market, draws=200, weeks=52, seed=4)
    assert res["verdict"]["overall"] == "KILL"
    for r in (45, 55, 65):
        assert res["verdict"]["by_roll"][f"roll{r}"]["verdict"] == "KILL"
    # rising roll cost can only reduce net P&L
    means = [res["arms"]["top1"][f"roll{r}"]["futures"]["mean_rs"] for r in (45, 55, 65)]
    assert means[0] > means[1] > means[2]
    # and the futures leg is dominated by the cash leg it is compared against, per rupee at risk
    assert res["cost_reference"]["cash_round_trip_bps_at_1L"] > \
        res["cost_reference"]["round_trip_bps_at_7p5L"]


def test_run_reports_both_expiry_conventions_and_the_proxy_lot_count():
    drift = {"AAA": 0.01, "FFF": -0.01}
    market, cal, syms = make_market(n_dates=200, drift=drift)
    scores = make_scores(cal, syms)
    res, _ = run(scores, market, draws=50, weeks=52, seed=5)
    e = res["expiry"]
    assert e["thursday"]["n_cross_strict"] <= e["thursday"]["n_cross_inclusive"]
    assert e["thursday"]["n_weeks"] == res["sample"]["n_weekly_cycles"]
    assert 0 < e["thursday"]["frac_cross_strict"] < 1
    assert "tuesday_sensitivity" in e
    # none of the synthetic names is in LOT_SIZES, so every leg is a proxy lot
    assert res["sample"]["frac_legs_proxy_lot"] == 1.0


def test_load_scores_enforces_the_contract(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["A"],
                  "score": [1.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="contract"):
        load_scores(str(p))
    q = tmp_path / "good.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 2, "symbol": ["A", "B"],
                  "score": [1.0, np.nan], "is_n50": [True, True]}).to_parquet(q)
    assert len(load_scores(str(q))) == 1        # the NaN score is not a scored cell
