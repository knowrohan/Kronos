"""Test 7 of NIFTY50_PLAN.md sec.8: the cost tables must reproduce their own cited figures.

Both cost models are pure rate tables whose only defence against a mistyped constant is that
someone worked an example by hand elsewhere and wrote the answer down. Two such answers exist:
the research agent's Rs 891.98 / 5.95 bps futures round trip (``data/nse/research/
futures_facts.json``) and the 22.4 bps cash round trip asserted in ``costs.py``'s own module
docstring. A rate typo that survives both is not a typo anyone will notice later, because every
downstream number -- the money bar in gate M2, the futures venue's viability in sec.6.2 -- is
quoted net of these rates and has no independent scale.

Slippage is set to zero wherever a cited figure is being reproduced: both worked examples are
statutory-plus-broker charges only, and slippage is a modelling assumption, not a rate. The
tests that follow re-add it explicitly so the default preset is pinned too.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse.costs import (  # noqa: E402
    FUTURES_PRESETS,
    LOT_SIZES,
    MARGIN_PCT_DEFAULT,
    MARGIN_PCT_RANGE,
    PRESETS,
    CostModel,
    FuturesCostModel,
)

FACTS_PATH = os.path.join(ROOT, "data", "nse", "research", "futures_facts.json")

LOT_NOTIONAL = 1_500_000.0     # the research agent's worked example: one lot, Rs 15 lakh
CASH_LEG = 1_000_000.0         # costs.py's docstring figure is quoted at Rs 10 lakh per leg


def _facts():
    """The researched rate table, or a skip -- data/nse/ is gitignored, so it may be absent."""
    if not os.path.exists(FACTS_PATH):
        pytest.skip(f"{FACTS_PATH} not on disk (data/ is gitignored); rate cross-check skipped")
    with open(FACTS_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------- futures worked example


def test_futures_round_trip_reproduces_worked_example():
    """5.95 bps on a Rs 15 lakh lot, within the +-0.5 bps the plan allows."""
    cm = FUTURES_PRESETS["worked_example"]
    got = cm.round_trip_bps(LOT_NOTIONAL)
    assert abs(got - 5.95) <= 0.5, f"round_trip_bps={got:.4f}, expected 5.95 +- 0.5"
    # tighter than the plan's tolerance, because it should agree to the rupee, not to the bp
    assert got == pytest.approx(5.9465, abs=0.001)


def test_futures_round_trip_matches_the_rupee_total():
    """Rs 891.98 total, and each line of the research agent's buy/sell decomposition."""
    cm = FUTURES_PRESETS["worked_example"]
    bd = cm.breakdown(LOT_NOTIONAL)
    assert bd["round_trip_rs"] == pytest.approx(891.98, abs=0.05)

    buy, sell = bd["buy"], bd["sell"]
    assert buy["brokerage"] == pytest.approx(20.0)          # capped, not 0.03% = Rs 450
    assert buy["exchange_txn"] == pytest.approx(25.95, abs=0.01)
    assert buy["sebi_fee"] == pytest.approx(1.5, abs=0.01)
    assert buy["stt"] == 0.0                                # no STT on the futures buy leg
    assert buy["stamp_duty"] == pytest.approx(30.0, abs=0.01)
    assert sell["stt"] == pytest.approx(750.0, abs=0.01)    # 0.05% sell side, from 2026-04-01
    assert sell["stamp_duty"] == 0.0
    assert sell["brokerage"] == pytest.approx(20.0)
    # GST is 18% of (brokerage + exchange + SEBI) on BOTH legs and nothing else
    assert buy["gst"] + sell["gst"] == pytest.approx(17.08, abs=0.02)
    # STT is 84% of the statutory round trip: the single line that decides this venue's cost
    assert sell["stt"] / bd["round_trip_rs"] > 0.8


def test_futures_rates_match_the_research_json():
    """The dataclass defaults are a transcription of futures_facts.json; assert they still are."""
    facts = _facts()
    by_topic = {f["topic"]: f for f in facts["facts"]}
    stt = next(f for t, f in by_topic.items() if t.startswith("STT on stock futures"))
    chg = next(f for t, f in by_topic.items() if t.startswith("Exchange txn charge"))
    cm = FuturesCostModel()
    assert cm.stt_sell == stt["rate_sell"]
    assert cm.stt_buy == stt["rate_buy"]
    assert cm.exchange_txn == chg["exchange_txn"]
    assert cm.sebi_fee == chg["sebi_fee"]
    assert cm.stamp_duty_buy == chg["stamp_buy"]
    assert cm.gst == chg["gst"]
    assert cm.brokerage_rate == chg["brokerage_rate"]
    assert cm.brokerage_cap == chg["brokerage_per_order_rs"]

    lots = next(f for f in facts["facts"] if "lots" in f)["lots"]
    assert LOT_SIZES == lots, "LOT_SIZES has drifted from the researched lot table"

    band = next(f for f in facts["facts"] if "roll_cost_bps_per_month_range" in f)
    lo, hi = band["roll_cost_bps_per_month_range"]
    assert FuturesCostModel().roll_cost_bps_per_month == pytest.approx((lo + hi) / 2.0)

    ex = facts["worked_example_round_trip_bps"]
    assert FUTURES_PRESETS["worked_example"].round_trip_bps(LOT_NOTIONAL) == pytest.approx(
        ex["round_trip_bps_on_single_lot_notional_15L"], abs=0.5)


def test_futures_slippage_and_carry_are_additive_and_optional():
    """The default preset is the worked example plus 4 bps of slippage; carry is separate."""
    base = FUTURES_PRESETS["worked_example"].round_trip_bps(LOT_NOTIONAL)
    dflt = FUTURES_PRESETS["zerodha_fut"].round_trip_bps(LOT_NOTIONAL)
    assert dflt - base == pytest.approx(4.0, abs=1e-9)      # 2.0 bps per side
    # holding cost is NOT in round_trip_bps -- it is charged per session held
    cm = FUTURES_PRESETS["zerodha_fut"]
    assert cm.holding_cost_bps(21) == pytest.approx(cm.roll_cost_bps_per_month)
    assert cm.holding_cost_bps(5) == pytest.approx(55.0 * 5 / 21)
    assert cm.holding_cost_bps(0) == 0.0
    # a 5-session hold pays more in carry than in the whole transaction round trip -- the
    # fact that decides sec.6.2: holding cost, not transaction cost, is what kills this venue
    assert cm.holding_cost_bps(5) > dflt
    assert FUTURES_PRESETS["zero"].round_trip_bps(LOT_NOTIONAL) == 0.0


def test_futures_margin_is_a_reported_range_not_a_constant():
    lo, hi = MARGIN_PCT_RANGE
    assert lo < MARGIN_PCT_DEFAULT < hi
    bd = FuturesCostModel().breakdown(LOT_NOTIONAL)
    assert bd["margin_rs"] == pytest.approx(MARGIN_PCT_DEFAULT * LOT_NOTIONAL)
    # the plan's pre-registered arithmetic: one lot's NOTIONAL is >= 150% of the Rs 10 lakh
    # book, and at the top of the margin band the initial margin alone takes over half of it
    book = 1_000_000.0
    assert LOT_NOTIONAL >= 1.5 * book
    assert hi * LOT_NOTIONAL > 0.5 * book


# --------------------------------------------------------------- cash model


def test_cash_zerodha_reproduces_the_module_docstring_figure():
    """~22.4 bps round trip at Rs 10 lakh legs, statutory + broker, before slippage."""
    cm = CostModel(slippage_bps_per_side=0.0)
    got = cm.round_trip_bps(CASH_LEG)
    assert got == pytest.approx(22.4, abs=0.05), f"round_trip_bps={got:.5f}, docstring says 22.4"
    # the default preset is that plus 5 bps of round-trip slippage, and nothing else
    assert PRESETS["zerodha"].round_trip_bps(CASH_LEG) == pytest.approx(got + 5.0, abs=1e-9)
    assert PRESETS["zerodha"].brokerage_rate == 0.0        # Zerodha delivery brokerage is zero


def test_cash_cost_falls_with_trade_size():
    """DP is flat and brokerage is capped, so bps falls with size -- the model is not scale-free.

    This is the property the plan's draft flat-bps model got wrong, and it is why the money bar
    is quoted at a stated capital: at Rs 10 lakh per leg the book pays ~1.4 bps less round trip
    than at Rs 1 lakh, all of it the DP charge on the sell leg.
    """
    cm = CostModel(slippage_bps_per_side=0.0)
    small, large = cm.round_trip_bps(100_000.0), cm.round_trip_bps(CASH_LEG)
    assert small > large
    floor = cm.round_trip_bps(1e9)
    assert large > floor >= 22.2                            # essentially all STT at the limit
    # the whole difference is the flat DP charge (plus its GST), scaled by notional
    dp_bps = cm.dp_charge_sell * (1 + cm.gst) * 1e4
    assert small - large == pytest.approx(dp_bps * (1 / 100_000.0 - 1 / CASH_LEG), rel=1e-9)


def test_cash_and_futures_breakdowns_are_self_consistent():
    """Every breakdown's total is the sum of its own lines and agrees with leg()/round_trip_bps."""
    for cm, notional in ((PRESETS["zerodha"], CASH_LEG),
                         (PRESETS["flat20"], CASH_LEG),
                         (FUTURES_PRESETS["zerodha_fut"], LOT_NOTIONAL),
                         (FUTURES_PRESETS["zerodha_fut_high_carry"], LOT_NOTIONAL)):
        bd = cm.breakdown(notional)
        for side in ("buy", "sell"):
            lines = {k: v for k, v in bd[side].items() if k != "total"}
            assert bd[side]["total"] == pytest.approx(sum(lines.values()))
            assert bd[side]["total"] == pytest.approx(cm.leg(notional, side))
        assert bd["round_trip_bps"] == pytest.approx(cm.round_trip_bps(notional))
        assert cm.one_way_bps(notional) == pytest.approx(cm.round_trip_bps(notional) / 2.0)
