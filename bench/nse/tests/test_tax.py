"""Test 8 of NIFTY50_PLAN.md sec.8: FIFO realisation, FY bucketing, set-off and the rates.

Gate M2 is stated in after-tax excess, so ``tax.py`` sits directly under a go/no-go decision
with no independent check on it -- an off-by-one in the financial-year boundary or a missed
loss set-off moves the verdict without moving anything visible. Every scenario below is built
so the right answer is arithmetic that fits on one line and is written in the test as a literal,
never as a re-derivation through the same code path.

The construction that makes that possible: a book holding ONE name at weight 1.0, whose daily
net return is that name's return. Then a fill's rupee notional is exactly the book value at the
open of the fill date, cost basis and proceeds are both readable off the equity curve, and the
realised gain is capital x (cumulative factor - 1). Costs are zeroed except where the STT
treatment is the thing under test, because the module's rule -- add STT back under ``stcg``,
leave it deducted under the business regimes -- is invisible at zero STT.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse import tax  # noqa: E402
from bench.nse.costs import CostModel  # noqa: E402

CAPITAL = 1_000_000.0

# No cost at all: the curve is the gross curve, so every realised gain is a clean literal.
ZERO_COST = CostModel(stt_buy=0, stt_sell=0, stamp_duty_buy=0, exchange_txn=0, sebi_fee=0,
                      brokerage_flat=0, dp_charge_sell=0, gst=0.0, slippage_bps_per_side=0)
# STT only, 0.10% both legs (the real delivery rate), nothing else: isolates the s.48 proviso.
STT_ONLY = CostModel(stt_buy=0.001, stt_sell=0.001, stamp_duty_buy=0, exchange_txn=0,
                     sebi_fee=0, brokerage_flat=0, dp_charge_sell=0, gst=0.0,
                     slippage_bps_per_side=0)


def _net(first: str, last: str, moves: dict[str, float]) -> pd.Series:
    """Daily net returns: zero on every business day in [first, last] except the named ones."""
    s = pd.Series(0.0, index=pd.bdate_range(first, last))
    for d, r in moves.items():
        s.loc[pd.Timestamp(d)] = r
    return s


def _blotter(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(exec_date, symbol, weight_delta) triples, the shape run_backtest(log_trades=True) emits."""
    return pd.DataFrame([{"exec_date": pd.Timestamp(d), "symbol": s, "weight_delta": float(w)}
                         for d, s, w in rows])


# --- The three-financial-year scenario every rate assertion below is stated against. -------
#
#     capital 10,00,000; one name held at a time, weight 1.0.
#     FY2023-24  buy 03 Apr 2023 @ 10,00,000  ->  sell 01 Aug 2023 @ 11,00,000   +1,00,000
#     FY2024-25  buy 01 Apr 2024 @ 11,00,000  ->  sell 01 Aug 2024 @ 10,45,000     -55,000
#     FY2025-26  buy 01 Apr 2025 @ 10,45,000  ->  sell 30 Jun 2025 @ 12,54,000   +2,09,000
#
THREE_FY_NET = _net("2023-04-03", "2025-06-30",
                    {"2023-06-01": 0.10, "2024-06-03": -0.05, "2025-06-02": 0.20})
THREE_FY_BLOTTER = _blotter([
    ("2023-04-03", "A", +1.0), ("2023-08-01", "A", -1.0),
    ("2024-04-01", "B", +1.0), ("2024-08-01", "B", -1.0),
    ("2025-04-01", "C", +1.0), ("2025-06-30", "C", -1.0),
])
PRE_TAX_FINAL = 1_254_000.0     # 10L x 1.10 x 0.95 x 1.20


def _parts(regime: str, cost_model=ZERO_COST):
    return tax.after_tax_curve(THREE_FY_NET, THREE_FY_BLOTTER, CAPITAL, regime,
                               cost_model, return_parts=True)


# ------------------------------------------------------------------ financial years


def test_fy_boundary_is_1_april():
    assert tax.fy_label("2023-03-31") == "FY2022-23"
    assert tax.fy_label("2023-04-01") == "FY2023-24"
    assert tax.fy_label("2024-03-31") == "FY2023-24"
    assert tax.fy_start_year("2024-01-15") == 2023
    assert tax.fy_start_year("2024-04-15") == 2024


# ------------------------------------------------------------------ fills and FIFO


def test_fills_are_weight_delta_times_the_opening_book_value():
    """Rupees come from the equity curve, and the first session transacts at `capital` itself."""
    f = tax.build_fills(THREE_FY_BLOTTER, THREE_FY_NET, CAPITAL, ZERO_COST)
    got = dict(zip(f["exec_date"].astype(str).str[:10], f["notional_rs"]))
    assert got["2023-04-03"] == pytest.approx(1_000_000.0)
    assert got["2023-08-01"] == pytest.approx(1_100_000.0)   # after the +10% on 01 Jun 2023
    assert got["2024-04-01"] == pytest.approx(1_100_000.0)   # nothing moved in between
    assert got["2024-08-01"] == pytest.approx(1_045_000.0)   # after the -5% on 03 Jun 2024
    assert got["2025-06-30"] == pytest.approx(1_254_000.0)
    assert list(f["side"]) == ["buy", "sell", "buy", "sell", "buy", "sell"]
    assert list(f["fy"]) == ["FY2023-24", "FY2023-24", "FY2024-25", "FY2024-25",
                             "FY2025-26", "FY2025-26"]


def test_fills_reject_a_blotter_that_is_not_from_the_same_run():
    """A fill on a session the curve never saw is a mismatched pair, not a rounding issue."""
    bad = _blotter([("2023-04-03", "A", 1.0), ("2030-01-02", "A", -1.0)])
    with pytest.raises(ValueError, match="not in the daily return index"):
        tax.build_fills(bad, THREE_FY_NET, CAPITAL, ZERO_COST)


def test_realised_gains_and_fy_bucketing():
    """Gain is proceeds minus FIFO cost, booked in the financial year of the SELL."""
    _, p = _parts("stcg")
    m = p["matches"]
    assert len(m) == 3
    assert list(m["gain_rs"]) == pytest.approx([100_000.0, -55_000.0, 209_000.0])
    assert list(m["fy"]) == ["FY2023-24", "FY2024-25", "FY2025-26"]
    assert list(m["term"]) == ["short", "short", "short"]      # 120, 122 and 90 days held
    assert list(m["holding_days"]) == [120, 122, 90]
    assert not m["partial"].any()
    # realised_by_fill collapses the matched pieces back to one row per sell fill
    per_fill = tax.realised_by_fill(m)
    assert len(per_fill) == 3
    assert per_fill["gain_rs"].sum() == pytest.approx(254_000.0)


def test_fifo_consumes_the_oldest_lot_first():
    """Two lots, a partial sell: the earliest buy is exhausted before the later one is touched.

    ``full_exit_frac`` is raised to 0.9 so the sell is read as a trim rather than an exit --
    the only path in which lot ordering is observable at all.
    """
    net = _net("2023-04-03", "2023-12-29", {"2023-05-01": 0.10, "2023-07-03": 0.10})
    bl = _blotter([("2023-04-03", "A", +0.4),    # 0.4 x 10,00,000 = 4,00,000
                   ("2023-06-01", "A", +0.4),    # 0.4 x 11,00,000 = 4,40,000
                   ("2023-08-01", "A", -0.4)])   # 0.4 x 12,10,000 = 4,84,000 proceeds
    f = tax.build_fills(bl, net, CAPITAL, ZERO_COST)
    m = tax.fifo_match(f, full_exit_frac=0.9)
    assert len(m) == 2
    assert list(m["buy_date"].astype(str).str[:10]) == ["2023-04-03", "2023-06-01"]
    # half the open cost basis (0.4 of 0.8 open weight) = 4,20,000, taken oldest-first:
    # the whole 4,00,000 first lot, then 20,000 of the second
    assert list(m["cost_rs"]) == pytest.approx([400_000.0, 20_000.0])
    # proceeds are split pro rata by the cost consumed -- the one approximation, flagged as partial
    assert list(m["proceeds_rs"]) == pytest.approx([484_000.0 * 400 / 420, 484_000.0 * 20 / 420])
    assert m["gain_rs"].sum() == pytest.approx(64_000.0)
    assert m["partial"].all()


def test_a_drifted_exit_is_read_as_a_full_exit():
    """The exit weight differs from the entry weight by the drift; 0.5 keeps that a full exit."""
    net = _net("2023-04-03", "2023-12-29", {"2023-05-01": 0.10})
    bl = _blotter([("2023-04-03", "A", +0.4),     # 4,00,000
                   ("2023-08-01", "A", -0.42)])   # 0.42 x 11,00,000 = 4,62,000
    m = tax.fifo_match(tax.build_fills(bl, net, CAPITAL, ZERO_COST))
    assert len(m) == 1
    assert not bool(m["partial"].iloc[0])
    assert m["cost_rs"].iloc[0] == pytest.approx(400_000.0)   # the whole lot, not 0.42/0.4 of it
    assert m["gain_rs"].iloc[0] == pytest.approx(62_000.0)


def test_a_sell_with_no_open_lot_refuses_to_guess_a_basis():
    net = _net("2023-04-03", "2023-12-29", {})
    bl = _blotter([("2023-08-01", "A", -0.4)])
    with pytest.raises(ValueError, match="no open lot"):
        tax.fifo_match(tax.build_fills(bl, net, CAPITAL, ZERO_COST))


# ------------------------------------------------------------------ rates and set-off


@pytest.mark.parametrize("regime,rate,fy1,fy3", [
    ("stcg", 0.208, 20_800.0, 32_032.0),
    ("business_31", 0.312, 31_200.0, 48_048.0),
    ("business_21", 0.208, 20_800.0, 32_032.0),
])
def test_rates_and_intra_window_loss_setoff(regime, rate, fy1, fy3):
    """20.8% / 31.2% on the taxable base, with FY2024-25's loss consumed by FY2025-26's gain.

    FY2025-26's taxable figure is 2,09,000 - 55,000 = 1,54,000 in all three regimes: a capital
    loss sets off against a later capital gain (s.74) and a business loss against a later
    business profit (s.72), so the mechanics coincide here even though the sections differ.
    """
    _, p = _parts(regime)
    t = p["tax"].set_index("fy")
    assert tax.REGIMES[regime]["rate"] == rate
    assert t.loc["FY2023-24", "tax_rs"] == pytest.approx(fy1)
    assert t.loc["FY2024-25", "tax_rs"] == 0.0
    assert t.loc["FY2025-26", "tax_rs"] == pytest.approx(fy3)
    assert t.loc["FY2024-25", "loss_carried_forward_rs"] == pytest.approx(55_000.0)
    assert t.loc["FY2025-26", "loss_brought_forward_rs"] == pytest.approx(55_000.0)
    assert t.loc["FY2025-26", "loss_setoff_rs"] == pytest.approx(55_000.0)
    assert t.loc["FY2025-26", "taxable_short_rs"] == pytest.approx(154_000.0)
    assert t.loc["FY2025-26", "loss_carried_forward_rs"] == 0.0
    assert t["tax_rs"].sum() == pytest.approx(fy1 + fy3)


def test_stt_is_added_back_under_stcg_and_left_deducted_under_business():
    """s.48 proviso: STT is not deductible against capital gains, but the curve already paid it.

    One round trip, gain 1,00,000 on a curve that is already net of 0.10% STT on each leg
    (Rs 1,000 buy + Rs 1,100 sell). Under ``stcg`` the base is 1,02,100; under the business
    regimes STT stays a deductible expense and the base is the 1,00,000 the curve shows.
    """
    net = _net("2023-04-03", "2024-03-29", {"2023-06-01": 0.10})
    bl = _blotter([("2023-04-03", "A", +1.0), ("2023-08-01", "A", -1.0)])
    f = tax.build_fills(bl, net, CAPITAL, STT_ONLY)
    assert list(f["stt_rs"]) == pytest.approx([1_000.0, 1_100.0])
    led = tax.fy_ledger(f, tax.fifo_match(f))
    assert led["realised_short_rs"].iloc[0] == pytest.approx(100_000.0)
    assert led["stt_rs"].iloc[0] == pytest.approx(2_100.0)

    st = tax.tax_by_fy(led, "stcg").iloc[0]
    assert st["taxable_short_rs"] == pytest.approx(102_100.0)
    assert st["tax_rs"] == pytest.approx(21_236.8)
    bz = tax.tax_by_fy(led, "business_31").iloc[0]
    assert bz["taxable_short_rs"] == pytest.approx(100_000.0)
    assert bz["tax_rs"] == pytest.approx(31_200.0)


def test_long_term_gain_in_the_book_gets_the_exemption_and_the_ltcg_rate():
    """A lot held past 365 days is long-term: 12.5% + cess above the Rs 1.25 lakh exemption."""
    net = _net("2023-04-03", "2025-03-31", {"2023-06-01": 0.30})
    bl = _blotter([("2023-04-03", "A", +1.0), ("2024-08-01", "A", -1.0)])
    f = tax.build_fills(bl, net, CAPITAL, ZERO_COST)
    m = tax.fifo_match(f)
    assert m["holding_days"].iloc[0] == 486
    assert m["term"].iloc[0] == "long"
    assert m["gain_rs"].iloc[0] == pytest.approx(300_000.0)
    t = tax.tax_by_fy(tax.fy_ledger(f, m), "stcg")
    row = t[t["fy"] == "FY2024-25"].iloc[0]
    assert row["realised_short_rs"] == 0.0
    assert row["ltcg_exemption_used_rs"] == pytest.approx(125_000.0)
    assert row["taxable_long_rs"] == pytest.approx(175_000.0)
    assert row["tax_rs"] == pytest.approx(0.13 * 175_000.0)     # 22,750
    # under a business regime the short/long split does not exist: one profit, one rate
    bus = tax.tax_by_fy(tax.fy_ledger(f, m), "business_31")
    assert bus[bus["fy"] == "FY2024-25"]["tax_rs"].iloc[0] == pytest.approx(0.312 * 300_000.0)


def test_carry_forward_expires_after_eight_financial_years():
    """s.74 / s.72: eight succeeding FYs, so a FY2010-11 loss dies unused in FY2019-20."""
    def ledger(gain_year: int) -> pd.DataFrame:
        rows = []
        for y in range(2010, gain_year + 1):
            rows.append({"fy": tax.fy_label(f"{y}-06-01"), "fy_start_year": y,
                         "realised_short_rs": (-100_000.0 if y == 2010 else
                                               100_000.0 if y == gain_year else 0.0),
                         "realised_long_rs": 0.0, "stt_rs": 0.0, "cost_rs": 0.0})
        return pd.DataFrame(rows)

    on_time = tax.tax_by_fy(ledger(2018), "stcg").iloc[-1]      # 8th succeeding year: still live
    assert on_time["loss_setoff_rs"] == pytest.approx(100_000.0)
    assert on_time["tax_rs"] == 0.0

    too_late = tax.tax_by_fy(ledger(2019), "stcg").iloc[-1]     # 9th: lapsed
    assert too_late["loss_lapsed_rs"] == pytest.approx(100_000.0)
    assert too_late["loss_setoff_rs"] == 0.0
    assert too_late["tax_rs"] == pytest.approx(20_800.0)


# ------------------------------------------------------------------ after-tax curve


def test_after_tax_curve_withdraws_at_fy_end_and_compounds_on_the_smaller_book():
    """Tax leaves the book on the last session of its FY, so later returns compound on less.

    The drag is therefore MORE than the tax paid: FY2023-24's Rs 20,800 is withdrawn on
    29 Mar 2024 and then misses the 0.95 x 1.20 = 1.14 the book earns afterwards.
    """
    curve, p = _parts("stcg")
    pre = tax.equity_curve(THREE_FY_NET, CAPITAL)
    assert pre.iloc[-1] == pytest.approx(PRE_TAX_FINAL)
    assert p["tax_paid_rs"] == pytest.approx(52_832.0)
    assert curve.iloc[-1] == pytest.approx(PRE_TAX_FINAL - 20_800.0 * 1.14 - 32_032.0)
    assert curve.iloc[-1] == pytest.approx(1_198_256.0)
    assert curve.iloc[-1] < PRE_TAX_FINAL - p["tax_paid_rs"]

    fy1_end = pd.Timestamp("2024-03-29")            # last business day of FY2023-24 in the index
    assert curve.loc[fy1_end] == pytest.approx(1_100_000.0 - 20_800.0)
    assert curve.loc[pd.Timestamp("2024-03-28")] == pytest.approx(1_100_000.0)
    # every FY in the index gets a row, including one with no trades at all
    assert list(p["ledger"]["fy"]) == ["FY2023-24", "FY2024-25", "FY2025-26"]

    assert _parts("business_31")[0].iloc[-1] == pytest.approx(1_170_384.0)
    assert _parts("business_21")[0].iloc[-1] == pytest.approx(1_198_256.0)


def test_after_tax_curve_is_a_bare_series_by_default():
    """The common call site wants a Series; return_parts is opt-in for the report."""
    c = tax.after_tax_curve(THREE_FY_NET, THREE_FY_BLOTTER, CAPITAL, "stcg", ZERO_COST)
    assert isinstance(c, pd.Series)
    assert c.name == "after_tax_stcg"
    assert len(c) == len(THREE_FY_NET)


def test_unknown_regime_is_refused():
    with pytest.raises(ValueError, match="unknown regime"):
        tax.tax_by_fy(_parts("stcg")[1]["ledger"], "business_15")


# ------------------------------------------------------------------ the tracker


def test_tracker_ltcg_exemption_arithmetic():
    """One buy, one sale after 12 months: 12.5% + cess on the gain above Rs 1.25 lakh."""
    b = _net("2023-04-03", "2025-06-30", {"2023-06-01": 0.30})
    assert (b.index[-1] - b.index[0]).days == 819           # comfortably long-term
    final = tax.tracker_after_tax(b, CAPITAL, realise=True)
    assert final == pytest.approx(1_300_000.0 - 0.13 * (300_000.0 - 125_000.0))
    assert final == pytest.approx(1_277_250.0)
    # the exemption is applied once, on exit -- not once per financial year
    assert tax.LTCG_EXEMPTION == 125_000.0
    assert tax.LTCG_RATE == pytest.approx(0.13)


def test_tracker_gain_inside_the_exemption_pays_nothing():
    b = _net("2023-04-03", "2025-06-30", {"2023-06-01": 0.10})
    assert tax.tracker_after_tax(b, CAPITAL, realise=True) == pytest.approx(1_100_000.0)


def test_tracker_short_window_pays_stcg_and_the_unrealised_variant_pays_nothing():
    """Under 12 months the tracker is short-term at 20.8%; if it is never sold, no tax at all."""
    short = _net("2024-01-01", "2024-06-28", {"2024-02-01": 0.30})
    assert (short.index[-1] - short.index[0]).days <= tax.LONG_TERM_DAYS
    assert tax.tracker_after_tax(short, CAPITAL, realise=True) == pytest.approx(
        1_300_000.0 - 0.208 * 300_000.0)
    long_win = _net("2023-04-03", "2025-06-30", {"2023-06-01": 0.30})
    assert tax.tracker_after_tax(long_win, CAPITAL, realise=False) == pytest.approx(1_300_000.0)
    # a losing tracker pays nothing either way
    losing = _net("2023-04-03", "2025-06-30", {"2023-06-01": -0.10})
    assert tax.tracker_after_tax(losing, CAPITAL, realise=True) == pytest.approx(900_000.0)


# ------------------------------------------------------------------ summary


def test_summary_reports_after_tax_return_per_regime_and_tax_per_fy():
    bench = _net("2023-04-03", "2025-06-30", {"2023-06-01": 0.30})
    s = tax.summary(THREE_FY_NET, THREE_FY_BLOTTER, CAPITAL, ZERO_COST, daily_bench=bench)
    assert s["n_days"] == 586
    assert s["pre_tax_final_rs"] == pytest.approx(PRE_TAX_FINAL)
    assert s["pre_tax_ann_return"] == pytest.approx(
        tax.annualised(PRE_TAX_FINAL, CAPITAL, 586))
    assert set(s["regimes"]) == {"stcg", "business_31", "business_21"}

    st = s["regimes"]["stcg"]
    assert st["after_tax_final_rs"] == pytest.approx(1_198_256.0)
    assert st["total_tax_rs"] == pytest.approx(52_832.0)
    assert st["tax_by_fy"] == {"FY2023-24": pytest.approx(20_800.0),
                               "FY2024-25": 0.0,
                               "FY2025-26": pytest.approx(32_032.0)}
    assert s["regimes"]["business_31"]["total_tax_rs"] == pytest.approx(79_248.0)
    assert st["after_tax_ann_return"] > s["regimes"]["business_31"]["after_tax_ann_return"]

    # the book realises everything it earns; the tracker realises once, on exit
    assert s["realised_gain_rs"] == pytest.approx(254_000.0)
    assert s["unrealised_gain_rs"] == pytest.approx(PRE_TAX_FINAL - CAPITAL - 254_000.0)
    assert s["n_fills"] == 6 and s["n_sell_matches"] == 3 and s["n_partial_sells"] == 0
    assert s["tracker"]["after_tax_final_rs_on_exit"] == pytest.approx(1_277_250.0)
    assert s["tracker"]["after_tax_final_rs_unrealised"] == pytest.approx(1_300_000.0)
    assert st["ann_excess_vs_tracker_after_tax"] == pytest.approx(
        st["after_tax_ann_return"] - s["tracker"]["after_tax_ann_return_on_exit"])
    # this synthetic book loses to the tracker after tax -- exactly the comparison gate M2 makes
    assert st["ann_excess_vs_tracker_after_tax"] < 0
    assert "not tax advice" in s["disclaimer"]


# ------------------------------------------------------------------ against a real backtest


def _synthetic_run(weighting: str, seed: int, n_dates: int = 260, n_sym: int = 12):
    """A real ``run_backtest`` on a synthetic market -- the only producer of the blotter shape."""
    import numpy as np

    from bench.nse.backtest import Market, run_backtest

    rng = np.random.default_rng(seed)
    cal = list(pd.bdate_range("2023-01-02", periods=n_dates))
    syms = [f"S{i:02d}" for i in range(n_sym)]
    px = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=(n_dates, n_sym)), axis=0))
    panel = pd.DataFrame([{"date": d, "symbol": s, "open": px[i, j], "close": px[i, j] * 1.001}
                          for i, d in enumerate(cal) for j, s in enumerate(syms)])
    scores = pd.DataFrame(rng.normal(size=(n_dates, n_sym)), index=cal, columns=syms)
    return run_backtest(scores, Market(panel, cal), topk=4, n_drop=1, hold_thresh=2,
                        rebalance=5, cost_model=ZERO_COST, capital=CAPITAL,
                        weighting=weighting, log_trades=True)


@pytest.mark.parametrize("weighting", ["drift", "rebalance"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reconstruction_closes_against_a_real_backtest(weighting, seed):
    """Realised + unrealised must equal the curve's total gain, to the rupee, on a real blotter.

    The identity that makes this a real check rather than a tautology: with no costs and a
    fully-invested long-only book, every rupee of realised gain is reinvested, so the open lots'
    cost basis must exceed the starting capital by EXACTLY the cumulative realised gain. Nothing
    in ``tax.py`` is told the book value of anything -- it reconstructs rupees from weights and
    the equity curve alone -- so if the drift accounting, the FIFO consumption or the pro-rata
    proceeds split of a partial sell were wrong, this would not close.

    ``weighting="rebalance"`` is included because it is the only mode that produces genuinely
    partial sells, which is where the reconstruction's single approximation lives.
    """
    res = _synthetic_run(weighting, seed)
    fills = tax.build_fills(res["trades"], res["daily"], CAPITAL, ZERO_COST)
    matches = tax.fifo_match(fills)
    realised = float(matches["gain_rs"].sum())
    open_cost = float(fills[fills["side"] == "buy"]["notional_rs"].sum()
                      - matches["cost_rs"].sum())
    final = float(tax.equity_curve(res["daily"], CAPITAL).iloc[-1])

    assert open_cost - CAPITAL == pytest.approx(realised, abs=0.01)
    unrealised = final - open_cost
    assert realised + unrealised == pytest.approx(final - CAPITAL, abs=0.01)
    assert len(matches) > 20                       # the identity is not vacuously true
    if weighting == "rebalance":
        assert matches["partial"].sum() > 0        # the approximation is exercised
    else:
        assert matches["partial"].sum() == 0       # topk-dropout only ever exits in full

    # summary() reports the same two halves it just proved add up
    s = tax.summary(res["daily"], res["trades"], CAPITAL, ZERO_COST)
    assert s["realised_gain_rs"] == pytest.approx(realised, abs=0.01)
    assert s["unrealised_gain_rs"] == pytest.approx(unrealised, abs=0.01)


def test_float_dust_in_the_blotter_is_not_a_fill():
    """A 1e-17 weight delta is renormalisation noise, not a trade, and must not become a lot."""
    res = _synthetic_run("drift", 2)
    bl = res["trades"]
    dust = bl[bl["weight_delta"].abs() < 1e-9]
    assert len(dust) == 1, "this seed is chosen because run_backtest emits exactly one dust row"
    assert abs(float(dust["weight_delta"].iloc[0])) < 1e-15
    fills = tax.build_fills(bl, res["daily"], CAPITAL, ZERO_COST)
    assert len(fills) == len(bl) - 1
    assert (fills["notional_rs"] > 1e-6).all()
    # keeping it would have flagged a spurious partial sell, which is the diagnostic that
    # reports how much of the reconstruction is approximate
    kept = tax.fifo_match(tax.build_fills(bl, res["daily"], CAPITAL, ZERO_COST, min_weight=0.0))
    assert int(kept["partial"].sum()) == 0    # harmless here, but the row is still not a trade
    assert len(tax.build_fills(bl, res["daily"], CAPITAL, ZERO_COST, min_weight=0.0)) == len(bl)


def test_summary_survives_an_empty_blotter():
    """A run that never traded must produce a ledger, not a KeyError."""
    s = tax.summary(THREE_FY_NET, pd.DataFrame(columns=["exec_date", "symbol", "weight_delta"]),
                    CAPITAL, ZERO_COST)
    for reg in ("stcg", "business_31", "business_21"):
        assert s["regimes"][reg]["total_tax_rs"] == 0.0
    assert s["realised_gain_rs"] == 0.0
    assert s["unrealised_gain_rs"] == pytest.approx(PRE_TAX_FINAL - CAPITAL)
