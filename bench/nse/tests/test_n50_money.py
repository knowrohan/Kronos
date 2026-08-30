"""Gates M2/M3 and the total-return benchmark leg: arithmetic checks on synthetic series.

Nothing here reads ``bench/out`` or ``data/nse``. Every number is either hand-computed in the
test or recomputed from ``tax.py``'s own primitives, so a failure means ``n50_money.py`` or the
benchmark construction in ``n50_backtest.py`` changed behaviour -- not that a data file moved.

The two things worth testing here are the two places a money number can go quietly wrong:
the benchmark's construction (a stale open feed, an accrual added the wrong way round, a
convention that measures a different holding period from the book's) and the after-tax excess
(a difference of annualised returns that must be taken over the same sessions, with the tracker
on LTCG-on-exit and the book on its own regime).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse import tax  # noqa: E402
from bench.nse.backtest import TRADING_DAYS, Market, annualise  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402
from bench.nse.n50_backtest import (BENCH_SPECS, TR_ACCRUAL_ANNUAL,  # noqa: E402
                                    TR_ACCRUAL_PER_SESSION, bench_series)
from bench.nse.n50_money import (DESC_BENCH, GATE_BENCH, M2_EXCESS_BAR,  # noqa: E402
                                 PRIMARY_CELL, accrual_sensitivity, after_tax_block,
                                 calendar_year_table, m2_gates, m3_gates, render_md,
                                 tracker_block)

CAPITAL = 1e6
CM = PRESETS["zerodha"]


# ------------------------------------------------------------------ the accrual (plan sec.11)


def test_tr_accrual_compounds_to_117bp_per_year():
    """252 sessions of the per-session accrual must compound to exactly +1.17 pp, to 1e-9.

    This is the one hand-entered number in the benchmark. Compounded, not divided: the naive
    0.0117/252 would land low, and the test pins which convention is in force.
    """
    assert TR_ACCRUAL_ANNUAL == 0.0117
    compounded = (1.0 + TR_ACCRUAL_PER_SESSION) ** TRADING_DAYS - 1.0
    assert compounded == pytest.approx(0.0117, abs=1e-9)
    # and the simple-division convention is measurably different, so the choice matters
    naive = (1.0 + 0.0117 / TRADING_DAYS) ** TRADING_DAYS - 1.0
    assert naive > compounded
    assert naive - compounded == pytest.approx(6.8e-5, abs=1e-5)


def _bench_parquet(tmp_path, n: int = 40) -> tuple[str, Market, pd.DatetimeIndex]:
    """A benchmark file whose open and close series move differently, plus a matching Market."""
    cal = pd.bdate_range("2024-01-01", periods=n)
    op = pd.Series(100.0 * 1.001 ** np.arange(n), index=cal)     # +10 bps a session on the open
    cl = pd.Series(200.0 * 1.002 ** np.arange(n), index=cal)     # +20 bps a session on the close
    b = pd.DataFrame({"date": cal, "nsei": cl.to_numpy(), "nsei_open": op.to_numpy(),
                      "niftybees_adj": cl.to_numpy(), "niftybees_adj_open": op.to_numpy()})
    path = str(tmp_path / "bench_n50.parquet")
    b.to_parquet(path, index=False)
    panel = pd.DataFrame({"date": cal, "symbol": "AAA", "open": 10.0, "close": 10.0})
    return path, Market(panel, list(cal)), cal


def test_niftybees_leg_is_close_to_close_and_nsei_legs_are_open_to_open(tmp_path):
    """The NIFTYBEES leg must read the CLOSE series (its open feed is stale, plan sec.11)."""
    path, market, cal = _bench_parquet(tmp_path)
    nb, _ = bench_series(market, "nsei_tr_proxy", path)
    px, _ = bench_series(market, "nsei", path)
    bees, _ = bench_series(market, "niftybees_adj", path)
    assert BENCH_SPECS["niftybees_adj"]["price"] == "close"
    assert BENCH_SPECS["nsei_tr_proxy"]["price"] == "open"
    # open series moves 10 bps a session, close series 20 bps: the legs are distinguishable
    assert float(px.iloc[0]) == pytest.approx(0.001, abs=1e-12)
    assert float(bees.iloc[0]) == pytest.approx(0.002, abs=1e-12)
    # the proxy is exactly the price leg plus the flat accrual, session by session
    assert np.allclose((nb - px).dropna().to_numpy(), TR_ACCRUAL_PER_SESSION, atol=1e-15)
    # ... and nothing is accrued past the end of the series, where the return is undefined
    assert np.isnan(nb.iloc[-1]) and np.isnan(px.iloc[-1])


def test_accrual_survives_a_forward_filled_hole(tmp_path):
    """A vendor hole is forward-filled on the PRICE; the dividend still accrues that session."""
    path, market, cal = _bench_parquet(tmp_path)
    b = pd.read_parquet(path)
    b.loc[5, "nsei_open"] = np.nan
    b.to_parquet(path, index=False)
    s, filled = bench_series(market, "nsei_tr_proxy", path)
    assert filled == 1
    assert float(s.iloc[4]) == pytest.approx(TR_ACCRUAL_PER_SESSION, abs=1e-12)  # flat + accrual
    assert float(s.iloc[5]) == pytest.approx(0.001 * 2 + 0.001 ** 2 + TR_ACCRUAL_PER_SESSION,
                                             abs=1e-12)                          # two-day move


# ------------------------------------------------------------------ after-tax arithmetic


def _book(n: int = 500, r: float = 0.0004, b_proxy: float = 0.0003,
          b_bees: float = 0.00025, sell_at: int = 250):
    """Constant-return book and trackers, with one round trip in the blotter."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    daily = pd.DataFrame({"net_return": r, "gross_return": r, "cost": 0.0, "turnover": 0.0},
                         index=idx)
    trades = pd.DataFrame([{"exec_date": idx[0], "symbol": "AAA", "weight_delta": 1.0},
                           {"exec_date": idx[sell_at], "symbol": "AAA", "weight_delta": -1.0}])
    bench = pd.DataFrame({GATE_BENCH: b_proxy, "niftybees_adj": b_bees}, index=idx)
    return daily, trades, bench, idx


def test_tracker_block_matches_the_section_112a_arithmetic_by_hand():
    """One purchase, one sale at the end: 12.5% + 4% cess above Rs 1.25 lakh, and nothing else."""
    n, b = 500, 0.0003
    idx = pd.bdate_range("2024-01-02", periods=n)
    s = pd.Series(b, index=idx)
    blk = tracker_block(s, CAPITAL, n)
    final = CAPITAL * (1.0 + b) ** n
    gain = final - CAPITAL
    assert blk["held_days"] > tax.LONG_TERM_DAYS and blk["term"] == "long"
    want = final - tax.LTCG_RATE * (gain - tax.LTCG_EXEMPTION)
    assert blk["after_tax_final_rs_on_exit"] == pytest.approx(want, rel=1e-12)
    assert blk["after_tax_ann_return_on_exit"] == pytest.approx(
        (want / CAPITAL) ** (TRADING_DAYS / n) - 1.0, rel=1e-12)
    # the never-sells variant pays nothing, so its annualised return is the raw one
    assert blk["after_tax_ann_return_unrealised"] == pytest.approx(
        (1.0 + b) ** TRADING_DAYS - 1.0, rel=1e-9)
    assert blk["tax_paid_on_exit_rs"] == pytest.approx(tax.LTCG_RATE * (gain - tax.LTCG_EXEMPTION),
                                                       rel=1e-12)


def test_after_tax_excess_is_book_minus_tracker_over_the_same_sessions():
    """The reported excess must equal the two annualised after-tax returns, recomputed here."""
    daily, trades, bench, idx = _book()
    n = len(idx)
    out = after_tax_block(daily, trades, bench, CAPITAL, CM)

    assert out["pre_tax_ann_return"] == pytest.approx((1.0 + 0.0004) ** TRADING_DAYS - 1.0,
                                                      rel=1e-9)
    for reg in ("stcg", "business_31", "business_21"):
        curve = tax.after_tax_curve(daily["net_return"], trades, CAPITAL, reg, CM)
        want_book = tax.annualised(float(curve.iloc[-1]), CAPITAL, n)
        assert out["regimes"][reg]["after_tax_ann_return"] == pytest.approx(want_book, rel=1e-12)
        for t in (GATE_BENCH, "niftybees_adj"):
            tr = tax.tracker_after_tax(bench[t], CAPITAL, realise=True)
            want = want_book - tax.annualised(tr, CAPITAL, n)
            assert out["after_tax_excess"][reg][t]["on_exit"] == pytest.approx(want, rel=1e-12)
            un = tax.tracker_after_tax(bench[t], CAPITAL, realise=False)
            assert out["after_tax_excess"][reg][t]["unrealised"] == pytest.approx(
                want_book - tax.annualised(un, CAPITAL, n), rel=1e-12)


def test_higher_tax_rate_never_raises_the_after_tax_return():
    """business_31 is 31.2% on the same base business_21 taxes at 20.8%: it cannot pay more."""
    daily, trades, bench, _ = _book()
    out = after_tax_block(daily, trades, bench, CAPITAL, CM)
    a = out["regimes"]["business_31"]["after_tax_ann_return"]
    c = out["regimes"]["business_21"]["after_tax_ann_return"]
    assert a <= c
    assert out["regimes"]["business_31"]["total_tax_rs"] >= \
        out["regimes"]["business_21"]["total_tax_rs"]
    # a book that never sells realises nothing and is taxed nothing
    flat = after_tax_block(daily, pd.DataFrame(columns=["exec_date", "symbol", "weight_delta"]),
                           bench, CAPITAL, CM)
    assert flat["regimes"]["stcg"]["total_tax_rs"] == 0.0
    assert flat["regimes"]["stcg"]["after_tax_ann_return"] == pytest.approx(
        flat["pre_tax_ann_return"], rel=1e-12)


# ------------------------------------------------------------------ calendar years


def test_calendar_year_excess_cumulative_and_annualised():
    """Both readings are exact on a constant-return series, and the partial flag is honest."""
    idx = (list(pd.bdate_range("2024-01-01", "2024-12-31"))
           + list(pd.bdate_range("2025-01-01", "2025-03-31")))
    idx = pd.DatetimeIndex(idx)
    daily = pd.DataFrame({"net_return": 0.0005}, index=idx)
    bench = pd.DataFrame({GATE_BENCH: 0.0002, "niftybees_adj": 0.0002}, index=idx)
    rows = calendar_year_table(daily, bench)
    assert [r["year"] for r in rows] == [2024, 2025]
    assert rows[0]["partial"] is False and rows[1]["partial"] is True
    for r in rows:
        n = r["n_days"]
        assert r["book_return_cum"] == pytest.approx(1.0005 ** n - 1.0, rel=1e-12)
        assert r[f"bench_{GATE_BENCH}_cum"] == pytest.approx(1.0002 ** n - 1.0, rel=1e-12)
        assert r[f"excess_{GATE_BENCH}_cum"] == pytest.approx(
            (1.0005 ** n - 1.0) - (1.0002 ** n - 1.0), rel=1e-12)
        assert r[f"excess_{GATE_BENCH}_ann"] == pytest.approx(
            (1.0005 ** TRADING_DAYS) - (1.0002 ** TRADING_DAYS), rel=1e-9)
        assert r[f"excess_{GATE_BENCH}_sign_disagrees"] is False


def test_calendar_year_flags_a_sign_disagreement_rather_than_hiding_it():
    """If the cumulative and annualised readings disagree in sign, the row says so."""
    idx = pd.DatetimeIndex(list(pd.bdate_range("2024-11-01", "2024-12-31")))
    daily = pd.DataFrame({"net_return": np.r_[np.full(len(idx) - 1, 0.001), -0.10]}, index=idx)
    bench = pd.DataFrame({GATE_BENCH: 0.0, "niftybees_adj": 0.0}, index=idx)
    rows = calendar_year_table(daily, bench)
    r = rows[0]
    # a loss compounds harder when annualised, so both readings stay negative here; the flag
    # must be a real comparison of the two signs, not a constant
    assert r[f"excess_{GATE_BENCH}_sign_disagrees"] == (
        (r[f"excess_{GATE_BENCH}_cum"] > 0) != (r[f"excess_{GATE_BENCH}_ann"] > 0))


# ------------------------------------------------------------------ the gates


def _met(ex_gate: float, k15: float = 0.01, k10d2: float = 0.01, ew: float = 0.01) -> dict:
    def cell(x):
        return {GATE_BENCH: {"ann_excess_net": x, "excess_ci95_annualised": [x - 0.05, x + 0.05],
                             "ann_return_net": 0.1, "ann_return_gross": 0.11, "n_days": 500,
                             "information_ratio": 0.5, "max_drawdown": -0.2, "turnover_ann": 10.0,
                             "cost_drag_ann_pct": 1.5, "one_way_cost_bps": 14.39},
                "niftybees_adj": {"ann_excess_net": x + 0.001,
                                  "excess_ci95_annualised": [x - 0.05, x + 0.05],
                                  "ann_return_net": 0.1, "ann_return_gross": 0.11, "n_days": 500,
                                  "information_ratio": 0.5, "max_drawdown": -0.2,
                                  "turnover_ann": 10.0, "cost_drag_ann_pct": 1.5,
                                  "one_way_cost_bps": 14.39},
                "ew_n50_drift": {"ann_excess_net": ew},
                "ew_n50_rebal": {"ann_excess_net": ew},
                "const_score": {"ann_excess_net": ew}}
    return {"books": {"k10_d1": cell(ex_gate), "k10_d2": cell(k10d2),
                      "k15_d1": cell(k15), "k15_d2": cell(k15)}}


def _atx(stcg: float, b31: float) -> dict:
    return {"after_tax_excess": {
        "stcg": {GATE_BENCH: {"on_exit": stcg, "unrealised": stcg},
                 "niftybees_adj": {"on_exit": stcg, "unrealised": stcg}},
        "business_31": {GATE_BENCH: {"on_exit": b31, "unrealised": b31},
                        "niftybees_adj": {"on_exit": b31, "unrealised": b31}},
        "business_21": {GATE_BENCH: {"on_exit": b31, "unrealised": b31},
                        "niftybees_adj": {"on_exit": b31, "unrealised": b31}}}}


def test_m2_passes_only_when_all_three_clauses_pass():
    """The bar is >= +4.1 pp, and both tax regimes must be positive; each flips on its own."""
    ok = m2_gates(_met(0.05), _atx(0.02, 0.01), "t")
    assert ok["verdict"] is True and all(c["pass"] for c in ok["clauses"])
    # exactly at the bar passes; a basis point below it does not
    assert m2_gates(_met(M2_EXCESS_BAR), _atx(0.02, 0.01), "t")["clauses"][0]["pass"] is True
    assert m2_gates(_met(M2_EXCESS_BAR - 1e-4), _atx(0.02, 0.01),
                    "t")["clauses"][0]["pass"] is False
    assert m2_gates(_met(0.05), _atx(-0.001, 0.01), "t")["verdict"] is False
    assert m2_gates(_met(0.05), _atx(0.02, -0.001), "t")["verdict"] is False
    # a failing gate still reports the description benchmark next to the gate one
    c = m2_gates(_met(0.02), _atx(0.02, 0.01), "t")["clauses"][0]
    assert c["also_vs_niftybees_adj"] == pytest.approx(0.021)


def _years(y24: float, y25: float, y26: float) -> list[dict]:
    out = []
    for y, v in ((2023, 0.5), (2024, y24), (2025, y25), (2026, y26)):
        out.append({"year": y, "n_days": 250, "partial": False,
                    "book_return_cum": v, "book_return_ann": v,
                    f"excess_{GATE_BENCH}_cum": v, f"excess_{GATE_BENCH}_ann": v,
                    f"excess_{GATE_BENCH}_sign_disagrees": False,
                    f"bench_{GATE_BENCH}_cum": 0.0, f"bench_{GATE_BENCH}_ann": 0.0,
                    "bench_niftybees_adj_cum": 0.0, "bench_niftybees_adj_ann": 0.0,
                    "excess_niftybees_adj_cum": v, "excess_niftybees_adj_ann": v,
                    "excess_niftybees_adj_sign_disagrees": False})
    return out


PROBE = {"contamination": {"checkpoint": "2025-06-30",
                           "pre": {"mean_ic": 0.01}, "post": {"mean_ic": 0.03}},
         "fingerprint": {"reversal_3d": 0.13, "momentum_10d": -0.12}}


def test_m3_counts_three_years_and_ignores_2023():
    """2023 is reported but never counted, so a huge 2023 cannot rescue two negative years."""
    g = m3_gates(_met(0.05), _years(-0.01, -0.02, -0.03), PROBE, "t")
    a = g["clauses"][0]
    assert a["pass"] is False and a["value"] == 0 and a["counted_years"] == [2024, 2025, 2026]
    assert m3_gates(_met(0.05), _years(0.01, 0.02, -0.03), PROBE, "t")["clauses"][0]["pass"] is True
    assert m3_gates(_met(0.05), _years(0.01, -0.02, -0.03),
                    PROBE, "t")["clauses"][0]["pass"] is False


def test_m3_variant_and_equal_weight_clauses():
    """k15_d1 and k10_d2 must BOTH be positive, and the book must beat the equal-weight leg."""
    g = m3_gates(_met(0.05, k15=0.01, k10d2=0.01, ew=0.01), _years(0.1, 0.1, 0.1), PROBE, "t")
    assert g["verdict"] is True
    assert m3_gates(_met(0.05, k15=-0.001), _years(0.1, 0.1, 0.1),
                    PROBE, "t")["clauses"][2]["pass"] is False
    assert m3_gates(_met(0.05, k10d2=-0.001), _years(0.1, 0.1, 0.1),
                    PROBE, "t")["clauses"][2]["pass"] is False
    assert m3_gates(_met(0.05, ew=-0.001), _years(0.1, 0.1, 0.1),
                    PROBE, "t")["clauses"][4]["pass"] is False


def test_m3_probe_clauses_and_the_undecided_case():
    """Fingerprint is a max over |rho|; with no probe the two clauses are UNDECIDED, not PASS."""
    g = m3_gates(_met(0.05), _years(0.1, 0.1, 0.1), PROBE, "t")
    assert g["clauses"][1]["pass"] is True and g["clauses"][1]["value"] == pytest.approx(0.02)
    assert g["clauses"][3]["value"] == pytest.approx(0.13) and g["clauses"][3]["pass"] is True
    hot = {"contamination": {"checkpoint": "x", "pre": {"mean_ic": 0.05},
                             "post": {"mean_ic": 0.01}},
           "fingerprint": {"reversal_3d": -0.44}}
    g2 = m3_gates(_met(0.05), _years(0.1, 0.1, 0.1), hot, "t")
    assert g2["clauses"][1]["pass"] is False and g2["clauses"][3]["pass"] is False
    g3 = m3_gates(_met(0.05), _years(0.1, 0.1, 0.1), None, "t")
    assert g3["verdict"] is None and g3["n_unknown_clauses"] == 2
    assert all(c["pass"] is None for c in g3["clauses"] if c["name"] in
               ("b_post_ge_pre_ic", "d_fingerprint"))


# ------------------------------------------------------------------ the report renders


def test_render_md_produces_every_section_and_no_crash():
    daily, trades, bench, idx = _book()
    atx = after_tax_block(daily, trades, bench, CAPITAL, CM)
    met = _met(0.05)
    years = calendar_year_table(daily, bench)
    payload = {
        "meta": {"written_at": "now", "capital_rs": CAPITAL, "cost_preset": "zerodha",
                 "tr_accrual_annual": TR_ACCRUAL_ANNUAL,
                 "tr_accrual_per_session": TR_ACCRUAL_PER_SESSION},
        "decision_tag": "t",
        "tags": {"t": {"window": {}, "books": met["books"],
                       "benchmark_books": {"ew_n50_drift": {
                           GATE_BENCH: {"ann_excess_net": 0.01, "ann_return_net": 0.06},
                           "niftybees_adj": {"ann_excess_net": 0.01, "ann_return_net": 0.06}}},
                       "null_percentile": 84.2,
                       "null": {"n_seeds": 500, "by_benchmark": {GATE_BENCH: {
                           "percentile_of_observed": 84.2, "p_value_one_sided": 0.158,
                           "z_vs_null": 0.93, "null_mean": -0.006, "null_sd": 0.046}}},
                       "after_tax": {PRIMARY_CELL: atx}, "years": {PRIMARY_CELL: years},
                       "m2": m2_gates(met, atx, "t"),
                       "m3": m3_gates(met, years, PROBE, "t")}},
    }
    md = render_md(payload)
    for section in ("# NIFTY 50 money test", "### M2 clauses", "### M3 clauses",
                    "## Windows", "## Net-of-cost books", "## After tax", "## Calendar years"):
        assert section in md
    assert "PASS" in md and "nsei_tr_proxy" in md
    json.dumps(payload, default=float)      # the same payload must be serialisable


def test_render_md_null_sentence_and_course_warning_track_the_numbers():
    """The two editorial sentences must be computed, not fixed at writing time.

    ``render_md`` closes the verdict block with a sentence about the random-score null and the
    cross-tag table with one about the tags' different windows. Both used to state a number
    ("a sixth", "486 sessions from 2024-08-26") that was true of one run and silently false of
    every other. This pins them to the payload: a null the book clears must not be described as
    one that beat it, and the course warning must name the two tags this run actually put
    furthest apart.
    """
    daily, trades, bench, idx = _book()
    atx = after_tax_block(daily, trades, bench, CAPITAL, CM)
    met = _met(0.05)
    years = calendar_year_table(daily, bench)

    def tag_block(p_value: float) -> dict:
        return {"window": {}, "books": met["books"], "benchmark_books": {},
                "null_percentile": 100.0 * (1.0 - p_value),
                "null": {"n_seeds": 500, "by_benchmark": {GATE_BENCH: {
                    "percentile_of_observed": 100.0 * (1.0 - p_value),
                    "p_value_one_sided": p_value, "z_vs_null": 0.93,
                    "null_mean": -0.006, "null_sd": 0.046}}},
                "after_tax": {PRIMARY_CELL: atx}, "years": {PRIMARY_CELL: years},
                "m2": m2_gates(met, atx, "t"), "m3": m3_gates(met, years, PROBE, "t")}

    base = {"meta": {"written_at": "now", "capital_rs": CAPITAL, "cost_preset": "zerodha",
                     "tr_accrual_annual": TR_ACCRUAL_ANNUAL,
                     "tr_accrual_per_session": TR_ACCRUAL_PER_SESSION},
            "decision_tag": "loser"}

    # p = 0.158 -> 79 of 500 draws beat it, roughly 1 in 6, and it is NOT an edge
    md = render_md({**base, "tags": {"loser": tag_block(0.158)}})
    assert "79 of the random score vectors (1 in 6) matched or beat" in md
    assert "not a demonstrated edge" in md
    assert "a sixth" not in md                       # the old hard-coded adjective is gone

    # the same renderer on a book the null does NOT contain must not call it a non-edge
    md_ok = render_md({**base, "decision_tag": "winner",
                       "tags": {"winner": tag_block(0.002)}})
    assert "outside the random-score distribution at the 5% level" in md_ok
    assert "not a demonstrated edge" not in md_ok

    # one tag -> no cross-tag warning to make; two tags -> it names the pair and the spread
    assert "no cross-tag comparison" in md
    two = {**base, "tags": {"loser": tag_block(0.158), "other": tag_block(0.158)}}
    two["tags"]["other"] = json.loads(json.dumps(two["tags"]["other"], default=float))
    two["tags"]["other"]["after_tax"] = {PRIMARY_CELL: dict(atx)}
    two["tags"]["other"]["after_tax"][PRIMARY_CELL] = {
        **atx, "trackers": {**atx["trackers"],
                            GATE_BENCH: {**atx["trackers"][GATE_BENCH],
                                         "pre_tax_ann_return":
                                             atx["trackers"][GATE_BENCH]["pre_tax_ann_return"]
                                             + 0.10}}}
    md2 = render_md(two)
    assert "`loser`" in md2 and "`other`" in md2
    assert "a spread of +10.00 pp/yr in the course itself" in md2


def test_accrual_sensitivity_strips_the_constant_back_out_and_finds_the_flip():
    """The hand-set TR accrual must be reported as a band, and the flip point must be real.

    ``nsei_tr_proxy`` is a price index plus +1.17 pp/yr that nobody measured on this book, and
    every pp of it comes off the excess one-for-one. A gate decided near the bar could therefore
    be an artefact of the constant, so the file has to say where the verdict would change.
    """
    idx = pd.bdate_range("2024-01-01", periods=504)
    # book at a flat +0.05%/session; benchmark = flat price leg + the live accrual
    book = pd.DataFrame({"net_return": np.full(len(idx), 0.0005)}, index=idx)
    per = TR_ACCRUAL_PER_SESSION
    bench = pd.DataFrame({GATE_BENCH: np.full(len(idx), 0.0002 + per),
                          DESC_BENCH: np.full(len(idx), 0.0002)}, index=idx)
    s = accrual_sensitivity(book, bench)

    # the reported excess at the live accrual must equal the excess the rest of the file computes
    live = [r for r in s["grid"] if abs(r["accrual_annual"] - TR_ACCRUAL_ANNUAL) < 1e-12][0]
    got = annualise(book["net_return"]) - annualise(bench[GATE_BENCH])
    assert abs(live["excess_ann"] - got) < 1e-12

    # raising the accrual can only lower the excess, and by the accrual GROSSED UP by the
    # benchmark's own growth: the accrual compounds on top of the price leg, so dropping it
    # returns accrual * (1 + price-leg annualised), not the bare accrual. Asserting the bare
    # number here would be asserting a subtraction the file does not do.
    ex = [r["excess_ann"] for r in s["grid"]]
    assert all(b <= a + 1e-15 for a, b in zip(ex, ex[1:])), "excess must fall as the accrual rises"
    zero = [r for r in s["grid"] if r["accrual_annual"] == 0.0][0]
    assert zero["excess_ann"] > live["excess_ann"]
    price_leg_ann = annualise(bench[GATE_BENCH] - TR_ACCRUAL_PER_SESSION)
    assert abs((zero["excess_ann"] - live["excess_ann"])
               - TR_ACCRUAL_ANNUAL * (1.0 + price_leg_ann)) < 2e-5

    # the breakeven must actually sit on the bar, and bracket PASS/FAIL
    be = s["breakeven_accrual_annual"]
    assert be is not None
    lo = [r for r in accrual_sensitivity(book, bench, grid=(be - 0.002,))["grid"]][0]
    hi = [r for r in accrual_sensitivity(book, bench, grid=(be + 0.002,))["grid"]][0]
    assert lo["passes_m2a"] is True and hi["passes_m2a"] is False
    assert abs(accrual_sensitivity(book, bench, grid=(be,))["grid"][0]["excess_ann"]
               - M2_EXCESS_BAR) < 1e-6

    # a book that cannot clear the bar at any accrual reports no flip point rather than a fake one
    flat = pd.DataFrame({"net_return": np.full(len(idx), 0.0)}, index=idx)
    s2 = accrual_sensitivity(flat, bench)
    assert s2["breakeven_accrual_annual"] is None
    assert s2["verdict_survives_measured_band"] is True
    assert not any(r["passes_m2a"] for r in s2["grid"])

    assert s["verdict_survives_measured_band"] is True

    # The survives-flag must read the MEASURED band, not the whole grid. This book is built to
    # clear the bar at a 0 pp accrual and miss it at every accrual anyone has measured -- exactly
    # A0_primary's shape. The flag must still say the verdict survives, because a 0 pp accrual is
    # a bare price index and not the total-return benchmark the gate is written against.
    edge = pd.DataFrame({"net_return": np.full(len(idx), 0.000377)}, index=idx)
    s3 = accrual_sensitivity(edge, bench)
    at = {round(r["accrual_annual"], 6): r for r in s3["grid"]}
    assert at[0.0]["passes_m2a"] is True
    assert at[0.0097]["passes_m2a"] is False and at[0.0144]["passes_m2a"] is False
    assert s3["verdict_survives_measured_band"] is True
    assert 0.0 < s3["breakeven_accrual_annual"] < 0.0097
