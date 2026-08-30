"""Test 9 of NIFTY50_PLAN section 8, plus the build-side invariants it depends on.

The claim under test is the one the whole money test rests on: **the book can only ever hold a
name the point-in-time NIFTY 50 contained on the signal date.** Survivorship leakage through the
universe is the cheapest way to manufacture a positive backtest, and it is invisible in the
summary statistics, so it is asserted on a synthetic panel where the answer is known by
construction rather than inferred from a real run.

The panel is synthetic and seeded; the two tests that touch real artifacts
(``data/nse/bench_n50.parquet``) skip when the file has not been fetched, so the suite still
runs on a machine that has never hit the network.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.nse.backtest import Market, run_backtest  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402
from bench.nse.n50 import evaluate, power_statement  # noqa: E402
from bench.nse.n50_backtest import (BENCH_COLS, constant_frame, eligible_frame,  # noqa: E402
                                    equal_weight_book, masked_scores, random_frame)
from bench.nse.n50_build import BENCH_N50, build_dataset_n50  # noqa: E402

N_DATES = 60
N_NAMES = 12
INDEX_SIZE = 6


def _panel(seed: int = 0):
    """A 12-name price panel on a contiguous business-day calendar, geometric random walk."""
    rng = np.random.default_rng(seed)
    cal = list(pd.bdate_range("2024-01-01", periods=N_DATES))
    syms = [f"S{i:02d}" for i in range(N_NAMES)]
    rows = []
    for j, s in enumerate(syms):
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, N_DATES))
        rows.append(pd.DataFrame({"date": cal, "symbol": s, "open": px,
                                  "close": px * (1.0 + rng.normal(0, 0.004, N_DATES))}))
    return pd.concat(rows, ignore_index=True), cal, syms


def _universe(cal, syms, switch=30):
    """PIT membership: S00..S05 for the first half; S00 is deleted and S06 added after.

    S06..S11 are never members before the switch and S00 is never a member after it, so both
    directions of the universe rule -- "not yet in" and "no longer in" -- are exercised.
    """
    rows = []
    for i, d in enumerate(cal):
        mem = syms[:INDEX_SIZE] if i < switch else syms[1:INDEX_SIZE + 1]
        rows += [{"date": d, "symbol": s} for s in mem]
    return pd.DataFrame(rows)


def _market(seed: int = 0):
    panel, cal, syms = _panel(seed)
    return Market(panel, cal), cal, syms


# ------------------------------------------------------------------ test 9


def test_outside_pit_names_are_never_bought():
    """The highest score every single day belongs to a name that is never in the index.

    If the universe mask leaks at all, S11 -- scored +100 on every session -- is the first name
    the top-k strategy buys. It must not appear in the blotter once.
    """
    market, cal, syms = _market()
    uni = _universe(cal, syms)
    elig = eligible_frame(uni)

    long = pd.DataFrame([{"date": d, "symbol": s, "score": (100.0 if s == "S11" else float(i))}
                         for d in cal for i, s in enumerate(syms)])
    wide = masked_scores(long, "score", elig)
    assert wide.columns.tolist() == sorted(uni["symbol"].unique())
    assert "S11" not in wide.columns  # never a member, so never even a column

    res = run_backtest(wide, market, topk=3, n_drop=1, hold_thresh=5, rebalance=5,
                       cost_model=PRESETS["zerodha"], capital=1e6, weighting="drift",
                       log_trades=True)
    traded = set(res["trades"]["symbol"])
    assert traded, "the smoke book traded nothing; the assertion below would be vacuous"
    assert "S11" not in traded

    # every fill, on the day it happened, was a member that day
    mem_by_date = {d: set(g) for d, g in uni.groupby("date")["symbol"]}
    for _, r in res["trades"].iterrows():
        if r["weight_delta"] <= 0:
            continue                      # a sale may legitimately be an ex-member
        sig = cal[market.idx[r["exec_date"]] - 1]
        assert r["symbol"] in mem_by_date[sig], f"bought {r['symbol']} outside the index on {sig}"


def test_deleted_member_is_exited_and_never_rebought():
    """S00 leaves the index at the switch: after it, it can be sold but never bought again.

    A deleted name's score becomes NaN, so ``topk_dropout_targets`` ranks it below every scored
    name and the n_drop guard sells it at the next rebalance that clears hold_thresh. What the
    test pins is the direction that matters -- it is never re-entered -- plus a bound on the
    exit lag, because "eventually" is not a universe rule.
    """
    market, cal, syms = _market(seed=3)
    uni = _universe(cal, syms)
    elig = eligible_frame(uni)
    # S00 is the best name while it is a member, so the book is certainly holding it at the switch
    long = pd.DataFrame([{"date": d, "symbol": s, "score": (10.0 if s == "S00" else -float(i))}
                         for d in cal for i, s in enumerate(syms)])
    res = run_backtest(masked_scores(long, "score", elig), market, topk=3, n_drop=2,
                       hold_thresh=5, rebalance=5, cost_model=PRESETS["zerodha"],
                       capital=1e6, weighting="drift", log_trades=True)
    tr = res["trades"]
    switch_date = cal[30]
    after = tr[tr["exec_date"] > switch_date]
    buys_after = after[(after["symbol"] == "S00") & (after["weight_delta"] > 0)]
    assert buys_after.empty, "a deleted member was re-bought"
    sells = tr[(tr["symbol"] == "S00") & (tr["weight_delta"] < 0)]
    assert len(sells), "the deleted member was never sold"
    lag = market.idx[sells["exec_date"].iloc[-1]] - market.idx[switch_date]
    assert 0 < lag <= 11, f"deletion took {lag} sessions to leave the book"


def test_equal_weight_benchmark_holds_only_members():
    """The equal-weight benchmark is subject to the same universe rule as the strategy.

    It is the benchmark that separates selection from the equal-weight tilt, so a version of it
    that quietly kept deleted names would flatter or damn the strategy for the wrong reason.
    """
    market, cal, syms = _market(seed=1)
    uni = _universe(cal, syms)
    res = equal_weight_book(market, uni, one_way=0.0015, rebalance=5, weighting="rebalance")
    d = res["daily"]
    assert len(d) > 20
    assert (d["n_held"] <= INDEX_SIZE).all()
    assert d["n_held"].max() == INDEX_SIZE
    # weekly reset to 1/N trades only the names that changed; turnover is 0 on non-rebalance days
    assert (d.loc[d["turnover"] > 0, "turnover"] <= 2.0 + 1e-9).all()
    assert res["one_way_cost"] == 0.0015


def test_masked_scores_and_placebo_frames_share_the_universe():
    """The constant-score placebo and every random-null draw see exactly the eligible cells.

    A null whose universe is wider than the strategy's is not a null for that strategy.
    """
    market, cal, syms = _market(seed=2)
    elig = eligible_frame(_universe(cal, syms))
    const = constant_frame(elig)
    rnd = random_frame(elig, 7)
    assert const.notna().equals(elig)
    assert rnd.notna().equals(elig)
    assert float(np.nanmax(np.abs(const.to_numpy()))) == 0.0
    assert not rnd.equals(random_frame(elig, 8))
    assert rnd.equals(random_frame(elig, 7))          # seeded, so the null is reproducible


# ------------------------------------------------------------------ build-side invariants


def test_build_dataset_n50_flags_and_counts_ineligible_members():
    """``is_n50`` is the intersection of eligibility and membership, and the gap is counted.

    A member with no dataset row is a name the 90-bar lookback rule has not admitted yet, not a
    hole in the data; the build reports those separately (JIOFIN on the real panel) so nobody
    reads a shrinking cross-section as missing prices.
    """
    _, cal, syms = _panel()
    uni = _universe(cal, syms)
    ds = uni.copy()
    ds = ds[~((ds["symbol"] == "S02") & (ds["date"] < cal[10]))]   # S02 ineligible for 10 dates
    ds["y_h10"] = 0.0
    out, info = build_dataset_n50(ds, uni)
    assert out["is_n50"].all()                     # every surviving row is a member
    assert info["ineligible_member_dates"]["count"] == 10
    assert set(info["ineligible_member_dates"]["by_symbol"]) == {"S02"}
    assert info["eligible_n50_per_date"]["max"] == INDEX_SIZE
    assert info["eligible_n50_per_date"]["min"] == INDEX_SIZE - 1


def test_evaluate_masks_scores_and_reports_both_metrics():
    """``n50.evaluate`` grades only the masked rows and returns IC and the top-k spread.

    The planted edge lives only inside the index members, so an evaluation that leaked
    non-members into the cross-section would dilute the IC below the masked one.
    """
    rng = np.random.default_rng(0)
    _, cal, syms = _panel()
    uni = _universe(cal, syms)
    ds = pd.DataFrame([{"date": d, "symbol": s} for d in cal for s in syms])
    mem = {(d, s) for d, s in zip(uni["date"], uni["symbol"])}
    inside = np.array([(d, s) in mem for d, s in zip(ds["date"], ds["symbol"])])
    score = rng.standard_normal(len(ds))
    ds["y_h5"] = np.where(inside, 0.9 * score, -0.9 * score) + 0.1 * rng.standard_normal(len(ds))

    res = evaluate(score, ds, 5, inside, min_names=5, ks=(3,))
    assert res["n_eval_rows"] == int(inside.sum())
    assert res["ic_summary"]["mean_ic"] > 0.5
    assert res["spread_summary"][3]["mean_ic"] > 0
    assert set(res["ic_by_year"]) == {"2024"}
    assert res["power"]["ic_sd_source"] == "measured"

    flat = evaluate(score, ds, 5, np.ones(len(ds), dtype=bool), min_names=5, ks=(3,))
    assert abs(flat["ic_summary"]["mean_ic"]) < res["ic_summary"]["mean_ic"]


def test_evaluate_reports_the_realised_autocorrelation_inflation():
    """`power_realised` prices the lag-20 bar at the ratio the series shows, not at 1.7.

    `power`/`power_asymptotic` multiply the naive SE by an ASSUMED `ar1_inflation`. On the
    real N50 series the observed se_nw_lag20/se_naive is ~1.16, so the assumed-1.7 bar is ~47%
    above the bar this series' own SE ladder implies -- two numbers made differently, printed
    side by side in the same JSON. `power_realised` closes that: its se_nw_lag20 must be the
    ic_summary's se_nw_lag20 exactly.
    """
    rng = np.random.default_rng(11)
    _, cal, syms = _panel()
    ds = pd.DataFrame([{"date": d, "symbol": s} for d in cal for s in syms])
    score = rng.standard_normal(len(ds))
    ds["y_h5"] = 0.4 * score + rng.standard_normal(len(ds))
    res = evaluate(score, ds, 5, np.ones(len(ds), dtype=bool), min_names=5, ks=(3,))

    pr, ic = res["power_realised"], res["ic_summary"]
    assert pr["ic_sd_source"] == "measured"
    assert pr["se_nw_lag20"] == pytest.approx(ic["se_nw_lag20"], rel=1e-9), \
        "power_realised does not reproduce the series' own lag-20 SE"
    assert pr["ar1_inflation"] == pytest.approx(ic["se_nw_lag20"] / ic["se_naive"], rel=1e-9)
    assert pr["bar_t3_nw_lag20"] == pytest.approx(3.0 * ic["se_nw_lag20"], rel=1e-9)
    # the assumed-1.7 statement is still there, unchanged, so nothing that quoted it moves
    assert res["power"]["ar1_inflation"] == 1.7
    assert res["power_asymptotic"]["ic_sd_source"] == "1/sqrt(N-1)"


def test_power_statement_matches_the_plans_arithmetic():
    """50 names x 685 dates -> naive SE ~0.0055 and a lag-20 t=3 bar in the +0.025-0.033 band.

    NIFTY50_PLAN section 4 states this before any result is looked at; if the helper drifts, the
    stated bar and the printed bar stop being the same number.
    """
    p = power_statement(50, 685)
    assert p["ic_sd_per_date"] == pytest.approx(0.1429, abs=0.001)
    assert p["se_naive"] == pytest.approx(0.0055, abs=0.0003)
    assert 0.008 <= p["se_nw_lag20"] <= 0.011
    assert 0.025 <= p["bar_t3_nw_lag20"] <= 0.033
    # more dates must lower the bar as 1/sqrt(T), and a wider cross-section must lower it too
    assert power_statement(50, 2740)["se_naive"] == pytest.approx(p["se_naive"] / 2, rel=0.01)
    assert power_statement(200, 685)["se_naive"] < p["se_naive"]
    # a measured per-date sd overrides the asymptotic one
    assert power_statement(50, 685, ic_sd_per_date=0.20)["ic_sd_source"] == "measured"
    assert power_statement(50, 685, ar1_inflation=1.0)["se_nw_lag20"] == p["se_naive"]


# ------------------------------------------------------------------ benchmark artifact (F8)


@pytest.mark.skipif(not os.path.exists(BENCH_N50), reason="bench_n50.parquet not fetched")
def test_bench_n50_columns_present_and_dates_monotone():
    """The TRI benchmark file has every column the backtest asks for, in date order.

    ``n50_backtest.bench_series`` indexes ``BENCH_COLS`` straight into this frame and reindexes
    it onto the exchange calendar, so a missing column or an unsorted date column would surface
    as a silently all-NaN benchmark rather than an error.
    """
    b = pd.read_parquet(BENCH_N50)
    want = {"date", "nsei", "nsei_open", "niftybees_adj", "niftybees_adj_open",
            "niftybees_close", "niftybees_open"}
    assert want <= set(b.columns)
    assert set(BENCH_COLS.values()) <= set(b.columns)
    assert b["date"].is_monotonic_increasing
    assert b["date"].is_unique
    assert str(b["date"].dtype) == "datetime64[ns]"
    assert len(b) > 500
    # the adjusted series must never be a constant multiple away from nonsense
    good = b["niftybees_close"].notna() & b["niftybees_adj"].notna()
    ratio = (b.loc[good, "niftybees_adj"] / b.loc[good, "niftybees_close"])
    assert ((ratio > 0.5) & (ratio <= 1.0 + 1e-9)).all()
    assert (b.loc[b["niftybees_open"].notna(), "niftybees_open"] > 0).all()
