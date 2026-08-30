"""Test 6 of NIFTY50_PLAN.md sec.8: the point-in-time NIFTY 50 universe is what it claims.

Every number in the money test is computed on `data/nse/membership_pit_n50.parquet`, so a
silent error here -- one name too many, a regime boundary a week late, a symbol the price panel
has never heard of -- would not surface as a crash but as a plausible-looking IC. The checks
below are therefore about the ARTIFACT, not about a re-derivation of it: they read the file the
probe will read and hold it against `n50_membership_meta.json`, the press-release-derived log
that produced it.

Two deviations from a naive "50 names on every date" are asserted rather than tolerated:

  * 2023-08-01 .. 2023-09-06 carries 51 real names. NSE published the index that way -- JIOFIN
    entered as a temporary 51st constituent on the Reliance demerger (ind_prs17072023.pdf) and
    left when it failed the price-band test (ind_prs05092023.pdf). The test asserts the window
    is exactly the one the meta declares as a sourced exception, so a reconstruction bug that
    happened to produce 51 names somewhere else still fails.
  * DUMMYTATAM/TMCV never appear at all. The Tata Motors CV placeholder was a real constituent
    at ZERO price with no divisor adjustment (ind_prs07102025.pdf); it is not a security anyone
    can buy, so it is stripped from the tradable universe.

The last test is the regression that protects the 2026-08-28 results: the NIFTY 200 path must
still reproduce `data/nse/membership_pit.parquet`. It compares FRAMES, not bytes -- pyarrow
25.0.1 writes different file-level metadata than the build that produced the on-disk artifact,
and pandas >= 2.2 infers datetime64[us] where the old one inferred ns, so byte equality is a
property of the environment while frame equality is a property of the code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "nse")
MEMB_N50 = os.path.join(DATA, "membership_pit_n50.parquet")
META_N50 = os.path.join(DATA, "n50_membership_meta.json")
CONFLICTS = os.path.join(DATA, "n50_membership_conflicts.json")
PANEL = os.path.join(DATA, "panel_price_pit.parquet")
MEMB_200 = os.path.join(DATA, "membership_pit.parquet")
SCRIPT = os.path.join(ROOT, "bench", "nse", "reconstruct_membership.py")

needs_build = pytest.mark.skipif(
    not (os.path.exists(MEMB_N50) and os.path.exists(META_N50)),
    reason="run `python bench/nse/reconstruct_membership.py --index nifty50` first")


@pytest.fixture(scope="module")
def memb() -> pd.DataFrame:
    return pd.read_parquet(MEMB_N50)


@pytest.fixture(scope="module")
def meta() -> dict:
    with open(META_N50) as f:
        return json.load(f)


@needs_build
def test_members_per_session(memb, meta):
    """50 names on every session, except inside the one sourced 51-name window."""
    per_date = memb.groupby("date")["symbol"].nunique()
    exceptions = meta["size_exceptions"]
    assert len(exceptions) == 1, "a new size exception needs a press release and a test update"
    lo, hi = pd.Timestamp(exceptions[0]["from"]), pd.Timestamp(exceptions[0]["to"])
    inside = (per_date.index >= lo) & (per_date.index < hi)
    assert (per_date[~inside] == 50).all(), \
        f"sessions off 50 outside the exception: {per_date[~inside][per_date[~inside] != 50]}"
    assert (per_date[inside] == exceptions[0]["n_real"]).all()
    assert inside.sum() > 0, "the exception window does not intersect Window A -- check --start"
    assert len(per_date) == meta["window"]["n_sessions"] == 756


@needs_build
def test_placeholders_absent(memb, meta):
    """Zero-price demerger placeholders are index members but never tradable universe rows."""
    holes = set(meta["placeholders"]["symbols"]) | {"TMCV", "DUMMYTATAM"}
    assert not (set(memb["symbol"]) & holes)
    # ... and the reason they are absent is recorded, not just the fact.
    assert meta["placeholders"]["reason"]


@needs_build
def test_regime_boundaries_match_meta(memb, meta):
    """Every membership change in the parquet is a regime boundary in the log, and vice versa."""
    sets = {d: frozenset(g) for d, g in memb.groupby("date")["symbol"]}
    dates = sorted(sets)
    observed = {}
    for prev, cur in zip(dates, dates[1:]):
        if sets[cur] != sets[prev]:
            observed[cur.date().isoformat()] = (frozenset(sets[prev] - sets[cur]),
                                                frozenset(sets[cur] - sets[prev]))

    holes = set(meta["placeholders"]["symbols"])
    lo, hi = pd.Timestamp(meta["window"]["start"]), pd.Timestamp(meta["window"]["end"])
    expected = {}
    for ev in meta["events"]:
        d = ev["effective_from"]
        # a boundary is only visible in the parquet if it falls strictly inside the window,
        # and only if it moves a tradable name
        if not (lo < pd.Timestamp(d) <= hi):
            continue
        out = frozenset(ev["excluded"]) - holes
        inn = frozenset(ev["included"]) - holes
        if out or inn:
            expected[d] = (out, inn)
    assert observed == expected

    # the regime list and the event list have to tell the same story
    starts = [r["from"] for r in meta["regimes"] if r["from"]]
    assert starts == [e["effective_from"] for e in meta["events"]]
    assert meta["regime_sizes"] == [r["n_real"] for r in meta["regimes"]]


@needs_build
@pytest.mark.skipif(not os.path.exists(PANEL), reason="price panel not built")
def test_members_are_panel_symbols(memb, meta):
    """Every N50 name is one of the 257 panel symbols -- the N50 is a subset of the PIT 200."""
    panel_syms = set(pd.read_parquet(PANEL, columns=["symbol"])["symbol"].unique())
    assert len(panel_syms) == 257
    assert not (set(memb["symbol"]) - panel_syms)
    assert meta["price_coverage"]["members_not_in_panel"] == []
    # and a subset of the 200 wherever the 200 is defined
    memb200 = pd.read_parquet(MEMB_200)
    pairs200 = set(map(tuple, memb200[["date", "symbol"]].to_numpy()))
    stray = [p for p in map(tuple, memb.to_numpy()) if p not in pairs200]
    assert stray == [], f"N50 members that are not PIT NIFTY 200 members: {stray[:5]}"


@needs_build
@pytest.mark.skipif(not os.path.exists(PANEL), reason="price panel not built")
def test_price_rows_exist_except_the_logged_ones(memb, meta):
    """Missing price rows are allowed only where the meta names them, symbol by symbol."""
    panel = pd.read_parquet(PANEL, columns=["date", "symbol"]).assign(_has=True)
    missing = memb.merge(panel, on=["date", "symbol"], how="left")
    missing = missing[missing["_has"].isna()]
    logged = meta["price_coverage"]["missing_by_symbol"]
    assert len(missing) == meta["price_coverage"]["n_missing_cells"]
    assert {str(s) for s in missing["symbol"].unique()} == set(logged)
    for sym, rec in logged.items():
        g = missing[missing["symbol"] == sym]
        assert len(g) == rec["n"]
        assert g["date"].min().date().isoformat() == rec["first"]
        assert g["date"].max().date().isoformat() == rec["last"]
    # the plan predicts exactly this hole: a name that was an index member before it listed
    assert set(logged) == {"JIOFIN"}
    assert meta["price_coverage"]["n_ineligible_lookback"] > 0


@needs_build
def test_conflicts_file_carries_no_errors():
    """Cross-validation against the research JSON: informational diffs only, no history diff."""
    with open(CONFLICTS) as f:
        conflicts = json.load(f)
    assert isinstance(conflicts, list)
    errors = [c for c in conflicts if c["severity"] == "error"]
    assert errors == [], f"press releases and research JSON disagree on history: {errors}"
    assert {c["kind"] for c in conflicts} <= {"source_url_mismatch",
                                              "rename_missing_from_research"}


@needs_build
def test_current_list_snapshot(meta):
    """The walk-back is anchored on a dated published list, and it matches the research list."""
    cur = meta["current_list"]
    assert cur["n"] == 50
    assert cur["matches_research"] is True
    assert cur["fetch"]["kind"] in ("primary", "snapshot")
    assert pd.Timestamp(cur["as_of"]) >= pd.Timestamp(meta["window"]["end"])


@pytest.mark.skipif(not os.path.exists(MEMB_200), reason="NIFTY 200 membership not built")
def test_nifty200_path_still_reproduces_membership_pit(tmp_path):
    """The refactor must not move the 200: re-run it and diff against the shipped artifact.

    `membership_pit.parquet` is `membership.parquet` restricted to Window A with the three
    names that have no price history dropped, so that derivation is spelled out here rather
    than trusted -- if it ever stops holding, this test says so instead of the probe.
    """
    prefix = tmp_path / "membership"
    r = subprocess.run([sys.executable, SCRIPT, "--index", "nifty200",
                        "--out-prefix", str(prefix)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    got = pd.read_parquet(str(prefix) + ".parquet")
    want_pit = pd.read_parquet(MEMB_200).sort_values(["date", "symbol"]).reset_index(drop=True)
    no_price = ["GUJGASLTD", "PEL", "TATAMTRDVR"]
    sub = got[(got["date"] >= "2023-08-01") & (got["date"] <= "2026-08-25")][["date", "symbol"]]
    sub = sub[~sub["symbol"].isin(no_price)].sort_values(["date", "symbol"]).reset_index(drop=True)
    assert sub.equals(want_pit)

    old = json.load(open(os.path.join(DATA, "membership.json")))
    new = json.load(open(str(prefix) + ".json"))
    assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
