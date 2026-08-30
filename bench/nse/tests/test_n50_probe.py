"""Tests for ``bench/nse/n50_probe.py`` -- the arm registry, the estimator path, the family null.

Synthetic panels only: nothing here reads ``bench/out`` or ``data/nse``, so the suite runs in
seconds on a machine that has never executed the pipeline and a failure is a real regression
rather than a data accident.

Three properties are worth a test:
  1. The PRE-REGISTERED core family is exactly the list NIFTY50_PLAN sec.5 names, in order.
     The family null's max is taken over this list, so an arm silently added or renamed
     changes a published p-value.
  2. A planted signal survives the whole probe path (build -> fm_scores -> evaluate on the
     `is_n50` subset). A probe that cannot see a signal it planted itself cannot be trusted
     with a null result.
  3. Every arm in one family-null draw sees the SAME permuted labels. If they did not, the
     per-draw max over arms would not be a family-wise null and clause (b) of M1 would be a
     number about nothing. Checked with a spy scorer, because the property is invisible in
     the output.
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

from bench.nse import n50_probe as np50  # noqa: E402
from bench.nse.mlcore import FREE_COLS, rank_within_date  # noqa: E402

H = 5

#: The pre-registration itself, restated here so a change to the module has to be a change to
#: the test as well -- which is the point of pre-registering.
CORE_EXPECTED = [
    "free",
    "hT_pre_cpca8", "hT_pre_cpca32", "hT_pre_cpca64", "hT_pre_cpca32_procrustes",
    "hT_res_cpca8", "hT_res_cpca32", "hT_res_cpca64",
    "raw540_cpca32",
    "free+hT_pre_cpca32", "free+hT_res_cpca32",
    "hT_pre_cpca8_fit50only",
]


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def registry_sandbox():
    """Snapshot/restore the module-global arm registry.

    The registry is global by design (a second agent registers the Stage B ladder at import
    time), so a test that adds an arm must not leak it into the next test's family. Built
    blocks are memoised on the context, not globally, so nothing else needs restoring.
    """
    saved = dict(np50.ARMS)
    try:
        yield
    finally:
        np50.ARMS.clear()
        np50.ARMS.update(saved)


def make_ctx(n_dates: int = 220, n_names: int = 60, n_n50: int = 30, signal: float = 0.30,
             seed: int = 0) -> np50.ProbeContext:
    """A balanced synthetic panel shaped exactly like ``dataset_n50.parquet``.

    Feature ``reversal_1d`` carries a planted cross-sectional edge and the first ``n_n50``
    names are the "index members", so the fit-wide/grade-narrow split of the real probe is
    exercised rather than assumed.
    """
    rng = np.random.default_rng(seed)
    ud = pd.bdate_range("2022-01-03", periods=n_dates)
    dates = np.repeat(ud.to_numpy(), n_names)
    syms = np.tile(np.array([f"S{i:03d}" for i in range(n_names)]), n_dates)
    dcode = np.repeat(np.arange(n_dates), n_names)
    n = n_dates * n_names
    Z = rng.standard_normal((n, len(FREE_COLS)))
    y_raw = signal * Z[:, 0] + np.sqrt(1 - signal ** 2) * rng.standard_normal(n)
    ds = pd.DataFrame({"date": dates, "symbol": syms})
    for j, c in enumerate(FREE_COLS):
        ds[c] = Z[:, j]
    for h in (1, 5, 10, 20):
        ds[f"y_h{h}"] = y_raw
    is_n50 = np.tile(np.arange(n_names) < n_n50, n_dates)
    ds["is_n50"] = is_n50
    return np50.ProbeContext(ds=ds, dates=dates, dcode=dcode, is_n50=is_n50,
                             min_hist=40, refit_every=30)


# ------------------------------------------------------- 1. the pre-registered family


def test_core_registry_is_the_preregistered_list():
    assert np50.arm_names("core") == CORE_EXPECTED
    assert np50.PRIMARY_ARM == "hT_pre_cpca32" and np50.PRIMARY_H == 5
    assert np50.PRIMARY_ARM in CORE_EXPECTED
    # every core arm carries a one-line rationale, so the family list is self-documenting
    assert all(np50.ARMS[n].doc for n in CORE_EXPECTED)


def test_arm_selector_forms():
    assert np50.arm_names("free,hT_pre_cpca32") == ["free", "hT_pre_cpca32"]
    assert np50.arm_names("all")[:len(CORE_EXPECTED)] == CORE_EXPECTED
    assert np50.arm_names("extended") == [n for n, a in np50.ARMS.items() if a.group == "extended"]
    with pytest.raises(SystemExit):
        np50.arm_names("no_such_arm")


def test_register_arm_rejects_duplicates(registry_sandbox):
    with pytest.raises(ValueError):
        np50.register_arm("free", lambda ctx: (ctx.free(), np.ones(ctx.n_rows, bool)))


def test_export_contract_targets_registered_arms():
    """Every scores_*.parquet the plan promises downstream must name a real arm/scorer."""
    for (arm, scorer), tag in np50.SCORE_EXPORTS.items():
        assert arm in np50.ARMS, arm
        assert scorer in np50.SCORERS, scorer
        assert tag


# ------------------------------------------------------------- 2. a planted signal is seen


def test_planted_signal_is_recovered(registry_sandbox):
    ctx = make_ctx()
    np50.register_arm("t_planted", np50._free_build, group="test", doc="free block w/ planted edge")
    sc, info = np50.score_arm(ctx, "t_planted", H)
    assert info["n_fit_rows"] == ctx.n_rows and info["n_features"] == len(FREE_COLS)
    ev = np50.evaluate(sc, ctx.ds, H, mask=ctx.is_n50, min_names=10)
    # the planted edge is large; anything at or below zero means the path is broken
    assert ev["ic_summary"]["mean_ic"] > 0.05, ev["ic_summary"]
    assert ev["ic_summary"]["t_nw_lag20"] > 3.0
    assert ev["spread_summary"][10]["mean_ic"] > 0.0
    # graded on the members only, fitted on everyone
    assert ev["n_eval_rows"] == int(ctx.is_n50.sum())


def test_pure_noise_arm_does_not_score(registry_sandbox):
    """The control for the test above: a block with no relation to the label reads ~0."""
    ctx = make_ctx()
    rng = np.random.default_rng(7)
    noise = rank_within_date(rng.standard_normal((ctx.n_rows, 6)), ctx.dcode)
    np50.register_arm("t_noise", lambda c: (noise, np.ones(c.n_rows, bool)), group="test",
                      doc="noise")
    sc, _ = np50.score_arm(ctx, "t_noise", H)
    ev = np50.evaluate(sc, ctx.ds, H, mask=ctx.is_n50, min_names=10)
    assert abs(ev["ic_summary"]["mean_ic"]) < 0.05, ev["ic_summary"]


def test_recovery_curve_is_monotone_in_the_planted_size():
    ctx = make_ctx(n_dates=250, n_names=80, n_n50=60, signal=0.0, seed=3)
    rows = np50.recovery_curve(ctx, H, targets=(0.05, 0.20), seed=1)
    assert len(rows) == 2
    assert rows[1]["planted_solo_ic_n50"] > rows[0]["planted_solo_ic_n50"] > 0
    assert rows[1]["increment"] > rows[0]["increment"]


def test_fit50only_arm_fits_only_members(registry_sandbox):
    """The under-determined control must actually be under-determined: its fit rows are the
    members, not the panel. If this ever silently widened, the arm would stop being a control."""
    ctx = make_ctx(n_dates=160, n_names=50, n_n50=25)
    blk = np50.rank_within_date(np.random.default_rng(0).standard_normal((ctx.n_rows, 4)),
                                ctx.dcode)
    np50.register_arm("t_sub", lambda c: (blk, c.is_n50.copy()), group="test", doc="members only")
    _, info = np50.score_arm(ctx, "t_sub", H)
    assert info["n_fit_rows"] == info["n_fit_n50_rows"] == int(ctx.is_n50.sum())


# ------------------------------------- 3. the family null shares one permutation per draw


def test_family_null_shares_one_permutation_across_arms(registry_sandbox):
    """Spy on the scorer: within a draw every arm must receive the identical label vector."""
    ctx = make_ctx(n_dates=150, n_names=40, n_n50=20)
    rng = np.random.default_rng(1)
    for i in range(3):
        b = rank_within_date(rng.standard_normal((ctx.n_rows, 3)), ctx.dcode)
        np50.register_arm(f"t_a{i}", lambda c, b=b: (b, np.ones(c.n_rows, bool)),
                          group="test", doc=f"spy arm {i}")
    names = [f"t_a{i}" for i in range(3)]

    seen: list[tuple[int, ...]] = []

    def spy(X, y, dates, horizon, **kw):
        seen.append(tuple(np.round(np.asarray(y, dtype=float), 12).tolist()))
        return np.asarray(X, dtype=float)[:, 0]

    n_draws = 4
    np50.SCORERS["t_spy"] = spy
    try:
        np50.family_null(ctx, names, H, n_draws, scorer="t_spy", block=10, seed=5,
                         min_names=10)
    finally:
        np50.SCORERS.pop("t_spy", None)

    assert len(seen) == n_draws * len(names)
    per_draw = [seen[i * len(names):(i + 1) * len(names)] for i in range(n_draws)]
    for d in per_draw:
        assert len(set(d)) == 1, "arms in one draw saw different permuted labels"
    # and the draws are not all the same permutation
    assert len({d[0] for d in per_draw}) > 1


def test_family_null_returns_one_series_per_arm_plus_the_max(registry_sandbox):
    ctx = make_ctx(n_dates=150, n_names=40, n_n50=20)
    rng = np.random.default_rng(2)
    for i in range(2):
        b = rank_within_date(rng.standard_normal((ctx.n_rows, 3)), ctx.dcode)
        np50.register_arm(f"t_b{i}", lambda c, b=b: (b, np.ones(c.n_rows, bool)),
                          group="test", doc=f"arm {i}")
    names = [f"t_b{i}" for i in range(2)]
    fam = np50.family_null(ctx, names, H, 6, block=10, seed=3, min_names=10)
    assert set(fam["arms"]) == set(names)
    for n in names:
        assert len(fam["arms"][n]) == 6
    stack = np.abs(np.vstack([fam["arms"][n] for n in names]))
    assert np.allclose(fam["max_abs"], np.nanmax(stack, axis=0), equal_nan=True)
    # a null distribution centred on zero is the whole point of permuting
    assert abs(float(np.nanmean(fam["max_abs"]))) < 0.25


def test_null_pvalues_convention():
    """(1 + #{ >= obs }) / (1 + n): never zero, and two-sided uses |null| vs |obs|."""
    draws = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    p = np50.null_pvalues(draws, 0.015)
    assert p["n_draws"] == 5
    assert p["p_one_sided"] == pytest.approx(2 / 6)      # 0.02 only
    assert p["p_two_sided"] == pytest.approx(3 / 6)      # -0.02, 0.02
    assert np50.null_pvalues(draws, 10.0)["p_two_sided"] == pytest.approx(1 / 6)


# ----------------------------------------------------------------- exports and gates


def test_write_scores_contract(tmp_path, registry_sandbox):
    ctx = make_ctx(n_dates=140, n_names=40, n_n50=20)
    np50.register_arm("t_free", np50._free_build, group="test", doc="free")
    sc, _ = np50.score_arm(ctx, "t_free", H)
    p = str(tmp_path / "scores_test.parquet")
    info = np50.write_scores(p, ctx, sc, {"unit": "test"}, {"arm": "t_free"})
    df = pd.read_parquet(p)
    assert list(df.columns) == ["date", "symbol", "score", "is_n50"]
    assert len(df) == int(np.isfinite(sc).sum()) == info["n_rows"]
    assert df["score"].notna().all() and df["is_n50"].dtype == bool
    assert df.sort_values(["date", "symbol"]).index.equals(df.index)
    assert os.path.exists(p.replace(".parquet", "_meta.json"))


def test_m1_gates_clauses_are_decided_by_their_own_number():
    """A hand-built result where (a) and (c) pass and (b) fails must report exactly that."""
    idx = pd.bdate_range("2023-01-02", periods=300)
    ic = pd.Series(np.full(len(idx), 0.03), index=idx)
    res = {"hT_pre_cpca32": {5: {
        "ic": ic, "ic_summary": {"mean_ic": 0.03},
        "spread_summary": {10: {"mean_ic": 0.002, "block_ci95_b20": [0.001, 0.003],
                                "t_nw_lag20": 4.0}},
        "power": {"bar_t3_nw_lag20": 0.028}, "power_realised": {"bar_t3_nw_lag20": 0.031}}}}
    incs = {"hT_pre_cpca32 - hT_res_cpca32": {"mean_increment": 0.02, "t_nw_lag20": 3.5,
                                              "block_ci95_b20": [0.005, 0.03]}}
    g = np50.m1_gates(res, incs, {"family": {"p_two_sided": 0.20, "observed": 0.03}}, 5)
    assert g["a_paired_pretrained_minus_reservoir"]["pass"] is True
    assert g["b_family_null_p"]["pass"] is False
    assert g["c_top10_spread"]["pass"] is True and g["c_top10_spread"]["via_ci"] is True
    assert g["verdict_M1"] is False
    assert g["ic_meets_t3_bar"] is True
    assert "money bar" in g["bands"]["where"]

    # the boundary cases: t = 2.99 and a CI that touches zero both fail clause (a)
    incs2 = {"hT_pre_cpca32 - hT_res_cpca32": {"mean_increment": 0.02, "t_nw_lag20": 2.99,
                                               "block_ci95_b20": [0.005, 0.03]}}
    assert np50.m1_gates(res, incs2, {}, 5)["a_paired_pretrained_minus_reservoir"]["pass"] is False
    incs3 = {"hT_pre_cpca32 - hT_res_cpca32": {"mean_increment": 0.02, "t_nw_lag20": 3.5,
                                               "block_ci95_b20": [-0.001, 0.03]}}
    assert np50.m1_gates(res, incs3, {}, 5)["a_paired_pretrained_minus_reservoir"]["pass"] is False


def test_conditional_ic_runs_and_reports_quintiles():
    ctx = make_ctx(n_dates=120, n_names=50, n_n50=40, signal=0.35, seed=11)
    y = ctx.ds["y_h5"].to_numpy(float)
    rng = np.random.default_rng(0)
    score = y + 0.5 * rng.standard_normal(ctx.n_rows)
    control = rng.standard_normal(ctx.n_rows)
    out = np50.conditional_ic(ctx, score, control, H, min_bucket=6)
    assert out["n_dates"] > 100
    assert len(out["per_quintile_mean"]) == 5
    # a score correlated with the label inside every bucket of an unrelated control
    assert out["summary"]["mean_ic"] > 0.2


# ============================================================================================
# Stage B representation ladder (arms A1-A7, F7, F12) -- see n50_probe.py's second half.
# Same rule as above: synthetic panels only, nothing reads bench/out.
# ============================================================================================

#: The ladder's pre-registration, restated here for the same reason the core list is: the
#: family null's max is taken over core + extended, so an arm silently added, renamed or
#: reordered changes a published family p-value.
EXTENDED_EXPECTED = [
    "hT_pre_L4_last_cpca32", "hT_pre_L8_last_cpca32", "hT_pre_L12_last_cpca32",
    "hT_pre_final_mean10_cpca32", "hT_pre_L12_mean10_cpca32", "hT_pre_L8_mean10_cpca32",
    "hT_pre_L4_mean10_cpca32",
    "hT_small_pre_cpca32", "hT_small_res_cpca32",
    "hT_mini_pre_cpca32",
    "tokz_last_raw20", "tokz_last_cpca8", "tokz_mean10_raw20", "tokz_mean10_cpca8",
]


#: The depth-matched random-weight controls (l4_control run). They are a THIRD group rather
#: than more `extended` arms so that `--arms extended` keeps reproducing the 26-arm family
#: null already published in probe_n50_extended.md; `--arms all` is the enlarged family.
CONTROL_EXPECTED = [
    "hT_res_L4_last_cpca32", "hT_res_L8_last_cpca32", "hT_res_L12_last_cpca32",
    "hT_res_L4_mean10_cpca32", "hT_res_L8_mean10_cpca32", "hT_res_L12_mean10_cpca32",
]


def test_extended_registry_is_the_preregistered_ladder():
    assert np50.arm_names("extended") == EXTENDED_EXPECTED
    assert np50.arm_names("control") == CONTROL_EXPECTED
    assert np50.arm_names("all") == CORE_EXPECTED + EXTENDED_EXPECTED + CONTROL_EXPECTED
    assert all(np50.ARMS[n].doc for n in EXTENDED_EXPECTED + CONTROL_EXPECTED)
    # the core family is untouched by the ladder, so `--arms core` still reproduces C1
    assert np50.arm_names("core") == CORE_EXPECTED


def test_layer_control_pairs_hold_the_read_fixed():
    """Every (layer, pool) contrast subtracts the SAME read off a random-weight trunk.

    `EXTENDED_PAIRS` already carried `ladder arm - hT_res_cpca32`, but that control is the
    FINAL/LAST state: subtracting it from a layer-4 mean-pooled arm prices depth, pooling and
    training in one number, so it cannot say which of the three did the work.
    """
    assert len(np50.LAYER_CONTROL_PAIRS) == 6
    for pre, res in np50.LAYER_CONTROL_PAIRS:
        key = next(k for k, a in np50.LAYER_ARMS.items() if a == pre)
        assert np50.RES_LAYER_ARMS[key] == res, key
        assert np50.ARM_SOURCE[res] == np50.RES_LAYER_SOURCES[key]
        assert (pre, res) in np50.EXTENDED_PAIRS
    # the reservoir ladder's (final, last) cell is the existing core control, not a new arm
    assert np50.RES_LAYER_ARMS[("final", "last")] == "hT_res_cpca32"
    assert np50.RES_LAYER_SOURCES[("final", "last")] == "ml:hT_reservoir"
    assert set(np50.RES_LAYER_ARMS) == set(np50.RES_LAYER_SOURCES)


def test_layer_map_is_consistent_with_the_registry():
    """The F12 table and the arm registry must name the same eight (layer, pool) states.

    The final/last cell deliberately points at the PRIMARY arm rather than a 27th arm: the
    Stage B pass re-extracted that state bit-identically, so registering it again would put
    the same array in the family null twice and inflate the family maximum.
    """
    assert set(np50.LAYER_SOURCES) == set(np50.LAYER_ARMS)
    assert len(np50.LAYER_ARMS) == 8
    assert np50.LAYER_ARMS[("final", "last")] == np50.PRIMARY_ARM
    for key, arm in np50.LAYER_ARMS.items():
        assert arm in np50.ARMS, (key, arm)
    # every non-primary layer arm is an extended arm, and BASE_LADDER_ARMS is exactly those
    ladder = [a for k, a in np50.LAYER_ARMS.items() if a != np50.PRIMARY_ARM]
    assert sorted(np50.BASE_LADDER_ARMS) == sorted(ladder)


def test_raw_cols_arm_keeps_every_column_and_ranks_within_date(registry_sandbox):
    """A5 asks whether the 20-column tokenizer code carries signal AT ALL, so the raw-column
    arm must pass all 20 through, rank-scaled within date like every other arm."""
    ctx = make_ctx(n_dates=60, n_names=40, n_n50=20)
    rng = np.random.default_rng(4)
    M = rng.standard_normal((ctx.n_rows, 7)).astype(np.float32)
    ctx._cache["n50:fake_z"] = (M, np.ones(ctx.n_rows, dtype=bool))
    np50.register_arm("t_raw", np50.raw_cols_arm("n50:fake_z"), group="test", doc="raw")
    blk, ok = np50.build_arm("t_raw", ctx)
    assert blk.shape == (ctx.n_rows, 7) and ok.all()
    assert blk.min() >= -1.0 and blk.max() <= 1.0
    for d in np.unique(ctx.dcode)[:3]:
        col = blk[ctx.dcode == d, 0]
        assert abs(float(col.mean())) < 1e-9          # uniform(-1,1) ranks are centred
    # rank scaling is monotone: the argsort of a column is the argsort of the source
    d0 = ctx.dcode == 0
    assert (np.argsort(blk[d0, 3]) == np.argsort(M[d0, 3])).all()


def test_raw_cols_arm_respects_a_partial_row_mask(registry_sandbox):
    """An array that covers only some rows (the N50-only extractions) must rank among the
    rows it covers and leave the rest invalid, not silently rank zeros alongside real data."""
    ctx = make_ctx(n_dates=40, n_names=30, n_n50=15)
    rng = np.random.default_rng(5)
    M = np.zeros((ctx.n_rows, 3), dtype=np.float32)
    valid = ctx.is_n50.copy()
    M[valid] = rng.standard_normal((int(valid.sum()), 3))
    ctx._cache["n50:fake_sub"] = (M, valid)
    np50.register_arm("t_sub_raw", np50.raw_cols_arm("n50:fake_sub"), group="test", doc="sub")
    blk, ok = np50.build_arm("t_sub_raw", ctx)
    assert (ok == valid).all()
    assert np.allclose(blk[~valid], 0.0)
    assert not np.allclose(blk[valid], 0.0)


def test_build_blocks_releases_the_source_but_keeps_the_block(registry_sandbox):
    """Eleven 832-d float32 panels do not fit in memory next to a 26-arm family null, so the
    sources are dropped after projection. The projected block must survive that."""
    ctx = make_ctx(n_dates=200, n_names=40, n_n50=20)
    rng = np.random.default_rng(6)
    ctx._cache["ml:fake_big"] = (rng.standard_normal((ctx.n_rows, 12)).astype(np.float32),
                                 np.ones(ctx.n_rows, dtype=bool))
    np50.register_arm("t_big", np50.cpca_arm("ml:fake_big", 3), group="test", doc="big")
    info = np50.build_blocks(ctx, ["t_big"], release=True, verbose=False)
    assert "ml:fake_big" not in ctx._cache          # source dropped
    assert info["t_big"]["n_features"] == 3
    blk, ok = np50.build_arm("t_big", ctx)          # memoised, not rebuilt from the source
    assert blk.shape == (ctx.n_rows, 3) and ok.sum() > 0


def test_extended_increments_cover_every_ladder_arm_and_the_controls(registry_sandbox):
    """Every ladder arm must be contrasted with the primary cell -- generated, not listed, so
    adding an arm cannot forget its increment."""
    ctx = make_ctx(n_dates=200, n_names=40, n_n50=20)
    rng = np.random.default_rng(7)
    names = [np50.PRIMARY_ARM, "hT_res_cpca32", "hT_small_pre_cpca32", "hT_small_res_cpca32"]
    res = {}
    idx = pd.bdate_range("2023-01-02", periods=200)
    for i, n in enumerate(names):
        res[n] = {H: {"ic": pd.Series(rng.normal(0.01 * i, 0.05, len(idx)), index=idx)}}
    incs = np50.extended_increments(res, H, ["hT_small_pre_cpca32", "hT_small_res_cpca32"])
    assert f"hT_small_pre_cpca32 - {np50.PRIMARY_ARM}" in incs
    assert f"hT_small_res_cpca32 - {np50.PRIMARY_ARM}" in incs
    assert "hT_small_pre_cpca32 - hT_small_res_cpca32" in incs
    # arms absent from `res` are skipped rather than crashing the table
    assert all(a in res and b in res for a, b in
               (k.split(" - ") for k in incs))


def test_extended_gates_are_decided_by_their_own_number():
    """G1 must fire on the PAIRED increment, not the level IC, and G2 must be about small's
    own reservoir. A hand-built result where G1 passes and G2 fails must report exactly that."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    ext = ["hT_pre_L8_last_cpca32", "hT_small_pre_cpca32"]
    res = {n: {H: {"ic_summary": {"mean_ic": 0.03}}} for n in ext + [np50.PRIMARY_ARM]}
    res["hT_small_res_cpca32"] = {H: {"ic_summary": {"mean_ic": 0.028}}}
    strong = np50.paired_increment(pd.Series(np.full(len(idx), 0.02), index=idx),
                                   pd.Series(np.zeros(len(idx)), index=idx), H)
    # a nearly-zero mean on top of real noise: positive but nowhere near t = 2
    rng = np.random.default_rng(3)
    weak = np50.paired_increment(
        pd.Series(0.0002 + rng.normal(0.0, 0.05, len(idx)), index=idx),
        pd.Series(np.zeros(len(idx)), index=idx), H)
    incs = {f"hT_pre_L8_last_cpca32 - {np50.PRIMARY_ARM}": strong,
            f"hT_small_pre_cpca32 - {np50.PRIMARY_ARM}": weak,
            "hT_small_pre_cpca32 - hT_small_res_cpca32": weak}
    g = np50.extended_gates(res, incs, H, ext)
    g1, g2 = g["G1_any_extended_beats_primary"], g["G2_small_beats_its_reservoir"]
    assert g1["pass"] is True
    assert [w["arm"] for w in g1["winners"]] == ["hT_pre_L8_last_cpca32"]
    assert g1["best_by_t20"]["arm"] == "hT_pre_L8_last_cpca32"
    # a constant series has zero NW SE -> infinite t; the weak arm is still ranked below it
    assert g1["all_ranked"][-1]["arm"] == "hT_small_pre_cpca32"
    assert g2["mean_increment"] == pytest.approx(weak["mean_increment"])
    # G2's verdict comes from the small-vs-its-own-reservoir contrast, not from any level
    assert g2["level_ic_pretrained"] == 0.03 and g2["level_ic_reservoir"] == 0.028


def test_extended_gates_fail_when_no_arm_clears_the_bar():
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(11)
    ext = ["hT_mini_pre_cpca32"]
    res = {n: {H: {"ic_summary": {"mean_ic": 0.01}}} for n in ext + [np50.PRIMARY_ARM]}
    noise = np50.paired_increment(pd.Series(rng.normal(0.0, 0.05, len(idx)), index=idx),
                                  pd.Series(np.zeros(len(idx)), index=idx), H)
    g = np50.extended_gates(res, {f"hT_mini_pre_cpca32 - {np50.PRIMARY_ARM}": noise}, H, ext)
    assert g["G1_any_extended_beats_primary"]["pass"] is False
    assert g["G1_any_extended_beats_primary"]["winners"] == []
    assert g["G2_small_beats_its_reservoir"]["pass"] is False   # contrast absent -> FAIL


def test_rank_average_is_a_within_date_rank_statistic():
    """A7 rank-averages rather than score-averages: components are fitted numbers on
    different scales and an unnormalised mean would be whichever has the largest variance."""
    ctx = make_ctx(n_dates=30, n_names=20, n_n50=10)
    rng = np.random.default_rng(8)
    a = rng.standard_normal(ctx.n_rows)
    b = 1000.0 * a + 3.0                       # same ranking, wildly different scale
    avg = np50.rank_average(ctx, [a, b])
    d0 = ctx.dcode == 0
    assert (np.argsort(avg[d0]) == np.argsort(a[d0])).all()
    # a NaN in any component drops the row rather than poisoning the average
    c = a.copy()
    c[5] = np.nan
    out = np50.rank_average(ctx, [a, c])
    assert np.isnan(out[5]) and np.isfinite(out[6])


def test_ensemble_drops_a_negative_component_and_says_so(registry_sandbox):
    """The pre-registration admits a component only if its own IC is positive; a negative one
    must be reported as dropped, never averaged in quietly."""
    ctx = make_ctx(n_dates=200, n_names=40, n_n50=20)
    rng = np.random.default_rng(9)
    idx = np.unique(ctx.dates)
    res, scores = {}, {}
    for n, ic in (("good_a", 0.02), ("good_b", 0.03), ("bad", -0.01)):
        res[n] = {H: {"ic_summary": {"mean_ic": ic},
                      "ic": pd.Series(np.full(len(idx), ic), index=pd.DatetimeIndex(idx))}}
        scores[n] = {H: rng.standard_normal(ctx.n_rows)}
    out = np50.ensemble_arm(ctx, res, scores, H, ["good_a", "good_b", "bad"])
    assert out["run"] is True
    assert out["admitted"] == ["good_a", "good_b"] and set(out["dropped"]) == {"bad"}
    assert out["dropped"]["bad"] == -0.01
    assert np.isfinite(out["scores"]).sum() == ctx.n_rows


def test_ensemble_not_formed_when_one_component_survives(registry_sandbox):
    ctx = make_ctx(n_dates=160, n_names=40, n_n50=20)
    idx = np.unique(ctx.dates)
    res = {n: {H: {"ic_summary": {"mean_ic": ic},
                   "ic": pd.Series(np.full(len(idx), ic), index=pd.DatetimeIndex(idx))}}
           for n, ic in (("good", 0.02), ("bad", -0.02))}
    scores = {n: {H: np.zeros(ctx.n_rows)} for n in res}
    out = np50.ensemble_arm(ctx, res, scores, H, ["good", "bad"])
    assert out["run"] is False and out["admitted"] == ["good"]


def test_family_p_convention():
    """The family p is |observed| against the per-draw max, with the same (1+k)/(1+n) floor."""
    mx = np.array([0.01, 0.012, 0.02, 0.03])
    assert np50._family_p(mx, 0.025) == pytest.approx(2 / 5)   # 0.03 only
    assert np50._family_p(mx, -0.025) == pytest.approx(2 / 5)  # sign-blind
    assert np50._family_p(mx, 10.0) == pytest.approx(1 / 5)    # the floor, never zero


def test_reservoir_seed_block_is_fitted_on_n50_rows_only(registry_sandbox, monkeypatch,
                                                        tmp_path):
    """F7's whole point is a like-for-like seed comparison, so all four blocks must see the
    SAME rows -- the N50 cells -- and the summary must carry the spread across seeds."""
    ctx = make_ctx(n_dates=260, n_names=60, n_n50=35)
    ctx.n50_dir = str(tmp_path)
    rng = np.random.default_rng(10)
    fake = {}
    # real .npy files, not a primed cache: `reservoir_seed_block` releases its source after
    # every block (the arrays are 250 MB each in the real run), so a primed cache would be
    # dropped after the first one and the test would not exercise the code that ships.
    rows = pd.DataFrame({"row": np.flatnonzero(ctx.is_n50)})
    rows.to_parquet(os.path.join(str(tmp_path), "rows_n50.parquet"), index=False)
    for label in ["pretrained_final_last", "reservoir_s7", "reservoir_s8", "reservoir_s9"]:
        M = rng.standard_normal((int(ctx.is_n50.sum()), 16)).astype(np.float32)
        np.save(os.path.join(str(tmp_path), f"fake_{label}.npy"), M)
        fake[label] = f"n50:fake_{label}"
    monkeypatch.setattr(np50, "SEED_SOURCES", fake)
    out = np50.reservoir_seed_block(ctx, h=H, k=4)
    assert out["label"] == "N50-only pooled"
    assert out["design"]["fit_rows"] == "is_n50 only"
    fit_rows = {v["n_fit_rows"] for v in out["arms"].values()}
    assert len(fit_rows) == 1 and fit_rows.pop() <= int(ctx.is_n50.sum())
    assert set(out["reservoir_seed_ics"]) == set(np50.RESERVOIR_SEEDS)
    assert np.isfinite(out["reservoir_seed_sd"])
    assert out["best_reservoir_seed"] in np50.RESERVOIR_SEEDS
    assert out["pretrained_minus_best_seed"]["n_dates"] > 0


def test_every_state_reading_arm_names_its_array():
    """The published effective-rank table is built from `ARM_SOURCE`, so an arm that reads a
    cached state and is missing from that map would silently vanish from the table."""
    for n in np50.arm_names("extended"):
        assert n in np50.ARM_SOURCE, n
    for n in ("hT_pre_cpca8", "hT_pre_cpca32", "hT_pre_cpca64", "hT_pre_cpca32_procrustes",
              "hT_res_cpca8", "hT_res_cpca32", "hT_res_cpca64", "hT_pre_cpca8_fit50only"):
        assert n in np50.ARM_SOURCE, n
    # the two arms that read no model state are deliberately absent
    assert "free" not in np50.ARM_SOURCE and "raw540_cpca32" not in np50.ARM_SOURCE
    for src in np50.ARM_SOURCE.values():
        assert src.startswith(("ml:", "n50:")), src
    # the layer map and the source map agree on where each layer arm reads from
    for (layer, pool), arm in np50.LAYER_ARMS.items():
        assert np50.ARM_SOURCE[arm] == np50.LAYER_SOURCES[(layer, pool)]


def test_score_agreement_is_a_per_date_rank_correlation():
    """The number that explains a paired increment's t: identical rankings must read 1.0 and
    a reversed one -1.0, both computed only on the N50 rows."""
    ctx = make_ctx(n_dates=40, n_names=50, n_n50=40)
    rng = np.random.default_rng(12)
    a = rng.standard_normal(ctx.n_rows)
    assert np50.score_agreement(ctx, a, 2.0 * a + 1.0, H)["mean_rho"] == pytest.approx(1.0)
    assert np50.score_agreement(ctx, a, -a, H)["mean_rho"] == pytest.approx(-1.0)
    ind = np50.score_agreement(ctx, a, rng.standard_normal(ctx.n_rows), H)
    assert abs(ind["mean_rho"]) < 0.2 and ind["n_dates"] == 40


def test_ladder_diagnostics_reports_both_m3_tripwires(registry_sandbox):
    """The promoted arm must carry its own fingerprint and contamination verdicts, not
    inherit the primary cell's."""
    ctx = make_ctx(n_dates=400, n_names=60, n_n50=40, seed=2)
    rng = np.random.default_rng(13)
    idx = pd.DatetimeIndex(np.unique(ctx.dates))
    sc = {"free": {H: rng.standard_normal(ctx.n_rows)},
          np50.PRIMARY_ARM: {H: rng.standard_normal(ctx.n_rows)},
          "t_promoted": {H: rng.standard_normal(ctx.n_rows)}}
    res = {n: {H: {"ic": pd.Series(rng.normal(0.02, 0.05, len(idx)), index=idx),
                   "ic_summary": {"mean_ic": 0.02}}} for n in sc}
    out = np50.ladder_diagnostics(ctx, res, sc, "t_promoted", H)
    assert out["arm"] == "t_promoted"
    assert set(out["fingerprint"]) == set(np50.FINGERPRINT)
    assert 0.0 <= out["fingerprint_max_abs_rho"] <= 1.0
    assert isinstance(out["m3_fingerprint_pass"], bool)
    assert isinstance(out["m3_post_ge_pre_pass"], bool)
    assert out["contamination"]["pre"]["n_dates"] + out["contamination"]["post"]["n_dates"] \
        == len(idx)
    assert "conditional_ic_within_free" in out
    assert "conditional_ic_within_primary" in out
    # random scores: agreement with the primary is near zero, and reported either way
    assert abs(out["agreement_with_primary"]["mean_rho"]) < 0.2


def _reading_fixture(rho: float, se_inc: float = 0.0144, se_arm: float = 0.0116,
                     se_pri: float = 0.0110) -> dict:
    """Minimal `out` dict for `_agreement_reading`: one promoted arm, one primary."""
    arm = "t_promoted"
    return {
        "meta": {"best_extended_arm": arm},
        "diagnostics": {arm: {"agreement_with_primary": {
            "n_dates": 565, "mean_rho": rho, "sd_rho": 0.16,
            "min_rho": -0.16, "max_rho": 0.68}}},
        "increments": {str(H): {f"{arm} - {np50.PRIMARY_ARM}": {"se_nw_lag20": se_inc}}},
        "arms": {arm: {H: {"ic_summary": {"se_nw_lag20": se_arm}}},
                 np50.PRIMARY_ARM: {H: {"ic_summary": {"se_nw_lag20": se_pri}}}},
    }


def test_agreement_reading_reverses_with_the_measured_rho():
    """The caveat that explains a failed G1 has to be read off the number, not asserted.

    At a high per-date rho the paired increment really does have little to detect. At a low
    one the opposite is true -- the difference carries almost the whole noise of both arms --
    and a sentence that claims 'high' regardless would soften a null with an agreement the
    data does not show. This is the guard on that.
    """
    lo = np50._agreement_reading(_reading_fixture(0.267), H)
    assert "LOW" in lo
    assert "+0.267" in lo
    assert "little to detect" not in lo
    # the three SEs that justify the reading are quoted, and the increment's is the largest
    for s in ("+0.0144", "+0.0116", "+0.0110"):
        assert s in lo
    assert "not evidence that the difference is zero" in lo

    hi = np50._agreement_reading(_reading_fixture(0.85), H)
    assert "little to detect" in hi
    assert "LOW" not in hi
    assert "+0.850" in hi


def test_agreement_reading_survives_a_missing_diagnostic():
    """No promoted-arm agreement (diagnostics skipped) must degrade to a plain caption."""
    out = _reading_fixture(0.267)
    out["diagnostics"] = {}
    s = np50._agreement_reading(out, H)
    assert "agreement" in s and "LOW" not in s and "little to detect" not in s
