"""Tests 1-5 of NIFTY50_PLAN.md sec.8 -- the estimator fixes F1-F5 in ``bench/nse/mlcore.py``.

Every panel here is synthetic and seeded, because the point of these tests is to pin
properties of the ESTIMATOR that a real panel cannot demonstrate: a leak shows up only
against a control where nothing is knowable, an attenuation factor is only interpretable
against a signal of known size, and an embargo boundary is only visible when exactly one
date carries a coefficient. Nothing reads ``bench/out`` or ``data/nse``, so the suite runs
in seconds on a machine that has never executed the pipeline.

The two mlcore behaviours the record depends on are pinned here as well, in the same
tests as their replacements: ``causal_pca(mode="refit")`` against
``attack_probe.causal_pca_block`` (bit-for-bit) and ``oos_scores(inner_embargo=False)``
against the un-embargoed inner split. A fix that cannot reproduce the number it replaces
is not a fix, it is a second unexplained number.
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

from bench.nse import mlcore as mc  # noqa: E402
from bench.nse.attack_probe import causal_pca_block  # noqa: E402

HORIZON = 5


# ------------------------------------------------------------------ local builders


def _dates(n_dates: int, n_names: int):
    """Date-major (date, name) index: unique dates, the per-row date, the per-row code."""
    ud = pd.bdate_range("2020-01-01", periods=n_dates).to_numpy()
    return ud, np.repeat(ud, n_names), np.repeat(np.arange(n_dates), n_names)


def _signal_panel(n_dates: int, n_names: int, n_feat: int, signal: float, seed: int):
    """Rank-transformed panel whose FIRST feature carries a planted cross-sectional edge.

    ``signal`` is the correlation planted in raw space; the solo IC each test reports is
    measured, never assumed, because ranking and the finite cross-section move it.
    """
    rng = np.random.default_rng(seed)
    ud, dates, dcode = _dates(n_dates, n_names)
    n = n_dates * n_names
    Z = rng.standard_normal((n, n_feat))
    y_raw = signal * Z[:, 0] + np.sqrt(1.0 - signal ** 2) * rng.standard_normal(n)
    return {
        "X": mc.rank_within_date(Z, dcode),
        "y": mc.rank_within_date(y_raw[:, None], dcode)[:, 0],
        "y_raw": y_raw, "dates": dates, "dcode": dcode, "ud": ud,
    }


def _factor_panel(n_dates: int, n_names: int, dim: int, seed: int, swap_at: int | None = None):
    """Low-rank feature block with a stable eigenbasis, for the causal_pca modes.

    A well-separated spectrum is what makes "the basis moved" a statement about the refit
    policy rather than about eigenvector noise. With ``swap_at`` the two leading factors
    trade amplitude part-way through, which is the covariance drift that makes a refitted
    basis rotate under a trailing coefficient average (NIFTY50_PLAN F2).
    """
    rng = np.random.default_rng(seed)
    ud, dates, dcode = _dates(n_dates, n_names)
    n = n_dates * n_names
    Q = np.linalg.qr(rng.standard_normal((dim, dim)))[0][:, :4]
    amp = np.tile(np.array([3.0, 2.8, 1.0, 0.5]), (n, 1))
    if swap_at is not None:
        amp[dcode >= swap_at] = np.array([2.8, 3.0, 1.0, 0.5])
    M = (rng.standard_normal((n, 4)) * amp) @ Q.T + 0.2 * rng.standard_normal((n, dim))
    return M, dates, dcode


def _recovery(scorer, p) -> tuple:
    """(solo IC of the planted feature, IC of the fitted score, ratio) on scored dates only.

    The solo IC is recomputed on exactly the rows the scorer covers -- fm_scores and
    pooled_scores both burn a warm-up -- so the ratio is a property of the estimator and
    not of which dates each one happens to reach.
    """
    s = scorer(p["X"], p["y"], p["dates"], HORIZON)
    ok = np.isfinite(s)
    solo = float(mc.daily_spearman(p["X"][ok, 0], p["y_raw"][ok], p["dates"][ok]).mean())
    got = float(mc.daily_spearman(s, p["y_raw"], p["dates"]).mean())
    return solo, got, got / solo


# ------------------------------------------------------------------------ test 1


def test_no_leak_shift(panel_factory, shift_forward):
    """Break the feature-label pairing and the primary scorer must fall into its own null.

    Rolling every date's labels one session along leaves the marginal distributions and the
    serial structure untouched and destroys only the association, so any IC that survives
    is machinery, not signal. The band is a date-block permutation null built with the SAME
    scorer as the observed statistic -- that identity is fix F1, and using oos_scores here
    while grading fm_scores is exactly the mismatch the plan flags.
    """
    p = panel_factory(n_dates=260, n_names=45, n_feat=3, signal=0.35, seed=0)
    fm = (lambda X, y, d, h: mc.fm_scores(X, y, d, h))

    ic_plain = float(mc.daily_spearman(fm(p["X"], p["y"], p["dates"], HORIZON),
                                       p["y_raw"], p["dates"]).mean())
    Xs, ys, yrs, ds = shift_forward(p, 1)
    ic_shift = float(mc.daily_spearman(fm(Xs, ys, ds, HORIZON), yrs, ds).mean())

    null = mc.refit_null(Xs, ys, ds, HORIZON, n_draws=60, block=20, seed=7, scorer=fm)
    lo, hi = (float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5)))
    print(f"\n  unshifted IC {ic_plain:+.4f} | shifted IC {ic_shift:+.4f} | "
          f"null sd {np.nanstd(null):.4f} 95% band [{lo:+.4f}, {hi:+.4f}]")

    assert np.isfinite(null).sum() == 60
    assert lo < ic_shift < hi, f"shifted labels still score {ic_shift:+.4f} outside [{lo},{hi}]"
    assert ic_plain > hi, f"planted signal {ic_plain:+.4f} did not clear the null band {hi:+.4f}"


# ------------------------------------------------------------------------ test 2


def test_planted_signal():
    """A signal of known solo IC must survive both scorers at >= 30% of its size.

    This is the ruler every measured increment is read against: a small number out of the
    pipeline is uninterpretable unless the pipeline's attenuation is known. At solo IC
    ~0.05 the per-seed recovery is itself noisy (the IC's own SE is a large fraction of
    the bar), so the 0.05 level is gated on the mean over four seeds and the per-seed bar
    is applied at solo IC ~0.09 where the estimate is stable.
    """
    scorers = {"fm_scores": mc.fm_scores, "pooled_scores": mc.pooled_scores}

    weak = {k: [] for k in scorers}
    for seed in range(4):
        p = _signal_panel(400, 50, 22, 0.05, seed)
        for k, f in scorers.items():
            solo, got, rec = _recovery(f, p)
            weak[k].append(rec)
            print(f"\n  solo~0.05 seed {seed} {k:13s} solo {solo:+.4f} "
                  f"got {got:+.4f} rec {rec:.3f}")
    for k, v in weak.items():
        assert float(np.mean(v)) >= 0.30, f"{k} mean recovery {np.mean(v):.3f} at solo IC 0.05"

    for seed in range(3):
        p = _signal_panel(400, 50, 22, 0.10, seed)
        for k, f in scorers.items():
            solo, got, rec = _recovery(f, p)
            print(f"\n  solo~0.09 seed {seed} {k:13s} solo {solo:+.4f} "
                  f"got {got:+.4f} rec {rec:.3f}")
            assert solo > 0.05, f"{k} planted level moved: solo {solo:+.4f}"
            assert rec >= 0.30, f"{k} recovery {rec:.3f} at solo IC {solo:+.4f}, seed {seed}"


# ------------------------------------------------------------------------ test 3


def test_fm_embargo_boundary():
    """b at date i-h-1 is the last coefficient fm_scores may average into date i's weight.

    The label of date s ends at open(s+1+h) and is therefore first known at the close of
    date s+h+1; a weight scoring date i may only average coefficients from dates whose
    labels closed before i, i.e. s <= i-h-1. Making exactly ONE date carry a coefficient
    turns that arithmetic into an observable: the labels are identically zero everywhere
    else, so b_t is exactly the zero vector there and the fitted score on a date that has
    not yet reached date S is exactly zero rather than merely small.
    """
    h, S, n_dates, n_names = HORIZON, 100, 200, 40
    ud, dates, dcode = _dates(n_dates, n_names)
    rng = np.random.default_rng(0)
    X = mc.rank_within_date(rng.standard_normal((n_dates * n_names, 3)), dcode)
    y = np.zeros(len(X))
    y[dcode == S] = X[dcode == S, 0]

    _, B = mc.fm_coeffs(X, y, dates)
    assert np.flatnonzero(np.abs(B).sum(axis=1) > 0).tolist() == [S], "more than one live date"

    sc = mc.fm_scores(X, y, dates, h)
    before, after = sc[dcode == S + h], sc[dcode == S + h + 1]
    r = float(np.corrcoef(after, X[dcode == S + h + 1, 0])[0, 1])
    print(f"\n  date S+h={S + h}: max|score| {np.abs(before).max():.3e}   "
          f"date S+h+1={S + h + 1}: max|score| {np.abs(after).max():.3e}  corr(X0) {r:+.4f}")

    assert np.isfinite(before).all() and np.isfinite(after).all(), "boundary dates unscored"
    # i = S+h: the last permitted coefficient is i-h-1 = S-1, so b_S is still embargoed
    assert np.all(before == 0.0), f"b_{S} used {h} sessions early"
    # i = S+h+1: i-h-1 = S, the first date on which b_S is knowable
    assert np.abs(after).max() > 0.0 and r > 0.9, f"b_{S} never used (corr {r:+.4f})"


def test_pooled_embargo_boundary():
    """pooled_scores' trailing window must stop h+1 dates before the block it scores.

    Same arithmetic as fm_scores, applied to a block rather than a date: the ridge that
    scores dates [i, i+refit_every) is fitted on [i-h-train_window, i-h-1], so the last
    training date is i-h-1. The pooled scorer exists because a 50-name cross-section
    cannot carry p = 54 regressors (F3); it is only worth having if its embargo is the
    same one, so the boundary is pinned here the same way -- one live date, zero labels
    everywhere else, hence an exactly-zero weight when that date is still embargoed.
    """
    h, i, n_dates, n_names = HORIZON, 140, 300, 40
    ud, dates, dcode = _dates(n_dates, n_names)
    rng = np.random.default_rng(0)
    X = mc.rank_within_date(rng.standard_normal((n_dates * n_names, 3)), dcode)
    blk = (dcode >= i) & (dcode < i + 20)

    got = {}
    for S in (i - h - 1, i - h):
        y = np.zeros(len(X))
        y[dcode == S] = X[dcode == S, 0]
        v = mc.pooled_scores(X, y, dates, h)[blk]
        assert np.isfinite(v).all(), f"block [{i},{i + 20}) unscored with S={S}"
        got[S] = v
    r = float(np.corrcoef(got[i - h - 1], X[blk, 0])[0, 1])
    print(f"\n  block [{i},{i + 20}): S={i - h - 1} max|score| {np.abs(got[i - h - 1]).max():.3e} "
          f"corr {r:+.4f} | S={i - h} max|score| {np.abs(got[i - h]).max():.3e}")
    assert np.abs(got[i - h - 1]).max() > 0.0 and r > 0.9, "the last usable date was dropped"
    assert np.all(got[i - h] == 0.0), f"b_{i - h} used before its label closed"


# ------------------------------------------------------------------------ test 4


def test_refit_null_scorer(monkeypatch):
    """`scorer` must be the estimator the null is built from, called once per draw.

    A pooled fit and a Fama-MacBeth fit have different sampling variance under the null, so
    a sigma computed with one and applied to the other is not a p-value (F1). The stub
    returns the permuted labels themselves, which forces every draw's IC to +1: any draw
    that is not exactly 1.0 came from somewhere other than the callable that was passed.
    """
    p = _signal_panel(160, 30, 3, 0.2, seed=0)
    calls = []

    def counting(X, y, dates, horizon):
        calls.append((X.shape, horizon))
        return y

    out = mc.refit_null(p["X"], p["y"], p["dates"], HORIZON, n_draws=7, block=20, seed=1,
                        scorer=counting)
    print(f"\n  scorer calls {len(calls)}  ICs {np.round(out, 6)}")
    assert len(calls) == 7, f"scorer called {len(calls)} times for 7 draws"
    assert all(c[1] == HORIZON for c in calls), "horizon not forwarded to the scorer"
    assert np.allclose(out, 1.0), f"nulls did not come from the injected scorer: {out}"

    # ... and scorer=None still routes to oos_scores, so null_causal.json reproduces
    hits = []
    real = mc.oos_scores
    monkeypatch.setattr(mc, "oos_scores",
                        lambda *a, **kw: (hits.append(kw.get("n_folds")), real(*a, **kw))[1])
    mc.refit_null(p["X"], p["y"], p["dates"], HORIZON, n_draws=2, block=20, seed=1, n_folds=3)
    assert hits == [3, 3], f"default scorer did not call oos_scores(n_folds=3): {hits}"


def test_refit_null_family_max_and_eval_mask():
    """The family null shares one label permutation across arms and grades on a subset.

    Sharing the draw is what makes the per-draw max |IC| a family-wise null: the arms are
    correlated, so drawing them independently would overstate the maximum and understate
    the correction (F6). `eval_mask` grades on the NIFTY 50 rows while the fit still sees
    every row -- the fit-on-200 / score-on-50 design of section 4 -- so the mask must
    change the IC without changing what the scorer was handed.
    """
    p = _signal_panel(200, 40, 4, 0.2, seed=2)
    widths = {"a2": p["X"][:, :2], "a4": p["X"]}
    # arms may carry a validity mask (causal_pca's unscored warm-up); pass one as a tuple
    valid = p["dcode"] >= 10
    widths["a4_masked"] = (p["X"], valid)
    seen_rows = []

    def scorer(X, y, dates, horizon):
        seen_rows.append(len(X))
        return mc.fm_scores(X, y, dates, horizon)

    half = (np.arange(len(p["X"])) % 40) < 25          # 25 of every 40 names
    res = mc.refit_null_family(widths, p["y"], p["dates"], HORIZON, scorer, n_draws=3,
                               block=20, seed=5, eval_mask=half)
    arms, mx = res["arms"], res["max_abs"]
    print(f"\n  arms {sorted(arms)}  draws {len(mx)}  rows/scorer call {sorted(set(seen_rows))}")
    assert sorted(arms) == ["a2", "a4", "a4_masked"]
    assert all(len(v) == 3 for v in arms.values()) and len(mx) == 3
    assert len(seen_rows) == 9, "3 arms x 3 draws"
    # the fit saw every finite row, not the 25/40 evaluation subset
    assert max(seen_rows) == int(np.isfinite(p["y"]).sum())
    for i in range(3):
        assert np.isclose(mx[i], max(abs(arms[n][i]) for n in arms)), "max_abs is not the arm max"


# ------------------------------------------------------------------------ test 5


def test_causal_pca_modes():
    """refit reproduces the record; frozen never moves; procrustes keeps the coordinates.

    Three claims, one per mode. (i) `refit` is bit-for-bit ``attack_probe.causal_pca_block``,
    so the 2026-08-28 causal numbers stay reproducible from mlcore. (ii) `frozen` fits once
    on the first ``min_hist`` dates, so its output cannot depend on ``refit_every``.
    (iii) `procrustes` rotates each new basis onto the previous one, which makes every
    diagonal of ``evec_new.T @ evec_prev`` non-negative by construction (the aligned
    cross-product is V S V', PSD) and keeps consecutive bases far closer than a raw refit
    -- the incoherence F2 says attenuates a trailing coefficient average.
    """
    M, dates, dcode = _factor_panel(300, 25, 8, seed=11, swap_at=150)
    k = 3

    ref_b, ref_ok = causal_pca_block(M, k, dates, refit_every=60, min_hist=120)
    got_b, got_ok = mc.causal_pca(M, k, dates, mode="refit", refit_every=60, min_hist=120)
    assert np.array_equal(ref_b, got_b) and np.array_equal(ref_ok, got_ok), "refit mode drifted"
    assert got_ok.sum() > 0 and not got_ok[dcode < 120].any(), "warm-up dates were scored"

    f60 = mc.causal_pca(M, k, dates, mode="frozen", refit_every=60)
    f20 = mc.causal_pca(M, k, dates, mode="frozen", refit_every=20)
    _, _, fb = mc.causal_pca(M, k, dates, mode="frozen", return_basis=True)
    assert np.array_equal(f60[0], f20[0]) and np.array_equal(f60[1], f20[1]), "frozen basis refit"
    assert len(fb) == 1 and fb[0]["start"] == 120, f"frozen fitted {len(fb)} bases"
    assert f60[1].sum() == int((dcode >= 120).sum()), "frozen left scoreable rows unscored"

    _, _, pb = mc.causal_pca(M, k, dates, mode="procrustes", refit_every=60, return_basis=True)
    _, _, rb = mc.causal_pca(M, k, dates, mode="refit", refit_every=60, return_basis=True)
    assert len(pb) == len(rb) > 1, "no refit boundary to align across"
    for i in range(1, len(pb)):
        dp = np.diag(pb[i]["evec"].T @ pb[i - 1]["evec"])
        dr = np.diag(rb[i]["evec"].T @ rb[i - 1]["evec"])
        gp = float(np.linalg.norm(pb[i]["evec"] - pb[i - 1]["evec"]))
        gr = float(np.linalg.norm(rb[i]["evec"] - rb[i - 1]["evec"]))
        print(f"\n  refit {i}: procrustes diag {np.round(dp, 4)} drift {gp:.4f} | "
              f"raw diag {np.round(dr, 4)} drift {gr:.4f}")
        assert (dp >= -1e-6).all(), f"procrustes left a negative alignment: {dp}"
        assert gp <= gr + 1e-9, "alignment moved the basis further from the previous one"
        assert gr > 3.0 * gp, f"alignment did nothing: drift {gp:.4f} vs raw {gr:.4f}"

    with pytest.raises(ValueError):
        mc.causal_pca(M, k, dates, mode="nope")


# ------------------------------------------------- F5: embargo on the inner alpha split


def test_oos_inner_embargo():
    """The h dates before the inner cut must leave the inner training set, and only those.

    Their labels run into the inner validation block, so leaving them in tilts the alpha
    choice toward the block it is scored on (F5). The date index is smuggled through as an
    extra feature column, which lets the ridge call itself report exactly which dates it
    was handed; ``inner_embargo=False`` restores the 2026-08-28 split so the older screen
    numbers stay reproducible.
    """
    h, n_dates, n_names = HORIZON, 300, 40
    ud, dates, dcode = _dates(n_dates, n_names)
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n_dates * n_names, 2))
    y_raw = 0.3 * Z[:, 0] + rng.standard_normal(len(Z))
    X = np.column_stack([mc.rank_within_date(Z, dcode), dcode.astype(float)])
    y = mc.rank_within_date(y_raw[:, None], dcode)[:, 0]

    real, seen = mc.ridge_fit, []

    def spy(Xa, ya, alpha):
        seen.append(np.unique(Xa[:, -1]).astype(int))
        return real(Xa, ya, alpha)

    # first fold trains on ud[:120-h]; the inner cut sits at int(0.8 * 115) = 92
    n_tr = 120 - h
    i_cut = int(0.8 * n_tr)
    got = {}
    try:
        mc.ridge_fit = spy
        for emb in (True, False):
            seen.clear()
            mc.oos_scores(X, y, dates, h, inner_embargo=emb)
            got[emb] = [s for s in seen if len(s) < n_tr][0]
    finally:
        mc.ridge_fit = real

    print(f"\n  i_cut {i_cut}: inner-train max date  embargoed {got[True].max()}  "
          f"un-embargoed {got[False].max()}")
    assert got[False].max() == i_cut - 1, "un-embargoed split moved"
    assert got[True].max() == i_cut - h - 1, "embargo did not drop exactly h dates"
    assert len(got[False]) - len(got[True]) == h, "embargo dropped the wrong number of dates"


# ------------------------------------------- subset IC, top-k spread, the NW lag ladder


def test_subset_ic_and_topk_spread():
    """subset_ic grades the masked rows only; daily_topk_spread grades the top of the book.

    The head is fitted on the wide universe and graded on the tradable subset (F3), so the
    mask is applied to scores that already exist and must never reach the fit. IC is a
    whole-cross-section rank statistic and can be positive while the top-k names are not,
    which is why the plan gates on the spread as well as on the IC.
    """
    p = _signal_panel(120, 50, 3, 0.3, seed=4)
    mask = (np.arange(len(p["X"])) % 50) < 35
    sub = mc.subset_ic(p["X"][:, 0], p["y_raw"], p["dates"], mask, min_names=30)
    ref = mc.daily_spearman(p["X"][mask, 0], p["y_raw"][mask], p["dates"][mask], min_names=30)
    assert sub.equals(ref) and len(sub) == 120

    # 30 eligible names is below min_names=30? no -- exactly at it; 29 must drop the date
    thin = (np.arange(len(p["X"])) % 50) < 29
    assert len(mc.subset_ic(p["X"][:, 0], p["y_raw"], p["dates"], thin, min_names=30)) == 0

    spread = mc.daily_topk_spread(p["X"][:, 0], p["y_raw"], p["dates"], k=10)
    perfect = mc.daily_topk_spread(p["y_raw"], p["y_raw"], p["dates"], k=10)
    print(f"\n  subset IC {sub.mean():+.4f}  top-10 spread {spread.mean():+.4f}  "
          f"oracle spread {perfect.mean():+.4f}")
    assert len(spread) == 120 and spread.mean() > 0.0
    assert perfect.mean() > spread.mean(), "a perfect score did not beat the planted feature"
    # the spread is measured against the whole cross-section, so k = N must give exactly 0
    assert np.allclose(mc.daily_topk_spread(p["X"][:, 0], p["y_raw"], p["dates"], k=50), 0.0)


def test_summarise_lag_ladder():
    """se_nw_lag{5,12,20} and a block-20 CI, with the record's keys left where they were.

    A fitted score is a trailing mean of past coefficients, so consecutive ICs are nearly
    the same number and the lag choice moves the t materially; reporting the ladder stops
    one lag being picked silently (F4). lag=h stays under se_newey_west/t_newey_west so
    the 2026-08-28 tables reproduce from the same call.
    """
    rng = np.random.default_rng(0)
    e = rng.standard_normal(400)
    x = np.zeros(400)
    for i in range(1, 400):                      # AR(1), the persistence F4 is about
        x[i] = 0.9 * x[i - 1] + e[i]
    s = pd.Series(0.02 + 0.01 * x)
    out = mc.summarise(s, HORIZON)
    inc = mc.paired_increment(s, pd.Series(np.zeros(400)), HORIZON)
    print(f"\n  se lag5 {out['se_nw_lag5']:.5f}  lag12 {out['se_nw_lag12']:.5f}  "
          f"lag20 {out['se_nw_lag20']:.5f}  t20 {out['t_nw_lag20']:+.2f}  "
          f"block20 CI [{out['block_ci95_b20'][0]:+.4f}, {out['block_ci95_b20'][1]:+.4f}]")
    for d in (out, inc):
        for L in (5, 12, 20):
            assert d[f"se_nw_lag{L}"] > 0 and np.isfinite(d[f"t_nw_lag{L}"])
        assert d["se_newey_west"] == d["se_nw_lag5"], "lag=h key no longer matches the ladder"
        assert d["t_newey_west"] == d["t_nw_lag5"]
        assert len(d["block_ci95_b20"]) == 2 and d["block_ci95_b20"][0] < d["block_ci95_b20"][1]
    assert out["se_nw_lag20"] > out["se_nw_lag5"], "longer lag did not widen a persistent SE"
    assert abs(out["t_nw_lag20"]) < abs(out["t_naive"]), "NW t not below the naive t"
    assert out["mean_ic"] == pytest.approx(inc["mean_increment"]), "A - 0 is not A"


def test_refit_null_eval_mask_matches_the_observed_cross_section():
    """The null must be graded on the SAME cross-section as the observed IC (F1, extended).

    A per-date Spearman over N names has sd ~ 1/sqrt(N-1). Grading the permutation null on the
    whole 200-name fitting panel while the observed IC is measured on the ~50-name NIFTY 50
    subset therefore shrinks the null sigma by ~2x and doubles the apparent z -- the same
    estimator-mismatch error the plan's F1 fixes, in the cross-section dimension. This asserts
    that ``eval_mask`` reaches the IC, that it does NOT reach the fit, and that the masked null
    is materially wider than the unmasked one on the same draws.
    """
    rng = np.random.default_rng(5)
    n_dates, n_names, n_feat = 160, 200, 3
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.standard_normal((n_dates * n_names, n_feat))
    y = rng.standard_normal(n_dates * n_names)
    # 50 of every 200 names are "in the index"
    sub = (np.tile(np.arange(n_names), n_dates) < 50)

    seen_rows = []

    def scorer(Xa, ya, da, h):
        seen_rows.append(len(ya))
        return Xa[:, 0]

    wide = mc.refit_null(X, y, dates, HORIZON, n_draws=12, block=20, seed=1,
                         scorer=scorer, min_names=20)
    narrow = mc.refit_null(X, y, dates, HORIZON, n_draws=12, block=20, seed=1,
                           scorer=scorer, eval_mask=sub, min_names=30)

    # the fit saw every row in both cases -- only the grading was masked
    assert set(seen_rows) == {n_dates * n_names}, seen_rows
    sd_wide, sd_narrow = float(np.std(wide, ddof=1)), float(np.std(narrow, ddof=1))
    print(f"\n  null sd on 200 names {sd_wide:.4f} vs on the 50-name subset {sd_narrow:.4f} "
          f"(ratio {sd_narrow / sd_wide:.2f}x; sqrt(199/49) = {np.sqrt(199 / 49):.2f})")
    assert not np.allclose(wide, narrow), "eval_mask did not reach the IC"
    assert sd_narrow > 1.4 * sd_wide, (
        f"the 50-name null is not wider than the 200-name one: {sd_narrow:.4f} vs {sd_wide:.4f}")

    # and min_names still bites: a mask thinner than min_names keeps no date at all
    thin = (np.tile(np.arange(n_names), n_dates) < 29)
    assert not np.isfinite(mc.refit_null(X, y, dates, HORIZON, n_draws=2, block=20, seed=1,
                                         scorer=scorer, eval_mask=thin, min_names=30)).any()


def test_refit_null_default_scorer_exposes_the_inner_embargo():
    """``scorer=None`` routes to oos_scores and the F5 embargo is reachable from here.

    oos_scores now defaults to inner_embargo=True, so the default null path no longer
    reproduces the 2026-08-28 null bit-for-bit; ``inner_embargo=False`` is the way back and
    this asserts the flag is actually forwarded rather than swallowed.
    """
    seen = []
    real = mc.oos_scores

    def spy(Xa, ya, da, h, **kw):
        seen.append(kw.get("inner_embargo"))
        return real(Xa, ya, da, h, **kw)

    rng = np.random.default_rng(6)
    n_dates, n_names = 200, 25
    dates = np.repeat(np.arange(n_dates), n_names)
    X = rng.standard_normal((n_dates * n_names, 2))
    y = rng.standard_normal(n_dates * n_names)
    try:
        mc.oos_scores = spy
        mc.refit_null(X, y, dates, HORIZON, n_draws=1, block=20, seed=0, n_folds=3)
        mc.refit_null(X, y, dates, HORIZON, n_draws=1, block=20, seed=0, n_folds=3,
                      inner_embargo=False)
    finally:
        mc.oos_scores = real
    print(f"\n  inner_embargo forwarded to oos_scores: {seen}")
    assert seen == [True, False]


# --------------------------------------------------- the null that respects persistence


def _regime_panel(n_dates: int, n_names: int, n_feat: int, seed: int, block: int = 40):
    """A panel whose signal STRENGTH switches regime every `block` dates.

    The observed per-date IC series of a Fama-MacBeth score on the real NIFTY panel has
    AR(1) ~0.46: the same weight works for stretches and then does not. An iid-strength
    synthetic panel cannot show that -- its IC series is per-date cross-sectional noise and
    is white -- so the memory has to be planted, in the only place it lives in the real data:
    the strength of the relation, not the features.
    """
    ud, dates, dcode = _dates(n_dates, n_names)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_dates * n_names, n_feat))
    strength = np.where((np.arange(n_dates) // block) % 2 == 0, 0.9, 0.1)
    y = np.repeat(strength, n_names) * X[:, 0] + rng.standard_normal(n_dates * n_names)
    Xr = mc.rank_within_date(X, dcode)
    yr = mc.rank_within_date(y[:, None], dcode)[:, 0]
    return dates, Xr, yr, y


def test_coef_permutation_null_keeps_the_series_memory_the_date_null_destroys():
    """NIFTY50_PLAN F4, applied to the NULL rather than to the observed statistic.

    The date-block label null permutes labels ACROSS dates and reassigns them to arbitrary
    names within the target date, so within a draw the fitted score is graded against a
    cross-section it has no relation to: the draw's IC series is white and its NW-lag-20 SE
    is no wider than the naive one. The observed statistic is not that object -- its score is
    a 250-session trailing mean of coefficients and its IC series is strongly persistent -- so
    the sd of the mean IC under that null is too small, and the z it produces is too large.

    The coefficient-level null permutes the per-date coefficient matrix in blocks and rebuilds
    the trailing mean through `fm_apply`, grading against the REAL labels; its draws are the
    same KIND of series as the observed one. That is what this pins: coefficient-null AR(1)
    within 0.1 of the observed, date-null AR(1) at zero.
    """
    dates, Xr, yr, y = _regime_panel(420, 60, 3, seed=4)
    obs = mc.fm_scores(Xr, yr, dates, HORIZON)
    ic_obs = mc.daily_spearman(obs, y, dates, min_names=30)
    a_obs, i_obs = mc.ar1(ic_obs), mc.nw_inflation(ic_obs, 20)

    coef = mc.coef_permutation_null(Xr, yr, dates, HORIZON, y, n_draws=16, block=20,
                                    seed=3, min_names=30)
    date = mc.date_permutation_null(Xr, yr, dates, HORIZON, mc.fm_scores, n_draws=16,
                                    block=20, seed=3, min_names=30)
    a_coef, a_date = float(np.nanmean(coef["ar1"])), float(np.nanmean(date["ar1"]))
    i_coef, i_date = float(np.nanmean(coef["inflation"])), float(np.nanmean(date["inflation"]))
    print(f"\n  observed AR(1) {a_obs:+.3f} inflation {i_obs:.2f}"
          f"\n  coef null AR(1) {a_coef:+.3f} inflation {i_coef:.2f}"
          f"\n  date null AR(1) {a_date:+.3f} inflation {i_date:.2f}")

    assert abs(a_coef - a_obs) < 0.1, (a_coef, a_obs)
    assert abs(a_date) < 0.1, a_date
    assert abs(a_date - a_obs) > 0.2, "the two nulls are not distinguishable on this panel"
    # and the diagnostic that matters for the sd of the mean: the date null is ~white
    assert i_date < 1.2 and i_coef > 1.5 * i_date
    # the coefficient null's observed block is the same statistic as the direct fit
    assert abs(coef["observed"]["mean_ic"] - float(ic_obs.mean())) < 1e-12
    assert coef["observed"]["ar1"] == pytest.approx(a_obs)


def test_date_permutation_null_is_refit_null_with_diagnostics():
    """Same null, same rng stream: the diagnostics wrapper must not be a second estimator."""
    dates, Xr, yr, _ = _regime_panel(200, 40, 2, seed=9)
    kw = dict(n_draws=6, block=20, seed=5, min_names=30)
    a = mc.refit_null(Xr, yr, dates, HORIZON, scorer=mc.fm_scores, **kw)
    b = mc.date_permutation_null(Xr, yr, dates, HORIZON, mc.fm_scores, **kw)["ic_means"]
    assert np.allclose(a, b, equal_nan=True), np.c_[a, b]


def test_block_permute_index_is_a_block_shuffle_of_the_rows():
    rng = np.random.default_rng(0)
    idx = mc.block_permute_index(53, 20, rng)
    assert sorted(idx.tolist()) == list(range(53))
    # every block survives intact and contiguous, only their order changes
    blocks = [list(range(b * 20, min((b + 1) * 20, 53))) for b in range(3)]
    seen, i = [], 0
    while i < len(idx):
        blk = next(b for b in blocks if b[0] == idx[i])
        assert idx[i:i + len(blk)].tolist() == blk
        seen.append(blk[0])
        i += len(blk)
    assert sorted(seen) == [0, 20, 40]


def test_fm_apply_is_the_second_half_of_fm_scores():
    """The null rebuilds scores through the SAME code path, not a re-implementation."""
    dates, Xr, yr, _ = _regime_panel(200, 40, 2, seed=1)
    ud, B = mc.fm_coeffs(Xr, yr, dates)
    direct = mc.fm_scores(Xr, yr, dates, HORIZON)
    viaapply = mc.fm_apply(Xr, dates, ud, B, HORIZON)
    assert np.array_equal(np.isnan(direct), np.isnan(viaapply))
    assert np.allclose(direct[~np.isnan(direct)], viaapply[~np.isnan(viaapply)], atol=0, rtol=0)
