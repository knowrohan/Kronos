"""Tests for the A8 trainable head (`bench/nse/n50_head.py`). Synthetic, seeded, seconds.

Three things have to be true before any A8 number is worth reading, and each is checked on a
panel whose answer is known by construction rather than inferred from a real run:

  1. **The optimiser works.** A ranking head that does not reduce its own loss cannot be used
     to argue that the objective was or was not the binding constraint -- a flat loss curve and
     a dead representation look identical in the output. So: loss falls, and on a panel with a
     planted linear edge the fitted score recovers it, while on a panel with no edge it does
     not manufacture one.
  2. **Nothing is knowable early.** The whole arm is a claim about out-of-sample ranking, so
     the fold table is audited directly: no training date lies within h sessions before any
     date that fold scores, and the same holds for the inner early-stopping split.
  3. **Seed averaging does what it says.** Three seeds in, one score column out, defined on
     exactly the rows the folds scored and NaN everywhere else.
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

from bench.nse.mlcore import daily_spearman, rank_within_date  # noqa: E402
from bench.nse.n50_head import (HeadConfig, _groups, check_folds, date_codes,  # noqa: E402
                                make_folds, run_head, sample_pairs, train_head,
                                usable_start)

pytest.importorskip("torch")

H = 5


def tiny_cfg(**kw) -> HeadConfig:
    """A fold schedule small enough to run in seconds but with the real embargo arithmetic."""
    base = dict(k=4, hidden=8, train_window=60, refit_every=10, val_frac=0.2,
                max_epochs=12, patience=4, pairs_per_date=60, dates_per_batch=8)
    base.update(kw)
    return HeadConfig(**base)


def panel(n_dates=140, n_names=30, k=4, signal=0.6, seed=0):
    """Date-major panel whose label is a monotone function of feature 0 plus noise.

    `signal = 0` gives a pure-noise panel, which is the control the head must NOT beat.
    """
    rng = np.random.default_rng(seed)
    ud = pd.bdate_range("2020-01-01", periods=n_dates).to_numpy()
    dates = np.repeat(ud, n_names)
    dcode = np.repeat(np.arange(n_dates), n_names)
    Z = rng.standard_normal((n_dates * n_names, k))
    eps = rng.standard_normal(n_dates * n_names)
    y_raw = signal * Z[:, 0] + np.sqrt(max(1e-9, 1 - signal ** 2)) * eps
    X = rank_within_date(Z, dcode)
    y = rank_within_date(y_raw[:, None], dcode)[:, 0]
    return X, y, y_raw, dates, dcode


# --------------------------------------------------------------- 1. the optimiser works


def test_training_loss_decreases():
    """One fold, one seed: the pairwise logistic loss falls and the val loss improves."""
    X, y, _, _, dcode = panel(n_dates=90, signal=0.6, seed=1)
    cfg = tiny_cfg(max_epochs=30, patience=30)
    tr, va = dcode < 60, (dcode >= 60) & (dcode < 90)
    _, hist = train_head(X[tr], y[tr], dcode[tr], X[va], y[va], dcode[va], cfg, seed=0)
    assert hist["train_loss"][-1] < hist["train_loss"][0], hist["train_loss"][:3]
    assert hist["best_val_loss"] < hist["val_loss"][0]
    # log(2) is the loss of a head that ranks at chance; a trained one must be under it
    assert hist["train_loss"][-1] < np.log(2.0)


def test_listwise_loss_also_decreases():
    """The declared alternative objective trains too, so the flag is a real option."""
    X, y, _, _, dcode = panel(n_dates=90, signal=0.6, seed=2)
    cfg = tiny_cfg(loss="listwise", max_epochs=30, patience=30)
    tr, va = dcode < 60, (dcode >= 60) & (dcode < 90)
    _, hist = train_head(X[tr], y[tr], dcode[tr], X[va], y[va], dcode[va], cfg, seed=0)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_head_recovers_a_planted_edge_and_not_a_missing_one():
    """Positive control and negative control from the same code path.

    Without the second half the first is untestable: any scorer that overfits the training
    block would pass "IC > 0" on a panel where the edge is real.
    """
    X, y, y_raw, dates, _ = panel(n_dates=140, signal=0.6, seed=3)
    got = run_head(X, y, dates, H, cfg=tiny_cfg(), seeds=(0, 1))
    ic = daily_spearman(got["scores"], y_raw, dates)
    assert ic.mean() > 0.10, ic.mean()

    Xn, yn, yn_raw, dn, _ = panel(n_dates=140, signal=0.0, seed=4)
    null = run_head(Xn, yn, dn, H, cfg=tiny_cfg(), seeds=(0, 1))
    icn = daily_spearman(null["scores"], yn_raw, dn)
    assert abs(icn.mean()) < 0.06, icn.mean()


# ------------------------------------------------------------------ 2. nothing leaks


def test_fold_table_embargo_at_every_fold():
    """No training date within h sessions before any date the fold scores.

    Checked on the table itself (the artifact `run_head` actually iterates), for every fold
    and for a range of h, so the invariant is not a property of one lucky horizon.
    """
    for h in (1, 5, 10, 20):
        folds = make_folds(500, h, first_date=0, train_window=250, refit_every=20)
        assert len(folds) > 0
        for _, f in folds.iterrows():
            assert f["test_lo"] - f["train_hi"] == h + 1
            assert f["val_lo"] - f["inner_hi"] == h + 1
            assert f["train_hi"] - f["train_lo"] + 1 == 250
            assert f["inner_lo"] == f["train_lo"] and f["val_hi"] == f["train_hi"]
            assert f["test_hi"] >= f["test_lo"]
        assert check_folds(folds, h)["outer_gap_min"] == h + 1


def test_fold_table_respects_the_block_start():
    """A frozen PCA basis leaves the first `min_hist` dates without features; no fold may
    train on them."""
    folds = make_folds(760, H, first_date=120, train_window=250, refit_every=20)
    assert folds["train_lo"].min() >= 120
    assert int(folds["refit_at"].min()) == 120 + 250 + H


def test_val_first_keeps_the_freshest_dates_in_the_fit():
    """`val_position='first'` is the staleness probe: the early-stopping block moves to the
    OLD end of the window, the fit keeps the freshest date, and the inner embargo flips
    direction rather than disappearing."""
    last = make_folds(500, H, train_window=250, refit_every=20, val_position="last")
    first = make_folds(500, H, train_window=250, refit_every=20, val_position="first")
    assert len(last) == len(first)
    for (_, a), (_, b) in zip(last.iterrows(), first.iterrows()):
        assert a["train_lo"] == b["train_lo"] and a["train_hi"] == b["train_hi"]
        assert a["inner_hi"] == a["train_hi"] - 50 - H        # assigned protocol is stale
        assert b["inner_hi"] == b["train_hi"]                 # probe keeps the freshest date
        assert b["inner_lo"] - b["val_hi"] == H + 1           # embargo, other direction
        # same number of fitted dates either way, so the two are comparable
        assert a["inner_hi"] - a["inner_lo"] == b["inner_hi"] - b["inner_lo"]
    assert check_folds(first, H)["inner_gap_min"] == H + 1
    with pytest.raises(ValueError, match="val_position"):
        make_folds(500, H, val_position="middle")


def test_check_folds_rejects_an_overlapping_inner_split():
    folds = make_folds(500, H, train_window=250, refit_every=20)
    folds.loc[0, "val_lo"] = folds.loc[0, "inner_hi"]          # fit and val share a date
    with pytest.raises(ValueError, match="inner split overlaps"):
        check_folds(folds, H)


def test_check_folds_rejects_a_hand_broken_table():
    """The audit is load-bearing, so it must actually fail on a table that violates it."""
    folds = make_folds(500, H, train_window=250, refit_every=20)
    folds.loc[0, "train_hi"] = folds.loc[0, "test_lo"] - H     # one session too close
    with pytest.raises(ValueError, match="outer embargo"):
        check_folds(folds, H)


def test_run_head_scores_only_post_embargo_dates():
    """End-to-end: every scored date is at least h+1 sessions after every date used to fit
    the model that scored it."""
    X, y, _, dates, dcode = panel(n_dates=140, signal=0.5, seed=5)
    cfg = tiny_cfg()
    got = run_head(X, y, dates, H, cfg=cfg, seeds=(0,))
    folds = got["folds"]
    scored = np.isfinite(got["scores"])
    assert scored.any()
    lo = int(folds["test_lo"].min())
    assert dcode[scored].min() == lo
    assert not scored[dcode < lo].any()
    for _, f in folds.iterrows():
        assert int(f["test_lo"]) - int(f["train_hi"]) >= H + 1


def test_usable_start_rejects_a_hole():
    """Session-count embargo arithmetic is only valid if the usable dates are a suffix."""
    dcode = np.repeat(np.arange(10), 3)
    ok = np.ones(30, dtype=bool)
    ok[:3] = False
    assert usable_start(ok, dcode, 10) == 1
    ok[15:18] = False                      # a hole in the middle
    with pytest.raises(ValueError, match="contiguous suffix"):
        usable_start(ok, dcode, 10)


# ------------------------------------------------------------- 3. seeds and pair sampling


def test_seed_averaging_shape_and_support():
    """Three seeds in, one score column out, on exactly the scored rows."""
    X, y, _, dates, dcode = panel(n_dates=140, signal=0.5, seed=6)
    got = run_head(X, y, dates, H, cfg=tiny_cfg(), seeds=(0, 1, 2))
    assert got["per_seed"].shape == (3, len(X))
    assert got["scores"].shape == (len(X),)
    scored = np.isfinite(got["scores"])
    assert scored.sum() == got["n_scored_rows"] > 0
    # every seed scores exactly the rows the ensemble scores -- no seed silently drops out
    for si in range(3):
        assert np.array_equal(np.isfinite(got["per_seed"][si]), scored)
    # the published score is the mean of the per-seed WITHIN-DATE RANKS, so it is bounded
    assert np.nanmax(np.abs(got["scores"])) <= 1.0
    r = np.vstack([rank_within_date(got["per_seed"][si][scored][:, None], dcode[scored])[:, 0]
                   for si in range(3)]).mean(axis=0)
    assert np.allclose(got["scores"][scored], r)
    # and it differs from any single seed, i.e. the averaging is not a no-op
    assert not np.allclose(got["scores"][scored], r[0])


def test_seeds_differ_but_agree_in_sign():
    """Different seeds are different fits (otherwise averaging is theatre) that agree."""
    X, y, y_raw, dates, _ = panel(n_dates=140, signal=0.6, seed=7)
    got = run_head(X, y, dates, H, cfg=tiny_cfg(), seeds=(0, 1, 2))
    s = got["per_seed"]
    m = np.isfinite(s[0])
    assert not np.allclose(s[0][m], s[1][m])
    for si in range(3):
        assert daily_spearman(s[si], y_raw, dates).mean() > 0.05


def test_sample_pairs_never_crosses_a_date():
    """A pair spanning two dates would be a market-direction bet, not a cross-sectional one."""
    rng = np.random.default_rng(0)
    dcode = np.repeat(np.arange(20), 7)
    y = rng.standard_normal(len(dcode))
    starts, counts = _groups(dcode)
    a, b, t = sample_pairs(starts, counts, y, 50, rng)
    assert len(a) > 0
    assert np.array_equal(dcode[a], dcode[b])
    assert np.array_equal(t, np.sign(y[a] - y[b]))
    assert (t != 0).all() and (a != b).all()


def test_sample_pairs_drops_ties_and_singleton_dates():
    y = np.array([1.0, 1.0, 1.0, 2.0, 3.0])          # date 0 all ties, date 1 a singleton
    starts, counts = np.array([0, 3, 4]), np.array([3, 1, 1])
    a, b, t = sample_pairs(starts, counts, y, 40, np.random.default_rng(0))
    assert len(a) == 0


def test_config_parameter_count_matches_the_plan():
    """Linear(64,32)-SiLU-Linear(32,1) is ~2.1k parameters; the plan budgets for that head."""
    import torch
    cfg = HeadConfig()
    assert cfg.n_params() == 2113
    from bench.nse.n50_head import build_head
    assert sum(p.numel() for p in build_head(cfg, 0).parameters()) == cfg.n_params()
    assert isinstance(build_head(cfg, 0)[1], torch.nn.SiLU)


def test_linear_head_ablation_builds_and_trains():
    """`hidden = 0` is the nonlinearity ablation the sensitivity grid needs: same objective,
    same protocol, a linear read-out. If it silently built an MLP the decomposition would be
    a tautology."""
    import torch
    cfg = tiny_cfg(hidden=0, max_epochs=20, patience=20)
    assert cfg.n_params() == cfg.k + 1
    from bench.nse.n50_head import build_head
    net = build_head(cfg, 0)
    assert len(net) == 1 and isinstance(net[0], torch.nn.Linear)
    assert sum(p.numel() for p in net.parameters()) == cfg.n_params()
    X, y, y_raw, dates, dcode = panel(n_dates=90, signal=0.6, seed=9)
    tr, va = dcode < 60, (dcode >= 60) & (dcode < 90)
    _, hist = train_head(X[tr], y[tr], dcode[tr], X[va], y[va], dcode[va], cfg, seed=0)
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_sensitivity_grid_is_declared_and_valid():
    """Every variant in the declared grid must be a legal HeadConfig override -- a typo here
    would silently drop an arm from a table that claims to report the grid whole."""
    from bench.nse.n50_head import SENSITIVITY
    from dataclasses import asdict, fields
    names = {f.name for f in fields(HeadConfig)}
    assert SENSITIVITY
    for kw in SENSITIVITY.values():
        assert set(kw) <= names, set(kw) - names
        HeadConfig(**{**asdict(HeadConfig()), **kw})


def test_date_codes_requires_sorted_rows():
    X, y, _, dates, _ = panel(n_dates=40, n_names=5, seed=8)
    perm = np.random.default_rng(0).permutation(len(dates))
    with pytest.raises(ValueError, match="sorted by date"):
        run_head(X[perm], y[perm], dates[perm], H, cfg=tiny_cfg())
