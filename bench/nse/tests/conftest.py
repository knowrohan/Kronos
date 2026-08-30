"""Fixtures for the estimator tests: synthetic panels only.

Nothing here reads ``bench/out`` or ``data/nse``, so the suite is fast and stays green on
a machine that has never run the pipeline; every number falls out of a fixed seed, so a
failure is a real regression and never a lucky draw.
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

from bench.nse.mlcore import rank_within_date  # noqa: E402


def pytest_configure(config):
    """Register `slow` locally rather than adding a repo-root pytest.ini.

    test_hidden.py marks the MPS pilot runs `slow`. With no ini file anywhere in the repo,
    pytest emits PytestUnknownMarkWarning on every run and `-m "not slow"` silently
    deselects nothing -- a filter that looks like it works and does not. Registering it
    here keeps the fix inside the test package: adding a pytest.ini at the repo root would
    move pytest's rootdir and change collection for every other suite in the tree.
    """
    config.addinivalue_line(
        "markers", "slow: runs the real Kronos model on MPS (minutes, needs the panel)")


def make_panel(n_dates: int = 240, n_names: int = 40, n_feat: int = 3,
               signal: float = 0.35, seed: int = 0) -> dict:
    """Balanced (date, name) panel whose feature 0 carries a planted cross-sectional edge.

    Rows are ordered date-major and every date has the same names, so a label vector can
    be reshaped to (n_dates, n_names) and rolled along the date axis to build the
    "nothing is knowable" control. Features and labels are rank-transformed within date,
    which is the only scaling the estimator is allowed to see.
    """
    rng = np.random.default_rng(seed)
    n = n_dates * n_names
    ud = pd.bdate_range("2020-01-01", periods=n_dates).to_numpy()
    dates = np.repeat(ud, n_names)
    dcode = np.repeat(np.arange(n_dates), n_names)
    Z = rng.standard_normal((n, n_feat))
    eps = rng.standard_normal(n)
    y_raw = signal * Z[:, 0] + np.sqrt(1.0 - signal ** 2) * eps
    return {
        "X": rank_within_date(Z, dcode),
        "Z": Z,
        "y": rank_within_date(y_raw[:, None], dcode)[:, 0],
        "y_raw": y_raw,
        "dates": dates,
        "dcode": dcode,
        "ud": ud,
        "n_dates": n_dates,
        "n_names": n_names,
    }


def shift_labels_forward(p: dict, k: int = 1):
    """Move every date's labels k sessions later, so feature and label are from different days.

    The first k dates lose their labels and their ROWS ARE DROPPED rather than NaN-filled:
    a NaN per-date coefficient would poison fm_scores' cumulative sum and turn a leak test
    into an all-NaN test. Returns (X, y_rank, y_raw, dates) on the surviving rows.
    """
    nd, nn = p["n_dates"], p["n_names"]
    keep = p["dcode"] >= k

    def roll(a):
        return np.roll(a.reshape(nd, nn), k, axis=0).reshape(-1)

    return p["X"][keep], roll(p["y"])[keep], roll(p["y_raw"])[keep], p["dates"][keep]


@pytest.fixture
def panel_factory():
    """The synthetic-panel builder, so each test picks its own size/signal/seed."""
    return make_panel


@pytest.fixture
def shift_forward():
    """The forward label shift used by the leak test."""
    return shift_labels_forward
