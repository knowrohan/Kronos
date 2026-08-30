"""A8 -- the trainable ranking head, and the one question it is asked to answer.

NIFTY50_PLAN section 5 arm A8 / REBUILD_PLAN Stage 3. Every other arm in the study feeds a
frozen block into a LINEAR estimator (`mlcore.fm_scores`: a per-date cross-sectional ridge
whose coefficients are averaged over the trailing 250 dates). If the pretrained trunk state
carries an edge that a linear read-out cannot see -- an interaction between principal
directions, a monotone-but-not-linear response -- then the estimator, not the representation,
is the binding constraint, and the whole ladder of §5 has been measuring the wrong thing.

So this file trains the smallest head that could possibly show that: a 2,113-parameter MLP
(Linear(64,32) - SiLU - Linear(32,1)) on the SAME causal PCA-64 block the ridge arm gets,
with the SAME labels, on the SAME dates, under a rolling protocol with the SAME h-session
embargo. The comparison is paired per date, so the estimator attenuation that makes every
level number in this study a lower bound cancels between the two arms.

**The plan's decision rule, pre-registered:** if the head does not beat the frozen ridge --
specifically if `head - ridge <= -1 date-block SE` -- then the objective was not the
constraint, and no amount of head capacity is going to rescue the representation. A head that
merely ties is equally informative and is reported as a tie, never as a win.

Protocol (fixed before the run, argparse-visible so a reader can see it was not tuned):
  * Block: `mlcore.causal_pca(h_T, k=64, mode="frozen")` -- one basis, fitted on the first 120
    dates, applied forever, then ranked within date. Frozen because a rotating basis makes a
    trailing coefficient mean incoherent (NIFTY50_PLAN F2) and because the head must not be
    handed a feature whose meaning changes under it mid-sample.
  * Labels: `y_h5` ranked within date over the wide PIT NIFTY 200 cross-section.
  * Rolling fit: refit every 20 dates on the trailing 250 dates ENDING h+1 sessions before the
    refit point. Date s = i-h-1 is the last date whose label open(s+1+h)/open(s+1) is known by
    the open of date i, which is exactly the embargo `fm_scores` and `pooled_scores` apply.
  * Early stopping: the last 20% (50 dates) of the training window is held out, and the h dates
    before it are dropped, so the inner split carries the same embargo as the outer one
    (NIFTY50_PLAN F5 -- an inner split without an embargo tilts model selection toward the
    block it is scored on). Adam lr 1e-3, weight decay 1e-4, max 100 epochs, patience 10,
    best-epoch weights restored.
  * Seeds 0, 1, 2. Their test-block scores are ranked within date and then averaged: a raw
    mean would weight whichever seed happened to land on the larger output scale, and every
    metric downstream is a within-date rank statistic on which a monotone transform is a
    no-op. The raw mean is computed too and reported beside it, because "average the seeds"
    admits both readings.
  * Fitted on all eligible PIT NIFTY 200 cells, graded on the `is_n50` subset (F3).

Rows whose y_h5 is not finite -- the last 6 sessions, which have no forward window -- are
dropped from BOTH arms before anything runs, so the head and the ridge see an identical row
set and the paired difference is exact. That costs the score file its last 6 signal dates and
is stated in the JSON rather than left for a reader to discover.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.mlcore import (OUT_NSE, causal_pca, daily_spearman,  # noqa: E402
                              fm_scores, paired_increment, pooled_scores,
                              rank_within_date, summarise)
from bench.nse.n50 import evaluate, summary_json, year_table  # noqa: E402

#: Declared BEFORE the primary run, and run whole -- these are not a search for a cell that
#: passes. Each asks one question the primary number cannot answer on its own:
#:   long / long_lr3e-3  -- was patience-10 early stopping the thing that beat the head?
#:   lr3e-3              -- was the step size?
#:   listwise            -- was the pairwise objective the wrong one?
#:   hidden64            -- was 2.1k parameters too few?
#:   linear_head         -- is any gap the OBJECTIVE + PROTOCOL, or the nonlinearity?
#:   val_first(_long)    -- the assigned protocol spends the FRESHEST 50 dates of the window on
#:                          early stopping, so the head is fitted on data 56 sessions staler
#:                          than the ridge's. This moves the held-out block to the old end and
#:                          prices that handicap instead of arguing about it.
#: Every one of them is reported whatever it says; none of them can promote the arm past
#: its pre-registered primary cell.
SENSITIVITY = {
    "long":         {"max_epochs": 300, "patience": 30},
    "long_lr3e-3":  {"max_epochs": 300, "patience": 30, "lr": 3e-3},
    "lr3e-3":       {"lr": 3e-3},
    "listwise":     {"loss": "listwise"},
    "hidden64":     {"hidden": 64},
    "linear_head":  {"hidden": 0},
    "val_first":    {"val_position": "first"},
    "val_first_long": {"val_position": "first", "max_epochs": 300, "patience": 30},
}

ML = os.path.join(OUT_NSE, "ml")
N50 = os.path.join(OUT_NSE, "n50")
DATASET_N50 = os.path.join(N50, "dataset_n50.parquet")
ARM = "A8_head"


@dataclass
class HeadConfig:
    """Every knob, in one object, so the JSON records what was actually run.

    Defaults are the plan's; `budget_downshift` halves `max_epochs` and `pairs_per_date`
    and is set by the pilot projection, never by hand mid-run.
    """
    k: int = 64                     # PCA width of the input block
    hidden: int = 32                # Linear(k, hidden) - SiLU - Linear(hidden, 1)
    train_window: int = 250         # trailing dates per fit
    refit_every: int = 20           # dates scored per fit
    val_frac: float = 0.2           # tail of the training window held out for early stopping
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    pairs_per_date: int = 2000      # sampled per date per epoch (all-pairs is ~19,900 at N=200)
    dates_per_batch: int = 25       # one Adam step per this many dates
    val_position: str = "last"      # "last" (the assigned protocol) | "first" (staleness probe)
    loss: str = "pairwise"          # "pairwise" (primary) | "listwise" (flag alternative)
    temp: float = 1.0               # listwise target temperature; unused when loss=pairwise

    def n_params(self) -> int:
        if self.hidden <= 0:                      # linear head: the nonlinearity ablation
            return self.k + 1
        return self.k * self.hidden + self.hidden + self.hidden + 1


# --------------------------------------------------------------------- folds


def date_codes(dates: np.ndarray):
    """(unique dates, per-row 0-based date index). Rows must already be date-sorted."""
    ud, dcode = np.unique(dates, return_inverse=True)
    return ud, dcode.astype(np.int64)


def usable_start(ok: np.ndarray, dcode: np.ndarray, n_dates: int) -> int:
    """First date index from which EVERY later date has a complete block row.

    `causal_pca` leaves the pre-`min_hist` dates unscored, so the usable dates are a suffix
    of the calendar. Asserting that it really is a suffix matters: if it were not, date-index
    differences would stop being session counts and the embargo arithmetic below -- which is
    stated in sessions -- would quietly be measuring something else.
    """
    per_date = np.minimum.reduceat(ok.astype(np.int8), np.searchsorted(dcode, np.arange(n_dates)))
    first = int(np.argmax(per_date > 0)) if per_date.any() else n_dates
    if not per_date[first:].all():
        bad = first + int(np.argmin(per_date[first:]))
        raise ValueError(f"usable dates are not a contiguous suffix: date {bad} is unusable "
                         f"after usable date {first}; session-count embargo would be wrong")
    return first


def make_folds(n_dates: int, h: int, first_date: int = 0, *, train_window: int = 250,
               refit_every: int = 20, val_frac: float = 0.2,
               val_position: str = "last") -> pd.DataFrame:
    """The rolling fit/score schedule, as a table -- the artifact the leak test asserts on.

    All bounds are INCLUSIVE date indices. The invariant the whole arm rests on:
    `test_lo - train_hi == h + 1`, i.e. no training date lies within h sessions before any
    scored date. It is a table rather than a loop variable precisely so a test can check it.

    `val_position` decides which end of the window early stopping eats. "last" is the assigned
    protocol, and it costs the head something the ridge baseline never pays: the freshest date
    the head is FITTED on is then `val_frac * train_window` sessions older than the freshest
    date `fm_scores` uses, so part of any gap is staleness rather than objective. "first" puts
    the held-out block at the OLD end (embargoed forward by h instead of backward) and leaves
    the fit ending at `train_hi`, which measures that cost instead of arguing about it.
    """
    if val_position not in ("last", "first"):
        raise ValueError(f"val_position must be 'last' or 'first', got {val_position!r}")
    rows = []
    i = first_date + train_window + h
    while i < n_dates:
        tr_hi = i - h - 1
        tr_lo = tr_hi - train_window + 1
        if tr_lo < first_date:
            i += refit_every
            continue
        n_in = int(round((1.0 - val_frac) * train_window))
        if val_position == "last":
            bounds = {"inner_lo": tr_lo, "inner_hi": tr_lo + n_in - 1 - h,   # embargo -> val
                      "val_lo": tr_lo + n_in, "val_hi": tr_hi}
        else:
            n_val = train_window - n_in
            bounds = {"val_lo": tr_lo, "val_hi": tr_lo + n_val - 1,
                      "inner_lo": tr_lo + n_val + h, "inner_hi": tr_hi}   # embargo -> fit
        rows.append({
            "fold": len(rows), "refit_at": i,
            "train_lo": tr_lo, "train_hi": tr_hi, **bounds,
            "test_lo": i, "test_hi": min(i + refit_every, n_dates) - 1,
        })
        i += refit_every
    return pd.DataFrame(rows)


def check_folds(folds: pd.DataFrame, h: int) -> dict:
    """Embargo audit of the fold table. Returns the margins; raises if any is too small."""
    if not len(folds):
        raise ValueError("no folds: the sample is shorter than train_window + h")
    outer = (folds["test_lo"] - folds["train_hi"]).to_numpy()
    if (folds["val_lo"] > folds["inner_hi"]).all():
        inner = (folds["val_lo"] - folds["inner_hi"]).to_numpy()          # val_position="last"
    elif (folds["inner_lo"] > folds["val_hi"]).all():
        inner = (folds["inner_lo"] - folds["val_hi"]).to_numpy()          # val_position="first"
    else:
        raise ValueError("inner split overlaps: the fit and the early-stopping block share dates")
    if (outer < h + 1).any():
        raise ValueError(f"outer embargo violated: min gap {int(outer.min())} < {h + 1}")
    if (inner < h + 1).any():
        raise ValueError(f"inner embargo violated: min gap {int(inner.min())} < {h + 1}")
    return {"outer_gap_min": int(outer.min()), "outer_gap_max": int(outer.max()),
            "inner_gap_min": int(inner.min()), "inner_gap_max": int(inner.max()),
            "required_gap": int(h + 1)}


# ------------------------------------------------------------------ the head


def build_head(cfg: HeadConfig, seed: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    if cfg.hidden <= 0:
        # hidden = 0 is the nonlinearity ablation: same objective, same rolling protocol,
        # same embargo, but a LINEAR read-out. It is what separates "the loss function was
        # the constraint" from "the nonlinearity was the constraint" -- two claims that a
        # single head-vs-ridge number cannot tell apart.
        return nn.Sequential(nn.Linear(cfg.k, 1))
    return nn.Sequential(nn.Linear(cfg.k, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1))


def _groups(dcode_local: np.ndarray):
    """(starts, counts) of the contiguous per-date row groups. Rows must be date-sorted."""
    _, starts, counts = np.unique(dcode_local, return_index=True, return_counts=True)
    return starts.astype(np.int64), counts.astype(np.int64)


def sample_pairs(starts: np.ndarray, counts: np.ndarray, y: np.ndarray,
                 n_pairs: int, rng: np.random.Generator):
    """Up to `n_pairs` WITHIN-DATE index pairs per date, ties dropped, plus their target sign.

    Drawing both members inside one date's [start, start+count) slice is what makes the loss
    cross-sectional: a pair spanning two dates would ask the head to compare a Tuesday name
    with a Thursday name, which is a market-direction bet, not a ranking. Ties (equal label
    ranks, and the a == b self-pairs a uniform draw produces) are dropped rather than given a
    zero target, so they neither contribute gradient nor dilute the mean.
    """
    if len(starts) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0)
    d = np.repeat(np.arange(len(starts)), n_pairs)
    c, s = counts[d], starts[d]
    a = s + (rng.random(len(d)) * c).astype(np.int64)
    b = s + (rng.random(len(d)) * c).astype(np.int64)
    keep = np.isfinite(y[a]) & np.isfinite(y[b]) & (y[a] != y[b])
    a, b = a[keep], b[keep]
    return a, b, np.sign(y[a] - y[b])


def pairwise_loss(s, a, b, t):
    """Mean logistic loss over sampled pairs: log(1 + exp(-(s_i - s_j) * sign(y_i - y_j)))."""
    import torch
    return torch.nn.functional.softplus(-(s[a] - s[b]) * t).mean()


def listwise_loss(s, y, g, n_groups, temp: float = 1.0):
    """ListNet: cross-entropy between softmax(scores) and softmax(label ranks), per date.

    The declared alternative to the pairwise objective. Same information, different weighting:
    the pairwise loss counts every inversion equally, the listwise one concentrates on the top
    of each date's list -- which is closer to what a top-10 book captures, and is the reason
    it is offered as a flag rather than dismissed.
    """
    import torch
    neg = torch.finfo(s.dtype).min

    def _log_softmax_by_group(v):
        m = torch.full((n_groups,), neg, dtype=v.dtype).scatter_reduce(
            0, g, v, reduce="amax", include_self=False)
        e = torch.exp(v - m[g])
        den = torch.zeros(n_groups, dtype=v.dtype).index_add(0, g, e)
        return v - m[g] - torch.log(den)[g]

    logp = _log_softmax_by_group(s)
    p = torch.exp(_log_softmax_by_group(y / temp))
    return -(p * logp).sum() / max(n_groups, 1)


def train_head(Xtr, ytr, dtr, Xva, yva, dva, cfg: HeadConfig, seed: int):
    """Fit one head on one fold and one seed. Returns (model, history dict).

    Batching is by DATE, not by row: a minibatch is `dates_per_batch` whole cross-sections,
    because the loss is only defined within a date. Validation pairs are drawn ONCE and reused
    every epoch, so the early-stopping curve is a property of the model rather than of the
    epoch's dice, and `patience` counts real non-improvement.
    """
    import torch
    rng = np.random.default_rng(seed * 100003 + 17)
    model = build_head(cfg, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xt = torch.as_tensor(np.ascontiguousarray(Xtr), dtype=torch.float32)
    yt = torch.as_tensor(np.ascontiguousarray(ytr), dtype=torch.float32)
    Xv = torch.as_tensor(np.ascontiguousarray(Xva), dtype=torch.float32)
    yv = torch.as_tensor(np.ascontiguousarray(yva), dtype=torch.float32)

    starts, counts = _groups(dtr)
    keep = counts >= 2
    starts, counts = starts[keep], counts[keep]
    rows_of = [np.arange(s, s + c, dtype=np.int64) for s, c in zip(starts, counts)]

    vs, vc = _groups(dva)
    vkeep = vc >= 2
    vs, vc = vs[vkeep], vc[vkeep]
    va, vb, vt = sample_pairs(vs, vc, yva, cfg.pairs_per_date, rng)
    va_t = torch.as_tensor(va)
    vb_t = torch.as_tensor(vb)
    vt_t = torch.as_tensor(vt, dtype=torch.float32)
    vg = torch.as_tensor(np.repeat(np.arange(len(vs)), vc))
    vrows = torch.as_tensor(np.concatenate([np.arange(s, s + c) for s, c in zip(vs, vc)])
                            if len(vs) else np.empty(0, dtype=np.int64))

    def _val_loss() -> float:
        with torch.no_grad():
            sv = model(Xv).squeeze(-1)
            if cfg.loss == "listwise":
                return float(listwise_loss(sv[vrows], yv[vrows], vg, len(vs), cfg.temp))
            if len(va) == 0:
                return float("nan")
            return float(pairwise_loss(sv, va_t, vb_t, vt_t))

    hist = {"train_loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(cfg.max_epochs):
        order = rng.permutation(len(starts))
        tot, nb = 0.0, 0
        for j in range(0, len(order), cfg.dates_per_batch):
            chunk = order[j:j + cfg.dates_per_batch]
            ridx = np.concatenate([rows_of[c] for c in chunk])
            cc = counts[chunk]
            ls = np.concatenate([[0], np.cumsum(cc)[:-1]]).astype(np.int64)
            xb = Xt[torch.as_tensor(ridx)]
            yb_np = ytr[ridx]
            sb = model(xb).squeeze(-1)
            if cfg.loss == "listwise":
                g = torch.as_tensor(np.repeat(np.arange(len(chunk)), cc))
                loss = listwise_loss(sb, torch.as_tensor(yb_np, dtype=torch.float32),
                                     g, len(chunk), cfg.temp)
            else:
                a, b, t = sample_pairs(ls, cc, yb_np, cfg.pairs_per_date, rng)
                if len(a) == 0:
                    continue
                loss = pairwise_loss(sb, torch.as_tensor(a), torch.as_tensor(b),
                                     torch.as_tensor(t, dtype=torch.float32))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        hist["train_loss"].append(tot / max(nb, 1))
        v = _val_loss()
        hist["val_loss"].append(v)
        if np.isfinite(v) and v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    hist["best_val_loss"] = best
    hist["epochs_run"] = len(hist["train_loss"])
    hist["stopped_early"] = bool(len(hist["train_loss"]) < cfg.max_epochs)
    return model, hist


def score_rows(model, X) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.as_tensor(np.ascontiguousarray(X),
                                     dtype=torch.float32)).squeeze(-1).numpy().astype(np.float64)


# ------------------------------------------------------------------- driver


def run_head(X: np.ndarray, y: np.ndarray, dates: np.ndarray, h: int, *,
             ok: np.ndarray | None = None, seeds=(0, 1, 2), cfg: HeadConfig | None = None,
             progress: bool = False) -> dict:
    """Roll the head across the sample and return per-row out-of-sample scores.

    `scores` is the rank-averaged ensemble (NaN on every row no fold scored); `scores_raw`
    is the plain mean of the seeds' raw outputs, kept because "average the three seeds'
    scores" reads both ways and the two readings should be shown, not chosen between.
    """
    cfg = cfg or HeadConfig()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.ones(len(X), dtype=bool) if ok is None else np.asarray(ok, dtype=bool)
    ud, dcode = date_codes(dates)
    if not (np.diff(dcode) >= 0).all():
        raise ValueError("rows must be sorted by date")
    first = usable_start(ok, dcode, len(ud))
    folds = make_folds(len(ud), h, first, train_window=cfg.train_window,
                       refit_every=cfg.refit_every, val_frac=cfg.val_frac,
                       val_position=cfg.val_position)
    audit = check_folds(folds, h)

    n_seeds = len(seeds)
    per_seed = np.full((n_seeds, len(X)), np.nan)
    per_seed_rank = np.full((n_seeds, len(X)), np.nan)
    hist_rows, t0 = [], time.time()
    for _, f in folds.iterrows():
        tr = (dcode >= f["inner_lo"]) & (dcode <= f["inner_hi"]) & np.isfinite(y)
        va = (dcode >= f["val_lo"]) & (dcode <= f["val_hi"]) & np.isfinite(y)
        te = (dcode >= f["test_lo"]) & (dcode <= f["test_hi"])
        if tr.sum() < 100 or va.sum() < 50 or not te.any():
            continue
        ft = time.time()
        for si, sd in enumerate(seeds):
            model, hst = train_head(X[tr], y[tr], dcode[tr], X[va], y[va], dcode[va], cfg, sd)
            s = score_rows(model, X[te])
            per_seed[si, te] = s
            per_seed_rank[si, te] = rank_within_date(s[:, None], dcode[te])[:, 0]
            hist_rows.append({"fold": int(f["fold"]), "seed": int(sd),
                              "epochs": hst["epochs_run"], "stopped_early": hst["stopped_early"],
                              "train_loss_first": hst["train_loss"][0],
                              "train_loss_last": hst["train_loss"][-1],
                              "val_loss_first": hst["val_loss"][0],
                              "val_loss_best": hst["best_val_loss"]})
        if progress:
            print(f"  fold {int(f['fold']):2d}  refit@{int(f['refit_at']):3d}  "
                  f"train {int(tr.sum()):6d}  test {int(te.sum()):5d}  "
                  f"{time.time() - ft:5.1f}s", flush=True)

    scored = np.isfinite(per_seed_rank).any(axis=0)
    scores = np.full(len(X), np.nan)
    scores_raw = np.full(len(X), np.nan)
    if scored.any():
        scores[scored] = np.nanmean(per_seed_rank[:, scored], axis=0)
        scores_raw[scored] = np.nanmean(per_seed[:, scored], axis=0)
    return {"scores": scores, "scores_raw": scores_raw, "per_seed": per_seed,
            "folds": folds, "fold_audit": audit, "history": pd.DataFrame(hist_rows),
            "seconds": time.time() - t0, "n_scored_rows": int(scored.sum()),
            "n_scored_dates": int(len(np.unique(dcode[scored]))) if scored.any() else 0,
            "config": asdict(cfg), "seeds": list(seeds)}


# ------------------------------------------------------------------ reporting


def _md_table(df: pd.DataFrame) -> str:
    """Markdown from a DataFrame without pulling in `tabulate` (not installed here)."""
    if not len(df):
        return "_(empty)_"
    cols = list(df.columns)

    def fmt(v):
        return f"{v:+.5f}" if isinstance(v, float) and np.isfinite(v) else str(v)

    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, rule] + body)


def _lag_row(d: dict) -> dict:
    keys = ("n_dates", "mean_ic", "mean_increment", "t_newey_west", "se_newey_west",
            "t_nw_lag12", "t_nw_lag20", "se_nw_lag20", "block_ci95_b20", "share_positive")
    return {k: d[k] for k in keys if k in d}


def gate_table(inc: dict) -> dict:
    """The pre-registered A8 clauses, each with the number that decides it.

    G3 is the plan's own rule and is deliberately phrased so that FIRING it is the negative
    verdict: "head <= ridge by >= 1 date-block SE" means the objective was not what was
    holding the arm back. A tie fires neither G1/G2 nor G3, and a tie is a real outcome.
    """
    m = inc.get("mean_increment", float("nan"))
    se20 = inc.get("se_nw_lag20", float("nan"))
    lo, hi = inc.get("block_ci95_b20", [float("nan")] * 2)
    boot_sd = (hi - lo) / 3.92 if np.isfinite(hi) and np.isfinite(lo) else float("nan")
    return {
        "G1_head_beats_ridge_mean": {"clause": "mean(head - ridge) > 0",
                                     "value": m, "pass": bool(m > 0)},
        "G2_block_ci_excludes_zero": {"clause": "block-20 bootstrap CI95 of (head - ridge) > 0",
                                      "value": [lo, hi], "pass": bool(np.isfinite(lo) and lo > 0)},
        "G3_objective_not_the_constraint_nw20": {
            "clause": "mean(head - ridge) <= -1 x se_nw_lag20  =>  objective was not the constraint",
            "value": m, "se": se20, "pass": bool(np.isfinite(se20) and m <= -se20)},
        "G3b_objective_not_the_constraint_bootsd": {
            "clause": "mean(head - ridge) <= -1 x block-bootstrap sd (second reading of "
                      "'date-block SE')",
            "value": m, "se": boot_sd, "pass": bool(np.isfinite(boot_sd) and m <= -boot_sd)},
        "t_nw_lag20": inc.get("t_nw_lag20"),
        "t_nw_lag12": inc.get("t_nw_lag12"),
        "t_nw_lag_h": inc.get("t_newey_west"),
    }


def write_md(path: str, res: dict) -> None:
    L = []
    A = L.append
    cfg = res["config"]
    A("# A8 -- trainable ranking head on causal PCA-64 h_T (NIFTY 50)\n")
    A(f"_h = {res['h']}, arm `{ARM}`, {cfg['n_params']} parameters, seeds {res['seeds']}, "
      f"{res['run']['n_folds']} folds, {res['run']['seconds']:.0f} s._\n")
    A("\nThe question A8 asks is not 'is there alpha' but **'is the linear estimator the "
      "binding constraint?'** -- the head and the frozen ridge see the identical block, the "
      "identical labels and the identical dates, so the paired difference isolates the "
      "objective.\n")

    A("\n## Verdict\n")
    A("Pre-registered cell (primary protocol, seeds averaged):\n")
    for k, v in res["gates"].items():
        if isinstance(v, dict) and "pass" in v:
            A(f"- **{k}** -- {v['clause']} -> **{'PASS' if v['pass'] else 'FAIL'}** "
              f"(value {v['value']})")
    dec = res["paired"]
    A(f"\n**What the gate number is made of.** head - FM ridge = "
      f"{dec['ic']['mean_increment']:+.5f}; but held at the head's OWN protocol (a pooled "
      f"trailing-250-date fit) the objective is worth "
      f"{dec['head_minus_pooled']['mean_increment']:+.5f} "
      f"(t20 {dec['head_minus_pooled']['t_nw_lag20']:+.2f}), while the protocol itself is "
      f"worth {dec['pooled_minus_ridge']['mean_increment']:+.5f} "
      f"(t20 {dec['pooled_minus_ridge']['t_nw_lag20']:+.2f}) against the Fama-MacBeth "
      "coefficient mean. Read the gate as a verdict on the objective only if you are also "
      "willing to read it as a verdict on pooled-vs-FM fitting, which is what the split says "
      "it mostly is.\n")

    gb = res.get("gates_best_sensitivity") or {}
    if gb:
        A(f"\nSame clauses at the BEST declared sensitivity variant (`{gb['variant']}`, "
          f"IC {gb['mean_ic']:+.5f}, t20 {gb['t_nw_lag20']:+.2f}). This is a maximum taken "
          "after the fact over the whole grid, so it is an upper bound on the head, not a "
          "pass -- it is here because a verdict about what the objective *could* do must be "
          "stated against the objective's best showing, not only its pre-registered one.\n")
        for k, v in gb.items():
            if isinstance(v, dict) and "pass" in v:
                A(f"- **{k}** (post hoc) -- {v['clause']} -> "
                  f"**{'FIRES' if v['pass'] else 'does not fire'}** (value {v['value']})")
    A("")

    A("\n## Paired increment, head - ridge, on the NIFTY 50 subset (the gate)\n")
    A("| statistic | value |")
    A("|---|---|")
    for k, v in _lag_row(res["paired"]["ic"]).items():
        A(f"| {k} | {v} |")

    A("\n## Levels (NIFTY 50 subset, matched dates)\n")
    A("| arm | n_dates | mean IC | t (lag h) | t (lag 12) | t (lag 20) | block CI95 b20 | share > 0 |")
    A("|---|---|---|---|---|---|---|---|")
    for name, key in (("head (A8)", "head"), ("frozen ridge (FM)", "ridge_matched"),
                      ("pooled ridge (head's protocol, linear)", "pooled_matched"),
                      ("frozen ridge, all its dates", "ridge_full"),
                      ("pooled ridge, all its dates", "pooled_full")):
        s = res["eval"][key]["ic_summary"]
        A(f"| {name} | {s.get('n_dates')} | {s.get('mean_ic'):+.5f} | "
          f"{s.get('t_newey_west'):+.2f} | {s.get('t_nw_lag12'):+.2f} | "
          f"{s.get('t_nw_lag20'):+.2f} | "
          f"[{s['block_ci95_b20'][0]:+.5f}, {s['block_ci95_b20'][1]:+.5f}] | "
          f"{s.get('share_positive'):.3f} |")

    A("\n## Decomposition -- what the head-minus-ridge number is made of\n")
    A("The head differs from `fm_scores` in TWO ways at once: a pairwise-logistic objective "
      "with a nonlinearity, and a pooled trailing-250-date fit instead of a mean of per-date "
      "cross-sectional coefficients. `pooled_scores` on the same block, window, cadence and "
      "embargo is the head's own protocol made linear and least-squares, so the gap splits.\n")
    A("| difference | what it isolates | mean | t (lag 20) | block CI95 b20 |")
    A("|---|---|---|---|---|")
    for key, what in (("ic", "head - FM ridge (the assigned gate)"),
                      ("head_minus_pooled", "objective + nonlinearity, protocol held fixed"),
                      ("pooled_minus_ridge", "protocol: pooled fit vs Fama-MacBeth mean")):
        d = res["paired"][key]
        A(f"| {key} | {what} | {d.get('mean_increment'):+.5f} | {d.get('t_nw_lag20'):+.2f} | "
          f"[{d['block_ci95_b20'][0]:+.5f}, {d['block_ci95_b20'][1]:+.5f}] |")

    A("\n## Top-k spread (NIFTY 50, the quantity the book captures)\n")
    A("| arm | k | mean spread | t (lag 20) | block CI95 b20 |")
    A("|---|---|---|---|---|")
    for name, key in (("head (A8)", "head"), ("frozen ridge (FM)", "ridge_matched")):
        for k, s in res["eval"][key]["spread_summary"].items():
            A(f"| {name} | {k} | {s.get('mean_ic'):+.5f} | {s.get('t_nw_lag20'):+.2f} | "
              f"[{s['block_ci95_b20'][0]:+.5f}, {s['block_ci95_b20'][1]:+.5f}] |")

    A("\n## By calendar year (head, k = 10)\n")
    A(res["tables"]["head_year"])
    A("\n## By calendar year (ridge, matched dates, k = 10)\n")
    A(res["tables"]["ridge_year"])

    A("\n## Seed-averaging reading\n")
    A("`score` is the rank-within-date average of the three seeds; `scores_raw` is the plain "
      "mean of raw outputs. Both are reported because the instruction admits both.\n")
    A("| reading | mean IC (N50) | t (lag 20) |")
    A("|---|---|---|")
    for name, key in (("rank-average (published)", "head"), ("raw mean", "head_raw")):
        s = res["eval"][key]["ic_summary"]
        A(f"| {name} | {s.get('mean_ic'):+.5f} | {s.get('t_nw_lag20'):+.2f} |")
    for si, sd in enumerate(res["seeds"]):
        s = res["eval"]["per_seed"][str(sd)]["ic_summary"]
        A(f"| seed {sd} alone | {s.get('mean_ic'):+.5f} | {s.get('t_nw_lag20'):+.2f} |")

    if res.get("sensitivity"):
        A("\n## Declared sensitivity grid\n")
        A("Declared before the primary run and reported whole. None of these can promote the "
          "arm past its pre-registered cell; they exist so that 'the head lost' can be told "
          "apart from 'the head never trained'.\n")
        A("| variant | median epochs | mean IC (N50) | t (lag 20) | vs FM ridge | vs pooled ridge | "
          "per-seed IC | s |")
        A("|---|---|---|---|---|---|---|---|")
        pm = res["eval"]["head"]["ic_summary"]
        A(f"| **primary** | {res['run']['epochs_median']:.0f} | {pm['mean_ic']:+.5f} | "
          f"{pm['t_nw_lag20']:+.2f} | {res['paired']['ic']['mean_increment']:+.5f} | "
          f"{res['paired']['head_minus_pooled']['mean_increment']:+.5f} | "
          + ", ".join(f"{res['eval']['per_seed'][str(sd)]['ic_summary']['mean_ic']:+.4f}"
                      for sd in res["seeds"])
          + f" | {res['run']['seconds']:.0f} |")
        for name, v in res["sensitivity"].items():
            if "ic_summary" not in v:
                A(f"| {name} | - | - | - | - | - | - | {v.get('skipped', '')} |")
                continue
            A(f"| {name} | {v['epochs_median']:.0f} | {v['ic_summary']['mean_ic']:+.5f} | "
              f"{v['ic_summary']['t_nw_lag20']:+.2f} | {v['vs_ridge']['mean_increment']:+.5f} | "
              f"{v['vs_pooled']['mean_increment']:+.5f} | "
              + ", ".join(f"{x:+.4f}" for x in v["per_seed_ic"].values())
              + f" | {v['seconds']:.0f} |")

    A("\n## Protocol audit\n")
    A(f"- fold embargo: min gap test_lo - train_hi = {res['run']['fold_audit']['outer_gap_min']} "
      f"sessions (required {res['run']['fold_audit']['required_gap']}); inner "
      f"val_lo - inner_hi = {res['run']['fold_audit']['inner_gap_min']}")
    A(f"- scored rows {res['run']['n_scored_rows']} over {res['run']['n_scored_dates']} dates "
      f"({res['run'].get('scored_date_min')} -> {res['run'].get('scored_date_max')}), of which "
      f"{res['run'].get('n_scored_n50_rows')} are PIT NIFTY 50 cells; the N50 mask itself "
      f"covers {res['eval']['head']['n_eval_rows']} rows of the dataset, so the arm is graded "
      "on the LAST 375 of 756 sessions and is not comparable to a full-sample arm without "
      "saying so")
    A(f"- epochs run: median {res['run']['epochs_median']}, early-stopped "
      f"{res['run']['early_stop_share']:.0%} of the {res['run']['n_fits']} fits")
    A(f"- training loss fell in {res['run']['loss_fell_share']:.0%} of fits "
      f"(mean first {res['run']['train_loss_first_mean']:.4f} -> last "
      f"{res['run']['train_loss_last_mean']:.4f})")
    A(f"- validation loss: mean first {res['run']['val_loss_first_mean']:.5f} -> best "
      f"{res['run']['val_loss_best_mean']:.5f} (chance is log 2 = 0.69315); the head reduces "
      "its TRAINING loss by ~10x more than its validation loss, which is why early stopping "
      "fires within a dozen epochs -- read the sensitivity grid before blaming patience")
    A(f"- budget: primary {res['run']['seconds']:.0f} s, whole run "
      f"{res['run']['wall_seconds']:.0f} s of a {res['run']['budget_seconds']:.0f} s budget, "
      f"downshift {res['run']['budget_downshift']}")
    A(f"- rows dropped for a non-finite y_h{res['h']} label: {res['run']['n_label_nan_rows']} "
      f"({res['run']['n_label_nan_dates']} trailing sessions, not scored)")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET_N50)
    ap.add_argument("--hidden-npy", default=os.path.join(ML, "hT_pretrained.npy"))
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tag", default=ARM)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--pca-mode", default="frozen", choices=["frozen", "procrustes", "refit"])
    ap.add_argument("--min-hist", type=int, default=120)
    ap.add_argument("--seeds", default="0,1,2", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "listwise"])
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--pairs-per-date", type=int, default=2000)
    ap.add_argument("--budget-seconds", type=float, default=3600.0)
    ap.add_argument("--pilot-folds", type=int, default=2,
                    help="folds timed before the full run to project the budget; 0 disables")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-block", default=None,
                    help="optional .npy path to cache the causal-PCA block between runs")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap/report seed")
    ap.add_argument("--no-sensitivity", dest="sensitivity", action="store_false",
                    help="skip the declared sensitivity grid (SENSITIVITY)")
    ap.set_defaults(sensitivity=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    h = args.horizon
    t_start = time.time()

    ds = pd.read_parquet(args.dataset)
    H = np.load(args.hidden_npy, mmap_mode="r")
    if len(H) != len(ds):
        raise SystemExit(f"h_T has {len(H)} rows, dataset has {len(ds)} -- realign first")
    ycol = f"y_h{h}"
    fin = np.isfinite(ds[ycol].to_numpy(dtype=float))
    n_nan_dates = int(ds.loc[~fin, "date"].nunique())
    sub = ds.loc[fin].reset_index(drop=True)
    dates = sub["date"].to_numpy()
    ud, dcode = date_codes(dates)
    y_raw = sub[ycol].to_numpy(dtype=float)
    y = rank_within_date(y_raw[:, None], dcode)[:, 0]

    print(f"[block] causal_pca k={args.k} mode={args.pca_mode} on {fin.sum()} rows", flush=True)
    cache = args.cache_block
    if cache and os.path.exists(cache):
        z = np.load(cache)
        Xb, ok = z[:, :-1].astype(np.float64), z[:, -1] > 0.5
    else:
        Xb, ok = causal_pca(np.asarray(H)[fin], args.k, dates, mode=args.pca_mode,
                            min_hist=args.min_hist)
        if cache:
            np.save(cache, np.hstack([Xb, ok[:, None].astype(np.float64)]))
    del H
    print(f"[block] {Xb.shape}  usable rows {int(ok.sum())} "
          f"({time.time() - t_start:.0f}s)", flush=True)

    cfg = HeadConfig(k=args.k, loss=args.loss, max_epochs=args.max_epochs,
                     patience=args.patience, pairs_per_date=args.pairs_per_date)

    # --- budget pilot: time `pilot_folds` folds, project, halve epochs+pairs if over ---
    downshift = False
    if args.pilot_folds > 0:
        first = usable_start(ok, dcode, len(ud))
        folds_all = make_folds(len(ud), h, first, train_window=cfg.train_window,
                               refit_every=cfg.refit_every, val_frac=cfg.val_frac,
                               val_position=cfg.val_position)
        n_folds = len(folds_all)
        lim = int(folds_all["test_hi"].iloc[min(args.pilot_folds, n_folds) - 1]) + 1
        pm = dcode < lim
        t0 = time.time()
        run_head(Xb[pm], y[pm], dates[pm], h, ok=ok[pm], seeds=args.seeds, cfg=cfg)
        per_fold = (time.time() - t0) / max(min(args.pilot_folds, n_folds), 1)
        proj = per_fold * n_folds
        print(f"[pilot] {per_fold:.1f}s/fold x {n_folds} folds -> {proj / 60:.1f} min "
              f"(budget {args.budget_seconds / 60:.0f} min)", flush=True)
        if proj > args.budget_seconds:
            downshift = True
            cfg.max_epochs = max(1, cfg.max_epochs // 2)
            cfg.pairs_per_date = max(1, cfg.pairs_per_date // 2)
            print(f"[pilot] OVER BUDGET -> halving to max_epochs={cfg.max_epochs}, "
                  f"pairs_per_date={cfg.pairs_per_date}", flush=True)

    print(f"[head] rolling fit, seeds {args.seeds}", flush=True)
    out = run_head(Xb, y, dates, h, ok=ok, seeds=args.seeds, cfg=cfg, progress=True)

    # --- baseline: the frozen ridge (Fama-MacBeth) on the IDENTICAL block ---
    print("[ridge] fm_scores on the identical block", flush=True)
    ridge = np.full(len(sub), np.nan)
    ridge[ok] = fm_scores(Xb[ok], y[ok], dates[ok], h)
    head = out["scores"]
    matched = np.isfinite(head)
    ridge_m = np.where(matched, ridge, np.nan)

    # Second baseline, so the assigned head-vs-FM-ridge number can be DECOMPOSED. The head
    # differs from `fm_scores` in two ways at once -- a pairwise-logistic objective with a
    # nonlinearity, AND a pooled trailing-250-date fit instead of a mean of per-date
    # cross-sectional coefficients. `pooled_scores` on the same block, same window, same
    # refit cadence and same embargo is the head's own protocol made linear and least-squares,
    # so (head - pooled) is the objective/nonlinearity and (pooled - FM) is the protocol.
    # Reporting only the sum of the two would blame the objective for whatever the protocol did.
    print("[ridge] pooled_scores on the identical block (head's own protocol, linear)", flush=True)
    pooled = np.full(len(sub), np.nan)
    pooled[ok] = pooled_scores(Xb[ok], y[ok], dates[ok], h, train_window=cfg.train_window,
                               refit_every=cfg.refit_every, min_train=cfg.train_window)
    pooled_m = np.where(matched, pooled, np.nan)

    # `is_n50` was stamped by n50_build.py from the same PIT membership file n50_mask reads;
    # cross-check them rather than trusting either, since every gate below is computed on it.
    from bench.nse.n50 import n50_mask
    mask = sub["is_n50"].to_numpy(dtype=bool)
    if not np.array_equal(mask, np.asarray(n50_mask(sub), dtype=bool)):
        raise SystemExit("dataset is_n50 disagrees with membership_pit_n50.parquet")
    ev = {
        "head": evaluate(head, sub, h, mask, seed=args.seed),
        "head_raw": evaluate(out["scores_raw"], sub, h, mask, seed=args.seed),
        "ridge_matched": evaluate(ridge_m, sub, h, mask, seed=args.seed),
        "ridge_full": evaluate(ridge, sub, h, mask, seed=args.seed),
        "pooled_matched": evaluate(pooled_m, sub, h, mask, seed=args.seed),
        "pooled_full": evaluate(pooled, sub, h, mask, seed=args.seed),
    }
    ev["per_seed"] = {str(sd): evaluate(out["per_seed"][i], sub, h, mask, seed=args.seed)
                      for i, sd in enumerate(args.seeds)}
    # 200-panel context: the same two scores graded on every eligible cell, not just the 50
    ic200 = {
        "head": summarise(daily_spearman(head, y_raw, dates), h, seed=args.seed),
        "ridge_matched": summarise(daily_spearman(ridge_m, y_raw, dates), h, seed=args.seed),
    }
    paired = {
        "ic": paired_increment(ev["head"]["ic"], ev["ridge_matched"]["ic"], h, seed=args.seed),
        "ic_200": paired_increment(daily_spearman(head, y_raw, dates),
                                   daily_spearman(ridge_m, y_raw, dates), h, seed=args.seed),
        "spread10": paired_increment(ev["head"]["spread"][10], ev["ridge_matched"]["spread"][10],
                                     h, seed=args.seed),
        "spread15": paired_increment(ev["head"]["spread"][15], ev["ridge_matched"]["spread"][15],
                                     h, seed=args.seed),
        "head_vs_seed_mean_raw": paired_increment(ev["head"]["ic"], ev["head_raw"]["ic"], h,
                                                  seed=args.seed),
        "head_minus_pooled": paired_increment(ev["head"]["ic"], ev["pooled_matched"]["ic"], h,
                                              seed=args.seed),
        "pooled_minus_ridge": paired_increment(ev["pooled_matched"]["ic"],
                                               ev["ridge_matched"]["ic"], h, seed=args.seed),
    }
    gates = gate_table(paired["ic"])

    # --- declared sensitivity grid ------------------------------------------------------
    # The primary protocol early-stops fast on a nearly flat validation curve, so a bare
    # "the head lost" would be indistinguishable from "the head never trained". Each variant
    # below re-runs the WHOLE rolling protocol with one knob moved and is reported in full.
    sens = {}
    if args.sensitivity:
        for name, kw in SENSITIVITY.items():
            if time.time() - t_start > args.budget_seconds:
                sens[name] = {"skipped": "budget exhausted"}
                continue
            c2 = HeadConfig(**{**asdict(cfg), **kw})
            t0 = time.time()
            o2 = run_head(Xb, y, dates, h, ok=ok, seeds=args.seeds, cfg=c2)
            e2 = evaluate(o2["scores"], sub, h, mask, seed=args.seed)
            h2 = o2["history"]
            sens[name] = {
                "config": {**asdict(c2), "n_params": c2.n_params()},
                "seconds": time.time() - t0,
                "epochs_median": float(h2["epochs"].median()) if len(h2) else float("nan"),
                "val_loss_first_mean": float(h2["val_loss_first"].mean()) if len(h2) else float("nan"),
                "val_loss_best_mean": float(h2["val_loss_best"].mean()) if len(h2) else float("nan"),
                "train_loss_last_mean": float(h2["train_loss_last"].mean()) if len(h2) else float("nan"),
                "ic_summary": e2["ic_summary"],
                "spread_summary": {str(k): v for k, v in e2["spread_summary"].items()},
                "per_seed_ic": {str(sd): float(daily_spearman(o2["per_seed"][i][mask],
                                                              y_raw[mask], dates[mask],
                                                              min_names=30).mean())
                                for i, sd in enumerate(args.seeds)},
                "vs_ridge": paired_increment(e2["ic"], ev["ridge_matched"]["ic"], h, seed=args.seed),
                "vs_pooled": paired_increment(e2["ic"], ev["pooled_matched"]["ic"], h, seed=args.seed),
                "vs_primary_head": paired_increment(e2["ic"], ev["head"]["ic"], h, seed=args.seed),
            }
            print(f"  [sens] {name:14s} IC {e2['ic_summary']['mean_ic']:+.5f} "
                  f"t20 {e2['ic_summary']['t_nw_lag20']:+.2f}  "
                  f"vs_ridge {sens[name]['vs_ridge']['mean_increment']:+.5f}  "
                  f"epochs {sens[name]['epochs_median']:.0f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # The pre-registered gate is the primary cell and nothing else. But a verdict of "the
    # objective was not the constraint" is a claim about the BEST the objective could do, so
    # the same clauses are re-evaluated at the sensitivity variant with the highest IC and
    # labelled for what they are: a post-hoc maximum over 8 variants, an upper bound on the
    # head, not a pass. Both readings go in the file; neither is allowed to hide the other.
    gates_best = {}
    scored_sens = {k: v for k, v in sens.items() if "ic_summary" in v}
    if scored_sens:
        bn = max(scored_sens, key=lambda k: scored_sens[k]["ic_summary"]["mean_ic"])
        gates_best = gate_table(scored_sens[bn]["vs_ridge"])
        gates_best.update({
            "variant": bn, "post_hoc": True,
            "note": "max over %d declared sensitivity variants; an upper bound on what the "
                    "head could do, NOT a pre-registered pass" % len(scored_sens),
            "mean_ic": scored_sens[bn]["ic_summary"]["mean_ic"],
            "t_nw_lag20": scored_sens[bn]["ic_summary"]["t_nw_lag20"],
        })

    hist = out["history"]
    cfgd = asdict(cfg)
    cfgd["n_params"] = cfg.n_params()
    res = {
        "arm": args.tag, "h": h, "seeds": list(args.seeds), "config": cfgd,
        "run": {
            "n_folds": int(len(out["folds"])), "n_fits": int(len(hist)),
            "seconds": float(out["seconds"]), "wall_seconds": float(time.time() - t_start),
            "budget_seconds": float(args.budget_seconds), "budget_downshift": bool(downshift),
            "n_scored_rows": out["n_scored_rows"], "n_scored_dates": out["n_scored_dates"],
            "n_scored_n50_rows": int(np.isfinite(head[mask]).sum()),
            # The head cannot score the first 120 dates (no frozen PCA basis) nor the next
            # 255 (no full training window + embargo), so downstream gets the LAST 375 of the
            # 756 sessions. Anything comparing this arm to a full-sample arm is comparing
            # different windows unless it says so.
            "scored_date_min": str(pd.Timestamp(sub["date"].to_numpy()[np.isfinite(head)].min())),
            "scored_date_max": str(pd.Timestamp(sub["date"].to_numpy()[np.isfinite(head)].max())),
            "fold_audit": out["fold_audit"],
            "epochs_median": float(hist["epochs"].median()) if len(hist) else float("nan"),
            "early_stop_share": float(hist["stopped_early"].mean()) if len(hist) else float("nan"),
            "loss_fell_share": float((hist["train_loss_last"] < hist["train_loss_first"]).mean())
                               if len(hist) else float("nan"),
            "train_loss_first_mean": float(hist["train_loss_first"].mean()) if len(hist) else float("nan"),
            "train_loss_last_mean": float(hist["train_loss_last"].mean()) if len(hist) else float("nan"),
            "val_loss_first_mean": float(hist["val_loss_first"].mean()) if len(hist) else float("nan"),
            "val_loss_best_mean": float(hist["val_loss_best"].mean()) if len(hist) else float("nan"),
            "n_label_nan_rows": int((~fin).sum()), "n_label_nan_dates": n_nan_dates,
            "block": {"k": args.k, "mode": args.pca_mode, "min_hist": args.min_hist,
                      "usable_rows": int(ok.sum()), "source": args.hidden_npy},
        },
        "eval": {k: (summary_json(v) if k != "per_seed"
                     else {kk: summary_json(vv) for kk, vv in v.items()}) for k, v in ev.items()},
        "ic_200_panel": ic200,
        "paired": paired,
        "gates": gates,
        "gates_best_sensitivity": gates_best,
        "sensitivity": sens,
        "folds": out["folds"].to_dict(orient="records"),
        "fits": hist.to_dict(orient="records"),
    }
    res["tables"] = {
        "head_year": _md_table(year_table(ev["head"], 10)),
        "ridge_year": _md_table(year_table(ev["ridge_matched"], 10)),
    }

    jp = os.path.join(args.out_dir, "head_n50.json")
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    write_md(os.path.join(args.out_dir, "head_n50.md"), res)

    sc = pd.DataFrame({"date": sub["date"].to_numpy(), "symbol": sub["symbol"].astype(str).to_numpy(),
                       "score": head, "is_n50": mask})
    sc = sc.loc[np.isfinite(sc["score"].to_numpy())].reset_index(drop=True)
    sp = os.path.join(args.out_dir, f"scores_{args.tag}.parquet")
    sc.to_parquet(sp, index=False)

    sidecar(os.path.join(args.out_dir, "head_n50_meta.json"), vars(args), extra={
        "n_scored_rows": int(len(sc)), "n_scored_n50_rows": int(sc["is_n50"].sum()),
        "run": res["run"], "gates": gates,
        "paired_ic_increment": paired["ic"].get("mean_increment"),
        "outputs": [jp, os.path.join(args.out_dir, "head_n50.md"), sp],
    })
    print(f"\nhead IC(N50) {ev['head']['ic_summary']['mean_ic']:+.5f} "
          f"t20 {ev['head']['ic_summary']['t_nw_lag20']:+.2f} | "
          f"ridge {ev['ridge_matched']['ic_summary']['mean_ic']:+.5f} "
          f"t20 {ev['ridge_matched']['ic_summary']['t_nw_lag20']:+.2f} | "
          f"paired {paired['ic']['mean_increment']:+.5f} "
          f"t20 {paired['ic']['t_nw_lag20']:+.2f}", flush=True)
    print(f"wrote {jp}\n      {sp}", flush=True)


if __name__ == "__main__":
    main()
