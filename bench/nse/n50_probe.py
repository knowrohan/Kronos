"""Stage C of NIFTY50_PLAN.md: the pre-registered arm family, graded on the PIT NIFTY 50.

WHAT THIS IS. One estimator, many representations. Every arm is a feature block over the
same 150,527 PIT NIFTY 200 cells; every arm goes through the identical scorer
(``mlcore.fm_scores``, window 250, ridge 10, h-session embargo); every arm is graded on the
identical evaluation rows (``is_n50``). So an "increment" is a difference of two numbers made
by the same machinery, and the family null below is a max over blocks that saw the same
permuted labels.

THE DESIGN, restated because it is the thing readers get wrong.
  * TRAIN rows = all eligible PIT NIFTY 200 cells. A 50-name cross-section cannot support a
    per-date ridge with 22 + 32 regressors (NIFTY50_PLAN F3), so the head is learnt wide.
  * EVAL rows = ``is_n50``. That is the tradable book, and it is the only cross-section the
    money test can be stated on. The mask is applied to scores AFTER they exist; no fit ever
    sees the mask (except the deliberately under-determined ``hT_pre_cpca8_fit50only`` arm,
    which exists to show what the alternative costs).
  * The gate lag is 20, not h: a Fama-MacBeth score is a 250-date trailing mean of
    coefficients, so consecutive per-date ICs are nearly the same number (F4).

PRE-REGISTERED PRIMARY CELL: h = 5, ``fm_scores``, Kronos-base h_T, ``causal_pca(k=32,
mode="frozen")``, rank-within-date, trained on 200, evaluated on the 50. Everything else in
the core family is secondary and enters the family null (F6).

EXTENDING IT. Arms live in a registry: ``register_arm(name, builder, group=...)`` where the
builder takes a :class:`ProbeContext` and returns ``(block, valid_row_mask)``. The Stage B
representation ladder (A1-A6) is the ``extended`` group and is added by a separate agent with
``ctx.hidden(...)`` / :func:`cpca_arm` / :func:`concat_arm`; nothing in the estimator, the
null, the gates or the report needs to change to admit it. ``--arms core|extended|all`` or an
explicit comma list selects what runs.

Outputs (all under ``bench/out/nse/n50/``): ``probe_n50.json`` (every number),
``probe_n50.md`` (the same numbers as tables), ``probe_n50_null.npz`` (per-draw null arrays,
so a later stage can pool them), and ``scores_{arm}.parquet`` for the arms the backtest needs
(columns: date, symbol, score, is_n50 -- one row per cell that received a score).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import sidecar  # noqa: E402
from bench.nse.mlcore import (FREE_COLS, OUT_NSE, causal_pca,  # noqa: E402
                              daily_spearman, fm_scores, paired_increment,
                              pooled_scores, rank_within_date, refit_null_family,
                              summarise)
from bench.nse.n50 import MIN_NAMES, evaluate, summary_json, year_table  # noqa: E402
from bench.nse.probe_ht import CHECKPOINT, FINGERPRINT  # noqa: E402

ML = os.path.join(OUT_NSE, "ml")
N50 = os.path.join(OUT_NSE, "n50")

#: The pre-registered primary cell (NIFTY50_PLAN sec.4). Named once, used by the gates, the
#: nulls, the fingerprint, the contamination split and the conditional IC.
PRIMARY_ARM = "hT_pre_cpca32"
PRIMARY_H = 5
PRIMARY_SCORER = "fm"

#: Bands from the record (NIFTY50_PLAN sec.0): below +0.014 is nothing, +0.014..+0.025 is
#: "write-up, not trade", >= +0.025 is the honest t=3 bar on 200 names and the money bar.
BAND_WRITEUP = 0.014
BAND_TRADE = 0.025

SCORERS: dict[str, Callable] = {"fm": fm_scores, "pooled": pooled_scores}

#: arm -> scores_{tag}.parquet, the CONTRACT downstream stages read (date, symbol, score,
#: is_n50). Keyed by (arm, scorer) because the pooled-scorer primary is a different file.
SCORE_EXPORTS = {
    ("hT_pre_cpca32", "fm"): "A0_primary",
    ("free", "fm"): "free",
    ("hT_pre_cpca64", "fm"): "A0_cpca64",
    ("free+hT_pre_cpca32", "fm"): "A0_free_plus",
    ("hT_res_cpca32", "fm"): "A0_reservoir",
    ("hT_pre_cpca32", "pooled"): "A0_pooled",
}


# --------------------------------------------------------------------------- context


@dataclass
class ProbeContext:
    """Everything an arm builder may read, loaded once and cached.

    The heavy arrays (832-d hidden states, the 540-value raw windows) are loaded lazily and
    memoised, so selecting ``--arms free`` never touches a 250 MB file, and running the whole
    family never loads one twice.
    """

    ds: pd.DataFrame
    dates: np.ndarray
    dcode: np.ndarray
    is_n50: np.ndarray
    ml_dir: str = ML
    n50_dir: str = N50
    min_hist: int = 120
    refit_every: int = 60
    _cache: dict = field(default_factory=dict, repr=False)
    _built: dict = field(default_factory=dict, repr=False)

    # -- loaders ----------------------------------------------------------------

    @classmethod
    def load(cls, dataset: str | None = None, ml_dir: str = ML, n50_dir: str = N50):
        ds = pd.read_parquet(dataset or os.path.join(n50_dir, "dataset_n50.parquet"))
        ds["date"] = pd.to_datetime(ds["date"])
        # Row alignment with the cached h_T is a correctness precondition, not a hope: the
        # .npy files carry no keys, so a silently reordered dataset would pair every hidden
        # state with the wrong (date, symbol) and still produce a plausible IC.
        ref = pd.read_parquet(os.path.join(ml_dir, "dataset.parquet"), columns=["date", "symbol"])
        ref["date"] = pd.to_datetime(ref["date"])
        if len(ref) != len(ds) or not (ref["date"].to_numpy() == ds["date"].to_numpy()).all() \
                or not (ref["symbol"].to_numpy() == ds["symbol"].to_numpy()).all():
            raise SystemExit("dataset_n50.parquet is not row-aligned with ml/dataset.parquet; "
                             "the cached h_T arrays cannot be used positionally")
        return cls(ds=ds, dates=ds["date"].to_numpy(),
                   dcode=pd.factorize(ds["date"], sort=True)[0],
                   is_n50=ds["is_n50"].to_numpy(dtype=bool),
                   ml_dir=ml_dir, n50_dir=n50_dir)

    @property
    def n_rows(self) -> int:
        return len(self.ds)

    def free(self) -> np.ndarray:
        """The 22 free OHLCV features, NaN-filled at the column median, ranked within date."""
        if "free" not in self._cache:
            A = np.array(self.ds[FREE_COLS].to_numpy(dtype=np.float64), copy=True)
            for j in range(A.shape[1]):
                c = A[:, j]
                b = ~np.isfinite(c)
                if b.any():
                    c[b] = np.nanmedian(c[~b])
            self._cache["free"] = rank_within_date(A, self.dcode)
        return self._cache["free"]

    def hidden(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """A cached hidden-state matrix as (float32 rows x d, valid_row_mask).

        ``key`` is either ``ml:<name>`` (a full-panel array in ``bench/out/nse/ml``) or
        ``n50:<name>`` (a Stage B array in ``bench/out/nse/n50``). An array shorter than the
        panel is scattered back to full length through its ``rows_sidecar`` parquet and the
        rows it does not cover are marked invalid -- that is how the N50-cell-only reservoir
        seeds (A6) and any other subset extraction join the family without special-casing.
        """
        if key in self._cache:
            return self._cache[key]
        where, _, name = key.partition(":")
        base = {"ml": self.ml_dir, "n50": self.n50_dir}.get(where)
        if base is None:
            raise ValueError(f"hidden(): key must start with 'ml:' or 'n50:', got {key!r}")
        path = os.path.join(base, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        M = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if len(M) == self.n_rows:
            out = (M, np.ones(self.n_rows, dtype=bool))
        else:
            meta_p = os.path.join(base, f"{name}_meta.json")
            meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
            side = meta.get("rows_sidecar", "rows_n50.parquet")
            rows = pd.read_parquet(os.path.join(base, side))["row"].to_numpy(dtype=int)
            if len(rows) != len(M):
                raise SystemExit(f"{name}: {len(M)} rows but sidecar {side} has {len(rows)}")
            full = np.zeros((self.n_rows, M.shape[1]), dtype=np.float32)
            valid = np.zeros(self.n_rows, dtype=bool)
            full[rows] = M
            valid[rows] = True
            out = (full, valid)
        self._cache[key] = out
        return out

    def raw_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """The raw 90x6 normalised windows flattened to 540 columns (the null-representation
        control: same estimator, no learned compression at all)."""
        if "raw540" not in self._cache:
            W = np.asarray(np.load(os.path.join(self.ml_dir, "windows.npy"), mmap_mode="r"),
                           dtype=np.float32)
            self._cache["raw540"] = (W.reshape(len(W), -1), np.ones(self.n_rows, dtype=bool))
        return self._cache["raw540"]

    def cpca(self, source: str, k: int, mode: str = "frozen",
             rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Causal PCA-k of a cached matrix, memoised on (source, k, mode, row-subset).

        ``rows`` restricts BOTH the basis fit and the within-date ranking to a row subset --
        that is what "fitted on the 50 only" has to mean if it is to mean anything, since a
        block ranked over 200 names and then fitted on 50 of them is a third thing that is
        neither arm.
        """
        tag = f"cpca:{source}:{k}:{mode}:{'sub' if rows is not None else 'all'}"
        if tag in self._cache:
            return self._cache[tag]
        M, valid = self.raw_windows() if source == "raw540" else self.hidden(source)
        sel = valid if rows is None else (valid & np.asarray(rows, dtype=bool))
        blk = np.zeros((self.n_rows, k), dtype=np.float64)
        ok = np.zeros(self.n_rows, dtype=bool)
        sub, sub_ok = causal_pca(M[sel], k, self.dates[sel], mode=mode,
                                 refit_every=self.refit_every, min_hist=self.min_hist)
        idx = np.flatnonzero(sel)
        blk[idx] = sub
        ok[idx[sub_ok]] = True
        self._cache[tag] = (blk, ok)
        return self._cache[tag]

    def release_sources(self) -> None:
        """Drop the raw 832-d / 540-d source matrices once every arm has been projected.

        Two 500 MB float32 hidden states plus the 325 MB window tensor are dead weight during
        a 300-draw null that only ever touches the already-built PCA blocks; the blocks
        themselves are memoised in ``_BUILT``, so nothing is recomputed.
        """
        for k in [k for k in self._cache
                  if k.startswith(("ml:", "n50:")) or k == "raw540"]:
            del self._cache[k]

    def labels(self, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(y_raw, y_rank, finite_mask) at horizon h. y_rank is ranked within date over the
        whole PIT 200 cross-section, once, so every arm and every null draw regresses on the
        identical label vector (a within-date monotone restriction to a subset preserves the
        ordering, so the fit-on-50 arm is not disadvantaged by sharing it)."""
        key = f"y{h}"
        if key not in self._cache:
            y_raw = self.ds[f"y_h{h}"].to_numpy(dtype=float)
            fin = np.isfinite(y_raw)
            y = np.full(len(y_raw), np.nan)
            y[fin] = rank_within_date(y_raw[fin, None], self.dcode[fin])[:, 0]
            self._cache[key] = (y_raw, y, fin)
        return self._cache[key]


# -------------------------------------------------------------------------- registry


@dataclass
class ArmSpec:
    name: str
    group: str
    build: Callable[[ProbeContext], tuple[np.ndarray, np.ndarray]]
    doc: str = ""


ARMS: dict[str, ArmSpec] = {}


def register_arm(name: str, build, group: str = "core", doc: str = "") -> None:
    """Add an arm. Registration order IS the pre-registration order the family null uses."""
    if name in ARMS:
        raise ValueError(f"arm {name!r} already registered")
    ARMS[name] = ArmSpec(name=name, group=group, build=build, doc=doc)


def cpca_arm(source: str, k: int, mode: str = "frozen", rows_key: str | None = None):
    """Builder factory: causal-PCA-k of a cached matrix. ``rows_key='is_n50'`` fits on the 50."""
    def build(ctx: ProbeContext):
        rows = ctx.is_n50 if rows_key == "is_n50" else None
        return ctx.cpca(source, k, mode=mode, rows=rows)
    return build


def concat_arm(*parts):
    """Builder factory: horizontal concatenation of other arms; valid = AND of their masks."""
    def build(ctx: ProbeContext):
        blocks, ok = [], np.ones(ctx.n_rows, dtype=bool)
        for p in parts:
            b, v = build_arm(p, ctx) if isinstance(p, str) else p(ctx)
            blocks.append(b)
            ok &= v
        return np.hstack(blocks), ok
    return build


def _free_build(ctx: ProbeContext):
    return ctx.free(), np.ones(ctx.n_rows, dtype=bool)


def build_arm(name: str, ctx: ProbeContext) -> tuple[np.ndarray, np.ndarray]:
    """Materialise one arm, memoised ON THE CONTEXT.

    The memo lives on `ctx`, not in a module global keyed by id(ctx): a global keyed by an
    object address hands a stale block to a second context that happens to be allocated where
    the first one was freed, which is the kind of bug that produces a plausible wrong number
    rather than a crash.
    """
    if name not in ctx._built:
        if name not in ARMS:
            raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
        ctx._built[name] = ARMS[name].build(ctx)
    return ctx._built[name]


def arm_names(selector: str) -> list[str]:
    """``all``, any registered group (``core``/``extended``/``control``), or a comma list.

    The six depth-matched reservoir controls are their own ``control`` group ON PURPOSE:
    ``--arms extended`` has to keep returning the fourteen arms the published
    ``probe_n50_extended.md`` family null maxed over, or a family p-value already on disk
    stops being reproducible from this file. ``--arms all`` includes them.
    """
    sel = selector.strip()
    if sel == "all":
        return list(ARMS)
    if sel in {a.group for a in ARMS.values()}:
        return [n for n, a in ARMS.items() if a.group == sel]
    want = [s.strip() for s in sel.split(",") if s.strip()]
    bad = [w for w in want if w not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s): {', '.join(bad)}\nknown: {', '.join(ARMS)}")
    return want


# ------------------------------------------------------- the pre-registered core family

register_arm("free", _free_build, doc="22 free OHLCV features (the block h_T must beat)")
register_arm("hT_pre_cpca8", cpca_arm("ml:hT_pretrained", 8), doc="Kronos-base h_T, causal PCA-8")
register_arm("hT_pre_cpca32", cpca_arm("ml:hT_pretrained", 32),
             doc="PRIMARY CELL: Kronos-base h_T, causal PCA-32, frozen basis")
register_arm("hT_pre_cpca64", cpca_arm("ml:hT_pretrained", 64), doc="Kronos-base h_T, causal PCA-64")
register_arm("hT_pre_cpca32_procrustes", cpca_arm("ml:hT_pretrained", 32, mode="procrustes"),
             doc="PCA-32 with Procrustes-aligned 60-date refits (F2 variant b)")
register_arm("hT_res_cpca8", cpca_arm("ml:hT_reservoir", 8), doc="random-weight control, PCA-8")
register_arm("hT_res_cpca32", cpca_arm("ml:hT_reservoir", 32), doc="random-weight control, PCA-32")
register_arm("hT_res_cpca64", cpca_arm("ml:hT_reservoir", 64), doc="random-weight control, PCA-64")
register_arm("raw540_cpca32", cpca_arm("raw540", 32), doc="raw 90x6 window, PCA-32 (no model)")
register_arm("free+hT_pre_cpca32", concat_arm("free", "hT_pre_cpca32"),
             doc="does h_T add anything to the free block?")
register_arm("free+hT_res_cpca32", concat_arm("free", "hT_res_cpca32"),
             doc="the same question asked of random weights")
register_arm("hT_pre_cpca8_fit50only", cpca_arm("ml:hT_pretrained", 8, rows_key="is_n50"),
             doc="UNDER-DETERMINED: basis, ranking and fit all on the ~50 names only")


# ------------------------------------------------------------------------ the estimator


def score_arm(ctx: ProbeContext, name: str, h: int, scorer: str = "fm") -> tuple[np.ndarray, dict]:
    """Out-of-sample score for every row the arm can score, at horizon h.

    The fit sees ``valid & finite(label)`` rows -- the whole PIT 200 cross-section for every
    core arm but the fit-on-50 control. Rows outside that set stay NaN and are graded by
    nobody.
    """
    X, valid = build_arm(name, ctx)
    y_raw, y, fin = ctx.labels(h)
    ok = valid & fin
    out = np.full(ctx.n_rows, np.nan)
    t0 = time.time()
    out[ok] = SCORERS[scorer](X[ok], y[ok], ctx.dates[ok], h)
    return out, {"arm": name, "scorer": scorer, "h": int(h), "n_features": int(X.shape[1]),
                 "n_fit_rows": int(ok.sum()),
                 "n_fit_n50_rows": int((ok & ctx.is_n50).sum()),
                 "n_scored_rows": int(np.isfinite(out).sum()),
                 "fit_seconds": round(time.time() - t0, 2)}


def full_panel_ic(scores: np.ndarray, ctx: ProbeContext, h: int) -> dict:
    """IC on the whole PIT 200 panel -- reference only. Its cross-section is 4x the book's, so
    its SE is ~half; it is not comparable to the N50 numbers and never enters a gate."""
    y_raw, _, _ = ctx.labels(h)
    ic = daily_spearman(scores, y_raw, ctx.dates, min_names=20)
    return summarise(ic, h)


def run_arms(ctx: ProbeContext, names, horizons, scorer: str = "fm", seed: int = 0,
             verbose: bool = True) -> dict:
    """Fit and grade every (arm, horizon) pair. Returns per-arm evaluation dicts + scores.

    ``res[arm][h]`` is :func:`bench.nse.n50.evaluate` output (IC with the lag ladder and the
    block-20 CI, top-10/15 spread, by-year) plus the full-200 reference IC; ``scores[arm][h]``
    is the row-aligned score vector, which the exports and the fingerprint reuse.
    """
    res: dict = {}
    scores: dict = {}
    for name in names:
        res[name], scores[name] = {}, {}
        for h in horizons:
            sc, info = score_arm(ctx, name, h, scorer=scorer)
            ev = evaluate(sc, ctx.ds, h, mask=ctx.is_n50, min_names=MIN_NAMES, seed=seed)
            ev["fit"] = info
            ev["full200_ic_summary"] = full_panel_ic(sc, ctx, h)
            res[name][h] = ev
            scores[name][h] = sc
            if verbose:
                s = ev["ic_summary"]
                print(f"  [{scorer}] h={h:<2d} {name:26s} IC {s['mean_ic']:+.4f}  "
                      f"t20 {s['t_nw_lag20']:+.2f}  CI20 [{s['block_ci95_b20'][0]:+.4f},"
                      f"{s['block_ci95_b20'][1]:+.4f}]  n={s['n_dates']}  "
                      f"({info['fit_seconds']}s)", flush=True)
    return {"eval": res, "scores": scores}


# -------------------------------------------------------------------------- increments


def increment_table(res: dict, pairs, h: int, seed: int = 0) -> dict:
    """Paired per-date A - B for the pre-registered contrasts, at the full lag ladder.

    A paired difference is the estimand every gate is stated on because the estimator's
    attenuation cancels in it (probe_ht's opening argument) -- and because the two arms are
    graded on the same dates and the same names, so their common cross-sectional noise
    cancels too.
    """
    out = {}
    for a, b in pairs:
        if a not in res or b not in res or h not in res[a] or h not in res[b]:
            continue
        out[f"{a} - {b}"] = paired_increment(res[a][h]["ic"], res[b][h]["ic"], h, seed=seed)
    return out


CORE_PAIRS = [
    ("hT_pre_cpca8", "hT_res_cpca8"),
    ("hT_pre_cpca32", "hT_res_cpca32"),
    ("hT_pre_cpca64", "hT_res_cpca64"),
    ("free+hT_pre_cpca32", "free"),
    ("free+hT_pre_cpca32", "free+hT_res_cpca32"),
    ("hT_pre_cpca32", "hT_pre_cpca32_procrustes"),
]


# -------------------------------------------------------------------------------- nulls


def family_null(ctx: ProbeContext, names, h: int, n_draws: int, scorer: str = "fm",
                block: int = 20, seed: int = 11, progress: int | None = None,
                min_names: int = MIN_NAMES) -> dict:
    """Date-block permutation null over the WHOLE family, one permutation per draw.

    Sharing the permutation across arms is what makes ``max_abs`` a family-wise null: it
    prices the search that produced the headline cell (F6). The scorer is the same callable
    that produced the observed statistic and the IC is computed on the same ``is_n50`` rows at
    the same ``min_names`` -- mixing either would make the sigma meaningless (F1).
    """
    blocks = {n: build_arm(n, ctx) for n in names}
    _, y, fin = ctx.labels(h)
    yv = np.where(fin, y, np.nan)
    return refit_null_family(blocks, yv, ctx.dates, h, SCORERS[scorer], n_draws=n_draws,
                             block=block, seed=seed, eval_mask=ctx.is_n50,
                             min_names=min_names, progress=progress)


def null_pvalues(draws: np.ndarray, observed: float) -> dict:
    """(1 + #{null >= obs}) / (1 + n), one- and two-sided. The +1 is Davison-Hinkley: a
    permutation p-value of exactly 0 is not attainable and claiming it is overstates."""
    a = np.asarray(draws, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return {"n_draws": 0}
    return {
        "n_draws": int(n), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
        "p95_abs": float(np.percentile(np.abs(a), 95)),
        "p99_abs": float(np.percentile(np.abs(a), 99)),
        "observed": float(observed),
        "z": float((observed - a.mean()) / a.std(ddof=1)) if a.std(ddof=1) > 0 else float("nan"),
        "p_one_sided": float((1 + int((a >= observed).sum())) / (n + 1)),
        "p_two_sided": float((1 + int((np.abs(a) >= abs(observed)).sum())) / (n + 1)),
    }


# ------------------------------------------------------------------- power, empirically


def recovery_curve(ctx: ProbeContext, h: int, targets=(0.01, 0.02, 0.04), seed: int = 0,
                   scorer: str = "fm", min_names: int = MIN_NAMES) -> list[dict]:
    """Plant a signal of KNOWN solo IC in the free block; report what the test registers.

    Without this a small increment is uninterpretable -- it could be a weak representation or
    a strong one seen through a lossy estimator on a 50-name cross-section. Everything is
    measured on the ``is_n50`` rows, because that is where the result has to be read.
    """
    y_raw, y, fin = ctx.labels(h)
    free = ctx.free()
    base_sc = SCORERS[scorer](free[fin], y[fin], ctx.dates[fin], h)
    base = np.full(ctx.n_rows, np.nan)
    base[fin] = base_sc
    base_ic = float(evaluate(base, ctx.ds, h, mask=ctx.is_n50, min_names=min_names,
                             seed=seed)["ic_summary"]["mean_ic"])
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        pr = target * y + np.sqrt(max(1e-9, 1 - target ** 2)) * rng.normal(size=ctx.n_rows)
        pl = np.full(ctx.n_rows, np.nan)
        pl[fin] = rank_within_date(pr[fin, None], ctx.dcode[fin])[:, 0]
        solo = np.full(ctx.n_rows, np.nan)
        solo[fin] = pl[fin]
        solo_n50 = float(evaluate(solo, ctx.ds, h, mask=ctx.is_n50, min_names=min_names,
                                  seed=seed)["ic_summary"]["mean_ic"])
        solo_200 = float(summarise(daily_spearman(solo, y_raw, ctx.dates, min_names=20),
                                   h)["mean_ic"])
        X = np.hstack([free, np.nan_to_num(pl)[:, None]])
        sc = np.full(ctx.n_rows, np.nan)
        sc[fin] = SCORERS[scorer](X[fin], y[fin], ctx.dates[fin], h)
        got = float(evaluate(sc, ctx.ds, h, mask=ctx.is_n50, min_names=min_names,
                             seed=seed)["ic_summary"]["mean_ic"])
        rows.append({"target": float(target), "planted_solo_ic_n50": solo_n50,
                     "planted_solo_ic_200": solo_200, "base_free_ic_n50": base_ic,
                     "combined_ic_n50": got, "increment": got - base_ic,
                     "recovery_pct": (100.0 * (got - base_ic) / solo_n50
                                      if abs(solo_n50) > 1e-4 else float("nan"))})
    return rows


# ------------------------------------------------- fingerprint / contamination / cond IC


def fingerprint(ctx: ProbeContext, scores: np.ndarray, h: int,
                cols=FINGERPRINT, min_names: int = 20) -> dict:
    """Per-date Spearman of the fitted score against each free factor, on the N50 rows.

    If the score is a repackaged reversal signal this is where it shows: M3 fails the arm at
    max |rho| >= 0.3.
    """
    m0 = ctx.is_n50 & np.isfinite(scores)
    d = ctx.dates
    out = {}
    for c in cols:
        v = ctx.ds[c].to_numpy(dtype=float)
        m = m0 & np.isfinite(v)
        rhos = []
        for dd in np.unique(d[m]):
            sel = m & (d == dd)
            if sel.sum() < min_names:
                continue
            rhos.append(sps.spearmanr(scores[sel], v[sel]).statistic)
        out[c] = float(np.nanmean(rhos)) if rhos else float("nan")
    return out


def contamination_split(ic: pd.Series, h: int, checkpoint=CHECKPOINT, seed: int = 0) -> dict:
    """IC before and after the checkpoint. If the edge lives only in the pre-checkpoint half,
    the trunk is remembering rather than forecasting."""
    idx = pd.DatetimeIndex(ic.index)
    pre, post = ic[idx < checkpoint], ic[idx >= checkpoint]
    return {"checkpoint": str(pd.Timestamp(checkpoint).date()),
            "pre": summarise(pre, h, seed=seed), "post": summarise(post, h, seed=seed),
            "post_ge_pre": bool(len(pre) and len(post)
                                and float(post.mean()) >= float(pre.mean()))}


def conditional_ic(ctx: ProbeContext, score: np.ndarray, control: np.ndarray, h: int,
                   n_bucket: int = 5, min_bucket: int = 8, seed: int = 0) -> dict:
    """IC of the score WITHIN quintiles of the free-block score, on the N50 rows.

    Conditioning rather than residualising, for the reason ``vol_signals.conditional_ic``
    gives: a residual can exceed the raw IC when the score is a nonlinear function of the
    control, which is not something real incremental information does. TRIPWIRE: a conditioned
    IC materially ABOVE the raw IC means the raw number is being suppressed by the control's
    own effect, and both numbers need explaining before either is traded.
    """
    y_raw, _, _ = ctx.labels(h)
    m0 = ctx.is_n50 & np.isfinite(score) & np.isfinite(control) & np.isfinite(y_raw)
    per_date, per_q = [], {q: [] for q in range(n_bucket)}
    for dd in np.unique(ctx.dates[m0]):
        sel = np.flatnonzero(m0 & (ctx.dates == dd))
        if len(sel) < min_bucket * n_bucket:
            continue
        cr = sps.rankdata(control[sel]) / len(sel)
        vals = []
        for q in range(n_bucket):
            b = sel[(cr > q / n_bucket) & (cr <= (q + 1) / n_bucket)]
            if len(b) < min_bucket:
                continue
            r = sps.spearmanr(score[b], y_raw[b]).statistic
            if np.isfinite(r):
                vals.append(r)
                per_q[q].append(r)
        if vals:
            per_date.append(np.mean(vals))
    s = pd.Series(per_date, dtype=float)
    return {"n_dates": int(len(s)), "n_bucket": n_bucket, "min_bucket": min_bucket,
            "summary": summarise(s, h, seed=seed) if len(s) else {},
            "per_quintile_mean": {str(q): (float(np.mean(v)) if v else float("nan"))
                                  for q, v in per_q.items()}}


# ---------------------------------------------------------------------------- exports


def write_scores(path: str, ctx: ProbeContext, scores: np.ndarray, args: dict,
                 extra: dict) -> dict:
    """Write the downstream CONTRACT: date, symbol, score, is_n50; one row per SCORED cell."""
    ok = np.isfinite(scores)
    df = pd.DataFrame({"date": ctx.dates[ok], "symbol": ctx.ds["symbol"].to_numpy()[ok],
                       "score": scores[ok].astype(np.float64),
                       "is_n50": ctx.is_n50[ok]})
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    df.to_parquet(path, index=False)
    info = {"path": path, "n_rows": int(len(df)), "n_n50_rows": int(df["is_n50"].sum()),
            "n_dates": int(df["date"].nunique()),
            "date_min": str(pd.Timestamp(df["date"].min()).date()),
            "date_max": str(pd.Timestamp(df["date"].max()).date())}
    sidecar(path.replace(".parquet", "_meta.json"), args, extra={**extra, **info})
    return info


# ------------------------------------------------------------------------------ gates


def m1_gates(res: dict, incs: dict, null: dict, h: int, primary: str = PRIMARY_ARM,
             reservoir: str = "hT_res_cpca32") -> dict:
    """NIFTY50_PLAN sec.6.3 M1, clause by clause, each with the number that decided it."""
    ev = res[primary][h]
    key = f"{primary} - {reservoir}"
    inc = incs.get(key, {})
    t20 = inc.get("t_nw_lag20", float("nan"))
    ci20 = inc.get("block_ci95_b20", [float("nan"), float("nan")])
    a_pass = bool(np.isfinite(t20) and t20 >= 3.0 and np.isfinite(ci20[0]) and ci20[0] > 0)

    fam_p = null.get("family", {}).get("p_two_sided", float("nan"))
    b_pass = bool(np.isfinite(fam_p) and fam_p < 0.05)

    sp = ev["spread_summary"][10]
    sp_ci = sp["block_ci95_b20"]
    sp_t = sp["t_nw_lag20"]
    c_ci = bool(sp["mean_ic"] > 0 and np.isfinite(sp_ci[0]) and sp_ci[0] > 0)
    c_t = bool(sp["mean_ic"] > 0 and np.isfinite(sp_t) and sp_t >= 2.0)
    c_pass = bool(c_ci or c_t)

    ic = ev["ic_summary"]["mean_ic"]
    return {
        "h": h, "primary_arm": primary, "reservoir_arm": reservoir,
        "a_paired_pretrained_minus_reservoir": {
            "pass": a_pass, "contrast": key,
            "mean_increment": inc.get("mean_increment", float("nan")),
            "t_nw_lag20": t20, "block_ci95_b20": ci20,
            "clause": "lag-20 NW t >= 3 AND block-20 CI lower bound > 0"},
        "b_family_null_p": {
            "pass": b_pass, "p_two_sided": fam_p,
            "observed_ic": null.get("family", {}).get("observed", float("nan")),
            "clause": "family max|IC| p < 0.05"},
        "c_top10_spread": {
            "pass": c_pass, "mean": sp["mean_ic"], "block_ci95_b20": sp_ci,
            "t_nw_lag20": sp_t, "via_ci": c_ci, "via_t": c_t,
            "clause": "mean > 0 AND (block-20 CI excludes 0 OR lag-20 t >= 2)"},
        "primary_ic": ic,
        "bands": {"writeup_floor": BAND_WRITEUP, "trade_bar": BAND_TRADE,
                  "where": ("below +0.014 (nothing)" if ic < BAND_WRITEUP else
                            "in [+0.014, +0.025): write-up, not trade" if ic < BAND_TRADE
                            else "at or above +0.025: the money bar")},
        "power_bar_t3_nw_lag20": ev["power"].get("bar_t3_nw_lag20", float("nan")),
        "power_bar_t3_realised": ev.get("power_realised", {}).get("bar_t3_nw_lag20", float("nan")),
        "ic_meets_t3_bar": bool(ic >= ev["power"].get("bar_t3_nw_lag20", np.inf)),
        "verdict_M1": bool(a_pass and b_pass and c_pass),
    }


# ----------------------------------------------------------------------------- report


def _f(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:+.{nd}f}"


def _arm_table(res: dict, h: int, names) -> str:
    head = ("| arm | n_dates | IC | t(h) | t(12) | t(20) | block-20 CI | share+ | "
            "top10 | t20(top10) | top15 | full-200 IC | t20(200) |\n"
            "|---|--:|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|\n")
    rows = []
    for n in names:
        e = (res.get(n) or {}).get(h, (res.get(n) or {}).get(str(h)))
        if e is None:
            continue
        sp = e["spread_summary"]          # summary_json stringifies the k keys
        s, s10, s15 = e["ic_summary"], sp.get(10, sp.get("10", {})), sp.get(15, sp.get("15", {}))
        f2 = e["full200_ic_summary"]
        rows.append(
            f"| `{n}` | {s['n_dates']} | {_f(s['mean_ic'])} | {_f(s[f't_nw_lag{h}'], 2)} | "
            f"{_f(s['t_nw_lag12'], 2)} | {_f(s['t_nw_lag20'], 2)} | "
            f"[{_f(s['block_ci95_b20'][0])}, {_f(s['block_ci95_b20'][1])}] | "
            f"{s['share_positive']:.2f} | {_f(s10['mean_ic'])} | {_f(s10['t_nw_lag20'], 2)} | "
            f"{_f(s15['mean_ic'])} | {_f(f2['mean_ic'])} | {_f(f2['t_nw_lag20'], 2)} |")
    return head + "\n".join(rows) + "\n"


def _inc_table(incs: dict, h: int) -> str:
    head = ("| contrast | n_dates | mean | t(h) | t(12) | t(20) | block-20 CI | share+ |\n"
            "|---|--:|--:|--:|--:|--:|:--|--:|\n")
    rows = []
    for k, v in incs.items():
        if not v.get("n_dates"):
            continue
        rows.append(
            f"| `{k}` | {v['n_dates']} | {_f(v['mean_increment'])} | "
            f"{_f(v[f't_nw_lag{h}'], 2)} | {_f(v['t_nw_lag12'], 2)} | {_f(v['t_nw_lag20'], 2)} | "
            f"[{_f(v['block_ci95_b20'][0])}, {_f(v['block_ci95_b20'][1])}] | "
            f"{v['share_positive']:.2f} |")
    return head + "\n".join(rows) + "\n"


def write_report(path: str, out: dict, names, horizons) -> None:
    """The same numbers as probe_n50.json, as tables a human can read without a JSON viewer."""
    L = ["# NIFTY 50 probe (Stage C) -- pre-registered core family", "",
         f"Written {out['meta']['written_at']}  |  arms: `{out['meta']['arms_selector']}`  |  "
         f"seed {out['meta']['seed']}  |  null draws {out['meta']['null_draws']}", "",
         f"Primary cell: **h={PRIMARY_H}, `{PRIMARY_ARM}`, scorer `{PRIMARY_SCORER}`**, "
         f"trained on {out['meta']['n_rows']} PIT NIFTY 200 cells, graded on "
         f"{out['meta']['n_n50_rows']} `is_n50` cells.", ""]

    g = out["gates"]
    L += ["## M1 verdict: **" + ("PASS" if g["verdict_M1"] else "FAIL") + "**", "",
          "| clause | verdict | deciding number |", "|---|:--|:--|"]
    a, b, c = (g["a_paired_pretrained_minus_reservoir"], g["b_family_null_p"],
               g["c_top10_spread"])
    L += [f"| (a) {a['clause']} | {'PASS' if a['pass'] else 'FAIL'} | "
          f"`{a['contrast']}` = {_f(a['mean_increment'])}, t20 = {_f(a['t_nw_lag20'], 2)}, "
          f"CI20 = [{_f(a['block_ci95_b20'][0])}, {_f(a['block_ci95_b20'][1])}] |",
          f"| (b) {b['clause']} | {'PASS' if b['pass'] else 'FAIL'} | "
          f"family p = {b['p_two_sided']:.4f} at observed IC {_f(b['observed_ic'])} |",
          f"| (c) {c['clause']} | {'PASS' if c['pass'] else 'FAIL'} | "
          f"top-10 spread = {_f(c['mean'], 5)}, CI20 = [{_f(c['block_ci95_b20'][0], 5)}, "
          f"{_f(c['block_ci95_b20'][1], 5)}], t20 = {_f(c['t_nw_lag20'], 2)} "
          f"(via CI: {c['via_ci']}, via t: {c['via_t']}) |", ""]
    L += [f"Primary IC {_f(g['primary_ic'])} -- {g['bands']['where']} "
          f"(bands: write-up floor +{BAND_WRITEUP:.3f}, trade bar +{BAND_TRADE:.3f}). "
          f"The N=50 t=3 bar from `power_statement` is "
          f"{_f(g['power_bar_t3_nw_lag20'])} at the assumed AR inflation and "
          f"{_f(g['power_bar_t3_realised'])} at the realised one; the measured IC "
          f"{'clears' if g['ic_meets_t3_bar'] else 'does NOT clear'} the assumed bar.", ""]

    for hh, gs in (out.get("gates_secondary") or {}).items():
        a2, c2 = gs["a_paired_pretrained_minus_reservoir"], gs["c_top10_spread"]
        L += [f"Secondary horizon h = {hh} (information only, not a decision): clause (a) "
              f"{'PASS' if a2['pass'] else 'FAIL'} at increment {_f(a2['mean_increment'])}, "
              f"t20 {_f(a2['t_nw_lag20'], 2)}, CI20 [{_f(a2['block_ci95_b20'][0])}, "
              f"{_f(a2['block_ci95_b20'][1])}]; clause (c) "
              f"{'PASS' if c2['pass'] else 'FAIL'} at top-10 spread {_f(c2['mean'], 5)}, "
              f"t20 {_f(c2['t_nw_lag20'], 2)}; clause (b) {gs['b_family_null_p']['note']}.", ""]

    for h in horizons:
        L += [f"## h = {h}", "", "### Arms (graded on the 50; full-200 IC for reference only)",
              "", _arm_table(out["arms"], h, names), "",
              "### Paired increments (lag ladder)", "", _inc_table(out["increments"][str(h)], h)]
        pooled = out.get("pooled", {}).get(str(h))
        if pooled:
            L += ["", f"Secondary scorer (`pooled_scores`) on the primary block: IC "
                  f"{_f(pooled['ic_summary']['mean_ic'])}, t20 "
                  f"{_f(pooled['ic_summary']['t_nw_lag20'], 2)}, block-20 CI "
                  f"[{_f(pooled['ic_summary']['block_ci95_b20'][0])}, "
                  f"{_f(pooled['ic_summary']['block_ci95_b20'][1])}], "
                  f"{pooled['ic_summary']['n_dates']} dates.", ""]
        yt = out["by_year"].get(str(h))
        if yt:
            L += ["", "### By calendar year (primary cell)", "",
                  "| year | n_dates | IC | t20 | top-10 spread | t20 |", "|---|--:|--:|--:|--:|--:|"]
            for r in yt:
                L += [f"| {r['year']} | {r['n_dates']} | {_f(r['ic_mean'])} | "
                      f"{_f(r['ic_t_nw20'], 2)} | {_f(r['spread10_mean'], 5)} | "
                      f"{_f(r['spread10_t_nw20'], 2)} |"]
            L += [""]

    n = out["nulls"]
    if not n:
        L += ["## Nulls", "", "Not run (`--null-draws 0`).", ""]
    else:
        L += ["## Nulls (date-block permutation, block 20, one permutation shared by all arms)", "",
          f"N1 -- primary cell `{PRIMARY_ARM}`: mean {_f(n['primary']['mean'])}, sd "
          f"{n['primary']['sd']:.4f}, p95|IC| {n['primary']['p95_abs']:.4f}, observed "
          f"{_f(n['primary']['observed'])}, z {n['primary']['z']:+.2f}, "
          f"p(one-sided) {n['primary']['p_one_sided']:.4f}, "
          f"p(two-sided) {n['primary']['p_two_sided']:.4f}, "
          f"{n['primary']['n_draws']} draws.", "",
          f"N2 -- family max|IC| against the same observed value: mean "
          f"{_f(n['family']['mean'])}, sd {n['family']['sd']:.4f}, p95 "
          f"{n['family']['p95_abs']:.4f}, p {n['family']['p_two_sided']:.4f}.", "",
          "| arm | null mean | null sd | null p95\\|IC\\| | observed | p (two-sided) |",
          "|---|--:|--:|--:|--:|--:|"]
        for k, v in n["per_arm"].items():
            L += [f"| `{k}` | {_f(v['mean'])} | {v['sd']:.4f} | {v['p95_abs']:.4f} | "
                  f"{_f(v['observed'])} | {v['p_two_sided']:.4f} |"]

    p = out["power"]
    L += ["", "## Power", "",
          f"Analytic (`n50.power_statement`, N={p['statement']['n_names']}, "
          f"T={p['statement']['n_dates']}): per-date IC sd "
          f"{p['statement']['ic_sd_per_date']:.4f} ({p['statement']['ic_sd_source']}), "
          f"naive SE {p['statement']['se_naive']:.4f}, lag-20 SE "
          f"{p['statement']['se_nw_lag20']:.4f}, t=3 bar "
          f"{_f(p['statement']['bar_t3_nw_lag20'])}, t=2 bar "
          f"{_f(p['statement']['bar_t2_nw_lag20'])}.", "",
          "Empirical recovery at N=50 (signal planted in the free block, fitted on all rows, "
          "graded on the 50):", "",
          "| target | planted solo IC (N50) | planted solo IC (200) | combined IC | increment | recovery |",
          "|--:|--:|--:|--:|--:|--:|"]
    for r in p["recovery"]:
        L += [f"| {r['target']:.2f} | {_f(r['planted_solo_ic_n50'])} | "
              f"{_f(r['planted_solo_ic_200'])} | {_f(r['combined_ic_n50'])} | "
              f"{_f(r['increment'])} | {r['recovery_pct']:.0f}% |"]

    c = out["contamination"]
    L += ["", "## Contamination split (primary cell)", "",
          f"Checkpoint {c['checkpoint']}: pre {_f(c['pre'].get('mean_ic'))} over "
          f"{c['pre'].get('n_dates', 0)} dates, post {_f(c['post'].get('mean_ic'))} over "
          f"{c['post'].get('n_dates', 0)} dates -- post >= pre: {c['post_ge_pre']}.", ""]

    fp = out["fingerprint"]
    top = sorted(fp.items(), key=lambda kv: -abs(kv[1]))
    L += ["## Fingerprint (per-date Spearman of the primary score vs the free factors, N50 rows)",
          "", "| factor | rho |", "|---|--:|"]
    L += [f"| `{k}` | {_f(v, 3)} |" for k, v in top]
    L += ["", f"max |rho| = {max(abs(v) for v in fp.values()):.3f} "
          f"(M3 threshold 0.30).", ""]

    ci = out["conditional_ic"]
    L += ["## Conditional IC (primary score within quintiles of the free-block score)", "",
          f"Conditioned mean {_f(ci['summary'].get('mean_ic'))} over {ci['n_dates']} dates "
          f"vs raw {_f(ci['raw_ic'])}; tripwire (conditioned > raw): {ci['tripwire']}.", "",
          "| free-score quintile | IC |", "|---|--:|"]
    L += [f"| Q{int(q) + 1} | {_f(v)} |" for q, v in ci["per_quintile_mean"].items()]
    L += ["", "## Score exports (contract: date, symbol, score, is_n50)", "",
          "| file | rows | N50 rows | dates |", "|---|--:|--:|--:|"]
    for k, v in out["exports"].items():
        L += [f"| `{os.path.basename(v['path'])}` | {v['n_rows']} | {v['n_n50_rows']} | "
              f"{v['n_dates']} |"]
    L += ["", "## Reading notes (things a reader would otherwise have to reverse-engineer)", "",
          f"* Exported scores are horizon {PRIMARY_H} only, one row per cell that received a "
          "score (the whole PIT 200 panel, with `is_n50` flagging the tradable subset).",
          "* Arms differ in `n_dates`: the causal-PCA arms cannot score the first "
          f"{out['meta'].get('min_hist', 120)} dates (no past basis) while `free` can, so the "
          "LEVEL rows are not on identical date sets. Every paired increment is an inner join "
          "on dates, so the contrasts are.",
          "* Clause (a) is measured against the single reservoir seed cached in "
          "`bench/out/nse/ml` (seed 7). The plan's \"best reservoir\" over seeds 8 and 9 "
          "(arm A6) lives in the `extended` group and is not in this run.",
          "* The fit-on-50 control shares the panel-wide label ranks; a within-date monotone "
          "restriction preserves the ordering, so it is not disadvantaged by that.",
          "* Every p-value is a permutation p-value with the (1+k)/(1+n) correction, so the "
          f"smallest attainable value at {out['meta']['null_draws']} draws is "
          f"{1.0 / (out['meta']['null_draws'] + 1):.4f}.", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# -------------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="core", help="core | extended | all | comma list")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--null-draws", type=int, default=300)
    ap.add_argument("--null-block", type=int, default=20)
    ap.add_argument("--null-pilot", type=int, default=5,
                    help="time this many draws first and project the full run (0 to skip)")
    ap.add_argument("--null-budget-min", type=float, default=60.0,
                    help="if the projection exceeds this, fall back to --null-draws-fallback")
    ap.add_argument("--null-draws-fallback", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tag", default="probe_n50")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    names = arm_names(args.arms)
    if not names:
        raise SystemExit(f"--arms {args.arms} selected no registered arm")

    t_start = time.time()
    ctx = ProbeContext.load(dataset=args.dataset, n50_dir=args.out_dir)
    print(f"dataset {ctx.n_rows} cells, {int(ctx.is_n50.sum())} is_n50, "
          f"{len(np.unique(ctx.dates))} dates", flush=True)
    print(f"arms ({len(names)}): {', '.join(names)}\n", flush=True)

    run = run_arms(ctx, names, horizons, scorer="fm", seed=args.seed)
    res, scores = run["eval"], run["scores"]

    # secondary scorer on the primary block only (pre-registered, not a new arm)
    pooled = {}
    if PRIMARY_ARM in names:
        prun = run_arms(ctx, [PRIMARY_ARM], horizons, scorer="pooled", seed=args.seed)
        pooled = {str(h): summary_json(prun["eval"][PRIMARY_ARM][h]) for h in horizons}
        pooled_scores_arr = prun["scores"][PRIMARY_ARM]
    else:
        pooled_scores_arr = {}

    ctx.release_sources()   # the PCA blocks are built; the 832-d sources are dead weight now
    incs = {str(h): increment_table(res, CORE_PAIRS, h, seed=args.seed) for h in horizons}

    # ---- nulls -------------------------------------------------------------------
    n_draws = args.null_draws
    pilot = {}
    if args.null_pilot and n_draws:
        t0 = time.time()
        family_null(ctx, names, PRIMARY_H, args.null_pilot, block=args.null_block,
                    seed=args.seed + 991)
        per = (time.time() - t0) / args.null_pilot
        proj = per * n_draws / 60.0
        pilot = {"pilot_draws": args.null_pilot, "seconds_per_draw": round(per, 2),
                 "projected_minutes_for_requested": round(proj, 1),
                 "budget_minutes": args.null_budget_min}
        print(f"null pilot: {per:.2f}s/draw -> {proj:.1f} min for {n_draws} draws", flush=True)
        if proj > args.null_budget_min:
            n_draws = args.null_draws_fallback
            pilot["reduced_to"] = n_draws
            print(f"  over budget -> reducing to {n_draws} draws", flush=True)
    nulls, null_arrays = {}, {}
    if n_draws:
        print(f"family null: {n_draws} draws x {len(names)} arms ...", flush=True)
        t0 = time.time()
        fam = family_null(ctx, names, PRIMARY_H, n_draws, block=args.null_block,
                          seed=args.seed + 11, progress=max(1, n_draws // 6))
        obs = res[PRIMARY_ARM][PRIMARY_H]["ic_summary"]["mean_ic"]
        nulls = {
            "n_draws": int(n_draws), "block": int(args.null_block), "scorer": PRIMARY_SCORER,
            "eval": "is_n50", "min_names": MIN_NAMES, "h": PRIMARY_H,
            "elapsed_s": round(time.time() - t0, 1),
            "primary": null_pvalues(fam["arms"][PRIMARY_ARM], obs),
            "family": null_pvalues(fam["max_abs"], obs),
            "per_arm": {k: null_pvalues(v, res[k][PRIMARY_H]["ic_summary"]["mean_ic"])
                        for k, v in fam["arms"].items()},
        }
        null_arrays = {f"arm__{k}": v for k, v in fam["arms"].items()}
        null_arrays["max_abs"] = fam["max_abs"]
        npz = os.path.join(args.out_dir, f"{args.tag}_null.npz")
        np.savez_compressed(npz, **null_arrays)
        sidecar(npz.replace(".npz", "_meta.json"), vars(args),
                extra={"arms": names, "n_draws": int(n_draws), "block": int(args.null_block),
                       "scorer": PRIMARY_SCORER, "h": PRIMARY_H, "eval_mask": "is_n50",
                       "min_names": MIN_NAMES, "pilot": pilot,
                       "keys": sorted(null_arrays), "note":
                       "per-draw mean IC on is_n50 rows; arms within a draw share one "
                       "permutation, so max_abs is a family-wise null"})
        print(f"  wrote {npz} ({nulls['elapsed_s']}s)", flush=True)

    # ---- power, fingerprint, contamination, conditional IC -----------------------
    print("power / fingerprint / contamination / conditional IC ...", flush=True)
    ev_p = res[PRIMARY_ARM][PRIMARY_H]
    power = {"statement": ev_p["power"], "statement_asymptotic": ev_p["power_asymptotic"],
             "statement_realised": ev_p.get("power_realised", {}),
             "recovery": recovery_curve(ctx, PRIMARY_H, seed=args.seed)}
    sc_p = scores[PRIMARY_ARM][PRIMARY_H]
    fp = fingerprint(ctx, sc_p, PRIMARY_H)
    contam = contamination_split(ev_p["ic"], PRIMARY_H, seed=args.seed)
    cond = conditional_ic(ctx, sc_p, scores["free"][PRIMARY_H], PRIMARY_H, seed=args.seed) \
        if "free" in scores else {}
    if cond:
        cond["raw_ic"] = ev_p["ic_summary"]["mean_ic"]
        cond["tripwire"] = bool(cond["summary"].get("mean_ic", -np.inf) > cond["raw_ic"])

    gates = m1_gates(res, incs[str(PRIMARY_H)], nulls, PRIMARY_H)
    # The secondary horizon gets the same clauses (a) and (c) for information. Clause (b) is
    # NOT evaluated there: the family null is a 300-draw refit at h=5 only, and scoring an
    # h=10 statistic against an h=5 null would be exactly the estimator mismatch F1 exists to
    # stop. It is reported as "not computed", never as a failure.
    gates_secondary = {}
    for h in horizons:
        if h == PRIMARY_H:
            continue
        g = m1_gates(res, incs[str(h)], {}, h)
        g["b_family_null_p"] = {"pass": None, "p_two_sided": None,
                                "clause": "family max|IC| p < 0.05",
                                "note": f"not computed: the family null was run at h={PRIMARY_H} "
                                        "only; an h=%d statistic has no h=%d null" % (h, h)}
        g["verdict_M1"] = None
        g["note"] = ("secondary horizon: clauses (a) and (c) only, no family null, "
                     "not a decision")
        gates_secondary[str(h)] = g

    # ---- exports -----------------------------------------------------------------
    exports = {}
    for (arm, sc_name), tag in SCORE_EXPORTS.items():
        if sc_name == "fm" and arm in scores:
            arr = scores[arm][PRIMARY_H]
        elif sc_name == "pooled" and arm == PRIMARY_ARM and pooled_scores_arr:
            arr = pooled_scores_arr[PRIMARY_H]
        else:
            continue
        p = os.path.join(args.out_dir, f"scores_{tag}.parquet")
        exports[f"{arm}@{sc_name}"] = write_scores(
            p, ctx, arr, vars(args),
            extra={"arm": arm, "scorer": sc_name, "horizon": PRIMARY_H,
                   "contract": "date, symbol, score, is_n50; one row per scored cell",
                   "fit_rows": "all eligible PIT NIFTY 200 cells"
                              if arm != "hT_pre_cpca8_fit50only" else "is_n50 rows only"})
        print(f"  {os.path.basename(p)}: {exports[f'{arm}@{sc_name}']['n_rows']} rows", flush=True)

    # ---- assemble ----------------------------------------------------------------
    out = {
        "meta": {"written_at": pd.Timestamp.now("UTC").isoformat(), "arms_selector": args.arms,
                 "arms": names, "arm_docs": {n: ARMS[n].doc for n in names},
                 "horizons": horizons, "seed": args.seed, "null_draws": int(n_draws),
                 "null_pilot": pilot, "primary_arm": PRIMARY_ARM, "primary_h": PRIMARY_H,
                 "primary_scorer": PRIMARY_SCORER,
                 "n_rows": int(ctx.n_rows), "n_n50_rows": int(ctx.is_n50.sum()),
                 "n_dates": int(len(np.unique(ctx.dates))), "min_hist": ctx.min_hist,
                 "refit_every": ctx.refit_every,
                 "elapsed_s": round(time.time() - t_start, 1)},
        "arms": {n: {h: summary_json(res[n][h]) for h in horizons} for n in names},
        "increments": incs,
        "pooled": pooled,
        "by_year": {str(h): year_table(res[PRIMARY_ARM][h], 10).to_dict("records")
                    for h in horizons},
        "nulls": nulls, "power": power, "fingerprint": fp, "contamination": contam,
        "conditional_ic": cond, "gates": gates, "gates_secondary": gates_secondary,
        "exports": exports,
    }
    jp = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    sidecar(jp.replace(".json", "_meta.json"), vars(args),
            extra={"arms": names, "horizons": horizons, "null_draws": int(n_draws),
                   "primary": f"{PRIMARY_ARM}@{PRIMARY_SCORER} h={PRIMARY_H}",
                   "outputs": [jp, os.path.join(args.out_dir, f"{args.tag}.md")]})
    # the markdown reads the JSON-shaped dict but needs the by-year records under str(h)
    write_report(os.path.join(args.out_dir, f"{args.tag}.md"), out, names, horizons)
    print(f"\nwrote {jp}\nwrote {os.path.join(args.out_dir, args.tag + '.md')}", flush=True)
    print(f"M1: {'PASS' if gates['verdict_M1'] else 'FAIL'}  "
          f"(a={gates['a_paired_pretrained_minus_reservoir']['pass']}, "
          f"b={gates['b_family_null_p']['pass']}, c={gates['c_top10_spread']['pass']})",
          flush=True)


# ==========================================================================================
# Stage B representation ladder: arms A1-A7 and F7 (NIFTY50_PLAN sec.5)
# ==========================================================================================
#
# Everything below is ADDITIVE. The core family above is untouched, so `--arms core` still
# reproduces C1's numbers bit-for-bit and the core-only family null stays a valid comparison.
#
# WHAT CHANGES AND WHAT DOES NOT. Every ladder arm is the SAME estimator on a DIFFERENT read
# of the same 150,527 windows: frozen causal PCA-32, rank-within-date, `fm_scores` fitted on
# the whole PIT NIFTY 200 panel, graded on `is_n50`. "Modifying Kronos" here means changing
# which internal state is read (which layer, which positions, which model size, tokenizer vs
# trunk) with zero edits to weights. So a difference between two ladder arms is attributable
# to the representation and nothing else.
#
# THE ONE PLACE THAT RULE IS BROKEN, deliberately and labelled: the F7 reservoir-seed block
# (A6). Seeds 8 and 9 were only extracted on the 37,800 N50 cells, so a like-for-like seed
# comparison cannot be fitted on the 200. That block is therefore fitted AND graded on the
# N50 rows only, through `pooled_scores` (whose sample size scales with N x T, which is what
# a 50-name cross-section needs -- F3), and is labelled "N50-only pooled" everywhere it
# appears. Its numbers are NOT comparable with the FM-on-200 arms and never enter M1.

#: Stage B arrays live in bench/out/nse/n50 under `n50:`; the round-3 cache lives in
#: bench/out/nse/ml under `ml:`. Keyed by (layer, pool) so the F12 layer table and the arm
#: registrations read from one place.
LAYER_SOURCES: dict[tuple[str, str], str] = {
    ("L4", "last"): "n50:hT_base_pretrained_L4_last",
    ("L8", "last"): "n50:hT_base_pretrained_L8_last",
    ("L12", "last"): "n50:hT_base_pretrained_L12_last",
    ("final", "last"): "ml:hT_pretrained",          # == n50:hT_base_pretrained_final_last
    ("L4", "mean10"): "n50:hT_base_pretrained_L4_mean10",
    ("L8", "mean10"): "n50:hT_base_pretrained_L8_mean10",
    ("L12", "mean10"): "n50:hT_base_pretrained_L12_mean10",
    ("final", "mean10"): "n50:hT_base_pretrained_final_mean10",
}

#: The SAME eight reads off a Kronos-base trunk with RANDOM weights (seed 7, identical
#: architecture, identical frozen tokenizer, identical extraction path). Layer depth and
#: position pooling are architecture, not learning: a mean over the last 10 positions of a
#: 12-block residual stream smooths the input window whether or not the blocks were trained,
#: and `hT_pre_L4_mean10_cpca32` was promoted off a level IC with no control at its own read.
#: `(final, last)` is deliberately absent: `ml:hT_reservoir` already IS that array (asserted
#: bit-identical by `check_reservoir_final_regression`), so registering it again would put
#: the same numbers into the family null twice.
RES_LAYER_SOURCES: dict[tuple[str, str], str] = {
    ("L4", "last"): "n50:hT_base_reservoir_s7_L4_last",
    ("L8", "last"): "n50:hT_base_reservoir_s7_L8_last",
    ("L12", "last"): "n50:hT_base_reservoir_s7_L12_last",
    ("final", "last"): "ml:hT_reservoir",       # == n50:hT_base_reservoir_s7_final_last
    ("L4", "mean10"): "n50:hT_base_reservoir_s7_L4_mean10",
    ("L8", "mean10"): "n50:hT_base_reservoir_s7_L8_mean10",
    ("L12", "mean10"): "n50:hT_base_reservoir_s7_L12_mean10",
}

RES_LAYER_ARMS: dict[tuple[str, str], str] = {
    ("L4", "last"): "hT_res_L4_last_cpca32",
    ("L8", "last"): "hT_res_L8_last_cpca32",
    ("L12", "last"): "hT_res_L12_last_cpca32",
    ("final", "last"): "hT_res_cpca32",
    ("L4", "mean10"): "hT_res_L4_mean10_cpca32",
    ("L8", "mean10"): "hT_res_L8_mean10_cpca32",
    ("L12", "mean10"): "hT_res_L12_mean10_cpca32",
}

#: (layer, pool) -> the registered arm that reads it at PCA-32. The final/last cell is the
#: pre-registered primary cell itself: the Stage B pass re-extracted that state and it is
#: bit-identical to the cached A0 array (checked by `check_final_last_regression`), so
#: registering a second arm for it would put the same numbers in the family null twice.
LAYER_ARMS: dict[tuple[str, str], str] = {
    ("L4", "last"): "hT_pre_L4_last_cpca32",
    ("L8", "last"): "hT_pre_L8_last_cpca32",
    ("L12", "last"): "hT_pre_L12_last_cpca32",
    ("final", "last"): PRIMARY_ARM,
    ("L4", "mean10"): "hT_pre_L4_mean10_cpca32",
    ("L8", "mean10"): "hT_pre_L8_mean10_cpca32",
    ("L12", "mean10"): "hT_pre_L12_mean10_cpca32",
    ("final", "mean10"): "hT_pre_final_mean10_cpca32",
}

#: The A1/A2 arms, i.e. every Kronos-base representation arm that is NOT the primary cell.
#: Each is contrasted against `hT_res_cpca32` -- the random-weight control at the same width
#: on the same estimator -- because a level IC says nothing without its control.
BASE_LADDER_ARMS = [LAYER_ARMS[k] for k in
                    [("L4", "last"), ("L8", "last"), ("L12", "last"),
                     ("final", "mean10"), ("L12", "mean10"), ("L8", "mean10"),
                     ("L4", "mean10")]]

#: F7 (A6): the four blocks of the N50-only pooled seed comparison. s7 is the round-3 cache
#: restricted to the N50 rows; s8/s9 exist only on those rows.
SEED_SOURCES = {
    "pretrained_final_last": "ml:hT_pretrained",
    "reservoir_s7": "ml:hT_reservoir",
    "reservoir_s8": "n50:hT_base_reservoir_s8_final_last_n50",
    "reservoir_s9": "n50:hT_base_reservoir_s9_final_last_n50",
}
RESERVOIR_SEEDS = ["reservoir_s7", "reservoir_s8", "reservoir_s9"]


def _num(x) -> float:
    """A sidecar field as a float; a missing one is NaN, not None, so a table can format it."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def raw_cols_arm(source: str):
    """Builder factory: the raw columns of a cached matrix, rank-within-date, no PCA.

    Only sane for a narrow block. It exists for the 20-column tokenizer embedding (A5): at
    d = 20 a PCA-32 is impossible and a PCA-8 throws away 60% of the columns, so the honest
    question "is the signal in the tokenizer's continuous code at all?" is asked of the code
    itself, and PCA-8 is run alongside as the compression-matched comparison.
    """
    def build(ctx: ProbeContext):
        M, valid = ctx.hidden(source)
        blk = np.zeros((ctx.n_rows, M.shape[1]), dtype=np.float64)
        idx = np.flatnonzero(valid)
        if len(idx):
            blk[idx] = rank_within_date(M[valid].astype(np.float64), ctx.dcode[valid])
        return blk, valid.copy()
    return build


# --- A1: residual stream at depth, last position -------------------------------------------
register_arm("hT_pre_L4_last_cpca32", cpca_arm(LAYER_SOURCES[("L4", "last")], 32),
             group="extended", doc="A1: residual stream after block 4 (pre-norm), last position")
register_arm("hT_pre_L8_last_cpca32", cpca_arm(LAYER_SOURCES[("L8", "last")], 32),
             group="extended", doc="A1: residual stream after block 8 (pre-norm), last position")
register_arm("hT_pre_L12_last_cpca32", cpca_arm(LAYER_SOURCES[("L12", "last")], 32),
             group="extended", doc="A1: residual stream after block 12 (pre-norm), last position")

# --- A2: position pooling (mean of the last 10 positions) ----------------------------------
register_arm("hT_pre_final_mean10_cpca32", cpca_arm(LAYER_SOURCES[("final", "mean10")], 32),
             group="extended", doc="A2: final post-norm state, mean of the last 10 positions")
register_arm("hT_pre_L12_mean10_cpca32", cpca_arm(LAYER_SOURCES[("L12", "mean10")], 32),
             group="extended", doc="A2: block-12 residual stream, mean of the last 10 positions")
register_arm("hT_pre_L8_mean10_cpca32", cpca_arm(LAYER_SOURCES[("L8", "mean10")], 32),
             group="extended", doc="A2: block-8 residual stream, mean of the last 10 positions")
register_arm("hT_pre_L4_mean10_cpca32", cpca_arm(LAYER_SOURCES[("L4", "mean10")], 32),
             group="extended", doc="A2: block-4 residual stream, mean of the last 10 positions")

# --- A3: capacity ladder, Kronos-small and its own reservoir -------------------------------
register_arm("hT_small_pre_cpca32", cpca_arm("n50:hT_small_pretrained_final_last", 32),
             group="extended", doc="A3: Kronos-small (512-d, 8 layers) h_T")
register_arm("hT_small_res_cpca32", cpca_arm("n50:hT_small_reservoir_s7_final_last", 32),
             group="extended", doc="A3 control: Kronos-small random weights (seed 7)")

# --- A4: Kronos-mini ------------------------------------------------------------------------
register_arm("hT_mini_pre_cpca32", cpca_arm("n50:hT_mini_pretrained_final_last", 32),
             group="extended", doc="A4: Kronos-mini (256-d, 4 layers, Tokenizer-2k) h_T")

# --- A5: tokenizer continuous embedding, pre-BSQ (20-d) -------------------------------------
register_arm("tokz_last_raw20", raw_cols_arm("n50:hT_tokz_last"),
             group="extended", doc="A5: tokenizer z (pre-BSQ), last position, all 20 columns")
register_arm("tokz_last_cpca8", cpca_arm("n50:hT_tokz_last", 8),
             group="extended", doc="A5: tokenizer z, last position, causal PCA-8")
register_arm("tokz_mean10_raw20", raw_cols_arm("n50:hT_tokz_mean10"),
             group="extended", doc="A5: tokenizer z, mean of last 10 positions, all 20 columns")
register_arm("tokz_mean10_cpca8", cpca_arm("n50:hT_tokz_mean10", 8),
             group="extended", doc="A5: tokenizer z, mean of last 10 positions, causal PCA-8")


# --- group "control": the random-weight control AT THE SAME READ (the gate F7/A1/A2 never had) ----------
# One level IC is not a result: `hT_res_cpca32` controls the final/last read, so contrasting
# a layer-4 mean-pooled state against it prices depth+pooling+training in one number. These
# six arms hold depth and pooling fixed and vary only whether the trunk was trained.
register_arm("hT_res_L4_last_cpca32", cpca_arm(RES_LAYER_SOURCES[("L4", "last")], 32),
             group="control", doc="control: random-weight block-4 residual stream, last pos")
register_arm("hT_res_L8_last_cpca32", cpca_arm(RES_LAYER_SOURCES[("L8", "last")], 32),
             group="control", doc="control: random-weight block-8 residual stream, last pos")
register_arm("hT_res_L12_last_cpca32", cpca_arm(RES_LAYER_SOURCES[("L12", "last")], 32),
             group="control", doc="control: random-weight block-12 residual stream, last pos")
register_arm("hT_res_L4_mean10_cpca32", cpca_arm(RES_LAYER_SOURCES[("L4", "mean10")], 32),
             group="control", doc="control: random-weight block-4 stream, mean of last 10 pos")
register_arm("hT_res_L8_mean10_cpca32", cpca_arm(RES_LAYER_SOURCES[("L8", "mean10")], 32),
             group="control", doc="control: random-weight block-8 stream, mean of last 10 pos")
register_arm("hT_res_L12_mean10_cpca32", cpca_arm(RES_LAYER_SOURCES[("L12", "mean10")], 32),
             group="control", doc="control: random-weight block-12 stream, mean of last 10 pos")


#: Which cached array each arm reads. Written out rather than introspected from the builder
#: closures, because the arrays-and-their-effective-ranks table is a published artefact and a
#: table built by guessing at closures is a table that can be silently wrong.
ARM_SOURCE: dict[str, str] = {
    PRIMARY_ARM: "ml:hT_pretrained", "hT_pre_cpca8": "ml:hT_pretrained",
    "hT_pre_cpca64": "ml:hT_pretrained", "hT_pre_cpca32_procrustes": "ml:hT_pretrained",
    "hT_pre_cpca8_fit50only": "ml:hT_pretrained",
    "hT_res_cpca8": "ml:hT_reservoir", "hT_res_cpca32": "ml:hT_reservoir",
    "hT_res_cpca64": "ml:hT_reservoir",
    "hT_pre_L4_last_cpca32": LAYER_SOURCES[("L4", "last")],
    "hT_pre_L8_last_cpca32": LAYER_SOURCES[("L8", "last")],
    "hT_pre_L12_last_cpca32": LAYER_SOURCES[("L12", "last")],
    "hT_pre_final_mean10_cpca32": LAYER_SOURCES[("final", "mean10")],
    "hT_pre_L12_mean10_cpca32": LAYER_SOURCES[("L12", "mean10")],
    "hT_pre_L8_mean10_cpca32": LAYER_SOURCES[("L8", "mean10")],
    "hT_pre_L4_mean10_cpca32": LAYER_SOURCES[("L4", "mean10")],
    "hT_small_pre_cpca32": "n50:hT_small_pretrained_final_last",
    "hT_small_res_cpca32": "n50:hT_small_reservoir_s7_final_last",
    "hT_mini_pre_cpca32": "n50:hT_mini_pretrained_final_last",
    "hT_res_L4_last_cpca32": RES_LAYER_SOURCES[("L4", "last")],
    "hT_res_L8_last_cpca32": RES_LAYER_SOURCES[("L8", "last")],
    "hT_res_L12_last_cpca32": RES_LAYER_SOURCES[("L12", "last")],
    "hT_res_L4_mean10_cpca32": RES_LAYER_SOURCES[("L4", "mean10")],
    "hT_res_L8_mean10_cpca32": RES_LAYER_SOURCES[("L8", "mean10")],
    "hT_res_L12_mean10_cpca32": RES_LAYER_SOURCES[("L12", "mean10")],
    "tokz_last_raw20": "n50:hT_tokz_last", "tokz_last_cpca8": "n50:hT_tokz_last",
    "tokz_mean10_raw20": "n50:hT_tokz_mean10", "tokz_mean10_cpca8": "n50:hT_tokz_mean10",
}


#: A3's own paired contrast plus every base-ladder arm against the random-weight control at
#: the same width. `EXTENDED_PAIRS` is fixed at import; the "extended arm minus the primary
#: cell" contrasts are generated in `extended_increments` so adding an arm cannot forget one.
EXTENDED_PAIRS = ([("hT_small_pre_cpca32", "hT_small_res_cpca32")]
                  + [(a, "hT_res_cpca32") for a in BASE_LADDER_ARMS]
                  + [("hT_mini_pre_cpca32", "hT_res_cpca32"),
                     ("tokz_last_cpca8", "hT_res_cpca8"),
                     ("tokz_mean10_cpca8", "hT_res_cpca8")])


#: THE gate of the layer-4 control run: pretrained minus reservoir AT THE SAME (layer, pool).
#: `EXTENDED_PAIRS` already contrasts every ladder arm with `hT_res_cpca32`, which is the
#: random-weight control at the FINAL/LAST read -- a different read from the arm it is
#: subtracted from, so that contrast prices depth, pooling and training together. These pairs
#: hold the read fixed.
LAYER_CONTROL_PAIRS = [(LAYER_ARMS[k], RES_LAYER_ARMS[k])
                       for k in [("L4", "mean10"), ("L8", "mean10"), ("L12", "mean10"),
                                 ("L4", "last"), ("L8", "last"), ("L12", "last")]]
EXTENDED_PAIRS = EXTENDED_PAIRS + [p for p in LAYER_CONTROL_PAIRS if p not in EXTENDED_PAIRS]


def check_reservoir_final_regression(ctx: ProbeContext, chunk: int = 20000) -> dict:
    """The seed-7 reservoir pass must reproduce the round-3 `ml:hT_reservoir` cache.

    Same argument as `check_final_last_regression` for the pretrained side: if the new pass's
    final/last state is bit-identical to the cached one, the random init is reproducible, the
    new layer arrays come from the same trunk as the control the record already used, and
    `hT_res_cpca32` can BE the (final, last) cell of the reservoir ladder instead of a
    duplicate arm scoring the same array inside the family null.
    """
    a = np.load(os.path.join(ctx.ml_dir, "hT_reservoir.npy"), mmap_mode="r")
    b = np.load(os.path.join(ctx.n50_dir, "hT_base_reservoir_s7_final_last.npy"), mmap_mode="r")
    if a.shape != b.shape:
        return {"identical": False, "reason": f"shape {a.shape} vs {b.shape}"}
    worst, n_diff = 0.0, 0
    for i in range(0, len(a), chunk):
        A = np.asarray(a[i:i + chunk], dtype=np.float32)
        B = np.asarray(b[i:i + chunk], dtype=np.float32)
        worst = max(worst, float(np.abs(A - B).max()))
        n_diff += int((A != B).sum())
    return {"identical": bool(n_diff == 0), "max_abs_diff": worst,
            "n_differing_elements": n_diff,
            "note": "bit-identical fp16 => hT_res_cpca32 IS the reservoir ladder's final/last cell"}


def extended_increments(res: dict, h: int, ext_names, seed: int = 0) -> dict:
    """Every pre-registered contrast at horizon h: ladder - primary, plus `EXTENDED_PAIRS`."""
    pairs = [(n, PRIMARY_ARM) for n in ext_names if n != PRIMARY_ARM]
    pairs += [p for p in EXTENDED_PAIRS if p not in pairs]
    return increment_table(res, pairs, h, seed=seed)


# ------------------------------------------------------------- alignment preconditions


def assert_windows_alignment(ctx: ProbeContext) -> dict:
    """Row i of every cached .npy is the window of row i of `dataset_n50.parquet`. Prove it.

    The arrays carry no keys. A silently reordered dataset would pair every hidden state with
    the wrong (date, symbol) and still produce a plausible IC, so this is checked against
    `windows_index.parquet` -- the file the extraction actually iterated -- and not merely
    against `ml/dataset.parquet` (which `ProbeContext.load` already checks).
    """
    wi = pd.read_parquet(os.path.join(ctx.ml_dir, "windows_index.parquet"))
    wi["date"] = pd.to_datetime(wi["date"])
    if len(wi) != ctx.n_rows:
        raise SystemExit(f"windows_index has {len(wi)} rows, dataset_n50 has {ctx.n_rows}")
    same = (bool((wi["date"].to_numpy() == ctx.dates).all())
            and bool((wi["symbol"].astype(str).to_numpy()
                      == ctx.ds["symbol"].astype(str).to_numpy()).all()))
    if not same:
        raise SystemExit("dataset_n50.parquet is not in windows_index.parquet order; the "
                         "cached hidden states cannot be used positionally")
    return {"n_rows": int(len(wi)), "windows_index_order": True}


def check_array_shapes(ctx: ProbeContext, sources) -> dict:
    """Every array is either full-panel or covers exactly the rows its sidecar names."""
    out = {}
    for src in sources:
        where, _, name = src.partition(":")
        base = {"ml": ctx.ml_dir, "n50": ctx.n50_dir}[where]
        M = np.load(os.path.join(base, f"{name}.npy"), mmap_mode="r")
        meta = array_meta(ctx, src)
        rec = {"rows": int(M.shape[0]), "d": int(M.shape[1]), "dtype": str(M.dtype),
               "full_panel": bool(M.shape[0] == ctx.n_rows),
               "effective_rank": meta.get("effective_rank"),
               "n_pcs_for_95pct_var": meta.get("n_pcs_for_95pct_var"),
               "rows_sidecar": meta.get("rows_sidecar")}
        if not rec["full_panel"]:
            side = meta.get("rows_sidecar") or "rows_n50.parquet"
            rows = pd.read_parquet(os.path.join(base, side))["row"].to_numpy(dtype=int)
            if len(rows) != M.shape[0]:
                raise SystemExit(f"{name}: {M.shape[0]} rows, sidecar {side} has {len(rows)}")
            rec["sidecar_matches_is_n50"] = bool(
                len(rows) == int(ctx.is_n50.sum())
                and (rows == np.flatnonzero(ctx.is_n50)).all())
            if not rec["sidecar_matches_is_n50"]:
                raise SystemExit(f"{name}: rows sidecar is not the is_n50 row set")
        if meta.get("n") not in (None, M.shape[0]):
            raise SystemExit(f"{name}: meta says n={meta['n']}, array has {M.shape[0]}")
        out[src] = rec
    return out


def array_meta(ctx: ProbeContext, source: str) -> dict:
    """The `_meta.json` sidecar written by `hidden.py` next to a cached array."""
    where, _, name = source.partition(":")
    base = {"ml": ctx.ml_dir, "n50": ctx.n50_dir}.get(where)
    p = os.path.join(base, f"{name}_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def check_final_last_regression(ctx: ProbeContext, chunk: int = 20000) -> dict:
    """Stage B re-extracted the A0 state; it must reproduce the round-3 cache.

    This is the regression check NIFTY50_PLAN sec.5 asks for ("A0 must be reproduced
    bit-for-bit from the same pass"). If it holds, the layer table's final/last row can be
    the primary arm itself instead of a thirteenth family member scoring the same array.
    """
    a = np.load(os.path.join(ctx.ml_dir, "hT_pretrained.npy"), mmap_mode="r")
    b = np.load(os.path.join(ctx.n50_dir, "hT_base_pretrained_final_last.npy"), mmap_mode="r")
    if a.shape != b.shape:
        return {"identical": False, "reason": f"shape {a.shape} vs {b.shape}"}
    worst, n_diff = 0.0, 0
    for i in range(0, len(a), chunk):
        A = np.asarray(a[i:i + chunk], dtype=np.float32)
        B = np.asarray(b[i:i + chunk], dtype=np.float32)
        d = np.abs(A - B)
        worst = max(worst, float(d.max()))
        n_diff += int((A != B).sum())
    return {"identical": bool(n_diff == 0), "max_abs_diff": worst, "n_differing_elements": n_diff,
            "note": "bit-identical fp16 => the layer table's final/last row IS the primary arm"}


# ------------------------------------------------------------------- block construction


def build_blocks(ctx: ProbeContext, names, release: bool = True, verbose: bool = True) -> dict:
    """Materialise every arm's feature block, dropping the source matrix after each one.

    Eleven full-panel 832-d float16 arrays are 2.75 GB on disk and 5.5 GB as float32; held
    together with the 540-column window tensor they do not fit alongside a 26-arm family
    null. Each source is re-read if a later arm needs it (a memory-mapped read of a file the
    OS has cached costs well under a second), while the PROJECTED blocks -- a few tens of MB
    each -- are memoised on the context and never rebuilt.
    """
    info = {}
    for n in names:
        t0 = time.time()
        X, ok = build_arm(n, ctx)
        if release:
            ctx.release_sources()
        info[n] = {"n_features": int(X.shape[1]), "n_valid_rows": int(ok.sum()),
                   "n_valid_n50_rows": int((ok & ctx.is_n50).sum()),
                   "build_seconds": round(time.time() - t0, 2)}
        if verbose:
            print(f"  built {n:30s} {X.shape[1]:3d} cols, {int(ok.sum()):6d} valid rows "
                  f"({info[n]['build_seconds']}s)", flush=True)
    return info


# ------------------------------------------------------ F7 / A6: reservoir seed variance


def reservoir_seed_block(ctx: ProbeContext, h: int = PRIMARY_H, k: int = 8, seed: int = 0,
                         train_window: int = 500, refit_every: int = 20,
                         alpha: float = 10.0) -> dict:
    """N50-ONLY POOLED. Three reservoir seeds and the pretrained state, like for like.

    F7's complaint is that a one-draw random control has no variance estimate, so "pretrained
    beats reservoir" could be a lucky seed. Seeds 8 and 9 were extracted on the 37,800 N50
    cells only, so the only comparison that holds the data fixed across seeds is one fitted
    on those rows: causal PCA-`k` whose basis and ranking use N50 rows only, then
    `pooled_scores` (trailing `train_window` dates x ~50 names, refit every `refit_every`,
    h-session embargo, fixed ridge) on the same rows. The pre-registered width is k = 8
    because the fit is now N x T with N = 50; the runner repeats it at k = 32 as a labelled
    sensitivity, since PCA-8 of the final state is a weak block for the pretrained arm in the
    FM-on-200 family too and a negative increment at k = 8 alone would be about the width.

    These numbers are NOT the FM-on-200 numbers and must never be quoted beside them: the
    universe, the width, the scorer and the fit rows all differ. What they support is one
    statement -- how much the reservoir IC moves when only the random seed changes.
    """
    y_raw, y, fin = ctx.labels(h)
    per = {}
    for label, src in SEED_SOURCES.items():
        t0 = time.time()
        X, ok = ctx.cpca(src, k, mode="frozen", rows=ctx.is_n50)
        m = ok & fin & ctx.is_n50
        sc = np.full(ctx.n_rows, np.nan)
        if m.sum() > k + 2:
            sc[m] = pooled_scores(X[m], y[m], ctx.dates[m], h, train_window=train_window,
                                  refit_every=refit_every, alpha=alpha)
        ev = evaluate(sc, ctx.ds, h, mask=ctx.is_n50, min_names=MIN_NAMES, seed=seed)
        meta = array_meta(ctx, src)
        per[label] = {"source": src, "ic": ev["ic"], "ic_summary": ev["ic_summary"],
                      "spread10_summary": ev["spread_summary"][10],
                      "effective_rank": _num(meta.get("effective_rank")),
                      "n_pcs_for_95pct_var": meta.get("n_pcs_for_95pct_var"),
                      "n_fit_rows": int(m.sum()), "seconds": round(time.time() - t0, 2)}
        ctx.release_sources()
        print(f"  [N50-only pooled] {label:24s} IC {ev['ic_summary']['mean_ic']:+.4f}  "
              f"t20 {ev['ic_summary']['t_nw_lag20']:+.2f}  n={ev['ic_summary']['n_dates']}",
              flush=True)

    seeds = [per[s]["ic_summary"]["mean_ic"] for s in RESERVOIR_SEEDS]
    best = max(RESERVOIR_SEEDS, key=lambda s: per[s]["ic_summary"]["mean_ic"])
    incs = {f"pretrained_final_last - {s}":
            paired_increment(per["pretrained_final_last"]["ic"], per[s]["ic"], h, seed=seed)
            for s in RESERVOIR_SEEDS}
    return {
        "label": "N50-only pooled",
        "design": {"h": h, "k": k, "pca_mode": "frozen", "scorer": "pooled_scores",
                   "train_window": train_window, "refit_every": refit_every, "alpha": alpha,
                   "fit_rows": "is_n50 only", "eval_rows": "is_n50",
                   "warning": "not comparable with the FM-on-200 arms: different universe, "
                              "width, scorer and fit rows"},
        "arms": {kk: {x: v[x] for x in v if x != "ic"} for kk, v in per.items()},
        "reservoir_seed_mean": float(np.mean(seeds)),
        "reservoir_seed_sd": float(np.std(seeds, ddof=1)),
        "reservoir_seed_ics": {s: per[s]["ic_summary"]["mean_ic"] for s in RESERVOIR_SEEDS},
        "best_reservoir_seed": best,
        "increments": incs,
        "pretrained_minus_best_seed": incs[f"pretrained_final_last - {best}"],
    }


# ----------------------------------------------------------------------- A7: ensemble


def rank_average(ctx: ProbeContext, arrays) -> np.ndarray:
    """Within-date rank average of several score vectors, on rows all of them scored.

    Rank-averaging, not score-averaging: the components are fitted numbers on different
    scales and an unnormalised mean would be whichever component happens to have the largest
    variance. Ranks are taken over the whole PIT 200 cross-section (the rows the components
    scored); every downstream metric is a within-date rank statistic on a subset, which a
    within-date monotone transform leaves alone.
    """
    S = np.column_stack([np.asarray(a, dtype=float) for a in arrays])
    ok = np.isfinite(S).all(axis=1)
    out = np.full(ctx.n_rows, np.nan)
    if ok.any():
        out[ok] = rank_within_date(S[ok], ctx.dcode[ok]).mean(axis=1)
    return out


def ensemble_arm(ctx: ProbeContext, res: dict, scores: dict, h: int, members,
                 seed: int = 0) -> dict:
    """A7, pre-registered as secondary: rank-average of A0 + the best A1/A2 + A3 small.

    The pre-registration has a condition attached -- a component enters only if its own h=5
    IC on `is_n50` is positive -- so a negative component cannot be laundered into the
    average. Which components were admitted, and which were dropped, is reported either way.
    """
    admitted, dropped = [], {}
    for n in members:
        ic = float(res[n][h]["ic_summary"]["mean_ic"])
        if ic > 0:
            admitted.append(n)
        else:
            dropped[n] = ic
    out = {"candidates": list(members), "admitted": admitted, "dropped": dropped,
           "rule": "component enters only if its own h=5 IC on is_n50 is positive"}
    if len(admitted) < 2:
        out["run"] = False
        out["note"] = ("fewer than two components cleared the positive-IC condition; the "
                       "ensemble is not formed")
        return out
    sc = rank_average(ctx, [scores[n][h] for n in admitted])
    ev = evaluate(sc, ctx.ds, h, mask=ctx.is_n50, min_names=MIN_NAMES, seed=seed)
    out["run"] = True
    out["scores"] = sc
    out["eval"] = ev
    out["vs_primary"] = (paired_increment(ev["ic"], res[PRIMARY_ARM][h]["ic"], h, seed=seed)
                         if PRIMARY_ARM in res else {"n_dates": 0,
                                                     "note": "primary cell not in this run"})
    out["vs_best_member"] = {
        n: paired_increment(ev["ic"], res[n][h]["ic"], h, seed=seed) for n in admitted}
    return out


# --------------------------------------------------- diagnostics for the promoted arm(s)


def score_agreement(ctx: ProbeContext, a: np.ndarray, b: np.ndarray, h: int,
                    min_names: int = MIN_NAMES) -> dict:
    """Mean per-date Spearman between two fitted scores on the N50 rows.

    This is the number that explains a paired increment's t. Two arms whose scores agree at
    rho = 0.8 differ by a small, noisy residual, so "arm X does not beat arm Y at t >= 2" can
    mean "X is much better" or "X and Y are nearly the same object seen twice" -- and only
    the agreement tells you which. Without it a failed G1 reads as a stronger negative than
    it is.
    """
    m0 = ctx.is_n50 & np.isfinite(a) & np.isfinite(b)
    rhos = []
    for dd in np.unique(ctx.dates[m0]):
        sel = m0 & (ctx.dates == dd)
        if sel.sum() < min_names:
            continue
        r = sps.spearmanr(a[sel], b[sel]).statistic
        if np.isfinite(r):
            rhos.append(r)
    s = pd.Series(rhos, dtype=float)
    return {"n_dates": int(len(s)), "mean_rho": float(s.mean()) if len(s) else float("nan"),
            "sd_rho": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
            "min_rho": float(s.min()) if len(s) else float("nan"),
            "max_rho": float(s.max()) if len(s) else float("nan")}


def ladder_diagnostics(ctx: ProbeContext, res: dict, scores: dict, arm: str, h: int,
                       score: np.ndarray | None = None, seed: int = 0,
                       ic: pd.Series | None = None) -> dict:
    """The M3 tripwires, run on the arm the ladder actually promotes.

    A selected arm is exactly the one whose fingerprint has to be checked: the search that
    found it would happily return a repackaged reversal factor, or a state that only works
    before the model's own training checkpoint. So the promoted arm gets the same three
    controls the primary cell got -- fingerprint vs the free factors (M3 fails at
    max |rho| >= 0.30), the contamination split at `CHECKPOINT`, and the conditional IC
    inside free-score quintiles -- plus two the primary cell did not need: the conditional IC
    inside the PRIMARY score's quintiles (does it add anything where the primary already
    ranks?) and the per-date agreement with the primary score.
    """
    sc = scores[arm][h] if score is None else score
    icser = res[arm][h]["ic"] if ic is None else ic
    out = {"arm": arm, "h": h,
           "fingerprint": fingerprint(ctx, sc, h),
           "contamination": contamination_split(icser, h, seed=seed)}
    out["fingerprint_max_abs_rho"] = float(max(abs(v) for v in out["fingerprint"].values()))
    out["fingerprint_worst_factor"] = max(out["fingerprint"],
                                          key=lambda k: abs(out["fingerprint"][k]))
    out["m3_fingerprint_pass"] = bool(out["fingerprint_max_abs_rho"] < 0.30)
    out["m3_post_ge_pre_pass"] = bool(out["contamination"]["post_ge_pre"])
    raw = float(res[arm][h]["ic_summary"]["mean_ic"]) if ic is None else float(icser.mean())
    if "free" in scores:
        c = conditional_ic(ctx, sc, scores["free"][h], h, seed=seed)
        c["raw_ic"] = raw
        c["tripwire"] = bool(c["summary"].get("mean_ic", -np.inf) > raw)
        out["conditional_ic_within_free"] = c
    if PRIMARY_ARM in scores and arm != PRIMARY_ARM:
        c = conditional_ic(ctx, sc, scores[PRIMARY_ARM][h], h, seed=seed)
        c["raw_ic"] = raw
        c["reading"] = ("IC of this arm INSIDE quintiles of the primary score: what survives "
                        "after the primary cell has already ranked the cross-section")
        out["conditional_ic_within_primary"] = c
        out["agreement_with_primary"] = score_agreement(ctx, sc, scores[PRIMARY_ARM][h], h)
    return out


# ------------------------------------------------------------- F12: the layer-depth table


def layer_depth_table(ctx: ProbeContext, res: dict, h: int) -> list[dict]:
    """IC and effective rank by depth and pooling -- the measurement F12 asks for.

    F12's observation is that the cached A0 state is post-`RMSNorm`, final layer, last
    position: a vector shaped for the next-token head, whose effective rank is 8.6 out of
    832. Whether that specialisation costs forecasting signal is answerable only by reading
    the residual stream underneath it, which is this table.
    """
    rows = []
    for layer in ("L4", "L8", "L12", "final"):
        rec: dict = {"layer": layer,
                     "position": "pre-norm residual stream" if layer != "final"
                                 else "post-norm (the next-token head's input)"}
        for pool in ("last", "mean10"):
            src = LAYER_SOURCES[(layer, pool)]
            arm = LAYER_ARMS[(layer, pool)]
            meta = array_meta(ctx, src)
            rec[f"{pool}_arm"] = arm
            rec[f"{pool}_effective_rank"] = _num(meta.get("effective_rank"))
            rec[f"{pool}_n_pcs_95pct"] = meta.get("n_pcs_for_95pct_var")
            e = res.get(arm, {}).get(h)
            s = e["ic_summary"] if e else {}
            rec[f"{pool}_ic"] = s.get("mean_ic", float("nan"))
            rec[f"{pool}_t20"] = s.get("t_nw_lag20", float("nan"))
            rec[f"{pool}_ci20"] = s.get("block_ci95_b20", [float("nan")] * 2)
            rec[f"{pool}_n_dates"] = s.get("n_dates", 0)
        rows.append(rec)
    return rows


# ------------------------------------------------------------------------ extended gates


def extended_gates(res: dict, incs: dict, h: int, ext_names, t_bar: float = 2.0) -> dict:
    """The two questions this assignment is pre-registered on, each with its own number.

    G1 -- does ANY ladder arm beat the primary cell by a PAIRED increment at lag-20 t >= 2?
      The level IC of a ladder arm being larger is not the question: two arms graded on the
      same dates share their cross-sectional noise, and only the paired difference prices
      that away. The comparison is one-sided by construction (an arm that is worse than the
      primary cell does not "beat" it), and the t is the lag-20 one because a Fama-MacBeth
      score is a trailing mean and its per-date series is autocorrelated far beyond h (F4).
    G2 -- does Kronos-small beat ITS OWN random-weight control? A capacity-ladder arm that
      cannot separate from its reservoir says the representation is doing nothing at that
      size, whatever its level IC is.
    Both are reported with the number that decided them, pass or fail. Neither is an M1
    clause: M1 is stated on the primary cell only, and nothing here can rescue it.
    """
    cands = []
    for n in ext_names:
        key = f"{n} - {PRIMARY_ARM}"
        v = incs.get(key)
        if not v or not v.get("n_dates"):
            continue
        cands.append({"arm": n, "contrast": key, "mean_increment": v["mean_increment"],
                      "t_nw_lag20": v["t_nw_lag20"], "t_nw_lag12": v["t_nw_lag12"],
                      f"t_nw_lag{h}": v[f"t_nw_lag{h}"],
                      "block_ci95_b20": v["block_ci95_b20"],
                      "share_positive": v["share_positive"],
                      "level_ic": res[n][h]["ic_summary"]["mean_ic"]})
    cands.sort(key=lambda r: -r["t_nw_lag20"])
    winners = [c for c in cands if np.isfinite(c["t_nw_lag20"]) and c["t_nw_lag20"] >= t_bar
               and c["mean_increment"] > 0]

    sm = incs.get("hT_small_pre_cpca32 - hT_small_res_cpca32", {})
    sm_t = sm.get("t_nw_lag20", float("nan"))
    sm_ci = sm.get("block_ci95_b20", [float("nan"), float("nan")])
    sm_pass = bool(np.isfinite(sm_t) and sm.get("mean_increment", 0) > 0
                   and (sm_t >= t_bar or (np.isfinite(sm_ci[0]) and sm_ci[0] > 0)))
    return {
        "h": h,
        "G1_any_extended_beats_primary": {
            "pass": bool(winners), "t_bar": t_bar,
            "clause": f"some extended arm - {PRIMARY_ARM} has mean > 0 and lag-20 NW t >= {t_bar}",
            "winners": winners,
            "best_by_t20": cands[0] if cands else {},
            "best_by_increment": max(cands, key=lambda r: r["mean_increment"]) if cands else {},
            "all_ranked": cands},
        "G2_small_beats_its_reservoir": {
            "pass": sm_pass,
            "clause": ("hT_small_pre_cpca32 - hT_small_res_cpca32 has mean > 0 and "
                       f"(lag-20 t >= {t_bar} or block-20 CI lower bound > 0)"),
            "mean_increment": sm.get("mean_increment", float("nan")),
            "t_nw_lag20": sm_t, "block_ci95_b20": sm_ci,
            "share_positive": sm.get("share_positive", float("nan")),
            "level_ic_pretrained": res.get("hT_small_pre_cpca32", {}).get(h, {})
                                      .get("ic_summary", {}).get("mean_ic", float("nan")),
            "level_ic_reservoir": res.get("hT_small_res_cpca32", {}).get(h, {})
                                     .get("ic_summary", {}).get("mean_ic", float("nan"))},
    }


# ----------------------------------------------------------------------------- report


def _layer_table_md(rows) -> str:
    head = ("| layer | state | last: eff.rank | last: IC | last: t20 | mean10: eff.rank | "
            "mean10: IC | mean10: t20 |\n|---|---|--:|--:|--:|--:|--:|--:|\n")
    body = []
    for r in rows:
        body.append(
            f"| {r['layer']} | {r['position']} | {r['last_effective_rank']:.1f} | "
            f"{_f(r['last_ic'])} | {_f(r['last_t20'], 2)} | "
            f"{r['mean10_effective_rank']:.1f} | {_f(r['mean10_ic'])} | "
            f"{_f(r['mean10_t20'], 2)} |")
    return head + "\n".join(body) + "\n"


def _sources_table_md(out: dict) -> list:
    """Every cached state this run read, with the effective rank its extraction measured.

    Effective rank is the participation ratio (exp of the spectral entropy) of the state's
    covariance: 8.6 out of 832 for the final post-norm state means the next-token head sees
    an almost eight-dimensional object. It is the one number that says how much of a 832-d
    array a PCA-32 can be losing, and it is read from the extraction sidecar rather than
    recomputed so the table and the array cannot drift apart.
    """
    used = {}
    for n in out["meta"]["arms"]:
        src = ARM_SOURCE.get(n)
        if src:
            used.setdefault(src, []).append(n)
    L = ["## Cached states read by this run (effective rank from each extraction sidecar)", "",
         "| array | rows | d | eff. rank | PCs for 95% var | arms reading it |",
         "|---|--:|--:|--:|--:|---|"]
    for src, arms in used.items():
        rec = out["array_shapes"].get(src)
        if rec is None:
            L.append(f"| `{src}` | | | | | {', '.join('`%s`' % a for a in arms)} |")
            continue
        er = rec.get("effective_rank")
        L.append(f"| `{src}` | {rec['rows']} | {rec['d']} | "
                 f"{(float(er) if er is not None else float('nan')):.1f} | "
                 f"{rec.get('n_pcs_for_95pct_var')} | "
                 f"{', '.join('`%s`' % a for a in arms)} |")
    return L + ["", "The raw 90x6 window block (`raw540_cpca32`) and the 22 free features "
                "have no extraction sidecar; they are not model states.", ""]


def _agreement_reading(out: dict, h: int) -> str:
    """What the tripwire table's agreement column implies for G1 -- read off the number.

    The implication REVERSES with the measured rho, so it cannot be asserted in advance.
    Two arms whose scores agree at a HIGH per-date rho differ by a small residual and their
    paired increment has little left to detect either way. At a LOW rho the opposite holds:
    the difference carries nearly the whole noise of both arms, its SE exceeds either arm's
    own level SE, and the paired test is the least powerful of the three -- a failed G1 is
    then a power fact about the contrast, not evidence that the two states are one object.
    Written from the numbers because asserting the first case unconditionally would explain
    away a failed gate with an agreement the data need not show.
    """
    d = out.get("diagnostics") or {}
    arm = (out.get("meta") or {}).get("best_extended_arm")
    ag = ((d.get(arm) or {}).get("agreement_with_primary") or {})
    rho = ag.get("mean_rho")
    if arm is None or rho is None or not np.isfinite(float(rho)):
        return "The last column is the promoted arm's mean per-date rank agreement with the primary score."
    rho = float(rho)
    inc = ((out.get("increments") or {}).get(str(h)) or {}).get(f"{arm} - {PRIMARY_ARM}", {})
    se_i = float(inc.get("se_nw_lag20", float("nan")))
    def _se(name):
        e = (out["arms"].get(name) or {})
        e = e.get(h, e.get(str(h)), ) or {}
        return float((e.get("ic_summary") or {}).get("se_nw_lag20", float("nan")))
    se_a, se_p = _se(arm), _se(PRIMARY_ARM)
    lead = (f"The last column prices the paired gates: `{arm}` agrees with the primary score "
            f"at mean per-date rho {rho:+.3f} over {ag.get('n_dates', 0)} dates "
            f"(sd {_f(ag.get('sd_rho'), 3)}, range {_f(ag.get('min_rho'), 3)} to "
            f"{_f(ag.get('max_rho'), 3)}). ")
    if rho >= 0.6:
        return (lead + "That is high: the two scores differ by a small noisy residual, so "
                "their paired increment has little to detect either way and a failed G1 is "
                "close to uninformative.")
    tail = (f"That is LOW, so the paired difference does NOT cancel much shared noise: its "
            f"lag-20 SE is {_f(se_i)} against {_f(se_a)} for the ladder arm's own level and "
            f"{_f(se_p)} for the primary cell's. "
            if all(np.isfinite(x) for x in (se_i, se_a, se_p)) else
            "That is LOW, so the paired difference does not cancel much shared noise. ")
    return (lead + tail + "The ladder arm and the primary cell are substantially different "
            "reads, and G1 asks the least powerful of the available questions about them: "
            "its failure bounds what 565 dates can resolve about the DIFFERENCE, and is not "
            "evidence that the difference is zero -- nor, on its own, evidence that it is "
            "positive.")


def write_extended_report(path: str, out: dict) -> None:
    """probe_n50_extended.md -- the same numbers as the JSON, as tables."""
    m = out["meta"]
    h0 = m["primary_h"]
    ext, core = m["extended_arms"], m["core_arms"]
    L = ["# NIFTY 50 probe -- Stage B representation ladder (arms A1-A7, F7, F12)", "",
         f"Written {m['written_at']}  |  seed {m['seed']}  |  family null draws "
         f"{m['null_draws']}  |  wall {m['elapsed_s']} s", "",
         f"Every ladder arm is the SAME estimator on a different read of the same "
         f"{m['n_rows']} PIT NIFTY 200 windows: frozen causal PCA (32, or 8 for the 20-column "
         f"tokenizer block), rank-within-date, `fm_scores` fitted on all "
         f"{m['n_rows']} cells, graded on the {m['n_n50_rows']} `is_n50` cells. So a "
         "difference between two arms is attributable to the representation and nothing "
         "else.", "",
         f"Primary cell (unchanged, from the core run): **h={h0}, `{PRIMARY_ARM}`**.", ""]

    g = out["gates_extended"]
    g1, g2 = g["G1_any_extended_beats_primary"], g["G2_small_beats_its_reservoir"]
    L += ["## The two pre-registered questions", "",
          "| question | verdict | deciding number |", "|---|:--|:--|"]
    b = g1["best_by_t20"]
    L += [f"| G1: {g1['clause']} | **{'PASS' if g1['pass'] else 'FAIL'}** | "
          f"best is `{b.get('arm', 'n/a')}` at increment {_f(b.get('mean_increment'))}, "
          f"t20 {_f(b.get('t_nw_lag20'), 2)}, CI20 "
          f"[{_f((b.get('block_ci95_b20') or [float('nan')] * 2)[0])}, "
          f"{_f((b.get('block_ci95_b20') or [float('nan')] * 2)[1])}]"
          + (f"; {len(g1['winners'])} arm(s) clear the bar" if g1["pass"] else "") + " |",
          f"| G2: {g2['clause']} | **{'PASS' if g2['pass'] else 'FAIL'}** | "
          f"increment {_f(g2['mean_increment'])}, t20 {_f(g2['t_nw_lag20'], 2)}, CI20 "
          f"[{_f(g2['block_ci95_b20'][0])}, {_f(g2['block_ci95_b20'][1])}] "
          f"(levels: pretrained {_f(g2['level_ic_pretrained'])}, reservoir "
          f"{_f(g2['level_ic_reservoir'])}) |", ""]

    for h in m["horizons"]:
        L += [f"## h = {h}", "", "### Extended arms (A1-A5), graded on the 50", "",
              _arm_table(out["arms"], h, ext), "",
              "### Core arms, re-run in this session (identical numbers to `probe_n50.md`)",
              "", _arm_table(out["arms"], h, core), "",
              "### Paired increments: every ladder arm minus the primary cell, and the "
              "pre-registered controls", "", _inc_table(out["increments"][str(h)], h), ""]

    L += ["## Layer depth and pooling (F12)", "",
          "Effective rank is the participation ratio of the state's covariance spectrum, "
          "read from each array's extraction sidecar; the IC is this run's, at "
          f"h={h0}, PCA-32.", "", _layer_table_md(out["layer_table"]), "",
          f"Regression check: the Stage B re-extraction of the final/last state is "
          f"{'bit-identical' if out['final_last_regression']['identical'] else 'NOT identical'}"
          f" to the cached A0 array (max abs diff "
          f"{out['final_last_regression']['max_abs_diff']:.3g}, "
          f"{out['final_last_regression']['n_differing_elements']} differing elements), so "
          "the final/last row of the table IS the primary arm and the family null does not "
          "contain the same array twice.", ""]

    L += _sources_table_md(out)
    L += ["## F7 / A6 -- reservoir seed variance (**N50-only pooled**)", "",
          "**These numbers are not comparable with anything above.** Seeds 8 and 9 exist "
          "only on the 37,800 N50 cells, so the like-for-like seed comparison is fitted AND "
          "graded on those rows: causal PCA with the basis and the ranking on N50 rows only, "
          "then `pooled_scores` (trailing-window ridge, h-session embargo). Different "
          "universe, width, scorer and fit rows from the FM-on-200 arms, so read this block "
          "for the SPREAD ACROSS SEEDS and never as a level.", ""]
    ctxt = out.get("reservoir_seeds_context") or {}
    if ctxt:
        L += [f"Scale check before reading it: in the FM-on-200 family at the same PCA-8 "
              f"width, `hT_pre_cpca8` reads {_f(ctxt.get('hT_pre_cpca8_fm200'))} and "
              f"`hT_res_cpca8` reads {_f(ctxt.get('hT_res_cpca8_fm200'))} -- PCA-8 of the "
              "final state is a poor block for BOTH arms, which is why the width is repeated "
              "below.", ""]
    for r in [out.get("reservoir_seeds"), out.get("reservoir_seeds_sensitivity")]:
        if not r:
            continue
        d = r["design"]
        L += [f"### {r['label']} -- PCA-{d['k']}, `pooled_scores` (train {d['train_window']}, "
              f"refit {d['refit_every']}, ridge {d['alpha']})", "",
              "| block | source | eff. rank | n_dates | IC | t(h) | t(12) | t(20) | "
              "block-20 CI |", "|---|---|--:|--:|--:|--:|--:|--:|:--|"]
        for k, v in r["arms"].items():
            q = v["ic_summary"]
            L += [f"| `{k}` | `{v['source']}` | {v['effective_rank']:.1f} | {q['n_dates']} | "
                  f"{_f(q['mean_ic'])} | {_f(q[f't_nw_lag{h0}'], 2)} | "
                  f"{_f(q['t_nw_lag12'], 2)} | {_f(q['t_nw_lag20'], 2)} | "
                  f"[{_f(q['block_ci95_b20'][0])}, {_f(q['block_ci95_b20'][1])}] |"]
        pb = r["pretrained_minus_best_seed"]
        L += ["", f"Reservoir over 3 seeds (7, 8, 9): mean {_f(r['reservoir_seed_mean'])}, "
              f"sd {r['reservoir_seed_sd']:.4f} "
              f"({', '.join(f'{k}={v:+.4f}' for k, v in r['reservoir_seed_ics'].items())}). "
              f"Best seed: `{r['best_reservoir_seed']}`.", "",
              f"Pretrained minus the BEST reservoir seed: {_f(pb['mean_increment'])}, "
              f"t(h) {_f(pb[f't_nw_lag{h0}'], 2)}, t(12) {_f(pb['t_nw_lag12'], 2)}, "
              f"t(20) {_f(pb['t_nw_lag20'], 2)}, block-20 CI [{_f(pb['block_ci95_b20'][0])}, "
              f"{_f(pb['block_ci95_b20'][1])}], share positive {pb['share_positive']:.2f}, "
              f"{pb['n_dates']} dates.", "",
              "| contrast | mean | t(20) | block-20 CI |", "|---|--:|--:|:--|"]
        for k, v in r["increments"].items():
            L += [f"| `{k}` | {_f(v['mean_increment'])} | {_f(v['t_nw_lag20'], 2)} | "
                  f"[{_f(v['block_ci95_b20'][0])}, {_f(v['block_ci95_b20'][1])}] |"]
        L += [""]

    d = out.get("diagnostics") or {}
    if d:
        L += ["## M3 tripwires on the arm(s) this ladder promotes", "",
              "A SELECTED arm is exactly the one whose fingerprint has to be checked: the "
              "search that found it would happily return a repackaged reversal factor, or a "
              "state that only works before the model's own training checkpoint.", "",
              "| arm | fingerprint max \\|rho\\| (factor) | M3 < 0.30 | pre-CP IC | post-CP IC "
              "| post >= pre | cond. IC in free quintiles | cond. IC in primary quintiles | "
              "mean per-date rho vs primary |", "|---|--:|:--|--:|--:|:--|--:|--:|--:|"]
        for k, v in d.items():
            cf = (v.get("conditional_ic_within_free") or {}).get("summary", {})
            cp = (v.get("conditional_ic_within_primary") or {}).get("summary", {})
            ag = v.get("agreement_with_primary") or {}
            c = v["contamination"]
            L += [f"| `{k}` | {v['fingerprint_max_abs_rho']:.3f} "
                  f"(`{v['fingerprint_worst_factor']}`) | "
                  f"{'PASS' if v['m3_fingerprint_pass'] else 'FAIL'} | "
                  f"{_f(c['pre'].get('mean_ic'))} ({c['pre'].get('n_dates', 0)}) | "
                  f"{_f(c['post'].get('mean_ic'))} ({c['post'].get('n_dates', 0)}) | "
                  f"{'PASS' if v['m3_post_ge_pre_pass'] else 'FAIL'} | "
                  f"{_f(cf.get('mean_ic'))} | {_f(cp.get('mean_ic'))} | "
                  f"{_f(ag.get('mean_rho'), 3)} |"]
        L += ["", f"Checkpoint: {list(d.values())[0]['contamination']['checkpoint']}. "
              + _agreement_reading(out, h0), ""]

    e = out["ensemble"]
    L += ["", "## A7 -- rank-average ensemble (pre-registered secondary)", ""]
    if not e.get("run"):
        L += [f"Not formed: {e.get('note', '')} Candidates {e['candidates']}, admitted "
              f"{e['admitted']}, dropped {e['dropped']}.", ""]
    else:
        s = e["eval"]["ic_summary"]
        sp = e["eval"]["spread_summary"]
        s10 = sp.get(10, sp.get("10", {}))
        L += [f"Rule: {e['rule']}. Candidates {e['candidates']}; admitted {e['admitted']}"
              + (f"; dropped {e['dropped']}" if e["dropped"] else "") + ".", "",
              f"IC {_f(s['mean_ic'])} over {s['n_dates']} dates, t(h) "
              f"{_f(s[f't_nw_lag{h0}'], 2)}, t(12) {_f(s['t_nw_lag12'], 2)}, t(20) "
              f"{_f(s['t_nw_lag20'], 2)}, block-20 CI [{_f(s['block_ci95_b20'][0])}, "
              f"{_f(s['block_ci95_b20'][1])}], share positive {s['share_positive']:.2f}; "
              f"top-10 spread {_f(s10.get('mean_ic'), 5)} at t20 "
              f"{_f(s10.get('t_nw_lag20'), 2)}.", "",
              f"Ensemble minus the primary cell: {_f(e['vs_primary']['mean_increment'])}, "
              f"t20 {_f(e['vs_primary']['t_nw_lag20'], 2)}, CI20 "
              f"[{_f(e['vs_primary']['block_ci95_b20'][0])}, "
              f"{_f(e['vs_primary']['block_ci95_b20'][1])}].", "",
              "| ensemble minus member | mean | t(20) | block-20 CI |", "|---|--:|--:|:--|"]
        for k, v in e["vs_best_member"].items():
            L += [f"| `{k}` | {_f(v['mean_increment'])} | {_f(v['t_nw_lag20'], 2)} | "
                  f"[{_f(v['block_ci95_b20'][0])}, {_f(v['block_ci95_b20'][1])}] |"]
        L += [""]

    n = out["nulls"]
    if not n:
        L += ["## Family null", "", "Not run (`--null-draws 0`).", ""]
    else:
        L += [f"## Family null over all {len(n['arms_in_family'])} arms "
              f"(core + extended), block {n['block']}, {n['n_draws']} draws", "",
              "One permutation per draw is shared by every arm, so the per-draw max |IC| is "
              "a family-wise null: it prices the search, which a single-cell null does not "
              "(F6). The scorer and the evaluation rows are the ones that produced the "
              "observed statistics (F1).", "",
              f"Family max|IC| null: mean {_f(n['family']['mean'])}, sd "
              f"{n['family']['sd']:.4f}, p95 {n['family']['p95_abs']:.4f}, LARGEST of the "
              f"{n['n_draws']} draws {n['family_max_draw']:.5f}.", "",
              f"Primary cell `{PRIMARY_ARM}`: observed {_f(n['primary']['observed'])} -> "
              f"family p = {n['primary_family_p']:.4f}, single-cell p = "
              f"{n['primary']['p_two_sided']:.4f}. It clears the largest family draw by "
              f"{100 * (abs(n['primary']['observed']) / n['family_max_draw'] - 1):.0f}%.", ""]
        be = n.get("best_extended", {})
        if be:
            L += [f"Best extended arm `{be['arm']}`: observed {_f(be['observed'])} -> family "
                  f"p = {be['family_p_two_sided']:.4f}, single-cell p = "
                  f"{be['cell_p_two_sided']:.4f}. It clears the largest family draw by "
                  f"{100 * (abs(be['observed']) / n['family_max_draw'] - 1):.0f}%. "
                  "Headroom over the largest draw is the honest reading of a p that sits on "
                  "its floor: two arms can share a p of "
                  f"{1.0 / (n['n_draws'] + 1):.4f} and be very differently far out.", ""]
        c = n.get("vs_core_only", {})
        if c:
            L += [f"Comparison with the core-only family null (C1, {c['n_draws']} draws over "
                  f"{c['n_arms']} arms): the family max distribution moved from mean "
                  f"{_f(c['family_mean'])} / p95 {c['family_p95_abs']:.4f} / sd "
                  f"{c['family_sd']:.4f} to mean {_f(n['family']['mean'])} / p95 "
                  f"{n['family']['p95_abs']:.4f} / sd {n['family']['sd']:.4f} over "
                  f"{len(n['arms_in_family'])} arms -- the wider search costs about "
                  f"{10000 * (n['family']['p95_abs'] - c['family_p95_abs']):.0f} bps of IC at "
                  f"the 95th percentile. Family p for the primary cell "
                  f"{c['family_p_two_sided']:.4f} -> {n['primary_family_p']:.4f}. "
                  + c.get("reading", ""), ""]
        L += ["| arm | null mean | null sd | null p95\\|IC\\| | observed | cell p | family p |",
              "|---|--:|--:|--:|--:|--:|--:|"]
        for k, v in n["per_arm"].items():
            L += [f"| `{k}` | {_f(v['mean'])} | {v['sd']:.4f} | {v['p95_abs']:.4f} | "
                  f"{_f(v['observed'])} | {v['p_two_sided']:.4f} | "
                  f"{n['family_p_per_arm'][k]:.4f} |"]
        L += ["", f"Every p-value is a permutation p-value with the (1+k)/(1+n) correction, "
              f"so the smallest attainable value at {n['n_draws']} draws is "
              f"{1.0 / (n['n_draws'] + 1):.4f}. A p at that floor means \"p <= "
              f"{1.0 / (n['n_draws'] + 1):.4f}\", not a small number.", ""]

    L += ["## Score exports (contract: date, symbol, score, is_n50)", "",
          "| file | arm | rows | N50 rows | dates |", "|---|---|--:|--:|--:|"]
    for k, v in out["exports"].items():
        L += [f"| `{os.path.basename(v['path'])}` | `{k}` | {v['n_rows']} | "
              f"{v['n_n50_rows']} | {v['n_dates']} |"]
    L += ["", "## Preconditions checked before any number was computed", "",
          f"* `dataset_n50.parquet` is in `windows_index.parquet` order "
          f"({out['alignment']['n_rows']} rows) -- the cached arrays carry no keys, so a "
          "reordered dataset would pair every state with the wrong cell and still produce a "
          "plausible IC.",
          "* Every array is either full-panel or covers exactly the `is_n50` row set named "
          "by its `rows_sidecar`; the row count in each `_meta.json` matches the array.",
          "* The core arms were re-fitted in this session and reproduce `probe_n50.json` "
          f"({out['core_reproduction']['note']}).", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


# -------------------------------------------------------------------------- extended CLI


def _family_p(max_abs: np.ndarray, observed: float) -> float:
    """p of |observed| against the family max|IC| null, (1+k)/(1+n)."""
    a = np.asarray(max_abs, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan")
    return float((1 + int((a >= abs(observed)).sum())) / (len(a) + 1))


def core_reproduction_check(out_dir: str, res: dict, h: int, tol: float = 1e-9) -> dict:
    """Did the core arms re-fitted here reproduce C1's `probe_n50.json`?

    The family null below is a max over core + extended arms, so it is only comparable with
    C1's core-only null if the core arms are the same numbers. Checked, not assumed.
    """
    p = os.path.join(out_dir, "probe_n50.json")
    if not os.path.exists(p):
        return {"checked": False, "note": "probe_n50.json not found"}
    prev = json.load(open(p))
    diffs = {}
    for n, per_h in prev.get("arms", {}).items():
        e = per_h.get(str(h)) or per_h.get(h)
        if not e or n not in res or h not in res[n]:
            continue
        a = float(e["ic_summary"]["mean_ic"])
        b = float(res[n][h]["ic_summary"]["mean_ic"])
        diffs[n] = {"probe_n50": a, "this_run": b, "abs_diff": abs(a - b)}
    worst = max((v["abs_diff"] for v in diffs.values()), default=float("nan"))
    return {"checked": True, "n_arms": len(diffs), "max_abs_ic_diff": worst,
            "identical": bool(np.isfinite(worst) and worst <= tol), "per_arm": diffs,
            "note": (f"{len(diffs)} core arms, max |IC difference| {worst:.2e}"
                     if np.isfinite(worst) else "no overlapping arms")}


def primary_scores_check(out_dir: str, ctx: ProbeContext, scores: np.ndarray) -> dict:
    """The rebuilt A0 primary score must equal C1's exported `scores_A0_primary.parquet`.

    The increments below are "ladder arm minus A0"; if the A0 rebuilt here were not the
    exported A0 the downstream stages read, the increments would be about a different
    baseline than the backtest is.
    """
    p = os.path.join(out_dir, "scores_A0_primary.parquet")
    if not os.path.exists(p):
        return {"checked": False, "note": "scores_A0_primary.parquet not found"}
    prev = pd.read_parquet(p)
    ok = np.isfinite(scores)
    cur = pd.DataFrame({"date": ctx.dates[ok], "symbol": ctx.ds["symbol"].to_numpy()[ok],
                        "score": scores[ok]}).sort_values(["date", "symbol"])
    prev = prev.sort_values(["date", "symbol"])
    if len(prev) != len(cur):
        return {"checked": True, "identical": False, "n_prev": int(len(prev)),
                "n_now": int(len(cur)), "note": "row counts differ"}
    same_keys = bool((prev["symbol"].to_numpy() == cur["symbol"].to_numpy()).all()
                     and (pd.to_datetime(prev["date"]).to_numpy()
                          == pd.to_datetime(cur["date"]).to_numpy()).all())
    d = float(np.abs(prev["score"].to_numpy() - cur["score"].to_numpy()).max())
    return {"checked": True, "identical": bool(same_keys and d == 0.0), "n_rows": int(len(cur)),
            "keys_match": same_keys, "max_abs_score_diff": d}


def main_extended(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="n50_probe.py extended",
        description="Stage B representation ladder (A1-A7, F7, F12) on the PIT NIFTY 50.")
    ap.add_argument("--arms", default="all",
                    help="the family the null maxes over; 'all' = core + extended")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--null-block", type=int, default=20)
    ap.add_argument("--null-pilot", type=int, default=5)
    ap.add_argument("--null-budget-min", type=float, default=75.0)
    ap.add_argument("--null-draws-fallback", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tag", default="probe_n50_extended")
    ap.add_argument("--pca-k", type=int, default=8, help="A6 (N50-only pooled) PCA width")
    ap.add_argument("--a6-extra-k", type=int, default=32,
                    help="second A6 width run as a labelled sensitivity (0 to skip)")
    ap.add_argument("--skip-a6", action="store_true")
    ap.add_argument("--no-release", action="store_true",
                    help="keep source matrices cached (faster, ~6 GB more RAM)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    names = arm_names(args.arms)
    ext = [n for n in names if ARMS[n].group == "extended"]
    core = [n for n in names if ARMS[n].group == "core"]
    if not ext:
        raise SystemExit(f"--arms {args.arms} selected no extended arm")

    t_start = time.time()
    ctx = ProbeContext.load(dataset=args.dataset, n50_dir=args.out_dir)
    align = assert_windows_alignment(ctx)
    sources = sorted({LAYER_SOURCES[k] for k in LAYER_SOURCES}
                     | set(SEED_SOURCES.values())
                     | {"n50:hT_small_pretrained_final_last",
                        "n50:hT_small_reservoir_s7_final_last",
                        "n50:hT_mini_pretrained_final_last",
                        "n50:hT_tokz_last", "n50:hT_tokz_mean10",
                        "n50:hT_base_pretrained_final_last"})
    shapes = check_array_shapes(ctx, sources)
    reg = check_final_last_regression(ctx)
    print(f"dataset {ctx.n_rows} cells, {int(ctx.is_n50.sum())} is_n50, "
          f"{len(np.unique(ctx.dates))} dates; windows_index order OK", flush=True)
    print(f"arrays checked: {len(shapes)}; Stage B final/last vs cached A0: "
          f"{'IDENTICAL' if reg['identical'] else 'DIFFERS (max %.3g)' % reg['max_abs_diff']}",
          flush=True)
    print(f"arms: {len(core)} core + {len(ext)} extended = {len(names)}\n", flush=True)

    print("building feature blocks ...", flush=True)
    build_info = build_blocks(ctx, names, release=not args.no_release)

    print("\nfitting and grading ...", flush=True)
    run = run_arms(ctx, names, horizons, scorer="fm", seed=args.seed)
    res, scores = run["eval"], run["scores"]

    repro = core_reproduction_check(args.out_dir, res, PRIMARY_H)
    a0_check = primary_scores_check(args.out_dir, ctx, scores[PRIMARY_ARM][PRIMARY_H])
    print(f"\ncore reproduction: {repro.get('note')}; A0 export match: "
          f"{a0_check.get('identical')}", flush=True)

    incs = {str(h): extended_increments(res, h, ext, seed=args.seed) for h in horizons}
    gates = extended_gates(res, incs[str(PRIMARY_H)], PRIMARY_H, ext)
    gates_secondary = {str(h): extended_gates(res, incs[str(h)], h, ext)
                       for h in horizons if h != PRIMARY_H}
    layers = layer_depth_table(ctx, res, PRIMARY_H)

    # ---- A6: reservoir seed variance, N50-only pooled ------------------------------
    seeds_block, seeds_block_alt = {}, {}
    if not args.skip_a6:
        print(f"\nF7 / A6 reservoir seed variance (N50-only pooled, PCA-{args.pca_k}) ...",
              flush=True)
        seeds_block = reservoir_seed_block(ctx, h=PRIMARY_H, k=args.pca_k, seed=args.seed)
        if args.a6_extra_k and args.a6_extra_k != args.pca_k:
            # The pre-registered A6 width is 8. If the PRETRAINED block is negative at 8 in
            # the FM-on-200 family too (`hT_pre_cpca8`), a negative A6 increment is about the
            # width, not about the seeds -- so the same comparison is repeated at a width
            # where the pretrained state is known to carry signal, and labelled a sensitivity.
            print(f"  sensitivity: same comparison at PCA-{args.a6_extra_k} ...", flush=True)
            seeds_block_alt = reservoir_seed_block(ctx, h=PRIMARY_H, k=args.a6_extra_k,
                                                   seed=args.seed)
            seeds_block_alt["label"] = f"N50-only pooled (PCA-{args.a6_extra_k} sensitivity)"

    # ---- A7: ensemble --------------------------------------------------------------
    ladder = [n for n in BASE_LADDER_ARMS if n in res]
    best_a1a2 = (max(ladder, key=lambda n: res[n][PRIMARY_H]["ic_summary"]["mean_ic"])
                 if ladder else None)
    members = [n for n in (PRIMARY_ARM, best_a1a2, "hT_small_pre_cpca32")
               if n and n in res]
    print(f"\nA7 ensemble candidates: {members} (best A1/A2 = {best_a1a2})", flush=True)
    ens = ensemble_arm(ctx, res, scores, PRIMARY_H, members, seed=args.seed)
    if ens.get("run"):
        s = ens["eval"]["ic_summary"]
        print(f"  ensemble IC {s['mean_ic']:+.4f}  t20 {s['t_nw_lag20']:+.2f}", flush=True)

    # ---- M3 tripwires on the arm the ladder promotes --------------------------------
    print("\ndiagnostics on the promoted arm(s) ...", flush=True)
    best_ext_arm = max(ext, key=lambda n: res[n][PRIMARY_H]["ic_summary"]["mean_ic"])
    diags = {best_ext_arm: ladder_diagnostics(ctx, res, scores, best_ext_arm, PRIMARY_H,
                                              seed=args.seed)}
    if ens.get("run"):
        diags["A7_ensemble"] = ladder_diagnostics(
            ctx, res, scores, PRIMARY_ARM, PRIMARY_H, score=ens["scores"],
            ic=ens["eval"]["ic"], seed=args.seed)
        diags["A7_ensemble"]["arm"] = "A7_ensemble"
    for k, v in diags.items():
        print(f"  {k:32s} fingerprint max|rho| {v['fingerprint_max_abs_rho']:.3f} "
              f"({v['fingerprint_worst_factor']}), post>=pre {v['m3_post_ge_pre_pass']}"
              + (f", rho vs primary {v['agreement_with_primary']['mean_rho']:+.3f}"
                 if "agreement_with_primary" in v else ""), flush=True)

    # ---- family null ----------------------------------------------------------------
    n_draws, pilot = args.null_draws, {}
    if args.null_pilot and n_draws:
        t0 = time.time()
        family_null(ctx, names, PRIMARY_H, args.null_pilot, block=args.null_block,
                    seed=args.seed + 991)
        per = (time.time() - t0) / args.null_pilot
        proj = per * n_draws / 60.0
        pilot = {"pilot_draws": args.null_pilot, "seconds_per_draw": round(per, 2),
                 "projected_minutes_for_requested": round(proj, 1),
                 "budget_minutes": args.null_budget_min, "n_arms": len(names)}
        print(f"\nnull pilot: {per:.2f}s/draw over {len(names)} arms -> {proj:.1f} min for "
              f"{n_draws} draws", flush=True)
        if proj > args.null_budget_min:
            n_draws = args.null_draws_fallback
            pilot["reduced_to"] = n_draws
            print(f"  over the {args.null_budget_min} min budget -> reducing to {n_draws} "
                  "draws", flush=True)
    nulls = {}
    if n_draws:
        print(f"family null: {n_draws} draws x {len(names)} arms ...", flush=True)
        t0 = time.time()
        fam = family_null(ctx, names, PRIMARY_H, n_draws, block=args.null_block,
                          seed=args.seed + 11, progress=max(1, n_draws // 8))
        obs = {k: res[k][PRIMARY_H]["ic_summary"]["mean_ic"] for k in names}
        per_arm = {k: null_pvalues(fam["arms"][k], obs[k]) for k in names}
        fam_p = {k: _family_p(fam["max_abs"], obs[k]) for k in names}
        best_ext = max(ext, key=lambda n: obs[n])
        prev_fam = {}
        pj = os.path.join(args.out_dir, "probe_n50.json")
        if os.path.exists(pj):
            pv = json.load(open(pj))
            pn = pv.get("nulls", {})
            if pn:
                prev_fam = {"n_draws": pn["n_draws"], "n_arms": len(pv["meta"]["arms"]),
                            "family_sd": pn["family"]["sd"],
                            "family_mean": pn["family"]["mean"],
                            "family_p95_abs": pn["family"]["p95_abs"],
                            "family_p_two_sided": pn["family"]["p_two_sided"],
                            "observed": pn["family"]["observed"]}
        nulls = {
            "n_draws": int(n_draws), "block": int(args.null_block), "scorer": PRIMARY_SCORER,
            "eval": "is_n50", "min_names": MIN_NAMES, "h": PRIMARY_H,
            "arms_in_family": names, "elapsed_s": round(time.time() - t0, 1),
            "primary": per_arm[PRIMARY_ARM],
            "family": null_pvalues(fam["max_abs"], obs[PRIMARY_ARM]),
            "family_max_draw": float(np.nanmax(fam["max_abs"])),
            "primary_family_p": fam_p[PRIMARY_ARM],
            "best_extended": {"arm": best_ext, "observed": obs[best_ext],
                              "family_p_two_sided": fam_p[best_ext],
                              "cell_p_two_sided": per_arm[best_ext]["p_two_sided"]},
            "per_arm": per_arm, "family_p_per_arm": fam_p,
        }
        if prev_fam:
            floor_here = 1.0 / (n_draws + 1)
            floor_there = 1.0 / (prev_fam["n_draws"] + 1)
            at_floor = [t for t, pv, fl in
                        [("this run", fam_p[PRIMARY_ARM], floor_here),
                         ("the core-only run", prev_fam["family_p_two_sided"], floor_there)]
                        if abs(pv - fl) < 1e-12]
            prev_fam["reading"] = (
                "A max over more arms is stochastically larger, so the same observed IC can "
                "only get a weakly LARGER family p in the wider family; the direction of the "
                f"change ({prev_fam['family_p_two_sided']:.4f} -> "
                f"{fam_p[PRIMARY_ARM]:.4f}) is therefore not news, its size is. "
                + (f"At the floor ({', '.join(at_floor)}), the p is a bound "
                   f"'p <= floor', not a measurement."
                   if at_floor else "Neither p is at its permutation floor."))
            nulls["vs_core_only"] = prev_fam
        npz = os.path.join(args.out_dir, f"{args.tag}_null.npz")
        arrs = {f"arm__{k}": v for k, v in fam["arms"].items()}
        arrs["max_abs"] = fam["max_abs"]
        np.savez_compressed(npz, **arrs)
        sidecar(npz.replace(".npz", "_meta.json"), vars(args),
                extra={"arms": names, "n_draws": int(n_draws), "block": int(args.null_block),
                       "scorer": PRIMARY_SCORER, "h": PRIMARY_H, "eval_mask": "is_n50",
                       "min_names": MIN_NAMES, "pilot": pilot, "keys": sorted(arrs),
                       "note": "per-draw mean IC on is_n50 rows; all arms in a draw share "
                               "one permutation, so max_abs is a family-wise null over "
                               f"{len(names)} arms (core + extended)"})
        print(f"  wrote {npz} ({nulls['elapsed_s']}s)", flush=True)

    # ---- exports --------------------------------------------------------------------
    exports = {}
    p = os.path.join(args.out_dir, f"scores_{best_ext_arm}.parquet")
    exports[best_ext_arm] = write_scores(
        p, ctx, scores[best_ext_arm][PRIMARY_H], vars(args),
        extra={"arm": best_ext_arm, "scorer": "fm", "horizon": PRIMARY_H,
               "selected_as": "best extended arm by h=5 IC on is_n50",
               "mean_ic_n50": res[best_ext_arm][PRIMARY_H]["ic_summary"]["mean_ic"],
               "contract": "date, symbol, score, is_n50; one row per scored cell",
               "fit_rows": "all eligible PIT NIFTY 200 cells"})
    if ens.get("run"):
        p = os.path.join(args.out_dir, "scores_A7_ensemble.parquet")
        exports["A7_ensemble"] = write_scores(
            p, ctx, ens["scores"], vars(args),
            extra={"arm": "A7_ensemble", "members": ens["admitted"], "scorer": "fm",
                   "horizon": PRIMARY_H, "combination": "within-date rank average",
                   "mean_ic_n50": ens["eval"]["ic_summary"]["mean_ic"],
                   "contract": "date, symbol, score, is_n50; one row per scored cell",
                   "fit_rows": "all eligible PIT NIFTY 200 cells"})
    for k, v in exports.items():
        print(f"  scores_{k}: {v['n_rows']} rows", flush=True)

    # ---- assemble --------------------------------------------------------------------
    ens_json = {k: v for k, v in ens.items() if k not in ("scores", "eval")}
    if ens.get("run"):
        ens_json["eval"] = summary_json(ens["eval"])
    out = {
        "meta": {"written_at": pd.Timestamp.now("UTC").isoformat(),
                 "arms_selector": args.arms, "arms": names, "core_arms": core,
                 "extended_arms": ext, "arm_docs": {n: ARMS[n].doc for n in names},
                 "horizons": horizons, "seed": args.seed, "null_draws": int(n_draws),
                 "null_pilot": pilot, "primary_arm": PRIMARY_ARM, "primary_h": PRIMARY_H,
                 "primary_scorer": PRIMARY_SCORER, "n_rows": int(ctx.n_rows),
                 "n_n50_rows": int(ctx.is_n50.sum()),
                 "n_dates": int(len(np.unique(ctx.dates))), "min_hist": ctx.min_hist,
                 "refit_every": ctx.refit_every, "best_extended_arm": best_ext_arm,
                 "best_a1a2_arm": best_a1a2,
                 "elapsed_s": round(time.time() - t_start, 1)},
        "alignment": align, "array_shapes": shapes, "final_last_regression": reg,
        "core_reproduction": repro, "a0_export_match": a0_check,
        "build_info": build_info,
        "arms": {n: {h: summary_json(res[n][h]) for h in horizons} for n in names},
        "increments": incs, "gates_extended": gates, "gates_secondary": gates_secondary,
        "layer_table": layers, "diagnostics": diags, "reservoir_seeds": seeds_block,
        "reservoir_seeds_sensitivity": seeds_block_alt,
        "reservoir_seeds_context": {
            "note": ("For scale: in the FM-on-200 family at the same PCA-8 width the "
                     "pretrained arm and its reservoir read as below. A6 changes universe, "
                     "scorer and fit rows all at once, so read it only for the SPREAD across "
                     "seeds, never as a level."),
            "hT_pre_cpca8_fm200": res.get("hT_pre_cpca8", {}).get(PRIMARY_H, {})
                                     .get("ic_summary", {}).get("mean_ic", float("nan")),
            "hT_res_cpca8_fm200": res.get("hT_res_cpca8", {}).get(PRIMARY_H, {})
                                     .get("ic_summary", {}).get("mean_ic", float("nan")),
            "hT_pre_cpca8_fit50only_fm": res.get("hT_pre_cpca8_fit50only", {})
                                            .get(PRIMARY_H, {}).get("ic_summary", {})
                                            .get("mean_ic", float("nan"))},
        "ensemble": ens_json,
        "nulls": nulls, "exports": exports,
    }
    jp = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    sidecar(jp.replace(".json", "_meta.json"), vars(args),
            extra={"arms": names, "extended_arms": ext, "horizons": horizons,
                   "null_draws": int(n_draws),
                   "primary": f"{PRIMARY_ARM}@{PRIMARY_SCORER} h={PRIMARY_H}",
                   "outputs": [jp, os.path.join(args.out_dir, f"{args.tag}.md")]
                              + [v["path"] for v in exports.values()]})
    write_extended_report(os.path.join(args.out_dir, f"{args.tag}.md"), out)
    print(f"\nwrote {jp}\nwrote {os.path.join(args.out_dir, args.tag + '.md')}", flush=True)
    g1, g2 = gates["G1_any_extended_beats_primary"], gates["G2_small_beats_its_reservoir"]
    print(f"G1 (any extended beats A0 at t20>=2): {'PASS' if g1['pass'] else 'FAIL'}  "
          f"G2 (small beats its reservoir): {'PASS' if g2['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    # Additive dispatch: `n50_probe.py extended [...]` runs the Stage B ladder; anything
    # else is the core run C1 published, byte-for-byte the same CLI it had.
    if len(sys.argv) > 1 and sys.argv[1] == "extended":
        main_extended(sys.argv[2:])
    else:
        main()
