"""Shared harness for the ranking-head experiments (REBUILD_PLAN Stage 1 / Stage 2).

Everything downstream -- the free-feature block, the lossless-window screen, the h_T
probe, the reservoir control -- fits through the estimator defined here, so that an
"incremental IC" is a difference of two numbers produced by identical machinery.

CORRECTNESS RULES, enforced in code rather than in a comment:
  * Labels come from the exchange calendar via ``forward_return``: open(t+1+h)/open(t+1)-1.
    Never a bare ``.shift(-h)`` -- see REBUILD_PLAN section 6 item 1.
  * The eligible (date, symbol) index is built with the SAME rule as signals.build_batch
    (traded on t, >= lookback bars, span <= max_stale), so every arm scores the identical
    cross-section and ICs are comparable across arms by construction.
  * PIT membership only. data/nse/processed/*.pkl is never read.
  * Folds are expanding and PIT-masked with an h-session embargo, so no label that
    overlaps the training window can leak into the test window.
  * Features are rank-transformed WITHIN each date and nothing else. No residualisation
    anywhere, no cross-date scaling, no full-sample moments.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.linalg as spl
import scipy.stats as sps

# Apple's Accelerate BLAS raises spurious divide-by-zero/overflow/invalid flags from its
# vectorised matmul kernel on large float64 products. Verified harmless: the product
# matches a chunked reference to 3.8e-14 relative and ridge_fit matches sklearn's
# cholesky solver to 3.8e-17 on coefficients (prediction correlation 1.0).
np.seterr(divide="ignore", over="ignore", invalid="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

DATA_NSE = os.path.join(ROOT, "data", "nse")
OUT_NSE = os.path.join(ROOT, "bench", "out", "nse")
FEATS = ["open", "high", "low", "close", "vol", "amt"]
CLOSE = 3
LOOKBACK = 90
MAX_STALE = 200
TRADING_DAYS = 252
HORIZONS = (1, 5, 10, 20)


# --------------------------------------------------------------------- loading


def load_panel(pit: bool = True):
    """Panel, calendar and PIT membership, shaped the way signals.py shapes them."""
    path = os.path.join(DATA_NSE, "panel_price_pit.parquet" if pit else "panel_price.parquet")
    panel = pd.read_parquet(path).sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["amt"] = (panel["open"] + panel["high"] + panel["low"] + panel["close"]) / 4 * panel["volume"]
    panel = panel.rename(columns={"volume": "vol"})
    with open(os.path.join(DATA_NSE, "calendar.json")) as f:
        calendar = [pd.Timestamp(d) for d in json.load(f)]
    memb = pd.read_parquet(os.path.join(DATA_NSE, "membership_pit.parquet"))
    member_of = {d: set(g["symbol"]) for d, g in memb.groupby("date")}
    return panel, calendar, member_of


def price_frames(panel: pd.DataFrame, calendar):
    opens = panel.pivot(index="date", columns="symbol", values="open").sort_index().reindex(calendar)
    closes = panel.pivot(index="date", columns="symbol", values="close").sort_index().reindex(calendar)
    return opens, closes


def forward_return(opens: pd.DataFrame, s_idx: int, horizon: int, lag: int = 1) -> pd.Series:
    """open(t+1+h)/open(t+1) - 1 -- what a strategy trading on t's close can capture.

    Identical to backtest.Market.forward_return(price="open", lag=1).
    """
    a, b = s_idx + lag, s_idx + lag + horizon
    if b >= len(opens):
        return pd.Series(dtype=float)
    return opens.iloc[b] / opens.iloc[a] - 1.0


# --------------------------------------------------------- eligibility / windows


def symbol_blocks(panel: pd.DataFrame):
    blocks = {}
    for sym, g in panel.groupby("symbol", sort=True):
        blocks[sym] = {
            "dates": g["date"].to_numpy(),
            "X": g[FEATS].to_numpy(dtype=np.float32),
            "pos": {d: i for i, d in enumerate(g["date"])},
        }
    return blocks


def eligible_on(blocks, day_syms, t, lookback=LOOKBACK, max_stale=MAX_STALE):
    """The exact eligibility test in signals.build_batch, without materialising windows."""
    out = []
    for sym in day_syms:
        b = blocks.get(sym)
        if b is None:
            continue
        i = b["pos"].get(t)
        if i is None or i + 1 < lookback:
            continue
        d0, d1 = b["dates"][i + 1 - lookback], b["dates"][i]
        if (d1 - d0) / np.timedelta64(1, "D") > max_stale:
            continue
        if not np.isfinite(b["X"][i + 1 - lookback: i + 1]).all():
            continue
        out.append(sym)
    return out


def normalised_window(blocks, sym, t, lookback=LOOKBACK):
    """The per-window z-scored, clipped 90x6 context Kronos is actually fed."""
    b = blocks[sym]
    i = b["pos"][t]
    x = b["X"][i + 1 - lookback: i + 1]
    m, s = x.mean(axis=0), x.std(axis=0)
    return np.clip((x - m) / (s + 1e-5), -5.0, 5.0), m, s


# ------------------------------------------------------------------- free block

FREE_COLS = [
    "reversal_1d", "reversal_2d", "reversal_3d", "reversal_5d", "reversal_10d",
    "momentum_10d", "momentum_20d", "momentum_60d", "momentum_90d", "momentum_120d",
    "rev1_over_vol20", "rev5_over_vol20", "mom20_over_vol20", "mom90_over_vol20",
    "trailing_vol_20", "trailing_vol_60", "parkinson_20", "anchor_z",
    "hl_range_20", "close_to_high_60", "close_to_low_60", "turnover_z_20",
]


def free_features(panel: pd.DataFrame, calendar) -> pd.DataFrame:
    """~22 free OHLCV features, all known at t's close. Long format (date, symbol, ...).

    Every column is a function of prices/volumes at or before t. anchor_z is the
    window-z-scored last close -- the mechanical part of the Kronos score -- included
    so that "does the model beat free features" is asked against a block that already
    contains the artefact, not against a strawman.
    """
    c = panel.pivot(index="date", columns="symbol", values="close").sort_index().reindex(calendar)
    h = panel.pivot(index="date", columns="symbol", values="high").sort_index().reindex(calendar)
    lo = panel.pivot(index="date", columns="symbol", values="low").sort_index().reindex(calendar)
    v = panel.pivot(index="date", columns="symbol", values="vol").sort_index().reindex(calendar)

    r1 = c / c.shift(1) - 1.0
    vol20 = r1.rolling(20).std()
    vol60 = r1.rolling(60).std()
    # Parkinson high-low volatility estimator
    park = np.sqrt((np.log(h / lo) ** 2).rolling(20).mean() / (4 * np.log(2)))
    turn = c * v
    f = {
        "reversal_1d": -(c / c.shift(1) - 1.0),
        "reversal_2d": -(c / c.shift(2) - 1.0),
        "reversal_3d": -(c / c.shift(3) - 1.0),
        "reversal_5d": -(c / c.shift(5) - 1.0),
        "reversal_10d": -(c / c.shift(10) - 1.0),
        "momentum_10d": c / c.shift(10) - 1.0,
        "momentum_20d": c / c.shift(20) - 1.0,
        "momentum_60d": c / c.shift(60) - 1.0,
        "momentum_90d": c / c.shift(90) - 1.0,
        "momentum_120d": c / c.shift(120) - 1.0,
        "rev1_over_vol20": -(c / c.shift(1) - 1.0) / (vol20 + 1e-9),
        "rev5_over_vol20": -(c / c.shift(5) - 1.0) / (vol20 + 1e-9),
        "mom20_over_vol20": (c / c.shift(20) - 1.0) / (vol20 + 1e-9),
        "mom90_over_vol20": (c / c.shift(90) - 1.0) / (vol20 + 1e-9),
        "trailing_vol_20": vol20,
        "trailing_vol_60": vol60,
        "parkinson_20": park,
        # window-z-scored last close: the anchor the sampler reverts toward
        "anchor_z": (c - c.rolling(LOOKBACK).mean()) / (c.rolling(LOOKBACK).std() + 1e-5),
        "hl_range_20": (h.rolling(20).max() - lo.rolling(20).min()) / c,
        "close_to_high_60": c / h.rolling(60).max() - 1.0,
        "close_to_low_60": c / lo.rolling(60).min() - 1.0,
        "turnover_z_20": (turn - turn.rolling(20).mean()) / (turn.rolling(20).std() + 1e-9),
    }
    out = None
    for k, df in f.items():
        s = df.stack(future_stack=True).rename(k)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out.index.names = ["date", "symbol"]
    return out.reset_index()


# ------------------------------------------------------------ rank / estimator


def rank_within_date(a: np.ndarray, dates_code: np.ndarray) -> np.ndarray:
    """Per-date rank -> uniform(-1, 1). The only scaling any arm is allowed."""
    out = np.empty_like(a, dtype=np.float64)
    for d in np.unique(dates_code):
        m = dates_code == d
        blk = a[m]
        n = blk.shape[0]
        if n < 2:
            out[m] = 0.0
            continue
        r = np.apply_along_axis(sps.rankdata, 0, blk)
        out[m] = 2.0 * (r - 0.5) / n - 1.0
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge on centred data. numpy only, no new dependency."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    n_f = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(n_f)
    return np.linalg.solve(A, Xc.T @ yc)


def causal_pca(M: np.ndarray, k: int, dates: np.ndarray, mode: str = "frozen",
               refit_every: int = 60, min_hist: int = 120, return_basis: bool = False):
    """Rank-transformed top-k PCs whose basis is fitted on PAST dates only.

    Three causal variants, because the choice changes the number materially:
      "frozen" (primary) -- the basis is estimated once on the first `min_hist` dates and
        applied forever, so every per-date coefficient a downstream Fama-MacBeth average
        mixes lives in ONE coordinate system.
      "procrustes" -- the basis is refitted every `refit_every` dates and each new basis is
        rotated onto the previous one by the orthogonal Procrustes solution, so the refits
        track the drifting covariance without rotating the coordinates under the average.
      "refit" -- the 2026-08-28 behaviour, bit-for-bit ``attack_probe.causal_pca_block``.
        Eigenvectors are sign- and rotation-indeterminate between refits, so a trailing
        mean of per-date coefficients spanning a refit boundary is incoherent and the arm
        is attenuated for a reason unrelated to look-ahead (NIFTY50_PLAN F2). Kept so the
        record's causal numbers reproduce.

    Dates before `min_hist` are never scored (no history to fit on) and are reported False
    in `ok_mask`. Returns (block_ranked, ok_mask); with return_basis=True also the list of
    {"start", "mu", "evec"} actually used, for the tests and the provenance sidecar.
    """
    if mode not in ("frozen", "procrustes", "refit"):
        raise ValueError(f"causal_pca: unknown mode {mode!r}")
    X = M.astype(np.float32)
    ud = np.unique(dates)
    out = np.full((len(X), k), np.nan)
    pos = {d: i for i, d in enumerate(ud)}
    dcode_all = np.array([pos[d] for d in dates])

    bases, prev = [], None
    start = min_hist
    while start < len(ud):
        # frozen fits once on the first min_hist dates and projects everything after it
        stop = len(ud) if mode == "frozen" else min(start + refit_every, len(ud))
        hist = dcode_all < (min_hist if mode == "frozen" else start)   # strictly past rows
        blk = (dcode_all >= start) & (dcode_all < stop)
        if hist.sum() > k * 10 and blk.any():
            Xh = X[hist]
            mu = Xh.mean(axis=0, keepdims=True)
            Xc = Xh - mu
            cov = (Xc.T @ Xc) / len(Xc)
            ev, evec = np.linalg.eigh(cov)
            evec = evec[:, np.argsort(ev)[::-1]][:, :k]
            if mode == "procrustes" and prev is not None:
                # rotate the new k-frame onto the old one; spans the same subspace, but the
                # coordinates (and hence the coefficients averaged over it) stay comparable
                R, _ = spl.orthogonal_procrustes(evec, prev)
                evec = evec @ R
            prev = evec
            out[blk] = (X[blk] - mu) @ evec
            bases.append({"start": int(start), "mu": mu, "evec": evec})
        start = stop

    ok = np.isfinite(out).all(axis=1)
    res = np.zeros_like(out)
    if ok.any():
        res[ok] = rank_within_date(out[ok].astype(np.float64), dcode_all[ok])
    return (res, ok, bases) if return_basis else (res, ok)


def expanding_folds(dates: np.ndarray, horizon: int, n_folds: int = 6, min_train: int = 120):
    """PIT-masked expanding windows with an h-session embargo between train and test.

    Returns (train_mask, test_mask) pairs over the unique sorted dates.
    """
    ud = np.unique(dates)
    n = len(ud)
    if n < min_train + n_folds * 10:
        n_folds = max(1, (n - min_train) // 20)
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        tr_end, te_end = edges[k], edges[k + 1]
        if te_end - tr_end < 5:
            continue
        # embargo: drop the last `horizon` training dates, whose labels overlap the test block
        tr_dates = ud[: max(0, tr_end - horizon)]
        te_dates = ud[tr_end:te_end]
        if len(tr_dates) < 30 or len(te_dates) < 5:
            continue
        folds.append((np.isin(dates, tr_dates), np.isin(dates, te_dates)))
    return folds


def oos_scores(X: np.ndarray, y: np.ndarray, dates: np.ndarray, horizon: int,
               alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0), n_folds: int = 6,
               inner_embargo: bool = True) -> np.ndarray:
    """Out-of-sample fitted score for every row, via expanding embargoed folds.

    Regularisation is selected INSIDE each fold on an inner split, so a permutation
    null that calls this function re-does the selection on every draw -- the
    requirement in REBUILD_PLAN section 1.1.

    The inner split carries the same h-session embargo as the outer one: the labels of
    the last `horizon` inner-training dates overlap the inner validation block, which
    tilts the alpha choice toward the block it is scored on (NIFTY50_PLAN F1 item F5).
    inner_embargo=False restores the 2026-08-28 behaviour for reproduction.
    """
    out = np.full(len(y), np.nan)
    for tr, te in expanding_folds(dates, horizon, n_folds=n_folds):
        Xtr, ytr, dtr = X[tr], y[tr], dates[tr]
        # inner split for alpha selection: last 20% of training dates
        ud = np.unique(dtr)
        i_cut = int(0.8 * len(ud))
        cut = ud[i_cut]
        # embargo: the h dates before the cut have labels running into the inner block
        emb = ud[max(0, i_cut - horizon)] if inner_embargo else cut
        itr, iva = dtr < emb, dtr >= cut
        if itr.sum() < 50 or iva.sum() < 20:
            best = alphas[len(alphas) // 2]
        else:
            best, best_ic = alphas[0], -np.inf
            for a in alphas:
                w = ridge_fit(Xtr[itr], ytr[itr], a)
                pred = Xtr[iva] @ w
                ic = mean_daily_spearman(pred, ytr[iva], dtr[iva])
                if np.isfinite(ic) and ic > best_ic:
                    best, best_ic = a, ic
        w = ridge_fit(Xtr, ytr, best)
        out[te] = X[te] @ w
    return out


def pooled_scores(X: np.ndarray, y: np.ndarray, dates: np.ndarray, horizon: int,
                  train_window: int = 500, refit_every: int = 20, alpha: float = 10.0,
                  min_train: int = 120) -> np.ndarray:
    """Pooled trailing-window ridge, refitted every `refit_every` dates.

    The block starting at unique date i is scored with a ridge fitted on the dates
    [i-horizon-train_window, i-horizon-1]: the last training date whose label
    open(s+1+h)/open(s+1) is known by the close of date i is s = i-h-1, which is the same
    h-session embargo fm_scores applies. Its sample size scales with N x T rather than
    with N, which is what a 50-name cross-section needs when p is 22 + 32 (NIFTY50_PLAN
    F3). Blocks with fewer than `min_train` training dates are left NaN.

    No intercept and no per-date centring: every metric downstream is a within-date rank
    statistic, which a per-date additive constant cannot move.
    """
    ud = np.unique(dates)
    n = len(ud)
    pos = {d: i for i, d in enumerate(ud)}
    dcode = np.array([pos[d] for d in dates])
    fin = np.isfinite(y)
    out = np.full(len(y), np.nan)
    for i in range(0, n, refit_every):
        hi = i - horizon                       # exclusive: date i-h-1 is the last usable
        lo = max(0, hi - train_window)
        if hi - lo < min_train:
            continue
        tr = (dcode >= lo) & (dcode < hi) & fin
        if tr.sum() < X.shape[1] + 2:
            continue
        w = ridge_fit(X[tr], y[tr], alpha)
        te = (dcode >= i) & (dcode < min(i + refit_every, n))
        if te.any():
            out[te] = X[te] @ w
    return out


# ------------------------------------------------------------------------- IC


def date_groups(dates: np.ndarray):
    """``(unique_dates, [row_index_array per date])`` with rows in ascending order.

    A pure speed-up of the ``for d in np.unique(dates): m = dates == d`` idiom that used to
    live in every per-date loop here: that idiom is O(n_dates x n_rows) and costs ~0.4 s per
    IC series on the 150,527-row panel, which a 300-draw x 24-arm null pays 7,200 times.
    `np.argsort(..., kind="stable")` puts each date's rows in the same ascending order a
    boolean mask does, so every downstream solve and correlation sees byte-identical inputs.
    """
    d = np.asarray(dates)
    ud, inv = np.unique(d, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(ud))
    ends = np.cumsum(counts)
    return ud, [order[e - c:e] for c, e in zip(counts, ends)]


def daily_spearman(score: np.ndarray, label: np.ndarray, dates: np.ndarray,
                   min_names: int = 20, groups=None) -> pd.Series:
    """Per-date Spearman rank IC, matching controls.compute_ic's min_names rule.

    `groups` is an optional precomputed :func:`date_groups` result for `dates`; passing it
    is what makes a 300-draw null affordable, and it changes no number.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ud, idx = date_groups(dates) if groups is None else groups
    out = {}
    for d, g in zip(ud, idx):
        gg = g[np.isfinite(score[g]) & np.isfinite(label[g])]
        if len(gg) < min_names:
            continue
        out[d] = sps.spearmanr(score[gg], label[gg]).statistic
    return pd.Series(out, dtype=float)


def mean_daily_spearman(score, label, dates, min_names=20) -> float:
    s = daily_spearman(score, label, dates, min_names)
    return float(s.mean()) if len(s) else float("nan")


def subset_ic(scores: np.ndarray, y_raw: np.ndarray, dates: np.ndarray,
              mask: np.ndarray, min_names: int = 30) -> pd.Series:
    """Per-date IC computed on a row subset only (the PIT NIFTY 50 cells).

    Fitting rows and evaluation rows differ by design -- the head is learnt on the wide
    universe and graded on the tradable subset (NIFTY50_PLAN F3) -- so the mask is applied
    to the scores AFTER they exist, never to the fit. min_names is 30 because a ~50-name
    cross-section can drop below the 20 the 200-name panel used.
    """
    m = np.asarray(mask, dtype=bool)
    return daily_spearman(np.asarray(scores, dtype=float)[m], np.asarray(y_raw, dtype=float)[m],
                          np.asarray(dates)[m], min_names=min_names)


def daily_topk_spread(score: np.ndarray, y_raw: np.ndarray, dates: np.ndarray, k: int,
                      min_names: int = 30) -> pd.Series:
    """Per date: mean y_raw of the top-k scores minus mean y_raw of every eligible name.

    This is the quantity a long-only top-k book actually captures. IC is a rank
    correlation over the whole cross-section and can be positive while the top of the
    book is not, so the plan gates on both (NIFTY50_PLAN section 4).
    """
    score = np.asarray(score, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    dates = np.asarray(dates)
    out = {}
    for d in np.unique(dates):
        m = (dates == d) & np.isfinite(score) & np.isfinite(y_raw)
        if m.sum() < min_names:
            continue
        s, yy = score[m], y_raw[m]
        top = np.argsort(-s, kind="stable")[: min(k, len(yy))]
        out[d] = float(yy[top].mean() - yy.mean())
    return pd.Series(out, dtype=float)


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """Verbatim the estimator in controls.py, so SEs are comparable to the record."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = float((e @ e) / n)
    for L in range(1, min(lag, n - 1) + 1):
        c = float((e[L:] @ e[:-L]) / n)
        var += 2.0 * (1.0 - L / (lag + 1.0)) * c
    return float(np.sqrt(max(var, 0.0) / n))


def date_block_bootstrap(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0):
    """CI for the mean of a per-date series under serial dependence (Kunsch 1989)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _nw_lags(a: np.ndarray, mean: float, horizon: int, lags) -> dict:
    """se_nw_lag{L}/t_nw_lag{L} for every requested truncation lag (None -> horizon).

    A fitted score is a trailing mean of past coefficients, so consecutive per-date ICs
    are nearly the same number and the series is autocorrelated well beyond h lags;
    reporting the ladder stops one lag being chosen silently (NIFTY50_PLAN F4).
    """
    out = {}
    for L in lags:
        LL = horizon if L is None else int(L)
        se = newey_west_se(a, LL)
        out[f"se_nw_lag{LL}"] = se
        out[f"t_nw_lag{LL}"] = mean / se if se > 0 else float("nan")
    return out


def summarise(ics: pd.Series, horizon: int, seed: int = 0, lags=(None, 12, 20)) -> dict:
    """Level statistics for a per-date IC series, at several Newey-West lags.

    The lag=horizon numbers stay under se_newey_west/t_newey_west so the record's tables
    reproduce; the ladder (`lags`, None meaning horizon) and the block-20 bootstrap are
    added because the gate lag is 20 (NIFTY50_PLAN F4).
    """
    a = ics.to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n_dates": 0}
    m = float(a.mean())
    se_nw = newey_west_se(a, horizon)
    se_naive = float(a.std(ddof=1) / np.sqrt(len(a)))
    lo, hi = date_block_bootstrap(a, block=max(horizon, 5), seed=seed)
    lo20, hi20 = date_block_bootstrap(a, block=20, seed=seed)
    out = {
        "n_dates": int(len(a)), "mean_ic": m,
        "se_newey_west": se_nw, "t_newey_west": m / se_nw if se_nw > 0 else float("nan"),
        "se_naive": se_naive, "t_naive": m / se_naive if se_naive > 0 else float("nan"),
        "block_ci95": [lo, hi], "block_ci95_b20": [lo20, hi20],
        "share_positive": float((a > 0).mean()),
    }
    out.update(_nw_lags(a, m, horizon, lags))
    return out


def paired_increment(ic_a: pd.Series, ic_b: pd.Series, horizon: int, seed: int = 0,
                     lags=(None, 12, 20)) -> dict:
    """The estimand every gate is stated on: the paired per-date difference A - B.

    Its SE is neither arm's level SE (REBUILD_PLAN section 1.1, third bullet). Same lag
    ladder and block-20 bootstrap as summarise, for the same reason (NIFTY50_PLAN F4).
    """
    j = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    d = (j["a"] - j["b"]).to_numpy()
    if len(d) == 0:
        return {"n_dates": 0}
    m = float(d.mean())
    se_nw = newey_west_se(d, horizon)
    lo, hi = date_block_bootstrap(d, block=max(horizon, 5), seed=seed)
    lo20, hi20 = date_block_bootstrap(d, block=20, seed=seed)
    out = {
        "n_dates": int(len(d)), "mean_increment": m,
        "se_newey_west": se_nw, "t_newey_west": m / se_nw if se_nw > 0 else float("nan"),
        "block_ci95": [lo, hi], "block_ci95_b20": [lo20, hi20],
        "share_positive": float((d > 0).mean()),
    }
    out.update(_nw_lags(d, m, horizon, lags))
    return out


# ----------------------------------------------------------- date-block null


def block_permute_labels(y: np.ndarray, dates: np.ndarray, block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Reassign whole BLOCKS of dates' labels onto other dates' features.

    Within-date cross-sectional structure is preserved intact and local serial
    dependence survives inside a block; only the feature-label association across
    time is destroyed. This is the null REBUILD_PLAN section 1.1 calls for -- the
    within-date shuffle in controls.py:108 destroys cross-date persistence and is
    therefore anti-conservative for exactly the persistent scores a fitted head emits.
    """
    ud = np.unique(dates)
    n_blocks = int(np.ceil(len(ud) / block))
    order = rng.permutation(n_blocks)
    src_dates = np.concatenate([ud[b * block:(b + 1) * block] for b in order])[: len(ud)]
    mapping = dict(zip(ud, src_dates))
    # group row indices by date once
    by_date: dict = {}
    for i, d in enumerate(dates):
        by_date.setdefault(d, []).append(i)
    out = np.full(len(y), np.nan)
    for d in ud:
        tgt, src = by_date[d], by_date[mapping[d]]
        k = min(len(tgt), len(src))
        # names are not aligned across dates, so take the first k of a shuffled source:
        # the label DISTRIBUTION for that date is preserved, its identity mapping is not
        sel = rng.permutation(src)[:k]
        out[np.array(tgt[:k])] = y[sel]
    return out


def refit_null(X, y, dates, horizon, n_draws=200, block=None, seed=0,
               n_folds=6, progress=None, scorer=None, eval_mask=None, min_names=20,
               inner_embargo=True):
    """Distribution of mean IC under the date-block null, refitting inside every draw.

    Regularisation selection happens inside the scorer, so it is re-done on every draw
    rather than fixed once on the real data -- otherwise the null is optimistic.

    `scorer` is the estimator the null is built from: a callable (X, y, dates, horizon)
    -> per-row scores. It MUST be the same callable that produced the observed statistic;
    a pooled fit and a Fama-MacBeth fit have different sampling variance under the null,
    so mixing them makes the resulting sigma meaningless (NIFTY50_PLAN F1).

    `eval_mask` (bool over rows) and `min_names` must ALSO match the observed statistic:
    the null IC has to be computed on the same cross-section the observed IC was. A per-date
    Spearman over ~200 names has sd ~1/sqrt(199) = 0.071; over ~50 names it is 0.14. Running
    the null on the full 200-name panel and comparing it to an IC measured on the NIFTY 50
    subset therefore halves the null sigma and doubles the apparent z -- the same
    estimator-mismatch error F1 fixes, in the cross-section dimension instead of the
    estimator dimension. The fit still sees every row; only the grading is masked, which is
    the fit-on-200 / score-on-50 design of section 4.

    `scorer=None` uses oos_scores with `n_folds` folds and `inner_embargo`. NOTE that
    oos_scores now defaults to inner_embargo=True (F5), so the default path does NOT
    reproduce null_causal.json bit-for-bit; pass inner_embargo=False for the 2026-08-28
    behaviour.
    """
    block = block or max(horizon, 5)
    rng = np.random.default_rng(seed)
    em = None if eval_mask is None else np.asarray(eval_mask, dtype=bool)

    def default_scorer(Xa, ya, da, h):
        return oos_scores(Xa, ya, da, h, n_folds=n_folds, inner_embargo=inner_embargo)

    fit = default_scorer if scorer is None else scorer
    out = []
    for k in range(n_draws):
        yp = block_permute_labels(y, dates, block, rng)
        ok = np.isfinite(yp)
        sc = fit(X[ok], yp[ok], dates[ok], horizon)
        sub = slice(None) if em is None else em[ok]
        out.append(mean_daily_spearman(np.asarray(sc, dtype=float)[sub], yp[ok][sub],
                                       dates[ok][sub], min_names=min_names))
        if progress and (k + 1) % progress == 0:
            a = np.asarray(out, dtype=float)
            print(f"    null draw {k + 1}/{n_draws}  p95|IC| "
                  f"{np.percentile(np.abs(a[np.isfinite(a)]), 95):.4f}", flush=True)
    return np.asarray(out, dtype=float)


def refit_null_family(blocks: dict, y, dates, horizon, scorer, n_draws=200, block=None,
                      seed=0, eval_mask=None, min_names=20, progress=None,
                      diagnostics=False) -> dict:
    """Date-block null run over the whole pre-registered arm family at once.

    Every arm sees the SAME permuted labels within a draw, so the per-draw maximum of
    |IC| across arms is a family-wise null: comparing the headline cell against it prices
    the >= 30-cell search that produced it, which a single-cell null does not
    (NIFTY50_PLAN F6). `scorer` is the same callable used for the observed arms (F1).

    `blocks` maps arm name -> feature matrix, or -> (matrix, valid_row_mask) for arms like
    causal_pca whose early dates have no basis. `eval_mask` (bool over rows) restricts the
    IC to a row subset -- the NIFTY 50 cells -- while the fit still uses every row, which
    is the fit-on-200/score-on-50 design of section 4.

    Returns {"arms": {name: array(n_draws) of mean IC}, "max_abs": array(n_draws)}, and with
    `diagnostics=True` also {"ar1": {name: array}, "inflation": {name: array}} -- the per-draw
    persistence of each draw's own IC series, which is how you check whether the null is even
    the same kind of object as the observed statistic (see :func:`coef_permutation_null`).
    """
    block = block or max(horizon, 5)
    rng = np.random.default_rng(seed)
    names = list(blocks)
    per_arm = {n: [] for n in names}
    diag = {n: {"ar1": [], "inflation": []} for n in names}
    max_abs = []
    em = None if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    for k in range(n_draws):
        yp = block_permute_labels(y, dates, block, rng)
        fin = np.isfinite(yp)
        draw = {}
        for n in names:
            b = blocks[n]
            X, valid = b if isinstance(b, tuple) else (b, None)
            ok = fin if valid is None else (fin & np.asarray(valid, dtype=bool))
            if ok.sum() < 2:
                draw[n] = float("nan")
                continue
            sc = scorer(X[ok], yp[ok], dates[ok], horizon)
            sub = slice(None) if em is None else em[ok]
            if diagnostics:
                ic = daily_spearman(np.asarray(sc, dtype=float)[sub], yp[ok][sub],
                                    dates[ok][sub], min_names=min_names)
                st = series_stats(ic)
                draw[n] = st["mean_ic"]
                diag[n]["ar1"].append(st["ar1"])
                diag[n]["inflation"].append(st["inflation"])
            else:
                draw[n] = mean_daily_spearman(np.asarray(sc, dtype=float)[sub], yp[ok][sub],
                                              dates[ok][sub], min_names=min_names)
        for n in names:
            per_arm[n].append(draw[n])
        vals = np.abs(np.asarray([draw[n] for n in names], dtype=float))
        max_abs.append(float(np.nanmax(vals)) if np.isfinite(vals).any() else float("nan"))
        if progress and (k + 1) % progress == 0:
            a = np.asarray(max_abs, dtype=float)
            a = a[np.isfinite(a)]
            print(f"    family null draw {k + 1}/{n_draws}  p95 max|IC| "
                  f"{np.percentile(a, 95):.4f}", flush=True)
    out = {"arms": {n: np.asarray(v, dtype=float) for n, v in per_arm.items()},
           "max_abs": np.asarray(max_abs, dtype=float)}
    if diagnostics:
        out["ar1"] = {n: np.asarray(diag[n]["ar1"], dtype=float) for n in names}
        out["inflation"] = {n: np.asarray(diag[n]["inflation"], dtype=float) for n in names}
    return out


# ------------------------------------------------- Fama-MacBeth style estimator


def fm_coeffs(X: np.ndarray, y: np.ndarray, dates: np.ndarray, ridge: float = 10.0,
              groups=None):
    """Per-date cross-sectional ridge coefficients b_t (Fama-MacBeth first stage).

    One small solve per date on that date's cross-section only, so nothing crosses
    a date boundary and the estimate is immune to the regime drift that makes a
    pooled fit transfer badly out of sample.
    """
    ud, idx = date_groups(dates) if groups is None else groups
    B = np.zeros((len(ud), X.shape[1]))
    n_f = X.shape[1]
    eye = np.eye(n_f)
    for i, g in enumerate(idx):
        Xd, yd = X[g], y[g]
        if len(yd) < max(20, 2):
            continue
        Xc = Xd - Xd.mean(axis=0, keepdims=True)
        yc = yd - yd.mean()
        B[i] = np.linalg.solve(Xc.T @ Xc + ridge * eye, Xc.T @ yc)
    return ud, B


def fm_scores(X: np.ndarray, y: np.ndarray, dates: np.ndarray, horizon: int,
              window: int = 250, ridge: float = 10.0, min_hist: int = 60,
              groups=None) -> np.ndarray:
    """Forecast with a trailing mean of past cross-sectional coefficients.

    The weight used on date t averages b_s over s in [t-window, t-horizon-1]: strictly
    past dates, with a `horizon`-session embargo so no label overlapping date t's own
    forward window can influence the weight that scores it.
    """
    groups = date_groups(dates) if groups is None else groups
    ud, B = fm_coeffs(X, y, dates, ridge=ridge, groups=groups)
    return fm_apply(X, dates, ud, B, horizon, window=window, min_hist=min_hist, groups=groups)


def fm_apply(X: np.ndarray, dates: np.ndarray, ud: np.ndarray, B: np.ndarray, horizon: int,
             window: int = 250, min_hist: int = 60, groups=None) -> np.ndarray:
    """The SECOND stage of :func:`fm_scores`, with the coefficient matrix supplied.

    Split out verbatim so the coefficient-permutation null (:func:`coef_permutation_null`)
    rebuilds the trailing-`window` mean and the scores through exactly the code path that
    produced the observed statistic, rather than a re-implementation of it.
    """
    _, idx = date_groups(dates) if groups is None else groups
    out = np.full(len(X), np.nan)
    csum = np.cumsum(np.vstack([np.zeros((1, B.shape[1])), B]), axis=0)
    for i, g in enumerate(idx):
        hi = i - horizon               # embargo
        lo = max(0, hi - window)
        if hi - lo < min_hist:
            continue
        w = (csum[hi] - csum[lo]) / (hi - lo)
        Xd = X[g]
        out[g] = (Xd - Xd.mean(axis=0, keepdims=True)) @ w
    return out


# ------------------------------------------- persistence diagnostics for an IC series


def ar1(x) -> float:
    """Lag-1 autocorrelation of a per-date series (the moment the two nulls differ on)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float("nan")
    e = a - a.mean()
    den = float(e @ e)
    return float((e[1:] @ e[:-1]) / den) if den > 0 else float("nan")


def nw_inflation(x, lag: int = 20) -> float:
    """se_nw(lag) / se_naive -- how much serial dependence widens the SE of the mean.

    A white series gives ~1.0 (a little under, since the Bartlett kernel shrinks the
    sample autocovariances); a persistent one gives >1. This is the single number that
    shows a permutation null is not the same kind of object as the observed statistic:
    if the null's IC series is white and the observed one is not, the null's sd of the
    MEAN is too small by roughly the ratio of the two inflations, and every z read off
    it is too large by the same factor (NIFTY50_PLAN F4, applied to the null itself).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return float("nan")
    se_naive = float(a.std(ddof=1) / np.sqrt(len(a)))
    se = newey_west_se(a, lag)
    return float(se / se_naive) if se_naive > 0 else float("nan")


def series_stats(ic: pd.Series, lag: int = 20) -> dict:
    """mean IC, n_dates, AR(1) and NW-lag inflation of a per-date IC series."""
    a = ic.to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return {"mean_ic": float(a.mean()) if len(a) else float("nan"),
            "n_dates": int(len(a)), "ar1": ar1(a), "inflation": nw_inflation(a, lag)}


# -------------------------------------------------- coefficient-level permutation null


def block_permute_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of ``range(n)`` built by shuffling contiguous blocks of `block` rows.

    Same construction as :func:`block_permute_labels` uses on the date axis, exposed on
    its own because the coefficient null permutes a matrix of per-date coefficients rather
    than a vector of labels.
    """
    n_blocks = int(np.ceil(n / block))
    order = rng.permutation(n_blocks)
    return np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in order])


def coef_permutation_null(X, y_fit, dates, horizon, y_grade, *, n_draws: int = 300,
                          block: int = 20, seed: int = 0, window: int = 250,
                          ridge: float = 10.0, min_hist: int = 60, eval_mask=None,
                          min_names: int = 30, lag: int = 20, progress: int | None = None,
                          keep_series: int = 0) -> dict:
    """Permutation null that destroys the signal WITHOUT destroying the score's memory.

    WHY THIS EXISTS. The date-block label null (:func:`refit_null`) permutes the labels and
    refits, so within a draw the fitted score is chasing a label series that has been
    scrambled across time: the resulting per-date IC series is close to white (AR(1) ~0.01
    on the real panel) while the OBSERVED IC series of a Fama-MacBeth score, whose weight is
    a 250-session trailing mean, has AR(1) ~0.46. The two objects are not the same kind of
    thing, and the null's sd of the mean IC is therefore ~1.8x too small -- every z read off
    it is ~1.8x too big.

    WHAT THIS DOES INSTEAD. Fit the per-date coefficients ``B`` on the REAL labels once
    (:func:`fm_coeffs`), then in each draw permute the ROWS of ``B`` in blocks of `block`
    dates and rebuild the trailing-`window` mean and the scores through :func:`fm_apply` --
    the same code path as the observed statistic -- and grade the IC against the REAL labels
    on `eval_mask`. The score on date t is still a mean of 250 coefficient vectors that
    changes by one in/one out per session, so the null IC series is as persistent as the
    observed one; what is destroyed is the correspondence between the date a coefficient was
    estimated on and the date it is used to score.

    WHAT IT IS A NULL OF -- read this before quoting a p-value from it. H0 here is "the
    coefficient time series is exchangeable in blocks", i.e. WHEN a coefficient was learnt
    carries no information about when it works. It is NOT "features and labels are
    unrelated": a signal carried by a time-STABLE mean coefficient survives the permutation
    intact, because the mean of a scrambled 250-date window of B is still an estimate of the
    same premium. So this null is conservative by construction for a stationary signal, and
    a large p from it must be read as "the timing of the coefficients is not what makes this
    work", never as "there is nothing here". The date-block label null answers the other
    question and understates its own sd; both are reported side by side for that reason.

    Returns ``{"ic_means", "ar1", "inflation", "series", "observed"}``; `ar1` and
    `inflation` are per-draw diagnostics of the draw's own IC series.
    """
    X = np.asarray(X, dtype=float)
    dates = np.asarray(dates)
    y_fit = np.asarray(y_fit, dtype=float)
    y_grade = np.asarray(y_grade, dtype=float)
    groups = date_groups(dates)
    ud, B = fm_coeffs(X, y_fit, dates, ridge=ridge, groups=groups)
    em = np.ones(len(X), dtype=bool) if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    egroups = date_groups(dates[em])
    ye = y_grade[em]

    obs = fm_apply(X, dates, ud, B, horizon, window=window, min_hist=min_hist, groups=groups)
    observed = series_stats(daily_spearman(obs[em], ye, dates[em], min_names=min_names,
                                            groups=egroups), lag)

    rng = np.random.default_rng(seed)
    means, ars, infl, series = [], [], [], []
    for k in range(n_draws):
        Bp = B[block_permute_index(len(ud), block, rng)]
        sc = fm_apply(X, dates, ud, Bp, horizon, window=window, min_hist=min_hist,
                      groups=groups)
        ic = daily_spearman(sc[em], ye, dates[em], min_names=min_names, groups=egroups)
        st = series_stats(ic, lag)
        means.append(st["mean_ic"])
        ars.append(st["ar1"])
        infl.append(st["inflation"])
        if len(series) < keep_series:
            series.append(ic)
        if progress and (k + 1) % progress == 0:
            a = np.asarray(means, dtype=float)
            print(f"    coef null draw {k + 1}/{n_draws}  sd {np.nanstd(a, ddof=1):.4f}  "
                  f"mean AR(1) {np.nanmean(ars):+.3f}", flush=True)
    return {"ic_means": np.asarray(means, dtype=float),
            "ar1": np.asarray(ars, dtype=float),
            "inflation": np.asarray(infl, dtype=float),
            "series": series, "observed": observed}


def coef_null_family(blocks: dict, y_fit, dates, horizon, y_grade, *, n_draws: int = 300,
                     block: int = 20, seed: int = 0, window: int = 250, ridge: float = 10.0,
                     min_hist: int = 60, eval_mask=None, min_names: int = 30, lag: int = 20,
                     progress: int | None = None) -> dict:
    """:func:`coef_permutation_null` over a whole arm family, one scramble per draw.

    `blocks` maps arm -> matrix or (matrix, valid_row_mask). Arms can have different sets of
    usable dates (a frozen causal PCA has no basis for the first `min_hist` dates), so the
    block scramble is drawn ONCE per draw on the global date axis and then restricted to each
    arm's own dates in that scrambled order. That keeps the per-draw maximum a family-wise
    statistic -- every arm sees the same reordering of calendar time -- without requiring the
    arms to share a row mask.

    Returns ``{"arms": {name: {...}}, "max_abs": array(n_draws)}``.
    """
    dates = np.asarray(dates)
    y_fit = np.asarray(y_fit, dtype=float)
    y_grade = np.asarray(y_grade, dtype=float)
    fin = np.isfinite(y_fit)
    em = np.ones(len(dates), dtype=bool) if eval_mask is None else np.asarray(eval_mask, bool)
    gud = np.unique(dates)

    fitted = {}
    for n, b in blocks.items():
        Xa, valid = b if isinstance(b, tuple) else (b, None)
        ok = fin if valid is None else (fin & np.asarray(valid, dtype=bool))
        Xa = np.asarray(Xa, dtype=float)[ok]
        da = dates[ok]
        g = date_groups(da)
        ud, B = fm_coeffs(Xa, y_fit[ok], da, ridge=ridge, groups=g)
        eok = em[ok]
        eg = date_groups(da[eok])
        fitted[n] = {"X": Xa, "dates": da, "groups": g, "ud": ud, "B": B, "em": eok,
                     "egroups": eg, "y": y_grade[ok][eok],
                     "pos": {d: i for i, d in enumerate(ud)}}

    rng = np.random.default_rng(seed)
    out = {n: {"ic_means": [], "ar1": [], "inflation": []} for n in blocks}
    max_abs = []
    for k in range(n_draws):
        scramble = gud[block_permute_index(len(gud), block, rng)]
        vals = []
        for n, f in fitted.items():
            order = np.asarray([f["pos"][d] for d in scramble if d in f["pos"]], dtype=int)
            sc = fm_apply(f["X"], f["dates"], f["ud"], f["B"][order], horizon, window=window,
                          min_hist=min_hist, groups=f["groups"])
            ic = daily_spearman(sc[f["em"]], f["y"], f["dates"][f["em"]],
                                min_names=min_names, groups=f["egroups"])
            st = series_stats(ic, lag)
            out[n]["ic_means"].append(st["mean_ic"])
            out[n]["ar1"].append(st["ar1"])
            out[n]["inflation"].append(st["inflation"])
            vals.append(st["mean_ic"])
        a = np.abs(np.asarray(vals, dtype=float))
        max_abs.append(float(np.nanmax(a)) if np.isfinite(a).any() else float("nan"))
        if progress and (k + 1) % progress == 0:
            m = np.asarray(max_abs, dtype=float)
            print(f"    coef family draw {k + 1}/{n_draws}  p95 max|IC| "
                  f"{np.percentile(m[np.isfinite(m)], 95):.4f}", flush=True)
    return {"arms": {n: {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()}
                     for n, v in out.items()},
            "max_abs": np.asarray(max_abs, dtype=float)}


def date_permutation_null(X, y, dates, horizon, scorer, *, n_draws: int = 300,
                          block: int = 20, seed: int = 0, eval_mask=None, valid=None,
                          min_names: int = 30, lag: int = 20, progress: int | None = None,
                          keep_series: int = 0) -> dict:
    """:func:`refit_null` with the per-draw IC SERIES diagnostics kept.

    Identical null and identical rng stream as ``refit_null(scorer=..., block=..., seed=...)``
    -- the same permuted labels in the same order -- so the two agree draw for draw; this one
    additionally returns each draw's AR(1) and lag-`lag` NW inflation, which is what shows
    the date-block null produces a nearly white IC series (and therefore an over-narrow sd
    for the mean) where the observed statistic's series is strongly persistent.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    dates = np.asarray(dates)
    em = None if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    means, ars, infl, series = [], [], [], []
    for k in range(n_draws):
        yp = block_permute_labels(y, dates, block, rng)
        ok = np.isfinite(yp) if valid is None else (np.isfinite(yp) & np.asarray(valid, bool))
        sc = np.asarray(scorer(X[ok], yp[ok], dates[ok], horizon), dtype=float)
        sub = slice(None) if em is None else em[ok]
        ic = daily_spearman(sc[sub], yp[ok][sub], dates[ok][sub], min_names=min_names)
        st = series_stats(ic, lag)
        means.append(st["mean_ic"])
        ars.append(st["ar1"])
        infl.append(st["inflation"])
        if len(series) < keep_series:
            series.append(ic)
        if progress and (k + 1) % progress == 0:
            a = np.asarray(means, dtype=float)
            print(f"    date null draw {k + 1}/{n_draws}  sd {np.nanstd(a, ddof=1):.4f}  "
                  f"mean AR(1) {np.nanmean(ars):+.3f}", flush=True)
    return {"ic_means": np.asarray(means, dtype=float),
            "ar1": np.asarray(ars, dtype=float),
            "inflation": np.asarray(infl, dtype=float), "series": series}
