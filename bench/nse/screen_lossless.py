"""Stage 1: the lossless-window screen. No model, no GPU. The cheapest decisive test.

Kronos can only ever re-encode the 90x6 window it is fed. So fit the SAME estimator on:

    free            22 free OHLCV features
    raw540          the raw normalised window, flattened -- the lossless representation
    free+raw540     does the window add anything the free block does not already have?
    free+PCA32/64   the compressed version, which is what a probe on h_T is competing with

If the LOSSLESS window adds nothing over free features, then no lossy re-encoding of it
can add anything either at this sample size, and the Kronos lane is priced before a
single GPU-second is spent. (Priced, not closed: the data-processing inequality bounds
information, not finite-sample estimators -- so a positive here is necessary, not
sufficient, for Stage 2 to be worth funding.)

A ridge on level/slope/curvature-dominated PCs is a known false-negative route, so every
feature block is also fitted with a nonlinear learner.

Gate (REBUILD_PLAN S1): incremental IC of the lossless window over the free block
< +0.007 at h in {5, 10} above the date-block null -> the prior collapses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bench.nse.mlcore import (FREE_COLS, OUT_NSE, daily_spearman,  # noqa: E402
                             oos_scores, paired_increment, rank_within_date,
                             refit_null, summarise)

ML = os.path.join(OUT_NSE, "ml")


def build_blocks(ds: pd.DataFrame, W: np.ndarray, n_pcs=(32, 64)):
    """Rank-transform within date, then assemble the feature blocks."""
    dcode = pd.factorize(ds["date"], sort=True)[0]

    free_raw = ds[FREE_COLS].to_numpy(dtype=np.float64)
    # median-impute within date before ranking, so a missing feature cannot drop a name
    for j in range(free_raw.shape[1]):
        col = free_raw[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0
    free = rank_within_date(free_raw, dcode)

    flat = W.reshape(len(W), -1).astype(np.float32)
    raw = rank_within_date(flat.astype(np.float64), dcode)

    blocks = {"free": free, "raw540": raw, "free+raw540": np.hstack([free, raw])}

    # PCA of the window, fitted on the whole sample: deliberately GENEROUS to the
    # window arm (a real deployment could not do this), so a null result is not an
    # artefact of a weak compression.
    Xc = flat - flat.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / len(Xc)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    for k in n_pcs:
        pcs = rank_within_date((Xc @ evecs[:, :k]).astype(np.float64), dcode)
        blocks[f"free+PCA{k}"] = np.hstack([free, pcs])
        blocks[f"PCA{k}"] = pcs
    return blocks, dcode


def nonlinear_oos(X, y, dates, horizon, n_folds=6, seed=0):
    """HistGradientBoosting through the identical fold structure as the ridge."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from bench.nse.mlcore import expanding_folds
    out = np.full(len(y), np.nan)
    for tr, te in expanding_folds(dates, horizon, n_folds=n_folds):
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=seed)
        m.fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--null-draws", type=int, default=0,
                    help="date-block permutation draws for the primary arms")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = pd.read_parquet(os.path.join(ML, "dataset.parquet"))
    W = np.load(os.path.join(ML, "windows.npy"), mmap_mode="r")
    W = np.asarray(W, dtype=np.float32)
    assert len(ds) == len(W), "dataset and window cache are misaligned"

    print("building feature blocks ...", flush=True)
    t0 = time.time()
    blocks, dcode = build_blocks(ds, W)
    dates = ds["date"].to_numpy()
    for k, v in blocks.items():
        print(f"  {k:14s} {v.shape}", flush=True)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)

    results = {}
    for h in [int(x) for x in args.horizons.split(",")]:
        y_raw = ds[f"y_h{h}"].to_numpy(dtype=float)
        ok = np.isfinite(y_raw)
        # rank the label within date too: the objective is cross-sectional ranking
        y = np.full(len(y_raw), np.nan)
        y[ok] = rank_within_date(y_raw[ok, None], dcode[ok])[:, 0]

        print(f"\n=== h={h}  ({ok.sum()} labelled cells) ===", flush=True)
        per_h, ic_series = {}, {}
        for name, X in blocks.items():
            t1 = time.time()
            sc = oos_scores(X[ok], y[ok], dates[ok], h)
            ics = daily_spearman(sc, y_raw[ok], dates[ok])
            st = summarise(ics, h, seed=args.seed)
            per_h[name] = st
            ic_series[name] = ics
            print(f"  ridge  {name:14s} IC {st['mean_ic']:+.4f}  "
                  f"t_NW {st['t_newey_west']:+.2f}  "
                  f"CI [{st['block_ci95'][0]:+.4f},{st['block_ci95'][1]:+.4f}]  "
                  f"({time.time() - t1:.0f}s)", flush=True)

            if args.nonlinear and name in ("free", "free+raw540", "free+PCA64"):
                t2 = time.time()
                scn = nonlinear_oos(X[ok], y[ok], dates[ok], h, seed=args.seed)
                icn = daily_spearman(scn, y_raw[ok], dates[ok])
                stn = summarise(icn, h, seed=args.seed)
                per_h[f"{name}|gbm"] = stn
                ic_series[f"{name}|gbm"] = icn
                print(f"  gbm    {name:14s} IC {stn['mean_ic']:+.4f}  "
                      f"t_NW {stn['t_newey_west']:+.2f}  ({time.time() - t2:.0f}s)", flush=True)

        # the estimand the gate is actually stated on: paired per-date differences
        incs = {}
        for a, b in [("free+raw540", "free"), ("free+PCA32", "free"),
                     ("free+PCA64", "free"), ("raw540", "free"),
                     ("free+raw540|gbm", "free|gbm")]:
            if a in ic_series and b in ic_series:
                incs[f"{a} - {b}"] = paired_increment(ic_series[a], ic_series[b], h,
                                                      seed=args.seed)
        print("  --- paired increments (the gate) ---", flush=True)
        for k, v in incs.items():
            if v.get("n_dates"):
                print(f"    {k:34s} {v['mean_increment']:+.4f}  "
                      f"t_NW {v['t_newey_west']:+.2f}  "
                      f"CI [{v['block_ci95'][0]:+.4f},{v['block_ci95'][1]:+.4f}]", flush=True)
        # ---- date-block permutation null on the primary arms ------------------
        nulls = {}
        if args.null_draws:
            for name in ("free", "free+raw540"):
                t3 = time.time()
                print(f"  null: {name} ({args.null_draws} draws, refitting alpha "
                      f"inside each) ...", flush=True)
                a = refit_null(blocks[name][ok], y[ok], dates[ok], h,
                               n_draws=args.null_draws, seed=args.seed + 11,
                               progress=max(1, args.null_draws // 4))
                a = a[np.isfinite(a)]
                nulls[name] = {
                    "n_draws": int(len(a)), "mean": float(a.mean()),
                    "sd": float(a.std(ddof=1)),
                    "p95_abs": float(np.percentile(np.abs(a), 95)),
                    "p95": float(np.percentile(a, 95)),
                    "elapsed_s": time.time() - t3,
                }
                obs = per_h[name]["mean_ic"]
                nulls[name]["observed"] = obs
                nulls[name]["exceeds_p95_abs"] = bool(abs(obs) > nulls[name]["p95_abs"])
                print(f"    null mean {a.mean():+.4f}  sd {a.std(ddof=1):.4f}  "
                      f"p95|IC| {nulls[name]['p95_abs']:.4f}  vs observed {obs:+.4f}  "
                      f"-> {'ABOVE null' if nulls[name]['exceeds_p95_abs'] else 'within null'}",
                      flush=True)
        results[f"h{h}"] = {"levels": per_h, "increments": incs, "nulls": nulls}

    json.dump(results, open(os.path.join(ML, "screen_lossless.json"), "w"),
              indent=2, default=float)

    print("\n=== S1 GATE: incremental IC of the lossless window over free ===", flush=True)
    for h in (5, 10):
        key = f"h{h}"
        if key in results:
            v = results[key]["increments"].get("free+raw540 - free", {})
            if v.get("n_dates"):
                m = v["mean_increment"]
                verdict = "PASS (fund Stage 2)" if m >= 0.007 else "FAIL (prior collapses)"
                print(f"  h={h:2d}  increment {m:+.4f}  vs bar +0.0070  -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
