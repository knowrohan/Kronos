"""Stage 2: fit the ranking head on the cached trunk state, against every control.

PRIMARY TEST -- pretrained h_T vs reservoir h_T.
Both are 832-dim blocks from the identical architecture, the identical frozen public
tokenizer, the identical windows and the identical estimator. Only the weights differ.
That makes it the one comparison in this study where the estimator's attenuation
CANCELS: whatever the fit loses, it loses equally on both arms. Every other comparison
here (h_T against free features, an increment over a 22-feature block) is attenuated by
a factor this script measures rather than assumes -- see the power calibration below.

If pretrained does not beat reservoir, pretraining bought nothing on this task, and no
amount of head capacity fixes that.

Reported with every null: the recovery curve, i.e. what solo IC the estimator would have
needed for the test to see it. A null result without its power curve is not a result.
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
                              fm_scores, paired_increment, rank_within_date,
                              refit_null, summarise)

ML = os.path.join(OUT_NSE, "ml")
FINGERPRINT = ["anchor_z", "reversal_1d", "reversal_2d", "reversal_3d", "reversal_5d",
               "reversal_10d", "momentum_10d", "momentum_20d", "momentum_60d",
               "momentum_90d", "trailing_vol_20", "parkinson_20", "rev1_over_vol20",
               "mom90_over_vol20"]
CHECKPOINT = pd.Timestamp("2025-06-30")


def pca_block(M: np.ndarray, k: int, dcode: np.ndarray) -> np.ndarray:
    """Top-k principal directions, then rank within date. PCA is fitted on the full
    sample -- deliberately generous to the representation arm, so a null is not an
    artefact of a weak compression."""
    X = M.astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / len(Xc)
    ev, evec = np.linalg.eigh(cov)
    evec = evec[:, np.argsort(ev)[::-1]]
    return rank_within_date((Xc @ evec[:, :k]).astype(np.float64), dcode)


def fit_arm(X, y, y_raw, dates, h, ridge=10.0):
    sc = fm_scores(X, y, dates, h, ridge=ridge)
    ics = daily_spearman(sc, y_raw, dates)
    return sc, ics, summarise(ics, h)


def power_curve(free, y, y_raw, dates, dcode, h, base_ic, ridge=10.0, seed=0):
    """What increment does a signal of KNOWN solo IC actually register as?

    Without this, a small measured increment is uninterpretable: it could be a weak
    representation, or a strong one seen through a lossy estimator.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for target in (0.01, 0.02, 0.04, 0.08):
        pr = target * y + np.sqrt(max(1e-9, 1 - target ** 2)) * rng.normal(size=len(y))
        pl = rank_within_date(pr[:, None], dcode)[:, 0]
        solo = summarise(daily_spearman(pl, y_raw, dates), h)["mean_ic"]
        sc = fm_scores(np.hstack([free, pl[:, None]]), y, dates, h, ridge=ridge)
        got = summarise(daily_spearman(sc, y_raw, dates), h)["mean_ic"]
        rows.append({"planted_solo_ic": solo, "combined_ic": got,
                     "increment": got - base_ic,
                     "recovery": (got - base_ic) / solo if abs(solo) > 1e-4 else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = pd.read_parquet(os.path.join(ML, "dataset.parquet"))
    dcode = pd.factorize(ds["date"], sort=True)[0]
    dates = ds["date"].to_numpy()

    A = ds[FREE_COLS].to_numpy(dtype=np.float64)
    for j in range(A.shape[1]):
        c = A[:, j]
        b = ~np.isfinite(c)
        if b.any():
            c[b] = np.nanmedian(c[~b])
    free = rank_within_date(A, dcode)

    blocks = {"free": free}
    meta = {}
    for arm in ("pretrained", "reservoir"):
        p = os.path.join(ML, f"hT_{arm}.npy")
        if not os.path.exists(p):
            print(f"!! missing {p} -- run hidden.py --arm {arm}", flush=True)
            continue
        H = np.load(p)
        assert len(H) == len(ds), f"{arm} h_T misaligned with dataset"
        mp = os.path.join(ML, f"hT_{arm}_meta.json")
        meta[arm] = json.load(open(mp)) if os.path.exists(mp) else {}
        for k in (32, 64):
            blocks[f"hT_{arm}_PCA{k}"] = pca_block(H, k, dcode)
        blocks[f"free+hT_{arm}_PCA32"] = np.hstack([free, blocks[f"hT_{arm}_PCA32"]])
        print(f"loaded hT_{arm} {H.shape}  eff_rank "
              f"{meta[arm].get('effective_rank', float('nan')):.1f}", flush=True)

    W = np.asarray(np.load(os.path.join(ML, "windows.npy"), mmap_mode="r"), dtype=np.float32)
    blocks["raw540_PCA32"] = pca_block(W.reshape(len(W), -1), 32, dcode)
    del W

    for k, v in blocks.items():
        print(f"  {k:26s} {v.shape}", flush=True)

    results = {"meta": meta}
    for h in [int(x) for x in args.horizons.split(",")]:
        y_raw = ds[f"y_h{h}"].to_numpy(dtype=float)
        ok = np.isfinite(y_raw)
        y = np.full(len(y_raw), np.nan)
        y[ok] = rank_within_date(y_raw[ok, None], dcode[ok])[:, 0]
        yo, yro, do, dco = y[ok], y_raw[ok], dates[ok], dcode[ok]

        print(f"\n=== h={h} ({ok.sum()} cells) ===", flush=True)
        lvl, ser = {}, {}
        for name, X in blocks.items():
            t0 = time.time()
            sc, ics, st = fit_arm(X[ok], yo, yro, do, h, ridge=args.ridge)
            lvl[name], ser[name] = st, ics
            print(f"  {name:26s} IC {st['mean_ic']:+.4f}  t_NW {st['t_newey_west']:+.2f}  "
                  f"CI [{st['block_ci95'][0]:+.4f},{st['block_ci95'][1]:+.4f}]  "
                  f"({time.time() - t0:.0f}s)", flush=True)

        # ---------------- PRIMARY GATE: pretrained vs reservoir -----------------
        print("  --- PRIMARY: pretrained vs reservoir (attenuation cancels) ---", flush=True)
        gates = {}
        for k in (32, 64):
            a, b = f"hT_pretrained_PCA{k}", f"hT_reservoir_PCA{k}"
            if a in ser and b in ser:
                inc = paired_increment(ser[a], ser[b], h, seed=args.seed)
                gates[f"{a} - {b}"] = inc
                verdict = ("pretrained > reservoir" if inc["mean_increment"] > 0
                           and inc["block_ci95"][0] > 0 else "NOT separated")
                print(f"    PCA{k}: {inc['mean_increment']:+.4f}  "
                      f"t_NW {inc['t_newey_west']:+.2f}  "
                      f"CI [{inc['block_ci95'][0]:+.4f},{inc['block_ci95'][1]:+.4f}]  "
                      f"-> {verdict}", flush=True)

        # ---------------- secondary increments over free ------------------------
        print("  --- secondary: over the free block (attenuated, see power) ---", flush=True)
        for a in [f"free+hT_pretrained_PCA32"]:
            if a in ser:
                inc = paired_increment(ser[a], ser["free"], h, seed=args.seed)
                gates[f"{a} - free"] = inc
                print(f"    {a} - free: {inc['mean_increment']:+.4f}  "
                      f"t_NW {inc['t_newey_west']:+.2f}  "
                      f"CI [{inc['block_ci95'][0]:+.4f},{inc['block_ci95'][1]:+.4f}]", flush=True)

        # ---------------- power calibration -------------------------------------
        print("  --- power: what a KNOWN signal registers as ---", flush=True)
        pc = power_curve(free[ok], yo, yro, do, dco, h, lvl["free"]["mean_ic"],
                         ridge=args.ridge, seed=args.seed)
        for r in pc:
            print(f"    solo {r['planted_solo_ic']:+.4f} -> increment "
                  f"{r['increment']:+.4f}  (recovery "
                  f"{100 * r['recovery'] if r['recovery'] else float('nan'):.0f}%)", flush=True)

        # ---------------- fingerprint + contamination ---------------------------
        fp, contam = {}, {}
        if "hT_pretrained_PCA32" in ser:
            sc = fm_scores(blocks["hT_pretrained_PCA32"][ok], yo, do, h, ridge=args.ridge)
            fin = np.isfinite(sc)
            for c in FINGERPRINT:
                v = ds[c].to_numpy(dtype=float)[ok]
                m = fin & np.isfinite(v)
                rho = [np.corrcoef(pd.Series(sc[m & (do == dd)]).rank(),
                                   pd.Series(v[m & (do == dd)]).rank())[0, 1]
                       for dd in np.unique(do) if (m & (do == dd)).sum() > 20]
                fp[c] = float(np.nanmean(rho))
            top = sorted(fp.items(), key=lambda kv: -abs(kv[1]))[:5]
            print("  --- fingerprint (per-date rho of fitted score vs free factors) ---",
                  flush=True)
            for c, r in top:
                print(f"    {c:20s} {r:+.3f}", flush=True)

            pre = ser["hT_pretrained_PCA32"][ser["hT_pretrained_PCA32"].index < CHECKPOINT]
            post = ser["hT_pretrained_PCA32"][ser["hT_pretrained_PCA32"].index >= CHECKPOINT]
            contam = {"pre_checkpoint": summarise(pre, h), "post_checkpoint": summarise(post, h)}
            print(f"  --- contamination split at {CHECKPOINT.date()} ---", flush=True)
            print(f"    pre  ({len(pre):3d} dates) IC {contam['pre_checkpoint'].get('mean_ic', float('nan')):+.4f}\n"
                  f"    post ({len(post):3d} dates) IC {contam['post_checkpoint'].get('mean_ic', float('nan')):+.4f}",
                  flush=True)

        nulls = {}
        if args.null_draws:
            for name in ("hT_pretrained_PCA32",):
                if name not in blocks:
                    continue
                print(f"  null: {name} ({args.null_draws} date-block draws) ...", flush=True)
                a = refit_null(blocks[name][ok], yo, do, h, n_draws=args.null_draws,
                               seed=args.seed + 11, progress=max(1, args.null_draws // 2))
                a = a[np.isfinite(a)]
                nulls[name] = {"n_draws": int(len(a)), "mean": float(a.mean()),
                               "p95_abs": float(np.percentile(np.abs(a), 95)),
                               "observed": lvl[name]["mean_ic"]}
                print(f"    p95|IC| {nulls[name]['p95_abs']:.4f} vs observed "
                      f"{lvl[name]['observed'] if 'observed' in lvl[name] else lvl[name]['mean_ic']:+.4f}",
                      flush=True)

        results[f"h{h}"] = {"levels": lvl, "gates": gates, "power": pc,
                            "fingerprint": fp, "contamination": contam, "nulls": nulls}

    json.dump(results, open(os.path.join(ML, "probe_ht.json"), "w"), indent=2, default=float)
    print(f"\nwrote {os.path.join(ML, 'probe_ht.json')}", flush=True)


if __name__ == "__main__":
    main()
