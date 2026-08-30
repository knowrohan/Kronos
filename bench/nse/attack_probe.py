"""Adversarial follow-up to probe_ht.py's positive result. Two attacks on my own design.

ATTACK 1 -- LOOKAHEAD IN THE BASIS.
probe_ht.py fits PCA on the whole sample. No labels are used, but the BASIS is chosen
with knowledge of the test period's feature distribution. For a null result that is
conservative; for a positive it is disqualifying. Here PCA is refitted causally: at each
refit point the basis comes from strictly past rows only, and is applied forward.

ATTACK 2 -- CAPACITY MISMATCH IN THE PRIMARY GATE.
The pretrained state has effective rank 8.6; the reservoir 226.9. PCA-32 therefore keeps
essentially all of the pretrained representation and discards most of the reservoir, so
"pretrained > reservoir at PCA-32" may be measuring compressibility rather than
pretraining. Here both arms are compared across a ladder of widths, including widths that
give the reservoir more components than the pretrained arm can even use.

A finding that does not survive both attacks is not a finding.
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
                              fm_scores, paired_increment, rank_within_date, summarise)

ML = os.path.join(OUT_NSE, "ml")


def causal_pca_block(M: np.ndarray, k: int, dates: np.ndarray, refit_every: int = 60,
                     min_hist: int = 120) -> np.ndarray:
    """Rank-transformed top-k PCs where the basis is fitted on PAST dates only.

    At each refit point the eigenbasis is estimated from rows whose date is strictly
    earlier than the block being projected, so nothing about the test period's feature
    distribution informs the directions used to score it.
    """
    X = M.astype(np.float32)
    ud = np.unique(dates)
    out = np.full((len(X), k), np.nan)
    pos = {d: i for i, d in enumerate(ud)}
    dcode_all = np.array([pos[d] for d in dates])

    start = min_hist
    while start < len(ud):
        stop = min(start + refit_every, len(ud))
        hist = dcode_all < start                     # strictly past rows
        blk = (dcode_all >= start) & (dcode_all < stop)
        if hist.sum() > k * 10 and blk.any():
            Xh = X[hist]
            mu = Xh.mean(axis=0, keepdims=True)
            Xc = Xh - mu
            cov = (Xc.T @ Xc) / len(Xc)
            ev, evec = np.linalg.eigh(cov)
            evec = evec[:, np.argsort(ev)[::-1]][:, :k]
            out[blk] = (X[blk] - mu) @ evec
        start = stop

    ok = np.isfinite(out).all(axis=1)
    res = np.zeros_like(out)
    if ok.any():
        res[ok] = rank_within_date(out[ok].astype(np.float64), dcode_all[ok])
    return res, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--widths", default="8,32,64,128,256")
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = pd.read_parquet(os.path.join(ML, "dataset.parquet"))
    dcode = pd.factorize(ds["date"], sort=True)[0]
    dates = ds["date"].to_numpy()
    widths = [int(x) for x in args.widths.split(",")]

    A = ds[FREE_COLS].to_numpy(dtype=np.float64)
    for j in range(A.shape[1]):
        c = A[:, j]
        b = ~np.isfinite(c)
        if b.any():
            c[b] = np.nanmedian(c[~b])
    free = rank_within_date(A, dcode)

    H = {a: np.load(os.path.join(ML, f"hT_{a}.npy")) for a in ("pretrained", "reservoir")}
    W = np.asarray(np.load(os.path.join(ML, "windows.npy"), mmap_mode="r"), dtype=np.float32)
    raw = W.reshape(len(W), -1)
    del W

    print("building CAUSAL PCA blocks (basis from past dates only) ...", flush=True)
    blocks, valid = {}, {}
    t0 = time.time()
    for k in widths:
        for arm in ("pretrained", "reservoir"):
            b, ok = causal_pca_block(H[arm], k, dates)
            blocks[f"hT_{arm}_cPCA{k}"] = b
            valid[f"hT_{arm}_cPCA{k}"] = ok
        b, ok = causal_pca_block(raw, k, dates)
        blocks[f"raw540_cPCA{k}"] = b
        valid[f"raw540_cPCA{k}"] = ok
        print(f"  k={k} done ({time.time() - t0:.0f}s)", flush=True)

    results = {}
    for h in [int(x) for x in args.horizons.split(",")]:
        y_raw = ds[f"y_h{h}"].to_numpy(dtype=float)
        base_ok = np.isfinite(y_raw)
        y = np.full(len(y_raw), np.nan)
        y[base_ok] = rank_within_date(y_raw[base_ok, None], dcode[base_ok])[:, 0]

        print(f"\n=== h={h} : CAUSAL PCA, capacity ladder ===", flush=True)
        lvl, ser = {}, {}

        ok_free = base_ok
        sc = fm_scores(free[ok_free], y[ok_free], dates[ok_free], h, ridge=args.ridge)
        ics = daily_spearman(sc, y_raw[ok_free], dates[ok_free])
        lvl["free"], ser["free"] = summarise(ics, h), ics
        print(f"  {'free':28s} IC {lvl['free']['mean_ic']:+.4f}  "
              f"t_NW {lvl['free']['t_newey_west']:+.2f}", flush=True)

        for name, X in blocks.items():
            ok = base_ok & valid[name]
            sc = fm_scores(X[ok], y[ok], dates[ok], h, ridge=args.ridge)
            ics = daily_spearman(sc, y_raw[ok], dates[ok])
            st = summarise(ics, h)
            lvl[name], ser[name] = st, ics
            print(f"  {name:28s} IC {st['mean_ic']:+.4f}  t_NW {st['t_newey_west']:+.2f}  "
                  f"CI [{st['block_ci95'][0]:+.4f},{st['block_ci95'][1]:+.4f}]  "
                  f"n={st['n_dates']}", flush=True)

        print("  --- PRIMARY GATE at matched width (causal basis) ---", flush=True)
        gates = {}
        for k in widths:
            a, b = f"hT_pretrained_cPCA{k}", f"hT_reservoir_cPCA{k}"
            inc = paired_increment(ser[a], ser[b], h, seed=args.seed)
            gates[f"cPCA{k}"] = inc
            sep = "SEPARATED" if inc["block_ci95"][0] > 0 else "not separated"
            print(f"    k={k:3d}: pretrained - reservoir {inc['mean_increment']:+.4f}  "
                  f"t_NW {inc['t_newey_west']:+.2f}  "
                  f"CI [{inc['block_ci95'][0]:+.4f},{inc['block_ci95'][1]:+.4f}]  {sep}",
                  flush=True)

        print("  --- capacity-unfairness check: reservoir at ITS best width vs "
              "pretrained at ITS best ---", flush=True)
        bp = max(widths, key=lambda k: lvl[f"hT_pretrained_cPCA{k}"]["mean_ic"])
        br = max(widths, key=lambda k: lvl[f"hT_reservoir_cPCA{k}"]["mean_ic"])
        inc = paired_increment(ser[f"hT_pretrained_cPCA{bp}"],
                               ser[f"hT_reservoir_cPCA{br}"], h, seed=args.seed)
        gates["best_vs_best"] = {"pretrained_width": bp, "reservoir_width": br, **inc}
        print(f"    pretrained@{bp} ({lvl[f'hT_pretrained_cPCA{bp}']['mean_ic']:+.4f}) "
              f"- reservoir@{br} ({lvl[f'hT_reservoir_cPCA{br}']['mean_ic']:+.4f}) = "
              f"{inc['mean_increment']:+.4f}  "
              f"CI [{inc['block_ci95'][0]:+.4f},{inc['block_ci95'][1]:+.4f}]", flush=True)

        results[f"h{h}"] = {"levels": lvl, "gates": gates}

    json.dump(results, open(os.path.join(ML, "attack_probe.json"), "w"),
              indent=2, default=float)
    print(f"\nwrote {os.path.join(ML, 'attack_probe.json')}", flush=True)


if __name__ == "__main__":
    main()
