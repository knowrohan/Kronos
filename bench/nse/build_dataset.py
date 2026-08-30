"""Build the shared (date, symbol) design matrix: free features + labels at h in {1,5,10,20}.

Writes bench/out/nse/ml/dataset.parquet -- the denominator of every experiment.

Then VALIDATES the pipeline against numbers already on disk, and refuses to hand the
dataset onward if it cannot reproduce them:
    reversal_1d      IC @ h=10 = +0.0157  (controls_pit.json baseline_ic, n_dates 745)
    anchor_z(-)      IC @ h=10 = -0.0092  (same)
    reversal_1d/s20  IC @ h=10 = +0.0141  (REBUILD_PLAN section 1.9 table)
A label pipeline that cannot reproduce a known IC is not a label pipeline.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bench.nse.mlcore import (DATA_NSE, HORIZONS, OUT_NSE, daily_spearman,  # noqa: E402
                             eligible_on, forward_return, free_features, load_panel,
                             price_frames, summarise, symbol_blocks)

ML = os.path.join(OUT_NSE, "ml")
os.makedirs(ML, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print("loading panel ...", flush=True)
    panel, calendar, member_of = load_panel(pit=True)
    opens, closes = price_frames(panel, calendar)
    blocks = symbol_blocks(panel)
    cal_idx = {d: i for i, d in enumerate(calendar)}

    # ---- eligible cross-section per PIT date, using signals.build_batch's exact rule
    dates = sorted(member_of)
    print(f"building eligibility over {len(dates)} PIT dates ...", flush=True)
    rows = []
    for k, t in enumerate(dates):
        syms = eligible_on(blocks, sorted(member_of[t]), t)
        if len(syms) < 20:
            continue
        rows.append(pd.DataFrame({"date": t, "symbol": syms}))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(dates)} {t.date()} n={len(syms)}", flush=True)
    idx = pd.concat(rows, ignore_index=True)
    print(f"eligible cells: {len(idx)}  dates: {idx['date'].nunique()}", flush=True)

    # ---- labels straight off the calendar, never a bare shift
    print("labels ...", flush=True)
    for h in HORIZONS:
        lab = {}
        for t in idx["date"].unique():
            si = cal_idx.get(pd.Timestamp(t))
            if si is None:
                continue
            fwd = forward_return(opens, si, h, lag=1)
            if len(fwd):
                lab[t] = fwd
        wide = pd.DataFrame(lab).T
        wide.index.name = "date"
        s = wide.stack(future_stack=True).rename(f"y_h{h}")
        s.index.names = ["date", "symbol"]
        idx = idx.merge(s.reset_index(), on=["date", "symbol"], how="left")

    # ---- free feature block
    print("free features ...", flush=True)
    free = free_features(panel, calendar)
    idx = idx.merge(free, on=["date", "symbol"], how="left")

    out = os.path.join(ML, "dataset.parquet")
    idx.to_parquet(out, index=False)
    print(f"wrote {out}  shape={idx.shape}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- validation gate ------------------------------------
    print("\n=== VALIDATION: reproduce known ICs from controls_pit.json ===", flush=True)
    ref = json.load(open(os.path.join(OUT_NSE, "controls_pit.json")))
    d = idx["date"].to_numpy()
    checks, ok = {}, True

    for name, col, sign, want in [
        ("reversal_1d", "reversal_1d", +1.0, 0.0157),
        ("anchor_reversal", "anchor_z", -1.0, -0.0092),
        ("reversal_1d/s20", "rev1_over_vol20", +1.0, 0.0141),
    ]:
        sc = sign * idx[col].to_numpy(dtype=float)
        ics = daily_spearman(sc, idx["y_h10"].to_numpy(dtype=float), d)
        got = float(ics.mean())
        stat = summarise(ics, 10)
        delta = abs(got - want)
        good = delta < 0.0015
        ok &= good
        checks[name] = {"got": got, "want": want, "delta": delta, "pass": bool(good),
                        "t_nw": stat["t_newey_west"], "n_dates": stat["n_dates"],
                        "se_nw": stat["se_newey_west"]}
        print(f"  {name:16s} got {got:+.4f}  want {want:+.4f}  "
              f"delta {delta:.4f}  t_NW {stat['t_newey_west']:+.2f}  "
              f"{'PASS' if good else 'FAIL'}", flush=True)

    # the NW-SE table the plan asks for (section 1.1), realised on our own labels
    print("\n=== realised NW SE by horizon (the honest t=3 bar) ===", flush=True)
    nw = {}
    for h in HORIZONS:
        y = idx[f"y_h{h}"].to_numpy(dtype=float)
        per_h = {}
        for col, sign in [("reversal_1d", +1.0), ("reversal_5d", +1.0),
                          ("momentum_20d", +1.0), ("momentum_60d", +1.0),
                          ("anchor_z", -1.0)]:
            ics = daily_spearman(sign * idx[col].to_numpy(dtype=float), y, d)
            st = summarise(ics, h)
            per_h[col] = st
        nw[f"h{h}"] = per_h
        se_rng = [per_h[c]["se_newey_west"] for c in per_h]
        print(f"  h={h:2d}  NW SE range {min(se_rng):.4f}-{max(se_rng):.4f}  "
              f"-> t=3 bar {3 * min(se_rng):.4f}-{3 * max(se_rng):.4f}", flush=True)

    json.dump({"checks": checks, "nw_se": nw, "n_cells": int(len(idx)),
               "n_dates": int(idx["date"].nunique()), "validated": bool(ok)},
              open(os.path.join(ML, "validation.json"), "w"), indent=2, default=float)

    if not ok:
        print("\nVALIDATION FAILED -- do not trust any arm fitted on this dataset.", flush=True)
        sys.exit(1)
    print("\nVALIDATION PASSED.", flush=True)


if __name__ == "__main__":
    main()
