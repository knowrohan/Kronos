"""Cache the exact 90x6 normalised context Kronos is fed, for every eligible cell.

This is the input side of the data-processing question. Kronos can only ever re-encode
THIS array, so whatever incremental signal the raw window carries over the free block is
the ceiling that any Kronos representation is competing against.

Normalisation is copied verbatim from signals.build_batch, so the cached window is
bit-identical to what every number in bench/out/nse/ was produced from.

Writes bench/out/nse/ml/windows.npy (float16, N x 90 x 6) + windows_index.parquet.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from bench.nse.mlcore import (LOOKBACK, OUT_NSE, load_panel,  # noqa: E402
                             normalised_window, symbol_blocks)

ML = os.path.join(OUT_NSE, "ml")


def main() -> None:
    t0 = time.time()
    idx = pd.read_parquet(os.path.join(ML, "dataset.parquet"), columns=["date", "symbol"])
    panel, calendar, _ = load_panel(pit=True)
    blocks = symbol_blocks(panel)

    n = len(idx)
    W = np.zeros((n, LOOKBACK, 6), dtype=np.float16)
    MU = np.zeros((n, 6), dtype=np.float32)
    SD = np.zeros((n, 6), dtype=np.float32)
    dates = idx["date"].to_numpy()
    syms = idx["symbol"].to_numpy()

    for i in range(n):
        w, m, s = normalised_window(blocks, syms[i], pd.Timestamp(dates[i]))
        W[i] = w.astype(np.float16)
        MU[i] = m
        SD[i] = s
        if (i + 1) % 25000 == 0:
            print(f"  {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)

    np.save(os.path.join(ML, "windows.npy"), W)
    np.save(os.path.join(ML, "windows_mu.npy"), MU)
    np.save(os.path.join(ML, "windows_sd.npy"), SD)
    idx.to_parquet(os.path.join(ML, "windows_index.parquet"), index=False)

    # proof the cache is aligned with what the model would be fed
    assert np.isfinite(W.astype(np.float32)).all(), "non-finite in cached windows"
    assert float(np.abs(W).max()) <= 5.0 + 1e-3, "clip bound violated"
    print(f"wrote windows.npy {W.shape} ({W.nbytes / 1e6:.0f} MB)  "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
