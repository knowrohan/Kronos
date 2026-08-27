"""Rolling Kronos forecasts over a single crypto series, keeping the sample paths.

Differences from bench/nse/signals.py, and why:

  * One asset, many origins -- so there is no cross-section to rank. The question here is
    calibration and timing, not selection.
  * sample_count is pinned to 1 and the batch is replicated `paths` times instead. The
    upstream call averages its sample_count paths before returning (`np.mean(preds, axis=1)`),
    which destroys exactly the distribution a calibration test needs. Replicating the batch
    costs the same compute and hands back the individual paths.
  * The full per-step path of closes is retained, not just the terminal bar, so p10/p90
    cone coverage can be measured at every step of the horizon.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import load_kronos, set_seed, sidecar  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402

FEATS = ["open", "high", "low", "close", "volume", "amount"]
CLOSE = 3
OUT = os.path.join(ROOT, "bench", "out", "btc")


def stamps_for(idx: pd.DatetimeIndex) -> np.ndarray:
    """[minute, hour, weekday, day, month] exactly as model.kronos.calc_time_stamps builds them."""
    return np.stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month],
                    axis=-1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--predict", type=int, default=10)
    ap.add_argument("--paths", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch-origins", type=int, default=8)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="1.0 by default: truncated sampling narrows every path, which "
                         "biases a calibration test toward overconfidence before the model "
                         "gets a say.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit-origins", type=int, default=0)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--tokenizer-path", default=None)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(a.csv, parse_dates=["timestamps"]).sort_values("timestamps")
    df = df.reset_index(drop=True)
    X = df[FEATS].to_numpy(dtype=np.float32)
    ts = pd.DatetimeIndex(df["timestamps"])
    n = len(df)

    origins = list(range(a.lookback, n - a.predict, a.stride))
    if a.limit_origins:
        origins = origins[: a.limit_origins]
    print(f"{a.tag}: {n} bars {ts[0]} .. {ts[-1]} | {len(origins)} origins "
          f"lookback={a.lookback} predict={a.predict} paths={a.paths}", flush=True)

    predictor = load_kronos(max_context=a.max_context, model_path=a.model_path,
                            tokenizer_path=a.tokenizer_path)
    dev = predictor.device
    rows: list[dict] = []
    path_store: list[np.ndarray] = []
    t_start = time.time()

    for b0 in range(0, len(origins), a.batch_origins):
        chunk = origins[b0: b0 + a.batch_origins]
        ctx = np.stack([X[i - a.lookback: i] for i in chunk])            # (B, L, 6)
        mu = ctx.mean(axis=1)
        sd = ctx.std(axis=1)
        xn = np.clip((ctx - mu[:, None, :]) / (sd[:, None, :] + 1e-5), -5, 5)
        xs = np.stack([stamps_for(ts[i - a.lookback: i]) for i in chunk])
        ys = np.stack([stamps_for(ts[i: i + a.predict]) for i in chunk])

        reps = a.paths
        xr = np.repeat(xn, reps, axis=0)
        xsr = np.repeat(xs, reps, axis=0)
        ysr = np.repeat(ys, reps, axis=0)
        set_seed(a.seed * 1_000_003 + b0)
        preds = auto_regressive_inference(
            predictor.tokenizer, predictor.model,
            torch.from_numpy(xr).to(dev), torch.from_numpy(xsr).to(dev),
            torch.from_numpy(ysr).to(dev),
            max_context=a.max_context, pred_len=a.predict, clip=5,
            T=a.T, top_k=a.top_k, top_p=a.top_p, sample_count=1, verbose=False)
        # (B*paths, L+predict, 6) -> (B, paths, predict) normalized future closes
        pr = np.asarray(preds).reshape(len(chunk), reps, -1, len(FEATS))
        futn = pr[:, :, -a.predict:, CLOSE]

        mc, sc_ = mu[:, CLOSE], sd[:, CLOSE]
        # affine and monotone, so denormalising the paths and the aggregates commute
        paths_px = futn * (sc_ + 1e-5)[:, None, None] + mc[:, None, None]
        path_store.append(paths_px.astype(np.float32))

        mean_futn = futn.mean(axis=1)                                     # (B, predict)
        anchor_z = xn[:, -1, CLOSE]
        for j, i in enumerate(chunk):
            close_now = float(X[i - 1, CLOSE])
            realized = X[i: i + a.predict, CLOSE].astype(np.float64)
            term = paths_px[j, :, -1].astype(np.float64)
            rows.append({
                "origin": ts[i - 1], "idx": i, "close_now": close_now,
                "anchor_z": float(anchor_z[j]),
                "mu_close": float(mc[j]), "sigma_close": float(sc_[j]),
                # aggregates in the same (aggregate - anchor) convention as the NSE study,
                # so the reversion regression is directly comparable across markets
                "last_z": float(mean_futn[j, -1] - anchor_z[j]),
                "mean_z": float(mean_futn[j].mean() - anchor_z[j]),
                "max_z": float(mean_futn[j].max() - anchor_z[j]),
                "min_z": float(mean_futn[j].min() - anchor_z[j]),
                "pred_last_px": float(term.mean()),
                "pred_ret": float(term.mean() / close_now - 1.0),
                "path_std_px": float(term.std()),
                # share of sampled paths ending above the anchor == the model's own P(up)
                "p_up": float((term > close_now).mean()),
                "realized_last_px": float(realized[-1]),
                "realized_ret": float(realized[-1] / close_now - 1.0),
                "realized_ret_1": float(realized[0] / close_now - 1.0),
                "realized_max": float(realized.max()), "realized_min": float(realized.min()),
            })

        done = b0 + len(chunk)
        if (b0 // a.batch_origins) % 20 == 0 or done >= len(origins):
            el = time.time() - t_start
            rate = done / max(el, 1e-9)
            print(f"  {done}/{len(origins)} origins  {el:6.1f}s  {rate:5.2f}/s  "
                  f"eta {(len(origins)-done)/max(rate,1e-9)/60:5.1f}m", flush=True)

    out_sig = os.path.join(OUT, f"signals_{a.tag}.parquet")
    pd.DataFrame(rows).to_parquet(out_sig, index=False)
    np.save(os.path.join(OUT, f"paths_{a.tag}.npy"),
            np.concatenate(path_store, axis=0))
    sidecar(os.path.join(OUT, f"signals_{a.tag}_meta.json"),
            {**vars(a), "n_origins": len(origins), "n_bars": n,
             "first": str(ts[0]), "last": str(ts[-1]),
             "runtime_sec": round(time.time() - t_start, 1)})
    print(f"wrote {out_sig}  ({len(rows)} rows, {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
