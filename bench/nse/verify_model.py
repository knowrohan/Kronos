"""Empirical verification of plan section 4 claims that concern the MODEL.

Covers 4.3 (batch throughput), 4.4 (signals live in normalized space),
4.5 (preds include reconstructed context; tokenizer reconstruction error),
4.7 (sample_count collapses the ensemble), 4.11 (MPS determinism, memory ceiling).

Writes bench/out/nse/verify_model.json.  Run with --quick to skip the slow sweeps.
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

from bench.nse.common import OUT_NSE, load_kronos, set_seed, sidecar  # noqa: E402
from model.kronos import auto_regressive_inference, calc_time_stamps  # noqa: E402

FEATS = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK, PREDICT = 90, 10


def make_windows(n_windows: int, lookback: int = LOOKBACK, predict: int = PREDICT):
    """Build n_windows normalized (x, x_stamp, y_stamp) triples from the RELIANCE daily CSV.

    Returns the normalized arrays plus the per-window mean/std so we can denormalize.
    """
    df = pd.read_csv(os.path.join(ROOT, "data", "reliance_1d.csv"), encoding="utf-8-sig")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    xs, xst, yst, means, stds, raw_last_close = [], [], [], [], [], []
    total = lookback + predict
    for k in range(n_windows):
        end = len(df) - k * predict
        start = end - total
        if start < 0:
            break
        w = df.iloc[start:end].reset_index(drop=True)
        ctx = w.iloc[:lookback]
        fut = w.iloc[lookback:]
        x = ctx[FEATS].values.astype(np.float32)
        m, s = x.mean(axis=0), x.std(axis=0)
        xn = np.clip((x - m) / (s + 1e-5), -5, 5)
        xs.append(xn)
        xst.append(calc_time_stamps(ctx["timestamps"]).values.astype(np.float32))
        yst.append(calc_time_stamps(fut["timestamps"]).values.astype(np.float32))
        means.append(m)
        stds.append(s)
        raw_last_close.append(float(ctx["close"].iloc[-1]))
    return (np.stack(xs), np.stack(xst), np.stack(yst),
            np.stack(means), np.stack(stds), np.array(raw_last_close))


def run(pred, x, xs_, ys_, sample_count=1, pred_len=PREDICT, T=1.0, top_p=0.9, top_k=0):
    dev = pred.device
    return auto_regressive_inference(
        pred.tokenizer, pred.model,
        torch.from_numpy(x).to(dev), torch.from_numpy(xs_).to(dev), torch.from_numpy(ys_).to(dev),
        max_context=512, pred_len=pred_len, clip=5, T=T, top_k=top_k, top_p=top_p,
        sample_count=sample_count, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip batch sweep and memory ceiling")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(OUT_NSE, "verify_model.json"))
    args = ap.parse_args()

    os.makedirs(OUT_NSE, exist_ok=True)
    res: dict = {}
    print("loading Kronos-base (revision-pinned)...", flush=True)
    predictor = load_kronos()
    print(f"device={predictor.device}  max_context={predictor.max_context}", flush=True)
    res["device"] = str(predictor.device)
    res["n_params_model"] = int(sum(p.numel() for p in predictor.model.parameters()))
    res["n_params_tokenizer"] = int(sum(p.numel() for p in predictor.tokenizer.parameters()))

    x, xs_, ys_, means, stds, raw_close = make_windows(8)
    print(f"windows: x={x.shape}  x_stamp={xs_.shape}  y_stamp={ys_.shape}", flush=True)

    # ---------------- 4.5 returned shape + reconstruction error ----------------
    set_seed(args.seed)
    preds = run(predictor, x, xs_, ys_, sample_count=1)
    res["4.5_pred_shape"] = list(preds.shape)
    res["4.5_expected_if_context_included"] = [x.shape[0], LOOKBACK + PREDICT, len(FEATS)]
    res["4.5_context_included"] = bool(preds.shape[1] == LOOKBACK + PREDICT)

    # ---------------- 4.4 normalized space ----------------
    close_norm = preds[:, :, 3]
    res["4.4_pred_close_abs_max"] = float(np.abs(close_norm).max())
    res["4.4_pred_close_mean"] = float(close_norm.mean())
    res["4.4_pred_close_std"] = float(close_norm.std())
    res["4.4_raw_close_level_mean"] = float(raw_close.mean())
    res["4.4_is_normalized_space"] = bool(np.abs(close_norm).max() < 20.0)
    # what the same values look like once denormalized -> should land near the real price
    denorm_close = close_norm * (stds[:, None, 3] + 1e-5) + means[:, None, 3]
    res["4.4_denorm_close_mean"] = float(denorm_close.mean())
    res["4.4_denorm_vs_raw_ratio"] = float(denorm_close.mean() / raw_close.mean())

    # ---------------- 4.5b tokenizer reconstruction error ----------------
    # preds[:, LOOKBACK-1, 3] is the model's reconstruction of the last context close;
    # x[:, -1, 3] is the true (normalized) last context close.
    recon_last = preds[:, LOOKBACK - 1, 3]
    true_last = x[:, -1, 3]
    err = recon_last - true_last
    res["4.5b_recon_err_mean"] = float(err.mean())
    res["4.5b_recon_err_std"] = float(err.std())
    res["4.5b_recon_err_absmean"] = float(np.abs(err).mean())
    # scale reference: the signal itself is preds[:, -1, 3] - true_last
    sig = preds[:, -1, 3] - true_last
    res["4.5b_signal_absmean"] = float(np.abs(sig).mean())
    res["4.5b_recon_err_as_pct_of_signal"] = float(
        np.abs(err).mean() / max(np.abs(sig).mean(), 1e-12) * 100)
    # full-context reconstruction, all 4 price columns
    recon_ctx = preds[:, :LOOKBACK, :4]
    true_ctx = x[:, :, :4]
    res["4.5b_ctx_recon_rmse_norm"] = float(np.sqrt(((recon_ctx - true_ctx) ** 2).mean()))
    res["4.5b_ctx_recon_corr"] = float(np.corrcoef(recon_ctx.ravel(), true_ctx.ravel())[0, 1])

    # ---------------- 4.7 sample_count collapses the ensemble ----------------
    set_seed(args.seed)
    p_sc5 = run(predictor, x[:2], xs_[:2], ys_[:2], sample_count=5)
    res["4.7_shape_with_sample_count_5"] = list(p_sc5.shape)
    res["4.7_ensemble_returned"] = bool(p_sc5.ndim == 4)  # False => collapsed to the mean
    # workaround: N identical rows, sample_count=1  -> N distinct paths survive
    n_paths = 8
    tile = np.repeat(x[:1], n_paths, axis=0)
    tile_s = np.repeat(xs_[:1], n_paths, axis=0)
    tile_y = np.repeat(ys_[:1], n_paths, axis=0)
    set_seed(args.seed)
    p_tile = run(predictor, tile, tile_s, tile_y, sample_count=1)
    term = p_tile[:, -1, 3]
    res["4.7_workaround_path_spread_std"] = float(term.std())
    res["4.7_workaround_n_distinct_paths"] = int(len(np.unique(np.round(term, 6))))
    res["4.7_workaround_recovers_distribution"] = bool(term.std() > 1e-6)

    # ---------------- 4.11a MPS determinism ----------------
    det = {}
    for label, fn in (("same_seed", lambda: set_seed(args.seed)), ("no_reseed", lambda: None)):
        fn()
        a = run(predictor, x[:4], xs_[:4], ys_[:4], sample_count=1)
        fn()
        b = run(predictor, x[:4], xs_[:4], ys_[:4], sample_count=1)
        det[label] = {
            "bitwise_identical": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.abs(a - b).max()),
            "terminal_close_diff_absmean": float(np.abs(a[:, -1, 3] - b[:, -1, 3]).mean()),
        }
    res["4.11a_determinism"] = det
    # deterministic decoding (T -> 0 is not supported; use top_k=1) as a fallback check
    set_seed(args.seed)
    g1 = run(predictor, x[:4], xs_[:4], ys_[:4], sample_count=1, top_k=1, top_p=1.0)
    set_seed(args.seed)
    g2 = run(predictor, x[:4], xs_[:4], ys_[:4], sample_count=1, top_k=1, top_p=1.0)
    res["4.11a_greedy_topk1_identical"] = bool(np.array_equal(g1, g2))

    # ---------------- 4.3 batch throughput sweep + 4.11b memory ceiling ----------------
    if not args.quick:
        big_x, big_s, big_y, _, _, _ = make_windows(1)
        sizes = [b for b in (1, 8, 16, 32, 64, 128, 256, 384, 512) if b <= args.max_batch]
        sweep = []
        for b in sizes:
            bx = np.repeat(big_x, b, axis=0)
            bs = np.repeat(big_s, b, axis=0)
            by = np.repeat(big_y, b, axis=0)
            try:
                _ = run(predictor, bx[:min(b, 4)], bs[:min(b, 4)], by[:min(b, 4)])  # warm
                if predictor.device == "mps":
                    torch.mps.synchronize()
                t0 = time.time()
                _ = run(predictor, bx, bs, by, sample_count=1)
                if predictor.device == "mps":
                    torch.mps.synchronize()
                dt = time.time() - t0
                row = {"batch": b, "wall_sec": round(dt, 3), "ms_per_series": round(dt / b * 1000, 1)}
                if predictor.device == "mps":
                    row["mps_alloc_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
                sweep.append(row)
                print(f"  batch {b:>4d}: {dt:6.2f}s  {row['ms_per_series']:8.1f} ms/series", flush=True)
            except RuntimeError as exc:
                sweep.append({"batch": b, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                print(f"  batch {b:>4d}: FAILED {type(exc).__name__}", flush=True)
                break
        res["4.3_batch_sweep"] = sweep
        ok = [r for r in sweep if "ms_per_series" in r]
        if ok:
            best = min(ok, key=lambda r: r["ms_per_series"])
            res["4.3_best"] = best
            res["4.3_speedup_vs_batch1"] = round(ok[0]["ms_per_series"] / best["ms_per_series"], 2)
            res["4.11b_max_batch_ok"] = max(r["batch"] for r in ok)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    sidecar(args.out.replace(".json", "_meta.json"), vars(args))
    print(json.dumps(res, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
