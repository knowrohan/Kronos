"""Why is every Kronos score negative? Dump the shape of the predicted path itself.

The cross-sectional scores are dominated by a systematic downward drift: predicted close falls
roughly monotonically over the 10-step horizon for essentially every name. This script measures
that directly rather than inferring it from the last/mean/max/min aggregates.

For each of a few signal dates it records, per prediction step h = 1..10 and averaged over the
whole cross-section:
  * mean predicted close in NORMALIZED units, vs the (normalized) last context close
  * the same as an implied percentage move off the raw close
  * the REALIZED cross-sectional mean move over the same horizon, for comparison
  * the model's reconstruction of the context, to separate round-trip error from drift

Also compares decoding regimes (sampled at the repo's T=0.6 vs greedy top_k=1) on the same
windows, which tells you whether the drift is a property of the model's mode or an artifact of
stochastic sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.common import DATA_NSE, OUT_NSE, load_kronos, set_seed, sidecar  # noqa: E402
from bench.nse.signals import build_batch, future_sessions, load_inputs, stamps_for  # noqa: E402
from model.kronos import auto_regressive_inference  # noqa: E402

CLOSE = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="*", default=["2023-09-05", "2024-09-04", "2025-09-04", "2026-06-01"])
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--predict", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=os.path.join(OUT_NSE, "drift_diagnostic.json"))
    args = ap.parse_args()

    blocks, calendar = load_inputs(os.path.join(DATA_NSE, "panel_price.parquet"),
                                   os.path.join(DATA_NSE, "calendar.json"))
    symbols = sorted(blocks)
    predictor = load_kronos()
    dev = predictor.device
    regimes = {"sampled_T0.6_topp0.9": dict(T=0.6, top_k=0, top_p=0.9),
               "greedy_topk1": dict(T=1.0, top_k=1, top_p=1.0)}
    out: dict = {"regimes": {}, "dates": []}

    for regime, kw in regimes.items():
        per_date = []
        for ds in args.dates:
            t = pd.Timestamp(ds)
            if t not in [c for c in calendar]:
                cand = [c for c in calendar if c >= t]
                if not cand:
                    continue
                t = cand[0]
            ti = calendar.index(t)
            b = build_batch(blocks, symbols, t, args.lookback, 220)
            if b is None:
                continue
            X, syms, means, stds, closes = b
            fut = future_sessions(calendar, ti, args.predict)
            i0 = blocks[syms[0]]["pos"][t]
            x_stamp = stamps_for(blocks[syms[0]]["dates"][i0 + 1 - args.lookback: i0 + 1])
            y_stamp = stamps_for(fut)

            chunks = []
            for c0 in range(0, len(syms), args.batch_size):
                xb = X[c0:c0 + args.batch_size]
                set_seed(args.seed + c0)
                p = auto_regressive_inference(
                    predictor.tokenizer, predictor.model,
                    torch.from_numpy(xb).to(dev),
                    torch.from_numpy(np.repeat(x_stamp[None], len(xb), 0)).to(dev),
                    torch.from_numpy(np.repeat(y_stamp[None], len(xb), 0)).to(dev),
                    max_context=512, pred_len=args.predict, clip=5, **kw,
                    sample_count=1, verbose=False)
                chunks.append(p)
            preds = np.concatenate(chunks, axis=0)

            anchor_z = X[:, -1, CLOSE]                       # normalized last context close
            recon_z = preds[:, args.lookback - 1, CLOSE]     # model's reconstruction of it
            fut_z = preds[:, -args.predict:, CLOSE]          # (N, 10) normalized
            sd, mu = stds[:, CLOSE], means[:, CLOSE]
            fut_px = fut_z * (sd[:, None] + 1e-5) + mu[:, None]
            pct = fut_px / closes[:, None] - 1.0             # implied % move per step

            # realized cross-sectional mean move over the same horizons, close-to-close
            real = []
            for h in range(1, args.predict + 1):
                r = []
                for k, s in enumerate(syms):
                    j = blocks[s]["pos"][t] + h
                    if j < len(blocks[s]["X"]):
                        r.append(blocks[s]["X"][j, CLOSE] / closes[k] - 1.0)
                real.append(float(np.mean(r)) if r else float("nan"))

            per_date.append({
                "date": str(t.date()), "n_symbols": len(syms),
                "anchor_z_mean": float(anchor_z.mean()),
                "recon_minus_anchor_z_mean": float((recon_z - anchor_z).mean()),
                "pred_z_by_step": [float(v) for v in fut_z.mean(axis=0)],
                "pred_z_minus_anchor_by_step": [float(v) for v in (fut_z - anchor_z[:, None]).mean(axis=0)],
                "pred_pct_by_step": [float(v) for v in pct.mean(axis=0)],
                "realized_pct_by_step": real,
                "share_symbols_negative_terminal": float((pct[:, -1] < 0).mean()),
            })
            print(f"[{regime}] {t.date()} n={len(syms)} anchor_z {anchor_z.mean():+.3f} "
                  f"pred%@10 {pct[:, -1].mean() * 100:+.2f}% realized%@10 {real[-1] * 100:+.2f}% "
                  f"neg-share {(pct[:, -1] < 0).mean():.2f}", flush=True)
        out["regimes"][regime] = per_date

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    steps = np.arange(1, args.predict + 1)
    ax = axes[0]
    for regime, rows in out["regimes"].items():
        if not rows:
            continue
        m = np.mean([r["pred_pct_by_step"] for r in rows], axis=0) * 100
        ax.plot(steps, m, "-o", lw=2, label=f"predicted, {regime}")
    rows = out["regimes"].get("sampled_T0.6_topp0.9", [])
    if rows:
        ax.plot(steps, np.mean([r["realized_pct_by_step"] for r in rows], axis=0) * 100,
                "k--s", lw=2, label="realized")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("prediction step (sessions ahead)")
    ax.set_ylabel("cross-sectional mean move (%)")
    ax.set_title("Kronos predicts a monotone decline for the whole cross-section")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for regime, rows in out["regimes"].items():
        if not rows:
            continue
        m = np.mean([r["pred_z_minus_anchor_by_step"] for r in rows], axis=0)
        ax.plot(steps, m, "-o", lw=2, label=regime)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("prediction step")
    ax.set_ylabel("predicted close − last context close (σ units)")
    ax.set_title("…because the sampler regresses toward the window mean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    rows = out["regimes"].get("sampled_T0.6_topp0.9", [])
    if rows:
        a = [r["anchor_z_mean"] for r in rows]
        t10 = [r["pred_z_minus_anchor_by_step"][-1] for r in rows]
        ax.scatter(a, t10, s=70, color="#dc2626")
        for r in rows:
            ax.annotate(r["date"], (r["anchor_z_mean"], r["pred_z_minus_anchor_by_step"][-1]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        lim = np.linspace(min(a + [0]) - 0.1, max(a + [0]) + 0.1, 10)
        ax.plot(lim, -lim, "k:", lw=1.5, label="score = −anchor_z (pure reversion)")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("cross-sectional mean anchor_z on that date")
        ax.set_ylabel("mean 10-step score (σ units)")
        ax.set_title("score tracks −anchor_z across dates")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = args.out.replace(".json", ".png")
    plt.savefig(png, dpi=130)
    plt.close(fig)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    sidecar(args.out.replace(".json", "_meta.json"), vars(args))
    print(f"-> {args.out}\n-> {png}")


if __name__ == "__main__":
    main()
