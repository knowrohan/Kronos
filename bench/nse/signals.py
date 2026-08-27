"""Batched Kronos inference -> one score per (date, symbol, signal_def, scaling).

The expensive stage. Crash-safe: one parquet part per rebalance date under
``bench/out/nse/signals_{tag}/parts/``; ``--resume`` skips dates already written.

FOUR SIGNAL DEFINITIONS, verbatim from finetune/qlib_test.py:
    last / mean / max / min   applied to the predicted future CLOSE path.

THREE SCALINGS (plan 4.4 / 4.5b), all free because they share one forward pass:
    z      f(pred_future_close) - x_norm[-1, close]
           The repo's exact definition. Lives in NORMALIZED space, i.e. units of the
           stock's own 90-bar close sigma -> a volatility-normalized score that
           systematically favours low-vol names.
    ret    (denorm(f(pred_future_close)) - close_t) / close_t
           Expected percentage return. The economically standard cross-sectional scaling.
    recon  f(pred_future_close) - pred[lookback-1, close]
           Stays entirely inside reconstruction space, so tokenizer round-trip error
           cancels instead of being folded into the score (measured at ~12% of signal
           magnitude, see bench/out/nse/verify_model.json).

CORRECTNESS RULES ENFORCED HERE
  * The context for signal date t ends at t's close. Nothing after t is read.
  * Future TIMESTAMPS (y_stamp) come from the exchange calendar, never from realized bars,
    so the last dates of the sample are still scoreable (plan 5.5).
  * Per-window normalization over the lookback slice only -- no full-sample moments.
  * Seeded per (date, chunk), so a resumed run reproduces an uninterrupted one exactly.
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
from bench.nse.common import (DATA_NSE, OUT_NSE, SIGNAL_DEFS, load_kronos,  # noqa: E402
                              set_seed, sidecar)
from model.kronos import auto_regressive_inference  # noqa: E402

FEATS = ["open", "high", "low", "close", "vol", "amt"]
CLOSE = 3
SCALINGS = ["z", "ret", "recon"]


# --------------------------------------------------------------------------- data


def load_inputs(panel_path: str, calendar_path: str):
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["amt"] = (panel["open"] + panel["high"] + panel["low"] + panel["close"]) / 4 * panel["volume"]
    panel = panel.rename(columns={"volume": "vol"})
    with open(calendar_path) as f:
        calendar = [pd.Timestamp(d) for d in json.load(f)]
    # per-symbol numpy blocks: much faster than repeated dataframe slicing
    blocks: dict[str, dict] = {}
    for sym, g in panel.groupby("symbol", sort=True):
        blocks[sym] = {
            "dates": g["date"].to_numpy(),
            "X": g[FEATS].to_numpy(dtype=np.float32),
            "pos": {d: i for i, d in enumerate(g["date"])},
        }
    return blocks, calendar


def stamps_for(dates) -> np.ndarray:
    """[minute, hour, weekday, day, month] exactly as calc_time_stamps builds them."""
    idx = pd.DatetimeIndex(dates)
    return np.stack([idx.minute, idx.hour, idx.weekday, idx.day, idx.month],
                    axis=1).astype(np.float32)


def future_sessions(calendar: list[pd.Timestamp], t_idx: int, n: int) -> list[pd.Timestamp]:
    """Next n exchange sessions after calendar[t_idx].

    Uses only the trading calendar -- which NSE publishes a year ahead -- never realized
    bars, so dates at the very end of the sample stay scoreable (plan 5.5). Past the end of
    the known calendar we extend with weekdays and the caller records that it happened.
    """
    out = calendar[t_idx + 1: t_idx + 1 + n]
    if len(out) < n:
        last = out[-1] if out else calendar[t_idx]
        extra = pd.bdate_range(last + pd.Timedelta(days=1), periods=n - len(out))
        out = list(out) + list(extra)
    return list(out)


def build_batch(blocks, symbols, t, lookback, max_stale):
    """Assemble normalized contexts for every eligible symbol on date t.

    Eligible = traded on t, has >= lookback bars at or before t, and those bars are not
    stitched across a long suspension (span <= max_stale sessions).
    """
    xs, syms, means, stds, closes = [], [], [], [], []
    for sym in symbols:
        b = blocks[sym]
        i = b["pos"].get(t)
        if i is None or i + 1 < lookback:
            continue
        sl = slice(i + 1 - lookback, i + 1)
        d0, d1 = b["dates"][sl.start], b["dates"][i]
        if (d1 - d0) / np.timedelta64(1, "D") > max_stale:
            continue
        x = b["X"][sl]
        if not np.isfinite(x).all():
            continue
        m, s = x.mean(axis=0), x.std(axis=0)
        xs.append(np.clip((x - m) / (s + 1e-5), -5.0, 5.0))
        syms.append(sym)
        means.append(m)
        stds.append(s)
        closes.append(float(x[-1, CLOSE]))
    if not xs:
        return None
    return (np.stack(xs), syms, np.stack(means), np.stack(stds), np.array(closes, dtype=np.float64))


# --------------------------------------------------------------------------- scoring


def score_block(preds, x_norm, means, stds, closes, lookback, predict):
    """preds: (B, lookback+predict, 6) in NORMALIZED space -> dict of 12 score arrays."""
    fut = preds[:, -predict:, CLOSE]                 # (B, predict) normalized closes
    anchor_z = x_norm[:, -1, CLOSE]                  # repo's `last_day_close`, normalized
    anchor_recon = preds[:, lookback - 1, CLOSE]     # model's own reconstruction of that bar
    mu, sd = means[:, CLOSE], stds[:, CLOSE]

    agg = {
        "last": fut[:, -1],
        "mean": fut.mean(axis=1),
        "max": fut.max(axis=1),
        "min": fut.min(axis=1),
    }
    out = {
        # The mechanical part of the score, kept so controls.py can ask whether Kronos
        # adds anything beyond "how extended is this name vs its own 90-bar mean".
        # An autoregressive sampler that regresses toward the window mean produces
        # score_z ~ -anchor_z by construction, i.e. a pure short-term reversal factor.
        "anchor_z": anchor_z.astype(np.float64),
        "anchor_recon": anchor_recon.astype(np.float64),
    }
    for name, a in agg.items():
        out[f"{name}_z"] = (a - anchor_z).astype(np.float64)
        # denormalization is affine and monotone, so denorm(f(p)) == f(denorm(p))
        denorm = a.astype(np.float64) * (sd + 1e-5) + mu
        out[f"{name}_ret"] = denorm / closes - 1.0
        out[f"{name}_recon"] = (a - anchor_recon).astype(np.float64)
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=os.path.join(DATA_NSE, "panel_price.parquet"))
    ap.add_argument("--calendar", default=os.path.join(DATA_NSE, "calendar.json"))
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--model-repo", default="NeoQuasar/Kronos-base")
    ap.add_argument("--model-path", default=None, help="local finetuned predictor dir")
    ap.add_argument("--tokenizer-path", default=None, help="local finetuned tokenizer dir")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--predict", type=int, default=10)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--T", type=float, default=0.6, help="repo default inference_T")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--sample-count", type=int, default=1)
    ap.add_argument("--n-paths", type=int, default=1,
                    help=">1 replicates each row to recover the collapsed ensemble (plan 4.7); "
                         "adds a dispersion column and costs n_paths x compute")
    ap.add_argument("--limit-dates", type=int, default=None, help="first N dates only (G0 pilot)")
    ap.add_argument("--stride", type=int, default=1, help="score every Nth session")
    ap.add_argument("--membership", default=None,
                    help="parquet of date,symbol point-in-time index membership; when given, "
                         "only actual index members are scored on each date")
    ap.add_argument("--max-stale-days", type=int, default=220,
                    help="reject contexts stitched across a suspension longer than this")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(OUT_NSE, f"signals_{args.tag}")
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    blocks, calendar = load_inputs(args.panel, args.calendar)
    symbols = sorted(blocks)
    member_of: dict | None = None
    if args.membership:
        mdf = pd.read_parquet(args.membership)
        mdf["date"] = pd.to_datetime(mdf["date"])
        member_of = {d: set(g) for d, g in mdf.groupby("date")["symbol"]}
        print(f"membership mask: {len(member_of)} dates, "
              f"{mdf['symbol'].nunique()} distinct symbols", flush=True)
    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    idxs = [i for i, d in enumerate(calendar) if lo <= d <= hi][:: args.stride]
    if args.limit_dates:
        idxs = idxs[: args.limit_dates]
    print(f"universe {len(symbols)} symbols | {len(idxs)} rebalance dates "
          f"{calendar[idxs[0]].date()} -> {calendar[idxs[-1]].date()}", flush=True)

    todo = []
    for i in idxs:
        p = os.path.join(parts_dir, f"{calendar[i].date()}.parquet")
        if args.resume and os.path.exists(p):
            continue
        todo.append(i)
    print(f"{len(idxs) - len(todo)} dates already done, {len(todo)} to compute", flush=True)

    if todo:
        predictor = load_kronos(max_context=args.max_context, model_repo=args.model_repo,
                                model_path=args.model_path, tokenizer_path=args.tokenizer_path)
        dev = predictor.device
        print(f"device={dev} batch={args.batch_size} T={args.T} top_p={args.top_p} "
              f"sample_count={args.sample_count} n_paths={args.n_paths}", flush=True)

    t_start = time.time()
    n_series = 0
    extended_cal_dates: list[str] = []
    divergent_stamp_dates: list[str] = []
    for k, ti in enumerate(todo):
        t = calendar[ti]
        day_syms = symbols if member_of is None else [
            s_ for s_ in symbols if s_ in member_of.get(t, ())]
        batch = build_batch(blocks, day_syms, t, args.lookback, args.max_stale_days)
        if batch is None:
            pd.DataFrame().to_parquet(os.path.join(parts_dir, f"{t.date()}.parquet"))
            continue
        X, syms, means, stds, closes = batch
        fut_dates = future_sessions(calendar, ti, args.predict)
        if ti + args.predict >= len(calendar):
            extended_cal_dates.append(str(t.date()))
        # x_stamp is built PER SYMBOL. On the current-membership panel every eligible symbol
        # shares the same last-`lookback` session dates, so this is identical to broadcasting
        # one stamp; on the point-in-time panel, names that were suspended or that listed
        # mid-window do not, and a broadcast stamp would misalign their time features.
        def _ctx(sym):
            i = blocks[sym]["pos"][t]
            return blocks[sym]["dates"][i + 1 - args.lookback: i + 1]
        per_sym_stamp = np.stack([stamps_for(_ctx(s_)) for s_ in syms])
        if not np.array_equal(per_sym_stamp, np.repeat(per_sym_stamp[:1], len(syms), 0)):
            divergent_stamp_dates.append(str(t.date()))
        y_stamp = stamps_for(fut_dates)

        rows = []
        for c0 in range(0, len(syms), args.batch_size):
            c1 = min(c0 + args.batch_size, len(syms))
            xb = X[c0:c1]
            nb = c1 - c0
            reps = max(1, args.n_paths)
            xr = np.repeat(xb, reps, axis=0) if reps > 1 else xb
            xs_b = (np.repeat(per_sym_stamp[c0:c1], reps, axis=0) if reps > 1
                    else per_sym_stamp[c0:c1])
            ys_b = np.repeat(y_stamp[None, ...], xr.shape[0], axis=0)
            # deterministic regardless of where a resumed run picks up
            set_seed(args.seed * 1_000_003 + ti * 1_009 + c0)
            preds = auto_regressive_inference(
                predictor.tokenizer, predictor.model,
                torch.from_numpy(xr).to(dev), torch.from_numpy(xs_b).to(dev),
                torch.from_numpy(ys_b).to(dev),
                max_context=args.max_context, pred_len=args.predict, clip=5,
                T=args.T, top_k=args.top_k, top_p=args.top_p,
                sample_count=args.sample_count, verbose=False)
            disp = None
            if reps > 1:
                pr = preds.reshape(nb, reps, preds.shape[1], preds.shape[2])
                disp = pr[:, :, -1, CLOSE].std(axis=1)
                preds = pr.mean(axis=1)
            sc = score_block(preds, xb, means[c0:c1], stds[c0:c1], closes[c0:c1],
                             args.lookback, args.predict)
            df = pd.DataFrame(sc)
            df.insert(0, "symbol", syms[c0:c1])
            df["close_t"] = closes[c0:c1]
            df["sigma_close"] = stds[c0:c1, CLOSE]
            if disp is not None:
                df["path_dispersion"] = disp
            rows.append(df)
            n_series += nb

        out = pd.concat(rows, ignore_index=True)
        out.insert(0, "date", t)
        out.to_parquet(os.path.join(parts_dir, f"{t.date()}.parquet"), index=False)

        el = time.time() - t_start
        rate = n_series / max(el, 1e-9)
        eta = (len(todo) - k - 1) * (el / (k + 1))
        print(f"[{k + 1:4d}/{len(todo)}] {t.date()} n={len(syms):3d} "
              f"{el / (k + 1):5.1f}s/date {1000 / max(rate, 1e-9):5.1f}ms/series "
              f"ETA {eta / 3600:5.2f}h", flush=True)

    # ---- consolidate -----------------------------------------------------
    parts = sorted(f for f in os.listdir(parts_dir) if f.endswith(".parquet"))
    frames = []
    for f in parts:
        d = pd.read_parquet(os.path.join(parts_dir, f))
        if len(d):
            frames.append(d)
    allsig = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    final = os.path.join(OUT_NSE, f"signals_{args.tag}.parquet")
    allsig.to_parquet(final, index=False)
    el = time.time() - t_start
    print(f"\n{len(allsig)} rows, {allsig['date'].nunique() if len(allsig) else 0} dates -> {final}")
    print(f"wall {el:.0f}s  {n_series} series  {el / max(n_series, 1) * 1000:.1f} ms/series")

    sidecar(os.path.join(OUT_NSE, f"signals_{args.tag}_meta.json"), vars(args), extra={
        "wall_clock_sec": round(el, 1), "series_computed": n_series,
        "ms_per_series": round(el / max(n_series, 1) * 1000, 2),
        "batch_size_used": args.batch_size, "rows": int(len(allsig)),
        "dates": int(allsig["date"].nunique()) if len(allsig) else 0,
        "score_columns": [c for c in allsig.columns
                          if any(c.startswith(d + "_") for d in SIGNAL_DEFS)],
        "dates_with_extended_calendar": extended_cal_dates,
        "dates_with_divergent_context_calendars": divergent_stamp_dates,
        "membership_mask": args.membership,
    })


if __name__ == "__main__":
    main()
