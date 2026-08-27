"""End-to-end Kronos evaluation: ensemble forecast + holdout scoring + baselines + charts.

Why not predictor.predict(sample_count=N)?
  auto_regressive_inference() reshapes to (batch, sample_count, L, D) then means over axis=1,
  so the ensemble is destroyed before it is returned. Passing a batch of N identical rows with
  sample_count=1 keeps the same GPU batching but makes that mean a no-op, so all N stochastic
  paths survive and a real predictive distribution can be built.
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "bench", "out")
PRICE = ["open", "high", "low", "close"]
FEATS = PRICE + ["volume", "amount"]


def predict_paths(predictor, x_df, x_ts, y_ts, pred_len, n_paths, T, top_p, top_k=0):
    """Return (n_paths, pred_len, 6) of denormalised sampled futures."""
    df = x_df.copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
        df["amount"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[PRICE].mean(axis=1)
    if df[FEATS].isnull().values.any():
        raise ValueError("NaN in input features")

    x = df[FEATS].values.astype(np.float32)
    x_stamp = calc_time_stamps(x_ts).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_ts).values.astype(np.float32)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    xn = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    tile = lambda a: np.repeat(a[np.newaxis, ...], n_paths, axis=0)
    preds = predictor.generate(tile(xn), tile(x_stamp), tile(y_stamp),
                               pred_len, T, top_k, top_p, sample_count=1, verbose=False)
    return np.asarray(preds) * (x_std + 1e-5) + x_mean


def directional_accuracy(actual, pred, anchor):
    a = np.sign(np.diff(np.concatenate([[anchor], actual])))
    p = np.sign(np.diff(np.concatenate([[anchor], pred])))
    mask = a != 0
    return float((a[mask] == p[mask]).mean()) if mask.any() else float("nan")


def score(name, y_true_close, p50, paths_close, anchor):
    err = p50 - y_true_close
    rmse = float(np.sqrt((err ** 2).mean()))
    naive = np.full_like(y_true_close, anchor)
    rmse_naive = float(np.sqrt(((naive - y_true_close) ** 2).mean()))
    p10 = np.percentile(paths_close, 10, axis=0)
    p90 = np.percentile(paths_close, 90, axis=0)
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": rmse,
        "mape_pct": float((np.abs(err) / np.abs(y_true_close)).mean() * 100),
        "rmse_pct_of_anchor": float(rmse / anchor * 100),
        "directional_accuracy": directional_accuracy(y_true_close, p50, anchor),
        "terminal_direction_correct": bool(
            np.sign(p50[-1] - anchor) == np.sign(y_true_close[-1] - anchor)),
        "cone_coverage_p10_p90": float(((y_true_close >= p10) & (y_true_close <= p90)).mean()),
        "baseline_naive_rmse": rmse_naive,
        "skill_score_vs_naive": float(1.0 - rmse / rmse_naive) if rmse_naive > 0 else float("nan"),
        "actual_move_pct": float((y_true_close[-1] - anchor) / anchor * 100),
        "predicted_move_pct": float((p50[-1] - anchor) / anchor * 100),
    }


def chart(name, hist, y_ts, y_true, paths, out_png, tail=160):
    close = paths[:, :, PRICE.index("close")]
    vol = paths[:, :, FEATS.index("volume")]
    p10, p50, p90 = (np.percentile(close, q, axis=0) for q in (10, 50, 90))
    p25, p75 = np.percentile(close, 25, axis=0), np.percentile(close, 75, axis=0)
    h = hist.iloc[-tail:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(h["timestamps"], h["close"], color="#1f2937", lw=1.3, label="history (actual)")
    ax1.plot(y_ts, y_true["close"], color="#2563eb", lw=2.0, label="holdout actual")
    ax1.plot(y_ts, p50, color="#dc2626", lw=2.0, label="Kronos p50")
    ax1.fill_between(y_ts, p10, p90, color="#dc2626", alpha=0.15, label="p10–p90")
    ax1.fill_between(y_ts, p25, p75, color="#dc2626", alpha=0.22, label="p25–p75")
    ax1.axvline(y_ts.iloc[0], color="#6b7280", ls="--", lw=1)
    ax1.axhline(h["close"].iloc[-1], color="#9ca3af", ls=":", lw=1, label="naive baseline")
    ax1.set_ylabel("Close")
    ax1.set_title(f"{name} — Kronos-base, {len(paths)} sampled paths, {len(y_ts)}-bar holdout")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(h["timestamps"], h["volume"], color="#1f2937", lw=1.0)
    ax2.plot(y_ts, y_true["volume"], color="#2563eb", lw=1.6, label="actual")
    ax2.plot(y_ts, np.median(vol, axis=0), color="#dc2626", lw=1.6, label="predicted p50")
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)


RUNS = [
    dict(name="btc_1h",      lookback=512, horizon=48, T=1.0, top_p=0.9),
    dict(name="btc_1d",      lookback=512, horizon=30, T=1.0, top_p=0.9),
    dict(name="nifty50_1h",  lookback=512, horizon=48, T=1.0, top_p=0.9),
    dict(name="nifty50_1d",  lookback=512, horizon=30, T=1.0, top_p=0.9),
    dict(name="reliance_1d", lookback=512, horizon=30, T=1.0, top_p=0.9),
    dict(name="reliance_1h", lookback=512, horizon=48, T=1.0, top_p=0.9),
]


def load(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), encoding="utf-8-sig")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def window_slice(df, lb, hz, k):
    """k=0 is the most recent window; each k steps the origin back by hz bars (non-overlapping holdouts)."""
    end = len(df) - k * hz
    start = end - (lb + hz)
    if start < 0:
        return None
    w = df.iloc[start:end].reset_index(drop=True)
    return dict(
        x_df=w.loc[:lb - 1, FEATS], x_ts=w.loc[:lb - 1, "timestamps"],
        y_true=w.loc[lb:, FEATS].reset_index(drop=True),
        y_ts=w.loc[lb:, "timestamps"].reset_index(drop=True),
        anchor=float(w.loc[lb - 1, "close"]), hist=df.iloc[:end],
        zero_vol_pct=round(float((w["volume"] == 0).mean() * 100), 2),
    )


def agg(rows):
    f = lambda k: [r[k] for r in rows if r[k] == r[k]]
    return {
        "n_windows": len(rows),
        "mape_pct_mean": float(np.mean(f("mape_pct"))),
        "mape_pct_median": float(np.median(f("mape_pct"))),
        "directional_accuracy_mean": float(np.mean(f("directional_accuracy"))),
        "skill_score_vs_naive_mean": float(np.mean(f("skill_score_vs_naive"))),
        "skill_score_vs_naive_median": float(np.median(f("skill_score_vs_naive"))),
        "beats_naive_win_rate": float(np.mean([r["skill_score_vs_naive"] > 0 for r in rows])),
        "terminal_direction_hit_rate": float(np.mean([r["terminal_direction_correct"] for r in rows])),
        "cone_coverage_p10_p90_mean": float(np.mean(f("cone_coverage_p10_p90"))),
        "mean_abs_predicted_move_pct": float(np.mean([abs(r["predicted_move_pct"]) for r in rows])),
        "mean_abs_actual_move_pct": float(np.mean([abs(r["actual_move_pct"]) for r in rows])),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20)
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--T", type=float, default=None, help="override sampling temperature")
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--tag", type=str, default="main", help="output subdir under bench/out")
    ap.add_argument("--charts", action="store_true", help="write chart+csv for window 0")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    kmodel = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(kmodel, tokenizer, max_context=512)
    outroot = os.path.join(OUT, args.tag); os.makedirs(outroot, exist_ok=True)
    print(f"device={predictor.device} ctx={predictor.max_context} paths={args.paths} "
          f"windows={args.windows} tag={args.tag}", flush=True)

    runs = [r for r in RUNS if args.only is None or r["name"] in args.only.split(",")]
    summary, t_start = {}, time.time()
    for r in runs:
        name, lb, hz = r["name"], r["lookback"], r["horizon"]
        T = args.T if args.T is not None else r["T"]
        top_p = args.top_p if args.top_p is not None else r["top_p"]
        if not os.path.exists(os.path.join(DATA, f"{name}.csv")):
            print(f"SKIP {name}: no csv", flush=True); continue
        df = load(name)
        per_window = []
        for k in range(args.windows):
            w = window_slice(df, lb, hz, k)
            if w is None:
                print(f"  {name} w{k}: not enough history, stopping", flush=True); break
            t0 = time.time()
            paths = predict_paths(predictor, w["x_df"], w["x_ts"], w["y_ts"], hz,
                                  args.paths, T, top_p)
            dt = time.time() - t0
            cp = paths[:, :, PRICE.index("close")]
            p50 = np.median(cp, axis=0)
            m = score(name, w["y_true"]["close"].values, p50, cp, w["anchor"])
            m.update(window=k, runtime_sec=round(dt, 1), T=T, top_p=top_p,
                     holdout_start=str(w["y_ts"].iloc[0]), holdout_end=str(w["y_ts"].iloc[-1]),
                     anchor_close=w["anchor"], zero_volume_pct=w["zero_vol_pct"])
            per_window.append(m)
            print(f"  {name:13s} w{k:<2d} {dt:5.1f}s MAPE {m['mape_pct']:6.2f}% dir {m['directional_accuracy']:.2f} "
                  f"skill {m['skill_score_vs_naive']:+7.3f} cov {m['cone_coverage_p10_p90']:.2f} "
                  f"pred {m['predicted_move_pct']:+7.2f}% act {m['actual_move_pct']:+6.2f}%", flush=True)

            if args.charts and k == 0:
                d = os.path.join(outroot, name); os.makedirs(d, exist_ok=True)
                chart(name, w["hist"], w["y_ts"], w["y_true"], paths, os.path.join(d, "forecast.png"))
                pd.DataFrame(cp.T, index=w["y_ts"],
                             columns=[f"path_{i}" for i in range(args.paths)]).to_csv(os.path.join(d, "paths_close.csv"))
                pd.DataFrame({"timestamps": w["y_ts"], "actual_close": w["y_true"]["close"], "pred_p50": p50,
                              "pred_p10": np.percentile(cp, 10, axis=0),
                              "pred_p90": np.percentile(cp, 90, axis=0)}).to_csv(
                    os.path.join(d, "forecast.csv"), index=False)

        if per_window:
            a = agg(per_window)
            a.update(lookback=lb, horizon=hz, n_paths=args.paths, T=T, top_p=top_p,
                     rows_available=len(df), zero_volume_pct=per_window[0]["zero_volume_pct"])
            summary[name] = {"aggregate": a, "windows": per_window}
            d = os.path.join(outroot, name); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "metrics.json"), "w") as f:
                json.dump(summary[name], f, indent=2)
            print(f"  -> {name}: MAPE {a['mape_pct_mean']:.2f}% | dirAcc {a['directional_accuracy_mean']:.3f} | "
                  f"skill {a['skill_score_vs_naive_mean']:+.3f} | beats-naive {a['beats_naive_win_rate']*100:.0f}% | "
                  f"cov {a['cone_coverage_p10_p90_mean']:.2f} | |pred| {a['mean_abs_predicted_move_pct']:.2f}% vs "
                  f"|act| {a['mean_abs_actual_move_pct']:.2f}%\n", flush=True)

    with open(os.path.join(outroot, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"total {time.time()-t_start:.0f}s -> {os.path.join(outroot, 'summary.json')}", flush=True)
