"""Does Kronos have an edge on BTC?

Five questions, in the order that lets an earlier answer kill a later one:

  A. MECHANISM   -- does the same normalization-driven reversion appear here as on
                    NSE-200 (70.1%) and the S&P 100 (68.9%)?
  B. CALIBRATION -- the sampled paths give a P(up). Does it match the realized hit rate?
                    This is the test the upstream API could not express, because
                    auto_regressive_inference averaged the paths away.
  C. SKILL       -- is predicted return correlated with realized return at all?
  D. BACKTEST    -- costed timing vs buy-and-hold, plus the Indian tax overlay.
  E. CONTROLS    -- random signals, the model's own negation, and price-only baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "bench", "out", "btc")

# Binance spot taker; the maker side is cheaper but a timing signal takes liquidity.
FEE_BPS = 10.0
SLIP_BPS = 2.0


# ------------------------------------------------------------------ inference helpers

def nw_tstat(x: np.ndarray, lag: int) -> float:
    """Newey-West t-stat for the mean. Overlapping horizons autocorrelate the series,
    so the naive t is inflated by roughly sqrt(horizon) without this."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    d = x - x.mean()
    # np.sum(d*d) rather than d @ d: numpy 2.2 on Apple Accelerate raises spurious
    # divide-by-zero/overflow warnings for 1-D matmul. Values are bit-identical
    # (verified: -0.973879 and -2.199779 either way); only the noise differs.
    g0 = float(np.sum(d * d)) / n
    if g0 <= 0:
        return float("nan")          # constant series: the t-stat is not defined
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(np.sum(d[k:] * d[:-k])) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk      # Bartlett
    se = np.sqrt(max(var, 1e-24) / n)
    return float(x.mean() / se)


def block_bootstrap_ci(x: np.ndarray, block: int, n_boot: int, seed: int) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    n_block = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i][:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[i] = x[idx].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ------------------------------------------------------------------ A. mechanism

def mechanism(d: pd.DataFrame) -> dict:
    x, y = d["anchor_z"].to_numpy(), d["last_z"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    slope, intercept = np.polyfit(x, y, 1)
    rho = float(np.corrcoef(x, y)[0, 1])
    # A "reversal factor" is just -anchor_z. If the score ranks like that, the model is
    # not adding information, it is restating how far price sits from its own window mean.
    rho_rev = float(np.corrcoef(d["mean_z"], -d["anchor_z"])[0, 1])
    return {
        "n": int(len(x)), "slope": float(slope), "intercept": float(intercept),
        "r2": rho ** 2, "reversion_pct": float(-slope * 100),
        "rho_score_vs_neg_anchor": rho_rev,
    }


# ------------------------------------------------------------------ B. calibration

def calibration(d: pd.DataFrame, paths: np.ndarray | None, n_bucket: int = 10) -> dict:
    p = d["p_up"].to_numpy()
    hit = (d["realized_ret"].to_numpy() > 0).astype(float)
    ok = np.isfinite(p) & np.isfinite(hit)
    p, hit = p[ok], hit[ok]
    brier = float(np.mean((p - hit) ** 2))
    # A constant forecast at the base rate is the bar any probabilistic model must clear.
    base = float(hit.mean())
    brier_base = float(np.mean((base - hit) ** 2))

    edges = np.linspace(0, 1, n_bucket + 1)
    rel = []
    for i in range(n_bucket):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < n_bucket - 1 else p <= 1.0)
        if sel.sum() >= 20:
            rel.append({"bucket": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(sel.sum()),
                        "stated": float(p[sel].mean()), "realized": float(hit[sel].mean())})

    out = {"n": int(len(p)), "mean_stated_p_up": float(p.mean()), "realized_up_rate": base,
           "brier": brier, "brier_constant_baseline": brier_base,
           # base rate 0 or 1 makes the constant forecast perfect and the ratio undefined
           "brier_skill_vs_constant": float(1 - brier / brier_base) if brier_base > 1e-9 else float("nan"),
           "reliability": rel}

    if paths is not None:
        term = paths[:, :, -1]
        lo, hi = np.percentile(term, 10, axis=1), np.percentile(term, 90, axis=1)
        real = d["realized_last_px"].to_numpy()[: len(lo)]
        out["cone_coverage_p10_p90"] = float(((real >= lo) & (real <= hi)).mean())
        out["cone_nominal"] = 0.80
        out["median_cone_width_pct"] = float(np.median((hi - lo) / d["close_now"].to_numpy()[:len(lo)] * 100))
    return out


# ------------------------------------------------------------------ C. skill

def skill(d: pd.DataFrame, horizon: int) -> dict:
    pr, rr = d["pred_ret"].to_numpy(), d["realized_ret"].to_numpy()
    ok = np.isfinite(pr) & np.isfinite(rr)
    pr, rr = pr[ok], rr[ok]
    ic = float(pd.Series(pr).corr(pd.Series(rr), method="spearman"))
    hit = (np.sign(pr) == np.sign(rr)).astype(float)
    # per-observation product, so the t-stat is about the signal, not the market's drift
    prod = np.sign(pr) * rr
    return {
        "n": int(len(pr)), "spearman_ic": ic,
        "hit_rate": float(hit.mean()), "hit_rate_t": nw_tstat(hit - 0.5, horizon),
        "signed_return_mean": float(prod.mean()), "signed_return_t": nw_tstat(prod, horizon),
        "signed_return_ci95": block_bootstrap_ci(prod, block=horizon * 2, n_boot=2000, seed=0),
        "pred_mean_ret": float(pr.mean()), "realized_mean_ret": float(rr.mean()),
    }


# ------------------------------------------------------------------ D. backtest

def backtest(d: pd.DataFrame, sig: np.ndarray, mode: str, horizon: int,
             fee_bps: float = FEE_BPS, slip_bps: float = SLIP_BPS) -> dict:
    """Non-overlapping: enter every `horizon` bars so each trade is independent, which is
    also the only way the turnover figure means anything."""
    step = horizon
    idx = np.arange(0, len(d) - 1, step)
    s = sig[idx]
    fwd = d["realized_ret"].to_numpy()[idx]
    pos = (s > 0).astype(float) if mode == "long_flat" else np.sign(s)

    gross = pos * fwd
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = turn * (fee_bps + slip_bps) / 1e4
    net = gross - cost

    bh = fwd                                    # buy and hold over the same bars
    # Annualise from elapsed calendar time. Deriving it from a bar count instead
    # silently assumes the sample is long, and on a short one it raises a large
    # per-trade return to a large power -- which is how a 24-origin pilot reported
    # six-million-percent annualised.
    ts = pd.to_datetime(d["origin"].to_numpy()[idx])
    bar_days = float(np.median(np.diff(ts.values).astype("timedelta64[s]").astype(float)) / 86400.0)
    years = max(((ts[-1] - ts[0]).total_seconds() / 86400.0 + bar_days) / 365.25, 1e-9)
    per_year = len(idx) / years                 # trades per year, for vol scaling

    def ann(r):
        return float((1 + pd.Series(r)).prod() ** (1.0 / years) - 1)

    def cum(r):
        return float((1 + pd.Series(r)).prod() - 1)

    vol = lambda r: float(np.std(r) * np.sqrt(per_year))
    return {
        "mode": mode, "n_trades": int(len(idx)), "exposure": float(np.mean(pos != 0)),
        "years": float(years),
        # compounding a handful of trades up to a year produces a real number that
        # means nothing; callers should read cum_return_* instead when this is False
        "annualisation_reliable": bool(years >= 1.0),
        "ann_return_gross": ann(gross), "ann_return_net": ann(net), "ann_return_bh": ann(bh),
        "ann_excess_net": ann(net) - ann(bh),
        "cum_return_net": cum(net), "cum_return_bh": cum(bh),
        "vol_ann": vol(net), "vol_ann_bh": vol(bh),
        "sharpe_net": float(np.mean(net) / (np.std(net) + 1e-12) * np.sqrt(per_year)),
        "turnover_per_year": float(turn.sum() * per_year / max(len(idx), 1)),
        "cost_drag_ann_pct": float(cost.mean() * per_year * 100),
        "hit_rate": float((net > 0).mean()),
        "max_drawdown": float(((1 + pd.Series(net)).cumprod() /
                               (1 + pd.Series(net)).cumprod().cummax() - 1).min()),
    }


def india_tax_overlay(bt: dict) -> dict:
    """Section 115BBH: 30% flat on crypto gains, no loss set-off, no expense deduction.
    Section 194S: 1% TDS withheld on every sell. The TDS is creditable against the 30%,
    so for a filer it is a cash-flow drag rather than an extra cost -- but the 30% with
    no loss offset is not, and it applies to gross gains, not to net-of-losses P&L.
    Not tax advice; the point is only the order of magnitude.
    """
    g, b = bt["ann_return_net"], bt["ann_return_bh"]
    # Buy-and-hold realises once; a timing strategy realises every trade, so the
    # no-set-off rule bites it and not the benchmark.
    return {
        "strategy_after_30pct": float(g * 0.70) if g > 0 else float(g),
        "buyhold_after_30pct": float(b * 0.70) if b > 0 else float(b),
        "tds_cashflow_drag_pct_of_notional": float(bt["turnover_per_year"] / 2 * 1.0),
        "note": "TDS is creditable against the 30%; shown as cash-flow drag, not a cost.",
    }


# ------------------------------------------------------------------ E. controls

def controls(d: pd.DataFrame, horizon: int, n_seed: int, mode: str) -> dict:
    out = {}
    for name, sig in {
        "kronos": d["pred_ret"].to_numpy(),
        "kronos_negated": -d["pred_ret"].to_numpy(),
        "anchor_reversal": -d["anchor_z"].to_numpy(),
        "momentum_10": d["close_now"].pct_change(10).fillna(0).to_numpy(),
        "always_long": np.ones(len(d)),
    }.items():
        out[name] = backtest(d, sig, mode, horizon)

    rng = np.random.default_rng(0)
    rnd = [backtest(d, rng.normal(size=len(d)), mode, horizon)["ann_excess_net"]
           for _ in range(n_seed)]
    rnd = np.array(rnd)
    real = out["kronos"]["ann_excess_net"]
    out["random_control"] = {
        "n_seed": n_seed, "mean": float(rnd.mean()), "p05": float(np.percentile(rnd, 5)),
        "p50": float(np.median(rnd)), "p95": float(np.percentile(rnd, 95)),
        "kronos_percentile": float((rnd < real).mean() * 100),
        "p_value_one_sided": float((rnd >= real).mean()),
    }
    return out


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--mode", default="long_flat", choices=["long_flat", "long_short"])
    ap.add_argument("--seeds", type=int, default=300)
    a = ap.parse_args()

    d = pd.read_parquet(os.path.join(OUT, f"signals_{a.tag}.parquet")).reset_index(drop=True)
    ppath = os.path.join(OUT, f"paths_{a.tag}.npy")
    paths = np.load(ppath) if os.path.exists(ppath) else None

    res = {
        "tag": a.tag, "n_origins": len(d),
        "first": str(d["origin"].iloc[0]), "last": str(d["origin"].iloc[-1]),
        "A_mechanism": mechanism(d),
        "B_calibration": calibration(d, paths),
        "C_skill": skill(d, a.horizon),
        "E_controls": controls(d, a.horizon, a.seeds, a.mode),
    }
    res["D_backtest"] = res["E_controls"]["kronos"]
    res["D_india_tax"] = india_tax_overlay(res["D_backtest"])

    path = os.path.join(OUT, f"analysis_{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)

    m, c, s = res["A_mechanism"], res["B_calibration"], res["C_skill"]
    print(f"\n=== {a.tag}: {len(d)} origins  {res['first'][:10]} .. {res['last'][:10]} ===")
    print(f"A mechanism    slope {m['slope']:+.4f}  R2 {m['r2']:.3f}  "
          f"reversion {m['reversion_pct']:+.1f}%  rho(score,-anchor) {m['rho_score_vs_neg_anchor']:+.3f}")
    print(f"B calibration  stated P(up) {c['mean_stated_p_up']:.3f} vs realized {c['realized_up_rate']:.3f}  "
          f"Brier {c['brier']:.4f} (constant {c['brier_constant_baseline']:.4f}, "
          f"skill {c['brier_skill_vs_constant']:+.3f})")
    if "cone_coverage_p10_p90" in c:
        print(f"               p10-p90 cone covers {c['cone_coverage_p10_p90']:.3f} of outcomes "
              f"(nominal 0.80), median width {c['median_cone_width_pct']:.1f}%")
    print(f"C skill        IC {s['spearman_ic']:+.4f}  hit {s['hit_rate']:.3f} (t {s['hit_rate_t']:+.2f})  "
          f"signed ret {s['signed_return_mean']*100:+.3f}% (t {s['signed_return_t']:+.2f})")
    b = res["D_backtest"]
    print(f"D backtest     net {b['ann_return_net']*100:+.2f}%/yr vs "
          f"buy-hold {b['ann_return_bh']*100:+.2f}%/yr  "
          f"excess {b['ann_excess_net']*100:+.2f}%  over {b['years']:.1f}y "
          f"(cum {b['cum_return_net']*100:+.0f}% vs {b['cum_return_bh']*100:+.0f}%)"
          + ("" if b["annualisation_reliable"] else
             "\n               [!] span < 1y: annualised figures are extrapolation, read cum_*"))
    print("E controls")
    for k, v in res["E_controls"].items():
        if k == "random_control":
            print(f"   random: mean {v['mean']*100:+.2f}%  p95 {v['p95']*100:+.2f}%  "
                  f"kronos at percentile {v['kronos_percentile']:.1f} (p={v['p_value_one_sided']:.3f})")
        else:
            print(f"   {k:18s} net {v['ann_return_net']*100:+7.2f}%/yr  excess {v['ann_excess_net']*100:+7.2f}%")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
