"""Consolidate everything into bench/out/nse/RESULTS.md + figures.

Section order is fixed by plan 8.7:
  1 pre-registered primary result with CI
  2 the full grid, labelled secondary
  3 strategy vs the random-signal DISTRIBUTION
  4 cost sensitivity 0/15/30/50 bps
  5 IC decay across horizons
  6 zero-shot vs finetuned vs momentum
  7 realised turnover and cost drag
  8 limitations, up front
  9 which section-4 claims held and which did not
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bench.nse.backtest import (TRADING_DAYS, bench_series, load_market,  # noqa: E402
                                metrics, run_backtest)
from bench.nse.common import OUT_NSE, sidecar  # noqa: E402
from bench.nse.costs import PRESETS  # noqa: E402

HORIZONS = [1, 5, 10, 20]


def j(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def pct(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{nd}f}%"


def num(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def fig_summary(grid, rnd, controls, out_png, tag):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # (0,0) IC decay
    ax = axes[0, 0]
    icg = controls.get("ic_grid", {})
    base = controls.get("baseline_ic", {})
    for col, style in (("last_ret", "-o"), ("mean_ret", "-s"), ("last_z", "--^")):
        if col in icg:
            y = [icg[col].get(f"open_h{h}", {}).get("mean_ic", np.nan) for h in HORIZONS]
            ax.plot(HORIZONS, y, style, label=col, lw=2)
    for col, style in (("momentum_10d", ":d"), ("anchor_reversal", ":v")):
        if col in base:
            y = [base[col].get(f"open_h{h}", {}).get("mean_ic", np.nan) for h in HORIZONS]
            ax.plot(HORIZONS, y, style, label=col, lw=1.6, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("forward horizon (sessions)")
    ax.set_ylabel("mean rank IC")
    ax.set_title("IC decay: Kronos vs free baselines")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) random distribution vs strategy
    ax = axes[0, 1]
    if rnd is not None and len(rnd):
        w = rnd[rnd["rebalance"] == "weekly"]["ann_excess_net"] * 100
        ax.hist(w, bins=45, color="#94a3b8", edgecolor="white")
        pre = grid[(grid.arm == "last_ret") & (grid.rebalance == "weekly") & (grid.cost_bps == 30.0)]
        if len(pre):
            ax.axvline(pre.iloc[0]["ann_excess_net"] * 100, color="#dc2626", lw=2.5,
                       label=f"Kronos {pre.iloc[0]['ann_excess_net'] * 100:+.2f}%")
        ax.axvline(np.percentile(w, 95), color="#2563eb", ls="--", lw=1.6, label="random p95")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("annualised excess vs ^CNX200 (%)")
        ax.set_title(f"random-signal distribution, n={len(w)} (weekly, 30 bps)")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,2) cost sensitivity
    ax = axes[0, 2]
    for arm, style in (("last_ret", "-o"), ("mean_ret", "-s"), ("last_z", "--^")):
        s = grid[(grid.arm == arm) & (grid.rebalance == "weekly")].sort_values("cost_bps")
        if len(s):
            ax.plot(s.cost_bps, s.ann_excess_net * 100, style, label=f"{arm} weekly", lw=2)
    s = grid[(grid.arm == "last_ret") & (grid.rebalance == "daily")].sort_values("cost_bps")
    if len(s):
        ax.plot(s.cost_bps, s.ann_excess_net * 100, "-.x", label="last_ret daily", lw=1.8)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(27.4, color="#059669", ls=":", lw=1.5, label="verified real cost 27.4 bps")
    ax.set_xlabel("round-trip cost (bps)")
    ax.set_ylabel("annualised excess (%)")
    ax.set_title("cost sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) full grid heat: excess by arm at 30 bps weekly
    ax = axes[1, 0]
    k = grid[(grid.family == "kronos") & (grid.cost_bps == 30.0)]
    piv = k.pivot_table(index="arm", columns="rebalance", values="ann_excess_net") * 100
    if len(piv):
        piv = piv.sort_values("weekly")
        y = np.arange(len(piv))
        ax.barh(y - 0.2, piv["weekly"], 0.4, label="weekly", color="#2563eb")
        if "daily" in piv:
            ax.barh(y + 0.2, piv["daily"], 0.4, label="daily", color="#f59e0b")
        ax.set_yticks(y)
        ax.set_yticklabels(piv.index, fontsize=8)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("annualised excess (%)")
        ax.set_title("all 12 Kronos arms @ 30 bps")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # (1,1) cumulative curves
    ax = axes[1, 1]
    for name, path in (("Kronos last_ret weekly", f"bt_primary_{tag}.parquet"),
                       ("Kronos REVERSED", f"bt_reversed_{tag}.parquet"),
                       ("momentum 10d weekly", f"bt_momentum_{tag}.parquet"),
                       ("anchor reversal weekly", f"bt_anchor_{tag}.parquet"),
                       ("equal-weight universe", f"bt_equalweight_{tag}.parquet")):
        p = os.path.join(OUT_NSE, path)
        if os.path.exists(p):
            d = pd.read_parquet(p)
            ax.plot(d.index, ((1 + d["net_return"]).cumprod() - 1) * 100, lw=1.8, label=name)
    p = os.path.join(OUT_NSE, f"bt_bench_{tag}.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p)
        ax.plot(d.index, ((1 + d["bench"]).cumprod() - 1) * 100, "k--", lw=1.6, label="^CNX200")
    ax.set_ylabel("cumulative return (%)")
    ax.set_title("cumulative return, net of 30 bps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,2) cumulative EXCESS
    ax = axes[1, 2]
    p = os.path.join(OUT_NSE, f"bt_primary_{tag}.parquet")
    pb = os.path.join(OUT_NSE, f"bt_bench_{tag}.parquet")
    if os.path.exists(p) and os.path.exists(pb):
        d = pd.read_parquet(p)
        b = pd.read_parquet(pb)["bench"].reindex(d.index).fillna(0)
        ax.plot(d.index, (d["net_return"] - b).cumsum() * 100, color="#dc2626", lw=1.8,
                label="net of cost")
        ax.plot(d.index, (d["gross_return"] - b).cumsum() * 100, color="#94a3b8", lw=1.4,
                label="gross")
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("cumulative excess (%)")
    ax.set_title("cumulative excess vs ^CNX200")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--ft-tag", default="finetuned")
    ap.add_argument("--pit-tag", default="pit", help="point-in-time-universe run to compare against")
    ap.add_argument("--us-tag", default="us", help="US generalisation run")
    ap.add_argument("--extra-tags", nargs="*", default=["greedy", "T1p0", "paths5"],
                    help="sampling-sensitivity runs to tabulate")
    ap.add_argument("--out", default=os.path.join(OUT_NSE, "RESULTS.md"))
    args = ap.parse_args()

    vm = j(os.path.join(OUT_NSE, "verify_model.json"), {})
    vp = j(os.path.join(OUT_NSE, "verify_pipeline.json"), {})
    vb = j(os.path.join(OUT_NSE, "verify_backtest.json"), {})
    controls = j(os.path.join(OUT_NSE, f"controls_{args.tag}.json"), {})
    gsum = j(os.path.join(OUT_NSE, f"grid_{args.tag}_summary.json"), {})
    ft = j(os.path.join(OUT_NSE, "finetune", "train_summary.json"))
    ftc = j(os.path.join(OUT_NSE, f"controls_{args.ft_tag}.json"))
    gp = os.path.join(OUT_NSE, f"grid_{args.tag}.parquet")
    grid = pd.read_parquet(gp) if os.path.exists(gp) else pd.DataFrame()
    rp = os.path.join(OUT_NSE, f"random_dist_{args.tag}.parquet")
    rnd = pd.read_parquet(rp) if os.path.exists(rp) else None
    smeta = j(os.path.join(OUT_NSE, f"signals_{args.tag}_meta.json"), {})
    ub = j(os.path.join(OUT_NSE, "universe_bias.json"))
    pitc = j(os.path.join(OUT_NSE, f"controls_{args.pit_tag}.json"))
    pitg = j(os.path.join(OUT_NSE, f"grid_{args.pit_tag}_summary.json"))
    pgp = os.path.join(OUT_NSE, f"grid_{args.pit_tag}.parquet")
    pitgrid = pd.read_parquet(pgp) if os.path.exists(pgp) else pd.DataFrame()
    usc = j(os.path.join(OUT_NSE, f"controls_{args.us_tag}.json"))
    memb = j(os.path.join(ROOT, "data", "nse", "membership.json"))
    extras = {t: j(os.path.join(OUT_NSE, f"controls_{t}.json")) for t in args.extra_tags}
    extras = {k: v for k, v in extras.items() if v}
    drift = j(os.path.join(OUT_NSE, "drift_diagnostic.json"))

    if len(grid):
        fig_summary(grid, rnd, controls, os.path.join(OUT_NSE, f"summary_{args.tag}.png"), args.tag)

    L = []
    A = L.append
    A("# Kronos on NSE — cross-sectional ranking backtest: RESULTS\n")
    A(f"Universe NIFTY 200 · benchmark `^CNX200` · test {controls.get('date_range', ['?', '?'])[0]} "
      f"→ {controls.get('date_range', ['?', '?'])[1]} · {controls.get('n_dates', '?')} signal dates\n")
    A("**Dividend convention: price-return.** Stock OHLC is split-adjusted but NOT "
      "dividend-adjusted, matching the `^CNX200` price index. Both legs use the same "
      "convention, and so does every control. The strategy is therefore charged the full "
      "dividend yield it does not receive, so reported excess return is CONSERVATIVE. The "
      "opposite choice would manufacture ~2%/yr of fake alpha.\n")

    # ---- 8 limitations, up front, not buried -----------------------------
    A("## Limitations — stated up front\n")
    for s in [
        "**Survivorship bias.** The constituent list is *current* NIFTY 200 membership "
        "back-projected. Names that fell out of the index are absent. The random-signal and "
        "equal-weight controls exist to quantify how much of any excess return this explains.",
        "**Single market, single ~3-year regime**, one of unusually strong Indian equity performance.",
        "**No market-impact model** beyond flat slippage. Top-50 NIFTY 200 names are liquid, so "
        "this is mild, but it worsens as `topk` shrinks.",
        "**Corporate actions beyond splits and dividends** (mergers, demergers, ticker changes) "
        "are not modelled.",
        "**The signal has a 10-session horizon but the daily arm rebalances every session** — a "
        "structural mismatch inherited from the repo defaults, which is why the weekly arm is primary.",
        "**Cash drag not modelled.** qlib deploys `risk_degree=0.95` of cash; this engine is fully "
        "invested. That flatters the strategy and the equal-weight control equally.",
    ]:
        A(f"- {s}")
    A("")

    # ---- 1 primary ------------------------------------------------------
    A("## 1. Pre-registered primary result\n")
    A("Fixed in `PREREG.md` before any IC or return was computed.\n")
    icg = controls.get("ic_grid", {})
    corr = controls.get("cross_sectional_rank_corr_with_kronos", {})
    p = icg.get("last_ret", {}).get("open_h10", {})
    A("**Primary metric — mean rank IC of `last_ret` vs the tradeable forward 10-session "
      "return (open t+1 → open t+11):**\n")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| mean rank IC | {num(p.get('mean_ic'), 4)} |")
    A(f"| Newey–West t (lag 10) | {num(p.get('t_newey_west'), 2)} |")
    A(f"| naive t (overstates) | {num(p.get('t_naive'), 2)} |")
    ci = p.get("bootstrap_ci95", [None, None])
    A(f"| block-bootstrap 95% CI | [{num(ci[0], 4)}, {num(ci[1], 4)}] |")
    A(f"| share of dates with IC > 0 | {num(p.get('share_dates_positive'), 3)} |")
    A(f"| n dates | {p.get('n_dates', '?')} |")
    A(f"| **significant (t>2 AND CI excludes 0)** | **{p.get('significant')}** |")
    A("")
    pc = icg.get("last_ret", {}).get("close_h10_model_target", {})
    A(f"Secondary diagnostic — same IC against the model's own close-to-close target: "
      f"{num(pc.get('mean_ic'), 4)} (NW t {num(pc.get('t_newey_west'), 2)}).\n")

    if len(grid):
        pre = grid[(grid.arm == "last_ret") & (grid.rebalance == "weekly") & (grid.cost_bps == 30.0)]
        if len(pre):
            r = pre.iloc[0]
            A("**Primary strategy result — weekly rebalance, topk=50, n_drop=5, `last_ret`, "
              "30 bps round trip:**\n")
            A("| quantity | value |")
            A("|---|---|")
            A(f"| annualised return, net | {pct(r['ann_return_net'])} |")
            A(f"| annualised benchmark (^CNX200) | {pct(r['ann_bench'])} |")
            A(f"| **annualised excess, net** | **{pct(r['ann_excess_net'])}** |")
            A(f"| annualised excess, gross | {pct(r['ann_excess_gross'])} |")
            if "ann_excess_vs_ew200" in r:
                A(f"| **annualised excess vs EQUAL-WEIGHT universe** | "
                  f"**{pct(r['ann_excess_vs_ew200'])}** |")
                A(f"| IR vs equal-weight universe | {num(r.get('ir_vs_ew200'), 2)} |")
            A(f"| bootstrap 95% CI on excess | [{pct(r['excess_ci95_annualised'][0])}, "
              f"{pct(r['excess_ci95_annualised'][1])}] |")
            A(f"| information ratio | {num(r['information_ratio'], 2)} |")
            A(f"| max drawdown | {pct(r['max_drawdown'])} |")
            A(f"| annual turnover | {r['turnover_ann']:.2f}× |")
            A(f"| cost drag | {r['cost_drag_ann_pct']:.2f}%/yr |")
            A("")
            if "prereg_percentile_in_random" in gsum:
                A(f"It sits at the **{gsum['prereg_percentile_in_random']:.1f}th percentile** of the "
                  f"random-signal distribution (one-sided empirical "
                  f"p = {gsum.get('prereg_empirical_p_one_sided')}).\n")

    # ---- 3b the reversal finding -----------------------------------------
    A("## 3b. What the signal actually is\n")
    A("An autoregressive sampler running on a per-window-normalized series regresses toward "
      "the window mean. Since `anchor_z = (close_t − mean_90)/std_90` is positive for a name "
      "trading above its own 90-bar mean, a prediction that regresses toward 0 gives "
      "`score_z ≈ −anchor_z` **by construction** — a short-term reversal factor, not a forecast.\n")
    A("| measurement | value |")
    A("|---|---|")
    A(f"| mean cross-sectional Spearman(`last_z`, `anchor_z`) | {num(corr.get('anchor_momentum'), 3)} |")
    A(f"| mean Spearman(`last_ret`, `−anchor_z`) | {num(corr.get('anchor_reversal'), 3)} |")
    A(f"| mean Spearman(`last_ret`, 10d momentum) | {num(corr.get('momentum_10d'), 3)} |")
    A("")
    orth = controls.get("orthogonalised_ic", {})
    if orth:
        A("**Orthogonalised IC (gate G2b).** Within each date, the Kronos rank-score is "
          "regressed on the free baseline's rank-score and the IC of the residual is measured. "
          "If the residual carries no IC, Kronos adds nothing the one-line feature does not.\n")
        A("| Kronos residualised against | IC 1d | IC 5d | IC 10d | IC 20d | NW t (10d) | significant |")
        A("|---|---|---|---|---|---|---|")
        for k, row in orth.items():
            r10 = row.get("open_h10", {})
            A(f"| {k.replace('kronos_resid_of_', '')} | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(r10.get('t_newey_west'), 2)} | {r10.get('significant')} |")
        A("")

    if drift:
        A("**The mechanism, measured directly.** Averaged over the cross-section and over "
          "sample dates, the predicted close path per step ahead:\n")
        A("| decoding | step 1 | step 5 | step 10 | share of names predicted down at 10 |")
        A("|---|---|---|---|---|")
        for regime, rows in drift.get("regimes", {}).items():
            if not rows:
                continue
            m = np.mean([r["pred_pct_by_step"] for r in rows], axis=0)
            neg = np.mean([r["share_symbols_negative_terminal"] for r in rows])
            A(f"| {regime} | {pct(m[0])} | {pct(m[4])} | {pct(m[9])} | {neg:.2f} |")
        rows = drift.get("regimes", {}).get("sampled_T0.6_topp0.9", [])
        if rows:
            m = np.mean([r["realized_pct_by_step"] for r in rows], axis=0)
            A(f"| **realized** | {pct(m[0])} | {pct(m[4])} | {pct(m[9])} | — |")
        A("")
        A("If the greedy row shows the same decline as the sampled row, the drift is the "
          "model's conditional **mode**, not sampling noise — see `drift_diagnostic.png`.\n")

    # ---- 3 random distribution -------------------------------------------
    A("## 3. Strategy vs the random-signal distribution\n")
    A("The critical control: what top-k-dropout earns on this survivorship-biased universe "
      "with **zero information**. Kronos must clear this distribution, not merely the index.\n")
    ue = gsum.get("universe_effects")
    if ue:
        A("**Before any signal at all, the universe alone explains most of the headline "
          "\"excess return\".** Over the same window:\n")
        A("| leg | annualised |")
        A("|---|---|")
        A(f"| cap-weighted `^CNX200` | {pct(ue['cap_weighted_index_ann'])} |")
        A(f"| EQUAL-weighted eligible universe (current NIFTY 200 membership) | "
          f"{pct(ue['equal_weight_eligible_universe_ann'])} |")
        A(f"| **gap: weighting + survivorship, zero signal** | "
          f"**{pct(ue['gap_weighting_plus_survivorship_ann'])}** |")
        A("")
        A("Any equal-weighted strategy drawn from this list starts with that gap in hand. "
          "Excess vs `^CNX200` is therefore not a measure of skill; excess vs the "
          "equal-weight universe, and position within the random-signal distribution, are.\n")
    if ub:
        d = ub["decomposition"]
        A("That gap splits into an investable part and a look-ahead part. Charging the "
          "equal-weight tilt at what **real, survivorship-free** NSE size indices actually "
          "delivered over the same sessions (NIFTY Next 50 and NIFTY Midcap 150) leaves the "
          "residual as survivorship:\n")
        A("| component | annualised |")
        A("|---|---|")
        A(f"| cap-weighted NIFTY 200 (`^CNX200`) | {pct(d['cap_weighted_nifty200'])} |")
        A(f"| + size tilt from equal-weighting (investable) | "
          f"{pct(d['plus_size_tilt_from_equal_weighting'])} |")
        A(f"| + **survivorship from back-projected membership (NOT investable)** | "
          f"**{pct(d['plus_survivorship_from_current_membership'])}** |")
        A(f"| = equal-weight of the current NIFTY 200 list | "
          f"{pct(d['equals_equal_weight_current_membership'])} |")
        A("")
        A("Reference points used, same window: " + ", ".join(
            f"{k} {pct(v['ann_return'])}" for k, v in ub["indices"].items()) + ".")
        A(f"\n{ub['note'].capitalize()}.\n")
    rw = gsum.get("random_weekly", {})
    if rw:
        A("| statistic | annualised excess |")
        A("|---|---|")
        for k in ("mean", "p05", "p50", "p95"):
            A(f"| random {k} | {pct(rw.get(k))} |")
        A(f"| random std | {pct(rw.get('std'))} |")
        A(f"| random mean turnover | {rw.get('mean_turnover_ann', float('nan')):.2f}× |")
        A("")

    # ---- 5 IC by horizon + baselines --------------------------------------
    A("## 5. IC decay and the free baselines\n")
    A("| signal | IC 1d | IC 5d | IC 10d | IC 20d | NW t (10d) | rank-corr w/ Kronos |")
    A("|---|---|---|---|---|---|---|")
    for col in ("last_ret", "last_z", "last_recon", "mean_ret", "mean_z",
                "max_ret", "max_z", "min_ret", "min_z"):
        if col in icg:
            row = icg[col]
            A(f"| `{col}` | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(row.get('open_h10', {}).get('t_newey_west'), 2)} | — |")
    for name, row in controls.get("baseline_ic", {}).items():
        A(f"| {name} | " + " | ".join(
            num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
          + f" | {num(row.get('open_h10', {}).get('t_newey_west'), 2)} "
            f"| {num(corr.get(name), 3)} |")
    A("")

    # ---- 4 cost sensitivity -----------------------------------------------
    A("## 4. Cost sensitivity\n")
    if len(grid):
        A("| arm | rebalance | 0 bps | 15 bps | 30 bps | 50 bps |")
        A("|---|---|---|---|---|---|")
        for arm in ("last_ret", "mean_ret", "last_z", "reversed_last_ret",
                    "momentum_10d", "momentum_20d", "anchor_reversal", "equal_weight_universe"):
            for reb in ("weekly", "daily"):
                s = grid[(grid.arm == arm) & (grid.rebalance == reb)]
                if len(s):
                    cells = []
                    for c in (0.0, 15.0, 30.0, 50.0):
                        v = s[s.cost_bps == c]["ann_excess_net"]
                        cells.append(pct(float(v.iloc[0])) if len(v) else "—")
                    A(f"| `{arm}` | {reb} | " + " | ".join(cells) + " |")
        A("")
        A("Verified real round-trip cost for a Rs 10 lakh delivery leg is "
          f"**{PRESETS['zerodha'].round_trip_bps(1e6):.1f} bps** "
          f"(statutory {PRESETS['zerodha'].round_trip_bps(1e6) - 5:.1f} + 5 bps slippage), "
          "so the 30 bps column is the realistic one and 50 bps is a stress case. "
          "See section 9 for why the plan's 30 bps estimate was right for the wrong reasons.\n")

    # ---- 7 turnover -------------------------------------------------------
    A("## 7. Realised turnover and cost drag\n")
    if len(grid):
        A("| arm | rebalance | turnover/yr | cost drag %/yr | turnover per rebalance |")
        A("|---|---|---|---|---|")
        for arm in ("last_ret", "momentum_10d", "anchor_reversal"):
            for reb in ("weekly", "daily"):
                s = grid[(grid.arm == arm) & (grid.rebalance == reb) & (grid.cost_bps == 30.0)]
                if len(s):
                    r = s.iloc[0]
                    A(f"| `{arm}` | {reb} | {r['turnover_ann']:.2f}× | "
                      f"{r['cost_drag_ann_pct']:.2f}% | {r['turnover_per_rebalance']:.3f} |")
        A("")
    A("Plan 4.2 predicted ~7.6%/yr of cost drag at `topk=50, n_drop=5` daily "
      "(0.20 × 0.0015 × 252). Compare against the realised numbers above — the prediction "
      "assumed every rebalance drops the full `n_drop`, but qlib's `comb` guard and "
      "`hold_thresh=5` both bind, so realised turnover is lower.\n")

    # ---- 6 finetune -------------------------------------------------------
    A("## 6. Zero-shot vs finetuned vs momentum\n")
    if ft:
        A("| quantity | value |")
        A("|---|---|")
        A(f"| zero-shot val loss | {ft.get('zero_shot_val_loss'):.5f} |")
        A(f"| best finetuned val loss | {ft.get('best_val_loss'):.5f} |")
        A(f"| improvement | {ft.get('val_loss_improvement'):.5f} |")
        A(f"| **G3 passed (val loss improved)** | **{ft.get('finetuning_helped_on_val')}** |")
        A(f"| steps / samples | {ft.get('steps_completed')} / {ft.get('samples_seen')} |")
        A(f"| wall clock | {ft.get('wall_clock_min')} min |")
        A("")
        A(f"What was cut: {ft.get('what_was_cut')}\n")
        if ftc:
            fp = ftc.get("ic_grid", {}).get("last_ret", {}).get("open_h10", {})
            A(f"Finetuned test IC (`last_ret`, 10d): {num(fp.get('mean_ic'), 4)} "
              f"(NW t {num(fp.get('t_newey_west'), 2)}) vs zero-shot {num(p.get('mean_ic'), 4)}.\n")
    else:
        A("_Finetuning not run — see the gate outcome in section 9._\n")

    # ---- 2 full grid ------------------------------------------------------
    A("## 2. Full grid — SECONDARY, reported whole\n")
    A("Every cell below is exploratory. A maximum over the grid is not an estimate.\n")
    if len(grid):
        A(f"Kronos arms span {pct(gsum.get('kronos_grid_excess_range', [np.nan, np.nan])[0])} to "
          f"{pct(gsum.get('kronos_grid_excess_range', [np.nan, np.nan])[1])} annualised excess; "
          f"{gsum.get('kronos_grid_share_positive_excess', float('nan')) * 100:.0f}% of cells are positive.\n")
        A("| family | arm | rebalance | cost bps | ann excess vs ^CNX200 | ann excess vs EW200 "
          "| IR | turnover/yr | maxDD |")
        A("|---|---|---|---|---|---|---|---|---|")
        for _, r in grid.sort_values(
                ["family", "arm", "rebalance", "cost_bps"]).iterrows():
            A(f"| {r['family']} | `{r['arm']}` | {r['rebalance']} | {r['cost_bps']:.0f} | "
              f"{pct(r['ann_excess_net'])} | {pct(r.get('ann_excess_vs_ew200'))} | "
              f"{num(r['information_ratio'], 2)} | "
              f"{r['turnover_ann']:.2f}× | {pct(r['max_drawdown'])} |")
        A("")

    # ---- 10 point-in-time universe ----------------------------------------
    A("## 10. Point-in-time universe — removing the survivorship bias\n")
    if memb:
        A("The plan listed survivorship as its headline limitation and noted that historical "
          "membership lists, if they exist, would mostly remove it. They do exist: NIFTY "
          "publishes every semi-annual reconstitution as a text-layer PDF, and **six documents "
          "cover this entire test window**. Walking backwards from today's list "
          "(`members_before = (members_after − included) + excluded`) rebuilds actual membership.\n")
        A("| press release | effective | excluded | included |")
        A("|---|---|---|---|")
        for e in memb.get("events", []):
            A(f"| {e['announced']} | {e['effective_from']} | {len(e['excluded'])} | "
              f"{len(e['included'])} |")
        A("")
        A(f"**{len(memb.get('ever_members', []))} distinct names were NIFTY 200 members at some "
          f"point in the window, against 200 today.** Prices were fetched for the removed "
          f"constituents; the reconstructed universe holds **199–200 names per session**. "
          f"Three names have no usable yfinance history and are the residual gap: "
          f"`PEL`, `TATAMTRDVR` (DVR shares cancelled), `GUJGASLTD` (merged).\n")
    if not pitgrid.empty and len(grid):
        A("**Same experiment, two universes.** The difference is the survivorship bias, measured:\n")
        A("| arm | current-membership excess | point-in-time excess | difference |")
        A("|---|---|---|---|")
        for arm in ("last_ret", "mean_ret", "last_z", "momentum_10d", "anchor_reversal",
                    "reversed_last_ret", "equal_weight_universe"):
            a = grid[(grid.arm == arm) & (grid.rebalance == "weekly") & (grid.cost_bps == 30.0)]
            b = pitgrid[(pitgrid.arm == arm) & (pitgrid.rebalance == "weekly")
                        & (pitgrid.cost_bps == 30.0)]
            if len(a) and len(b):
                d = float(b.iloc[0]["ann_excess_net"]) - float(a.iloc[0]["ann_excess_net"])
                A(f"| `{arm}` | {pct(a.iloc[0]['ann_excess_net'])} | "
                  f"{pct(b.iloc[0]['ann_excess_net'])} | {pct(d)} |")
        A("")
    if pitc:
        pp = pitc.get("ic_grid", {}).get("last_ret", {}).get("open_h10", {})
        A(f"Primary IC on the point-in-time universe: {num(pp.get('mean_ic'), 4)} "
          f"(NW t {num(pp.get('t_newey_west'), 2)}, significant {pp.get('significant')}), "
          f"against {num(p.get('mean_ic'), 4)} on the current-membership universe.\n")
        po = pitc.get("orthogonalised_ic", {})
        if po:
            A("| Kronos residualised against (point-in-time) | IC 10d | NW t | significant |")
            A("|---|---|---|---|")
            for k, row in po.items():
                r10 = row.get("open_h10", {})
                A(f"| {k.replace('kronos_resid_of_', '')} | {num(r10.get('mean_ic'), 4)} | "
                  f"{num(r10.get('t_newey_west'), 2)} | {r10.get('significant')} |")
            A("")

    # ---- 11 sensitivity and generalisation --------------------------------
    if extras or usc:
        A("## 11. Does the result survive decoding changes and a second market?\n")
    if extras:
        A("Sampling-regime sensitivity, same dates, same universe:\n")
        A("| decoding | IC 1d | IC 5d | IC 10d | IC 20d | NW t (10d) |")
        A("|---|---|---|---|---|---|")
        base = icg.get("last_ret", {})
        A("| repo default T=0.6, top_p=0.9 | " + " | ".join(
            num(base.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
          + f" | {num(base.get('open_h10', {}).get('t_newey_west'), 2)} |")
        for t, c in extras.items():
            row = c.get("ic_grid", {}).get("last_ret", {})
            A(f"| {t} | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(row.get('open_h10', {}).get('t_newey_west'), 2)} |")
        A("")
        A("`greedy` decodes the model's conditional **mode** (top_k=1). If it shows the same "
          "near-zero IC and the same downward drift, the result is a property of the model, "
          "not of the sampler.\n")
    if usc:
        A("**Second market (extension beyond the plan).** Same code, same horizons, S&P 100 "
          "cross-section, same test window:\n")
        A("| signal | IC 1d | IC 5d | IC 10d | IC 20d | NW t (10d) | significant |")
        A("|---|---|---|---|---|---|---|")
        for col in ("last_ret", "mean_ret", "last_z"):
            row = usc.get("ic_grid", {}).get(col, {})
            r10 = row.get("open_h10", {})
            A(f"| `{col}` | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(r10.get('t_newey_west'), 2)} | {r10.get('significant')} |")
        for name, row in usc.get("baseline_ic", {}).items():
            r10 = row.get("open_h10", {})
            A(f"| {name} | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(r10.get('t_newey_west'), 2)} | {r10.get('significant')} |")
        A("")
        uc = usc.get("cross_sectional_rank_corr_with_kronos", {})
        A(f"Rank correlation of the Kronos score with the free baselines on US data: "
          + ", ".join(f"{k} {num(v, 3)}" for k, v in uc.items()) + ".\n")

    # ---- 12 model-free signal sweep ----------------------------------------
    blsum = j(os.path.join(OUT_NSE, "grid_baselines_summary.json"))
    blp = os.path.join(OUT_NSE, "grid_baselines.parquet")
    bl = pd.read_parquet(blp) if os.path.exists(blp) else pd.DataFrame()
    if blsum:
        A("## 12. What does carry information — model-free sweep\n")
        A(f"All {blsum.get('n_sessions')} test sessions, same engine, same standard errors. "
          "These need no model, so they can be evaluated at full daily resolution.\n")
        A("| signal | IC 1d | IC 5d | IC 10d | IC 20d | NW t (1d) | significant at 1d |")
        A("|---|---|---|---|---|---|---|")
        for k, row in blsum.get("ic", {}).items():
            r1 = row.get("open_h1", {})
            A(f"| `{k}` | " + " | ".join(
                num(row.get(f'open_h{h}', {}).get('mean_ic'), 4) for h in HORIZONS)
              + f" | {num(r1.get('t_newey_west'), 2)} | {r1.get('significant')} |")
        A("")
        A("`kronos_emulator` is `−0.701 × (close − mean₉₀)/std₉₀`, the fitted reproduction of "
          "what Kronos actually computes. `zscore_reversal` is the same factor unscaled.\n")
    if not bl.empty:
        A("Best model-free arms at 30 bps, ranked against the equal-weight universe:\n")
        A("| arm | rebalance | return net | excess vs ^CNX200 | excess vs EW | IR vs EW | "
          "turnover/yr | cost drag |")
        A("|---|---|---|---|---|---|---|---|")
        best = bl[bl.cost_bps == 30.0].sort_values("ann_excess_vs_ew200", ascending=False)
        for _, r in best.head(15).iterrows():
            A(f"| `{r['arm']}` | {r['rebalance']} | {pct(r['ann_return_net'])} | "
              f"{pct(r['ann_excess_net'])} | {pct(r['ann_excess_vs_ew200'])} | "
              f"{num(r['ir_vs_ew200'], 2)} | {r['turnover_ann']:.1f}× | "
              f"{r['cost_drag_ann_pct']:.2f}% |")
        A("")
        r1 = bl[(bl.arm == "reversal_1d") & (bl.rebalance == "daily")].sort_values("cost_bps")
        if len(r1):
            A("Cost sensitivity of the strongest-IC arm (1-day reversal, daily rebalance):\n")
            A("| round-trip cost | excess vs EW | turnover/yr | cost drag |")
            A("|---|---|---|---|")
            for _, r in r1.iterrows():
                A(f"| {r['cost_bps']:.0f} bps | {pct(r['ann_excess_vs_ew200'])} | "
                  f"{r['turnover_ann']:.1f}× | {r['cost_drag_ann_pct']:.2f}%/yr |")
            A("")

    # ---- 9 verification ---------------------------------------------------
    A("## 9. Verification outcomes — which plan claims held\n")
    A("Full detail in `VERIFIED.md`. Machine-readable in `verify_model.json`, "
      "`verify_pipeline.json`, `verify_backtest.json`.\n")
    A("| check | result |")
    A("|---|---|")
    for k, v in list(vp.items()) + list(vb.items()):
        A(f"| `{k}` | {'PASS' if v.get('pass') else 'FAIL'} |")
    A("")
    A("| plan claim | verdict |")
    A("|---|---|")
    A(f"| 4.3 batching is a ~32× win | **REFUTED** — {vm.get('4.3_speedup_vs_batch1')}× at "
      f"lookback=90, best {vm.get('4.3_best', {}).get('ms_per_series')} ms/series at batch "
      f"{vm.get('4.3_best', {}).get('batch')} |")
    A(f"| 4.4 signals live in normalized space | **CONFIRMED** — |pred close| max "
      f"{vm.get('4.4_pred_close_abs_max', 0):.2f} z vs price level "
      f"{vm.get('4.4_raw_close_level_mean', 0):.0f} |")
    A(f"| 4.5 preds include reconstructed context | **CONFIRMED** — shape "
      f"{vm.get('4.5_pred_shape')} |")
    A(f"| 4.5b tokenizer recon error is negligible | **REFUTED** — "
      f"{vm.get('4.5b_recon_err_as_pct_of_signal', 0):.1f}% of signal magnitude |")
    A(f"| 4.7 sample_count>1 destroys the ensemble | **CONFIRMED** — returns "
      f"{vm.get('4.7_shape_with_sample_count_5')}, not a 4-D array |")
    A(f"| 4.11 MPS sampling is nondeterministic | **REFUTED** — bit-identical under a "
      f"fixed seed ({vm.get('4.11a_determinism', {}).get('same_seed', {}).get('bitwise_identical')}) |")
    A("")

    A("---\n")
    A(f"Signals: {smeta.get('rows', '?')} rows over {smeta.get('dates', '?')} dates, "
      f"{smeta.get('ms_per_series', '?')} ms/series, "
      f"{smeta.get('wall_clock_sec', 0) / 3600:.2f} h wall clock. "
      f"Model `{smeta.get('model_repo')}` @ `{str(smeta.get('model_revision'))[:12]}`, "
      f"tokenizer @ `{str(smeta.get('tokenizer_revision'))[:12]}`.\n")

    # Sections were emitted in reading order but numbered against plan section 8.7; renumber
    # them to document order and record the mapping so the plan's checklist stays traceable.
    import re as _re
    mapping, n = [], 0
    out_lines = []
    for line in L:
        m = _re.match(r"^## (\d+b?)\. (.*)$", line)
        if m:
            n += 1
            mapping.append((str(n), m.group(1), m.group(2)))
            line = f"## {n}. {m.group(2)}"
        out_lines.append(line)
    if mapping:
        idx = ["", "---", "", "### Section index",
               "", "| # | plan §8.7 item | section |", "|---|---|---|"]
        idx += [f"| {a} | {b} | {c} |" for a, b, c in mapping]
        out_lines += idx
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    sidecar(args.out.replace(".md", "_meta.json"), vars(args))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
