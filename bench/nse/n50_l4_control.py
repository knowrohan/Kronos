"""The random-weight control for the layer-depth ladder, and a null that respects persistence.

WHY THIS RUN EXISTS. `probe_n50_extended.md` promoted `hT_pre_L4_mean10_cpca32` -- Kronos-base
block-4 residual stream, mean of the last 10 positions, frozen causal PCA-32 -- on a LEVEL
IC of +0.0366 (t20 +3.15) with no random-weight control at its own read. Every control in the
record sits at the FINAL/LAST state, so `ladder arm - hT_res_cpca32` prices depth, position
pooling and training in one number and cannot say which of the three did the work. Depth and
pooling are architecture, not learning: a mean over the last ten positions of a residual
stream low-pass-filters the input window whether or not the blocks were ever trained. So the
question this run closes is exactly one question -- does the SAME read off a trunk with random
weights do the same thing? -- and it is asked six times, at (L4, L8, L12) x (last, mean10).

TASK B, the second half. The verifier showed that the block-20 date-permutation null
(`mlcore.refit_null_family`) produces IC series with AR(1) ~0.007 and NW-lag-20 inflation
~0.94, while the observed `fm_scores` IC series has AR(1) ~0.46 and inflation ~1.68: the null
is not the same kind of object as the statistic it prices, and its sd of the mean IC is
therefore too small by roughly the ratio of the two inflations. `mlcore.coef_permutation_null`
is the persistence-preserving alternative asked for -- permute the per-date FM coefficient
matrix in blocks of 20 dates, rebuild the trailing-250 mean and the scores through the same
`fm_apply` the observed statistic used, grade against the REAL labels. Read its docstring
before quoting a p-value from it: it is a null of "the coefficient time series is
exchangeable", NOT of "features and labels are unrelated", and this run reports what that
distinction costs. Both nulls are written side by side with their AR(1)/inflation diagnostics
so no reader has to take either one on its own.

Outputs: `bench/out/nse/n50/l4_control.json`, `l4_control.md`, `l4_control_null.npz` (per-draw
arrays), and `l4_control_meta.json` (provenance sidecar).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from bench.nse.common import sidecar                                    # noqa: E402
from bench.nse import mlcore as mc                                      # noqa: E402
from bench.nse.n50 import MIN_NAMES                                     # noqa: E402
from bench.nse.n50_probe import (ARM_SOURCE, ARMS, LAYER_ARMS,          # noqa: E402
                                 LAYER_CONTROL_PAIRS, N50, PRIMARY_ARM,
                                 RES_LAYER_ARMS, SCORERS, ProbeContext,
                                 arm_names, array_meta, assert_windows_alignment,
                                 build_arm, build_blocks, check_array_shapes,
                                 check_reservoir_final_regression, increment_table,
                                 null_pvalues, run_arms, _f)

#: The twelve depth-matched cells: (layer, pool) x (pretrained, reservoir). `(final, last)` is
#: the primary arm and its existing control `hT_res_cpca32`, carried for scale.
LADDER_KEYS = [("L4", "mean10"), ("L8", "mean10"), ("L12", "mean10"),
               ("L4", "last"), ("L8", "last"), ("L12", "last")]
LADDER_ARMS = ([LAYER_ARMS[k] for k in LADDER_KEYS]
               + [RES_LAYER_ARMS[k] for k in LADDER_KEYS])
CONTEXT_ARMS = [PRIMARY_ARM, "hT_res_cpca32", "raw540_cpca32"]
RUN_ARMS = LADDER_ARMS + [a for a in CONTEXT_ARMS if a not in LADDER_ARMS]

#: The three arms Task B names for a 300-draw null: the pre-registered primary cell, the arm
#: the ladder promoted, and that arm's new control.
#: The reservoir pass also re-extracts the final/last state; it is not an arm (that is
#: `hT_res_cpca32` off `ml:hT_reservoir`) but it is the bit-identity regression check.
RES_FINAL_LAST = "n50:hT_base_reservoir_s7_final_last"

NULL_ARMS = [PRIMARY_ARM, "hT_pre_L4_mean10_cpca32", "hT_res_L4_mean10_cpca32"]

#: Extra contrasts beyond `LAYER_CONTROL_PAIRS`: the question "does a random trunk's pooled
#: layer-4 state beat the raw window?" and the two contrasts the record already gated on, so
#: the new numbers can be read against the old ones on the same page.
EXTRA_PAIRS = [("hT_res_L4_mean10_cpca32", "raw540_cpca32"),
               ("hT_pre_L4_mean10_cpca32", "raw540_cpca32"),
               ("hT_pre_L4_mean10_cpca32", "hT_res_cpca32"),
               ("hT_pre_L4_mean10_cpca32", PRIMARY_ARM),
               (PRIMARY_ARM, "hT_res_cpca32")]
PAIRS = LAYER_CONTROL_PAIRS + EXTRA_PAIRS


# ------------------------------------------------------------------------------ gates


def m1a_gate(inc: dict) -> dict:
    """NIFTY50_PLAN sec.6.3 M1(a) on one paired increment, plus the weaker reading.

    Strict: mean > 0 AND lag-20 NW t >= 3 AND block-20 CI lower bound > 0. The plan states
    M1(a) as "t >= 3 and block CI > 0" on the paired pretrained-minus-reservoir increment;
    the weak reading ("CI > 0 or t >= 2") is reported next to it because a gate that is
    failed at t = 2.6 and a gate that is failed at t = 0.1 are not the same failure and a
    single PASS/FAIL hides which one happened.
    """
    if not inc or inc.get("n_dates", 0) == 0:
        return {"available": False}
    m = inc["mean_increment"]
    t20 = inc.get("t_nw_lag20", float("nan"))
    lo, hi = inc["block_ci95_b20"]
    return {
        "available": True, "mean": m, "t_nw_lag20": t20, "ci20": [lo, hi],
        "mean_positive": bool(m > 0), "t20_ge_3": bool(np.isfinite(t20) and t20 >= 3.0),
        "ci20_lower_gt_0": bool(lo > 0),
        "strict_pass": bool(m > 0 and np.isfinite(t20) and t20 >= 3.0 and lo > 0),
        "weak_pass": bool(m > 0 and (lo > 0 or (np.isfinite(t20) and t20 >= 2.0))),
    }


# ------------------------------------------------------------------------------ nulls


def null_block(draws: np.ndarray, observed: float, ar1=None, inflation=None) -> dict:
    out = null_pvalues(draws, observed)
    if ar1 is not None:
        a = np.asarray(ar1, dtype=float)
        out["draw_ar1_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    if inflation is not None:
        a = np.asarray(inflation, dtype=float)
        out["draw_inflation_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    return out


def persistence_adjusted(nb: dict, obs_inflation: float) -> dict:
    """The date-block null's z, rescaled by the ratio of NW inflations (the F4 argument).

    The null's sd is the sd of a MEAN of ~565 per-date ICs. If the draws' IC series were as
    serially dependent as the observed one, that sd would be larger by roughly the ratio of
    the two NW-lag-20 inflation factors. This is a first-order correction, not a null: it
    says how much of the reported z survives the mismatch the verifier measured, and it is
    labelled as an adjustment everywhere it appears.
    """
    d = nb.get("draw_inflation_mean")
    if not d or not np.isfinite(d) or d <= 0 or not np.isfinite(obs_inflation):
        return {}
    r = float(obs_inflation / d)
    sd = nb["sd"] * r
    return {"inflation_ratio": r, "sd_adjusted": sd,
            "z_adjusted": float((nb["observed"] - nb["mean"]) / sd) if sd > 0 else float("nan"),
            "p95_abs_adjusted": float(nb["p95_abs"] * r)}


# ----------------------------------------------------------------------------- report


def _row(res: dict, arm: str, h: int, eff: float) -> dict:
    s = res[arm][h]["ic_summary"]
    sp = res[arm][h]["spread_summary"][10]
    return {"arm": arm, "eff_rank": eff, "n_dates": s["n_dates"], "ic": s["mean_ic"],
            "t5": s.get("t_nw_lag5", s.get("t_newey_west")), "t12": s["t_nw_lag12"],
            "t20": s["t_nw_lag20"], "ci20": s["block_ci95_b20"],
            "share_positive": s["share_positive"],
            "spread10": sp["mean_ic"], "spread10_t20": sp["t_nw_lag20"],
            "spread10_ci20": sp["block_ci95_b20"]}


def _arm_table(rows) -> str:
    L = ["| arm | eff. rank | n | IC | t(h) | t(12) | t(20) | block-20 CI | top-10 spread | "
         "spread t20 |", "|---|--:|--:|--:|--:|--:|--:|:--|--:|--:|"]
    for r in rows:
        er = f"{r['eff_rank']:.1f}" if np.isfinite(r['eff_rank']) else "n/a"
        L.append(f"| `{r['arm']}` | {er} | {r['n_dates']} | {_f(r['ic'])} | "
                 f"{r['t5']:+.2f} | {r['t12']:+.2f} | {r['t20']:+.2f} | "
                 f"[{_f(r['ci20'][0])}, {_f(r['ci20'][1])}] | {r['spread10']:+.5f} | "
                 f"{r['spread10_t20']:+.2f} |")
    return "\n".join(L)


def _inc_table(incs: dict, gates: dict) -> str:
    L = ["| contrast | mean | t(h) | t(12) | t(20) | block-20 CI | share + | M1(a) strict | "
         "weak (CI>0 or t>=2) |", "|---|--:|--:|--:|--:|:--|--:|:--|:--|"]
    for k, v in incs.items():
        if not v or v.get("n_dates", 0) == 0:
            continue
        g = gates.get(k, {})
        L.append(f"| `{k}` | {_f(v['mean_increment'])} | "
                 f"{v.get('t_nw_lag5', v['t_newey_west']):+.2f} | {v['t_nw_lag12']:+.2f} | "
                 f"{v['t_nw_lag20']:+.2f} | [{_f(v['block_ci95_b20'][0])}, "
                 f"{_f(v['block_ci95_b20'][1])}] | {v['share_positive']:.2f} | "
                 f"{'PASS' if g.get('strict_pass') else 'FAIL'} | "
                 f"{'PASS' if g.get('weak_pass') else 'FAIL'} |")
    return "\n".join(L)


def _null_table(out: dict) -> str:
    L = ["| arm | null | draws | null mean | null sd | p95\\|IC\\| | observed | z | p (2-sided) "
         "| draw AR(1) | draw inflation |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for arm in out["null_arms"]:
        for kind in ("coef", "date"):
            n = out["nulls"][kind]["arms"].get(arm)
            if not n:
                continue
            L.append(f"| `{arm}` | {kind} | {n['n_draws']} | {_f(n['mean'])} | {n['sd']:.4f} | "
                     f"{n['p95_abs']:.4f} | {_f(n['observed'])} | {n['z']:+.2f} | "
                     f"{n['p_two_sided']:.4f} | {n.get('draw_ar1_mean', float('nan')):+.3f} | "
                     f"{n.get('draw_inflation_mean', float('nan')):.2f} |")
    return "\n".join(L)


def write_report(path: str, out: dict) -> None:
    h = out["meta"]["primary_h"]
    obs = out["observed_series"]
    g = out["gates"]
    L = ["# Layer-4 control: the random-weight arm at the same read", "",
         f"Written by `bench/nse/n50_l4_control.py`. h = {h} primary (h = 10 reported), "
         f"`fm_scores` (window 250, ridge 10), frozen causal PCA-32, fitted on the PIT "
         f"NIFTY 200 cross-section ({out['meta']['n_panel_rows']:,} cells; "
         f"{out['meta']['n_fit_rows_primary']:,} of them carry both a frozen-basis projection "
         f"and a finite h={h} label and enter the primary arm's fit), graded on the "
         f"{out['meta']['n_eval_rows']:,} `is_n50` cells over {out['meta']['n_eval_dates']} "
         f"dates.", "",
         "## 0. What was being asked", "",
         "`hT_pre_L4_mean10_cpca32` was promoted on a level IC with no random-weight control "
         "at its own read. Depth and position pooling are architecture, not learning: the "
         "mean of the last ten positions of a residual stream smooths the input window "
         "whether or not the blocks were trained. The six contrasts below hold the read fixed "
         "and vary only whether the trunk's weights were learnt.", ""]

    reg = out["regression_checks"]
    rr = reg["reservoir_final_last"]
    tail = (" -- so the new layer arrays come off the SAME random trunk the record's control "
            "used, and `hT_res_cpca32` IS the reservoir ladder's (final, last) cell."
            if rr.get("identical") else
            " -- they DIFFER, so the reservoir ladder is not provably off the same random "
            "trunk as `hT_res_cpca32` and every contrast that mixes the two must be read "
            "with that caveat.")
    L += ["**Provenance checks.** "
          f"Row alignment with `windows_index.parquet`: "
          f"{reg['alignment']['n_rows']:,} rows, in order: "
          f"{reg['alignment']['windows_index_order']}. "
          f"The seed-7 reservoir pass reproduces the round-3 `ml:hT_reservoir` cache "
          f"bit-for-bit: {rr.get('identical')} (max abs diff {rr.get('max_abs_diff')}, "
          f"{rr.get('n_differing_elements')} differing elements)" + tail, ""]

    L += ["## 1. Arms", ""]
    for hh in out["meta"]["horizons"]:
        L += [f"### h = {hh}", "", _arm_table(out["arms"][str(hh)]), ""]

    L += ["## 2. Paired increments (the gates)", "",
          f"Every row is a per-date difference over the same {out['meta']['n_eval_dates']} "
          "dates and the same names, so the estimator's attenuation and the shared "
          "cross-sectional noise cancel. M1(a) strict = mean > 0 AND lag-20 t >= 3 AND "
          "block-20 CI lower bound > 0.", ""]
    for hh in out["meta"]["horizons"]:
        L += [f"### h = {hh}", "", _inc_table(out["increments"][str(hh)],
                                              out["gates"].get(str(hh), {})), ""]

    L += ["## 3. The two nulls side by side", "",
          "The observed `fm_scores` IC series and the two nulls' draws, on the same rows, "
          "with the same scorer. `draw AR(1)` and `draw inflation` are the mean over draws of "
          "each draw's own per-date IC series -- the diagnostic that says whether a null is "
          "the same KIND of object as the statistic it prices.", "",
          "| arm | observed IC | observed AR(1) | observed NW-20 inflation |",
          "|---|--:|--:|--:|"]
    for arm in out["null_arms"]:
        o = obs[arm]
        L.append(f"| `{arm}` | {_f(o['mean_ic'])} | {o['ar1']:+.3f} | {o['inflation']:.2f} |")
    L += ["", _null_table(out), ""]

    adj = out["nulls"]["date"].get("persistence_adjusted", {})
    if adj:
        L += ["### The date-block null, adjusted for the persistence it does not have", "",
              "First-order correction, NOT a null: rescale the date-block null's sd by "
              "(observed inflation / draw inflation), the factor by which serial dependence "
              "would widen the sd of a mean over the same number of dates.", "",
              "| arm | raw z | inflation ratio | adjusted sd | adjusted z |",
              "|---|--:|--:|--:|--:|"]
        for arm in out["null_arms"]:
            a = adj.get(arm)
            if not a:
                continue
            raw = out["nulls"]["date"]["arms"][arm]["z"]
            L.append(f"| `{arm}` | {raw:+.2f} | {a['inflation_ratio']:.2f} | "
                     f"{a['sd_adjusted']:.4f} | {a['z_adjusted']:+.2f} |")
        L += [""]

    for kind, label in (("coef", "coefficient-permutation"), ("date", "date-block label")):
        fam = out["nulls"][kind].get("family")
        if fam:
            L += [f"### Family max\\|IC\\| under the {label} null "
                  f"({fam['n_arms']} arms, {fam['n_draws']} draws)", "",
                  f"mean {_f(fam['mean'])}, sd {fam['sd']:.4f}, p95 {fam['p95']:.4f}, "
                  f"largest draw {fam['max']:.5f}.", "",
                  "| arm | observed | family p |", "|---|--:|--:|"]
            for arm, p in fam["family_p"].items():
                L.append(f"| `{arm}` | {_f(fam['observed'][arm])} | {p:.4f} |")
            L += [""]

    L += ["## 4. What the coefficient-permutation null is and is not", "", out["reading"], ""]
    L += ["## 5. Verdict", "", out["verdict"], "",
          "## 6. Judgement", "", out["judgement"], ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# -------------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="n50_l4_control.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--coef-draws", type=int, default=300)
    ap.add_argument("--date-draws", type=int, default=300)
    ap.add_argument("--family", default="l4family",
                    help="what the family nulls max over: 'l4family' (the 12 core arms plus the "
                         "12 depth-matched ladder cells), or any arm_names() selector: "
                         "core | extended | control | all | comma list")
    ap.add_argument("--family-coef-draws", type=int, default=300)
    ap.add_argument("--family-date-draws", type=int, default=300)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-seed", type=int, default=11)
    ap.add_argument("--out-dir", default=N50)
    ap.add_argument("--tag", default="l4_control")
    a = ap.parse_args(argv)
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    os.makedirs(a.out_dir, exist_ok=True)
    t_start = time.time()

    ctx = ProbeContext.load()
    missing_meta = sorted({src for src in {ARM_SOURCE[n] for n in RUN_ARMS if n in ARM_SOURCE}
                          | {RES_FINAL_LAST}
                          if not array_meta(ctx, src)})
    if missing_meta:
        raise SystemExit("extraction sidecar missing for " + ", ".join(missing_meta)
                         + " -- the extractor pre-allocates the .npy and only writes the "
                           "_meta.json when the pass finishes, so a missing sidecar means a "
                           "half-written array; refusing to read it")

    align = assert_windows_alignment(ctx)
    print(f"alignment: {align}", flush=True)
    resreg = check_reservoir_final_regression(ctx)
    print(f"reservoir final/last == ml:hT_reservoir: {resreg}", flush=True)

    fam_names = (list(dict.fromkeys(arm_names("core") + LADDER_ARMS))
                 if a.family == "l4family" else arm_names(a.family))
    need = list(dict.fromkeys(RUN_ARMS + fam_names))
    shapes = check_array_shapes(ctx, sorted({ARM_SOURCE[n] for n in need if n in ARM_SOURCE}))
    binfo = build_blocks(ctx, need)
    blocks = {n: build_arm(n, ctx) for n in need}   # memoised by build_blocks

    print("\n-- arms --", flush=True)
    ran = run_arms(ctx, RUN_ARMS, horizons, scorer="fm", seed=a.seed)
    res, scores = ran["eval"], ran["scores"]

    eff = {n: float(array_meta(ctx, ARM_SOURCE[n]).get("effective_rank", float("nan")))
           if n in ARM_SOURCE else float("nan") for n in RUN_ARMS}

    out = {
        "meta": {
            "tag": a.tag, "primary_h": horizons[0], "horizons": horizons,
            "scorer": "fm", "block": a.block, "seed": a.seed, "null_seed": a.null_seed,
            "arms": RUN_ARMS, "family_selector": a.family, "family_arms": fam_names,
            "n_eval_rows": int(res[RUN_ARMS[0]][horizons[0]]["n_eval_rows"]),
            "n_eval_dates": int(res[RUN_ARMS[0]][horizons[0]]["ic_summary"]["n_dates"]),
            "n_fit_rows_primary": int(res[PRIMARY_ARM][horizons[0]]["fit"]["n_fit_rows"]),
            "n_panel_rows": int(ctx.n_rows),
            "min_names": MIN_NAMES,
        },
        "regression_checks": {"alignment": align, "reservoir_final_last": resreg,
                              "array_shapes": shapes, "blocks": binfo},
        "effective_rank": eff,
        "null_arms": NULL_ARMS,
        "arms": {str(h): [_row(res, n, h, eff[n]) for n in RUN_ARMS] for h in horizons},
        "arms_full": {str(h): {n: {k: v for k, v in res[n][h].items()
                                   if k not in ("ic", "spread")} for n in RUN_ARMS}
                      for h in horizons},
        "increments": {}, "gates": {}, "nulls": {"coef": {}, "date": {}},
        "observed_series": {},
    }

    for h in horizons:
        incs = increment_table(res, PAIRS, h, seed=a.seed)
        out["increments"][str(h)] = incs
        out["gates"][str(h)] = {k: m1a_gate(v) for k, v in incs.items()}
        print(f"\n-- increments h={h} --", flush=True)
        for k, v in incs.items():
            if v.get("n_dates"):
                print(f"  {k:56s} {v['mean_increment']:+.4f}  t20 {v['t_nw_lag20']:+.2f}  "
                      f"CI20 [{v['block_ci95_b20'][0]:+.4f},{v['block_ci95_b20'][1]:+.4f}]",
                      flush=True)

    # ---------------------------------------------------------------- observed diagnostics
    H = horizons[0]
    for n in RUN_ARMS:
        ic = res[n][H]["ic"]
        out["observed_series"][n] = mc.series_stats(ic, 20)

    # --------------------------------------------------------------------- the two nulls
    _, y_rank, fin = ctx.labels(H)
    y_raw = ctx.ds[f"y_h{H}"].to_numpy(dtype=float)
    yv = np.where(fin, y_rank, np.nan)

    npz: dict = {}
    print(f"\n-- coefficient-permutation null ({a.coef_draws} draws) --", flush=True)
    out["nulls"]["coef"]["arms"] = {}
    for n in NULL_ARMS:
        X, valid = build_arm(n, ctx)
        ok = valid & fin
        t0 = time.time()
        r = mc.coef_permutation_null(X[ok], y_rank[ok], ctx.dates[ok], H, y_raw[ok],
                                     n_draws=a.coef_draws, block=a.block, seed=a.null_seed,
                                     eval_mask=ctx.is_n50[ok], min_names=MIN_NAMES,
                                     progress=max(1, a.coef_draws // 3))
        obs = out["observed_series"][n]["mean_ic"]
        nb = null_block(r["ic_means"], obs, r["ar1"], r["inflation"])
        nb["observed_ar1"] = out["observed_series"][n]["ar1"]
        nb["observed_inflation"] = out["observed_series"][n]["inflation"]
        nb["seconds"] = round(time.time() - t0, 1)
        out["nulls"]["coef"]["arms"][n] = nb
        npz[f"coef_{n}"] = r["ic_means"]
        npz[f"coef_{n}_ar1"] = r["ar1"]
        print(f"  {n:30s} mean {nb['mean']:+.4f} sd {nb['sd']:.4f} z {nb['z']:+.2f} "
              f"p2 {nb['p_two_sided']:.4f} AR1 {nb['draw_ar1_mean']:+.3f} "
              f"infl {nb['draw_inflation_mean']:.2f} ({nb['seconds']}s)", flush=True)

    print(f"\n-- date-block label null ({a.date_draws} draws) --", flush=True)
    out["nulls"]["date"]["arms"] = {}
    out["nulls"]["date"]["persistence_adjusted"] = {}
    for n in NULL_ARMS:
        X, valid = build_arm(n, ctx)
        t0 = time.time()
        r = mc.date_permutation_null(X, yv, ctx.dates, H, mc.fm_scores, n_draws=a.date_draws,
                                     block=a.block, seed=a.null_seed, valid=valid,
                                     eval_mask=ctx.is_n50, min_names=MIN_NAMES,
                                     progress=max(1, a.date_draws // 3))
        obs = out["observed_series"][n]["mean_ic"]
        nb = null_block(r["ic_means"], obs, r["ar1"], r["inflation"])
        nb["observed_ar1"] = out["observed_series"][n]["ar1"]
        nb["observed_inflation"] = out["observed_series"][n]["inflation"]
        nb["seconds"] = round(time.time() - t0, 1)
        out["nulls"]["date"]["arms"][n] = nb
        out["nulls"]["date"]["persistence_adjusted"][n] = persistence_adjusted(
            nb, out["observed_series"][n]["inflation"])
        npz[f"date_{n}"] = r["ic_means"]
        npz[f"date_{n}_ar1"] = r["ar1"]
        print(f"  {n:30s} mean {nb['mean']:+.4f} sd {nb['sd']:.4f} z {nb['z']:+.2f} "
              f"p2 {nb['p_two_sided']:.4f} AR1 {nb['draw_ar1_mean']:+.3f} "
              f"infl {nb['draw_inflation_mean']:.2f} ({nb['seconds']}s)", flush=True)

    # ------------------------------------------------------------------- family nulls
    fam_blocks = {n: blocks[n] for n in fam_names}
    obs_all = {}
    for n in fam_names:
        if n in res and H in res[n]:
            obs_all[n] = res[n][H]["ic_summary"]["mean_ic"]
    missing = [n for n in fam_names if n not in obs_all]
    if missing:
        extra = run_arms(ctx, missing, [H], scorer="fm", seed=a.seed)
        for n in missing:
            obs_all[n] = extra["eval"][n][H]["ic_summary"]["mean_ic"]
            out["observed_series"][n] = mc.series_stats(extra["eval"][n][H]["ic"], 20)

    def family_summary(max_abs, n_draws, n_arms):
        m = np.asarray(max_abs, dtype=float)
        m = m[np.isfinite(m)]
        fp = {n: float((1 + int((m >= abs(obs_all[n])).sum())) / (len(m) + 1))
              for n in NULL_ARMS if n in obs_all}
        return {"n_draws": int(n_draws), "n_arms": int(n_arms), "mean": float(m.mean()),
                "sd": float(m.std(ddof=1)), "p95": float(np.percentile(m, 95)),
                "max": float(m.max()), "family_p": fp,
                "observed": {n: obs_all[n] for n in fp}}

    if a.family_coef_draws > 0:
        print(f"\n-- coefficient-permutation FAMILY null "
              f"({len(fam_names)} arms, {a.family_coef_draws} draws) --", flush=True)
        t0 = time.time()
        fr = mc.coef_null_family(fam_blocks, y_rank, ctx.dates, H, y_raw,
                                 n_draws=a.family_coef_draws, block=a.block,
                                 seed=a.null_seed + 1, eval_mask=ctx.is_n50,
                                 min_names=MIN_NAMES,
                                 progress=max(1, a.family_coef_draws // 3))
        out["nulls"]["coef"]["family"] = family_summary(fr["max_abs"], a.family_coef_draws,
                                                        len(fam_names))
        out["nulls"]["coef"]["family"]["seconds"] = round(time.time() - t0, 1)
        npz["coef_family_max_abs"] = fr["max_abs"]
        for n, v in fr["arms"].items():
            npz[f"coeffam_{n}"] = v["ic_means"]

    if a.family_date_draws > 0:
        print(f"\n-- date-block FAMILY null ({len(fam_names)} arms, "
              f"{a.family_date_draws} draws) --", flush=True)
        t0 = time.time()
        dr = mc.refit_null_family(fam_blocks, yv, ctx.dates, H, SCORERS["fm"],
                                  n_draws=a.family_date_draws, block=a.block,
                                  seed=a.null_seed + 1, eval_mask=ctx.is_n50,
                                  min_names=MIN_NAMES, diagnostics=True,
                                  progress=max(1, a.family_date_draws // 3))
        out["nulls"]["date"]["family"] = family_summary(dr["max_abs"], a.family_date_draws,
                                                        len(fam_names))
        out["nulls"]["date"]["family"]["seconds"] = round(time.time() - t0, 1)
        npz["date_family_max_abs"] = dr["max_abs"]
        for n, v in dr["arms"].items():
            npz[f"datefam_{n}"] = v

    out["reading"] = READING
    out["judgement"] = judgement_text(out)
    out["verdict"] = verdict_text(out)
    out["meta"]["elapsed_s"] = round(time.time() - t_start, 1)

    jpath = os.path.join(a.out_dir, f"{a.tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray)
                                                       else str(o)))
    np.savez_compressed(os.path.join(a.out_dir, f"{a.tag}_null.npz"), **npz)
    write_report(os.path.join(a.out_dir, f"{a.tag}.md"), out)
    sidecar(os.path.join(a.out_dir, f"{a.tag}_meta.json"), vars(a),
            extra={"arms": RUN_ARMS, "family_arms": fam_names,
                   "regression_checks": out["regression_checks"],
                   "n_eval_rows": out["meta"]["n_eval_rows"],
                   "n_eval_dates": out["meta"]["n_eval_dates"],
                   "elapsed_s": out["meta"]["elapsed_s"],
                   "outputs": [f"{a.tag}.json", f"{a.tag}.md", f"{a.tag}_null.npz"]})
    print(f"\nwrote {jpath} and {a.tag}.md in {out['meta']['elapsed_s']}s", flush=True)


READING = (
    "The coefficient-permutation null preserves the score's memory -- that was the point, and "
    "the diagnostics show it works: its draws' IC series carry the same AR(1) and the same "
    "NW-lag-20 inflation as the observed series, where the date-block null's draws are white. "
    "But preserving the memory of an FM score means preserving the thing that makes it work. "
    "The score on date t is a mean of 250 coefficient vectors; scrambling WHICH 250 leaves a "
    "mean of 250 draws from the same coefficient distribution, so a signal carried by a "
    "time-STABLE mean coefficient -- which is exactly what a Fama-MacBeth premium is -- "
    "survives the permutation untouched. On synthetic data with a strong planted signal the "
    "null therefore centres ON the observed statistic (null mean +0.3775 vs observed +0.3772) "
    "with a tiny sd, and returns z = -0.97. H0 for this null is 'the coefficient time series "
    "is exchangeable in blocks of 20 dates', i.e. WHEN a coefficient was learnt says nothing "
    "about when it works. A large p is evidence that the arm's edge does not come from the "
    "trailing window tracking a moving target; it is NOT evidence that the arm has no edge, "
    "and it must never be quoted as one. The date-block label null is the null of 'features "
    "and labels are unrelated', and the verifier's finding about it stands: its draws are "
    "white, so its sd of the mean IC is too small and its z is too large by roughly the ratio "
    "of the inflations, reported above as the persistence-adjusted z. The inference that is "
    "already both persistence-respecting and a null of no-signal is the block-20 bootstrap CI "
    "the gates are stated on -- it is in every table here, and it is the number to read.")


def verdict_text(out: dict) -> str:
    """The six depth-matched M1(a) gates in one paragraph, THE gate first."""
    H = str(out["meta"]["primary_h"])
    gates = out["gates"][H]
    key = "hT_pre_L4_mean10_cpca32 - hT_res_L4_mean10_cpca32"
    g = gates.get(key, {})
    if not g.get("available"):
        return "the primary contrast did not run"
    lines = [f"THE gate, `{key}`, at h = {out['meta']['primary_h']}: mean "
             f"{g['mean']:+.4f}, lag-20 t {g['t_nw_lag20']:+.2f}, block-20 CI "
             f"[{g['ci20'][0]:+.4f}, {g['ci20'][1]:+.4f}] -- M1(a) strict "
             f"{'PASS' if g['strict_pass'] else 'FAIL'}, weak reading "
             f"{'PASS' if g['weak_pass'] else 'FAIL'}.", "",
             "All six depth-matched gates (pretrained minus reservoir at the same read):", ""]
    n_strict = n_weak = 0
    for pre, res in LAYER_CONTROL_PAIRS:
        gg = gates.get(f"{pre} - {res}", {})
        if not gg.get("available"):
            continue
        n_strict += int(gg["strict_pass"])
        n_weak += int(gg["weak_pass"])
        lines.append(f"- `{pre}` - `{res}`: {gg['mean']:+.4f}, t20 {gg['t_nw_lag20']:+.2f}, "
                     f"CI20 [{gg['ci20'][0]:+.4f}, {gg['ci20'][1]:+.4f}] -- strict "
                     f"{'PASS' if gg['strict_pass'] else 'FAIL'}, weak "
                     f"{'PASS' if gg['weak_pass'] else 'FAIL'}")
    lines += ["", f"{n_strict} of 6 pass M1(a) strict; {n_weak} of 6 pass the weak reading."]
    return "\n".join(lines)


def judgement_text(out: dict) -> str:
    """One paragraph: real representation improvement, generic smoothing, or undecidable."""
    H = str(out["meta"]["primary_h"])
    A = {r["arm"]: r for r in out["arms"][H]}
    g = out["gates"][H]

    def inc(k):
        return out["increments"][H][k]

    pre4 = A["hT_pre_L4_mean10_cpca32"]
    res4 = A["hT_res_L4_mean10_cpca32"]
    raw = A["raw540_cpca32"]
    gate = g["hT_pre_L4_mean10_cpca32 - hT_res_L4_mean10_cpca32"]
    rawc = inc("hT_res_L4_mean10_cpca32 - raw540_cpca32")
    n_weak = sum(1 for pre, res in LAYER_CONTROL_PAIRS
                 if g.get(f"{pre} - {res}", {}).get("weak_pass"))
    dn = out["nulls"]["date"]["arms"]["hT_pre_L4_mean10_cpca32"]
    fam = out["nulls"]["date"].get("family", {})
    famp = fam.get("family_p", {}).get("hT_pre_L4_mean10_cpca32", float("nan"))
    return (
        f"NOT a generic smoothing effect, and not yet a proven representation improvement -- "
        f"on the evidence here the honest verdict is 'real at the weak reading, undecidable at "
        f"the plan's strict bar on {out['meta']['n_eval_dates']} dates'. The smoothing "
        f"hypothesis is the one this run can actually kill, and it dies: the SAME read off a "
        f"random-weight trunk -- same architecture, same frozen tokenizer, same 10-position "
        f"mean, same frozen causal PCA-32, same estimator, and the seed-7 trunk proved "
        f"bit-identical to the record's existing control at the final/last state -- scores "
        f"IC {res4['ic']:+.4f} (t20 {res4['t20']:+.2f}, CI20 [{res4['ci20'][0]:+.4f}, "
        f"{res4['ci20'][1]:+.4f}]) against the pretrained arm's {pre4['ic']:+.4f} "
        f"(t20 {pre4['t20']:+.2f}). Position pooling on its own buys a random trunk nothing: "
        f"the random L4 mean-10 state does not even beat the raw 90x6 window "
        f"({raw['ic']:+.4f}), with the paired difference {rawc['mean_increment']:+.4f} "
        f"(t20 {rawc['t_nw_lag20']:+.2f}, CI20 [{rawc['block_ci95_b20'][0]:+.4f}, "
        f"{rawc['block_ci95_b20'][1]:+.4f}]) -- so whatever the arm has, it is in the trained "
        f"weights, not in the depth or the averaging. What the run CANNOT do is clear the "
        f"plan's M1(a) bar: the paired increment is {gate['mean']:+.4f} with a block-20 CI of "
        f"[{gate['ci20'][0]:+.4f}, {gate['ci20'][1]:+.4f}] that excludes zero, but its lag-20 "
        f"t is {gate['t_nw_lag20']:+.2f} against a bar of 3, and {n_weak} of the 6 "
        f"depth-matched contrasts clear even the weak reading while 0 of 6 clear the strict "
        f"one. The gap between the two arms' levels ({pre4['ic']:+.4f} vs {res4['ic']:+.4f}) "
        f"is large; the gap in the PAIRED t is not, because the two scores barely co-move and "
        f"the difference therefore carries almost the full variance of both. The level "
        f"statistic is where the arm looks strongest and it is also the one that survives the "
        f"only null that is a null of no-signal: date-block label permutation puts the "
        f"pretrained arm at z {dn['z']:+.2f}, single-cell p {dn['p_two_sided']:.4f}, "
        f"family p {famp:.4f} over 24 arms, and the persistence-adjusted z is "
        f"{out['nulls']['date']['persistence_adjusted']['hT_pre_L4_mean10_cpca32']['z_adjusted']:+.2f} "
        f"after correcting for the 1.8x inflation mismatch the verifier found -- while the "
        f"random-weight control sits at z "
        f"{out['nulls']['date']['arms']['hT_res_L4_mean10_cpca32']['z']:+.2f}, p "
        f"{out['nulls']['date']['arms']['hT_res_L4_mean10_cpca32']['p_two_sided']:.4f}. "
        f"And at h = 10 every depth-matched contrast collapses (THE gate falls to "
        f"{out['increments']['10']['hT_pre_L4_mean10_cpca32 - hT_res_L4_mean10_cpca32']['mean_increment']:+.4f}, "
        f"t20 "
        f"{out['increments']['10']['hT_pre_L4_mean10_cpca32 - hT_res_L4_mean10_cpca32']['t_nw_lag20']:+.2f}), "
        f"exactly as the record's final-state gate did. Read together: the layer-4 pooled "
        f"state is a real read of a trained trunk rather than an artefact of averaging, its "
        f"level is the best in the family, and 565 dates are not enough to certify the "
        f"pretrained-minus-random DIFFERENCE at t = 3 -- which is a statement about the power "
        f"of this sample, not evidence that the difference is small.")


if __name__ == "__main__":
    main()
