"""Consolidate rolling backtest + temperature sweep into one figure and a markdown report."""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "bench/out"
roll = json.load(open(f"{OUT}/rolling/summary.json"))
names = list(roll.keys())
A = {k: v["aggregate"] for k, v in roll.items()}

sweep = {}
for p in sorted(glob.glob(f"{OUT}/tsweep_T*/summary.json")):
    T = float(os.path.basename(os.path.dirname(p)).replace("tsweep_T", ""))
    sweep[T] = {k: v["aggregate"] for k, v in json.load(open(p)).items()}
Ts = sorted(sweep)

fig, ax = plt.subplots(2, 2, figsize=(15, 9))

a = ax[0][0]
sk = [A[n]["skill_score_vs_naive_mean"] for n in names]
a.barh(names, sk, color=["#16a34a" if s > 0 else "#dc2626" for s in sk])
a.axvline(0, color="k", lw=1)
a.set_title("Skill score vs naive baseline (8 windows each)\n>0 beats 'tomorrow = today'")
a.set_xlabel("skill = 1 - RMSE_model/RMSE_naive"); a.grid(alpha=.3, axis="x")

a = ax[0][1]
pr = [A[n]["mean_abs_predicted_move_pct"] for n in names]
ac = [A[n]["mean_abs_actual_move_pct"] for n in names]
y = np.arange(len(names)); h = .38
a.barh(y - h/2, ac, h, label="actual |move|", color="#2563eb")
a.barh(y + h/2, pr, h, label="predicted |move|", color="#dc2626")
a.set_yticks(y); a.set_yticklabels(names); a.legend(); a.grid(alpha=.3, axis="x")
a.set_title("Systematic over-extension\nmodel predicts bigger moves than occur")
a.set_xlabel("mean |move| over horizon (%)")

a = ax[1][0]
cov = [A[n]["cone_coverage_p10_p90_mean"] for n in names]
a.barh(names, cov, color="#7c3aed")
a.axvline(.80, color="#16a34a", ls="--", lw=2, label="calibrated p10–p90 = 0.80")
a.set_xlim(0, 1); a.legend(); a.grid(alpha=.3, axis="x")
a.set_title("Predictive-interval calibration\nfraction of actuals inside p10–p90")
a.set_xlabel("coverage")

a = ax[1][1]
for n in sweep[Ts[0]]:
    a.plot(Ts, [sweep[T][n]["skill_score_vs_naive_mean"] for T in Ts], marker="o", lw=2, label=n)
a.axhline(0, color="k", lw=1, ls=":")
a.set_xlabel("sampling temperature T"); a.set_ylabel("skill vs naive")
a.set_title("Temperature sweep (4 windows each)\noptimal T flips direction by asset")
a.legend(); a.grid(alpha=.3)

plt.tight_layout(); plt.savefig(f"{OUT}/summary.png", dpi=130); plt.close()

allw = [w for v in roll.values() for w in v["windows"]]
wins = sum(w["skill_score_vs_naive"] > 0 for w in allw)
lines = [
 "# Kronos-base end-to-end evaluation", "",
 f"Model: Kronos-base (102.3M) + Kronos-Tokenizer-base, 512 context, device=mps (Apple M5, 32GB).",
 f"Protocol: rolling-origin, non-overlapping holdouts, {allw[0]['n_paths'] if 'n_paths' in allw[0] else 8} sampled paths per forecast, p50 as point estimate.",
 f"Baseline: naive persistence (forecast = last observed close).", "",
 f"**{len(allw)} forecasts total. Beat naive in {wins}/{len(allw)} ({wins/len(allw)*100:.0f}%). "
 f"Mean directional accuracy {np.mean([w['directional_accuracy'] for w in allw]):.3f}. "
 f"Mean p10-p90 coverage {np.mean([w['cone_coverage_p10_p90'] for w in allw]):.2f} vs 0.80 nominal.**", "",
 "| dataset | MAPE% | dirAcc | skill | beats naive | p10-p90 cov | pred/actual move |",
 "|---|---|---|---|---|---|---|"]
for n in names:
    x = A[n]
    lines.append(f"| `{n}` | {x['mape_pct_mean']:.2f} | {x['directional_accuracy_mean']:.3f} | "
                 f"{x['skill_score_vs_naive_mean']:+.3f} | {x['beats_naive_win_rate']*100:.0f}% | "
                 f"{x['cone_coverage_p10_p90_mean']:.2f} | "
                 f"{x['mean_abs_predicted_move_pct']/x['mean_abs_actual_move_pct']:.2f}x |")
lines += ["", "## Temperature sweep (skill vs naive, 4 windows)", "",
          "| dataset | " + " | ".join(f"T={T}" for T in Ts) + " |", "|---" * (len(Ts)+1) + "|"]
for n in sweep[Ts[0]]:
    lines.append(f"| `{n}` | " + " | ".join(f"{sweep[T][n]['skill_score_vs_naive_mean']:+.3f}" for T in Ts) + " |")
open(f"{OUT}/RESULTS.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwrote {OUT}/summary.png and {OUT}/RESULTS.md")
