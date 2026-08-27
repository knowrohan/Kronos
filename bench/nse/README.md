# `bench/nse/` — Kronos as a cross-sectional stock ranker on the NIFTY 200

Implementation of `bench/NSE_RANKING_PLAN.md`. Nothing under `model/`, `finetune/`, `webui/` or
`examples/` is modified; where repo behaviour had to change it was copied here and changed in
the copy, and `finetune/config.py` is adjusted only by patching `Config.__init__` at runtime.

**Start here:** [`RESULTS.md`](../out/nse/RESULTS.md) for the full grid ·
[`findings.html`](../out/nse/findings.html) for the writeup ·
[`VERIFIED.md`](VERIFIED.md) for what the plan got right and wrong ·
[`PREREG.md`](PREREG.md) for the pre-registration and its deviations.

## Reproducing

```bash
cd /Users/rap/Documents/z-projects/Kronos
bash bench/nse/run_all.sh          # ~8 GPU-hours on an M5 / 32 GB, every stage resumable
```

Each script is independently runnable, takes `--help`, and writes a JSON provenance sidecar
(argv, git SHA, package versions, pinned HF revisions) beside whatever it produces.

## Pipeline

| stage | script | what it does | cost |
|---|---|---|---|
| data | `fetch_universe.py` | NIFTY 200 constituents + daily OHLCV + benchmarks. Keeps both `close` (split-adjusted) and `adj_close` (total-return) so the dividend convention is an explicit downstream choice. | 15 s |
| data | `preprocess.py` | Emits the repo's exact pickle contract (`vol`/`amt` naming, `amt=(o+h+l+c)/4·vol`), the trading calendar, and the train/val/test split. | 10 s |
| data | `reconstruct_membership.py` | Rebuilds **point-in-time** NIFTY 200 membership from six NSE reconstitution PDFs, removing survivorship bias. | 20 s |
| verify | `verify_model.py` | Plan §4.3/4.4/4.5/4.7/4.11 — batch throughput, normalized-space check, ensemble collapse, MPS determinism, memory ceiling. | 4 min |
| verify | `verify_pipeline.py` | Plan §10.1/10.2/10.3/10.5 — pickle contract through the *unmodified* `QlibDataset`, seed reproducibility, cross-check against `bench/kronos_eval.py`, lookahead assertions. | 2 min |
| verify | `verify_backtest.py` | Plan §10.4/10.6/10.7/10.8 — perfect foresight, constant signal, shifted signal, cost reconciliation, universe integrity, IC machinery in both directions. | 10 s |
| signal | `signals.py` | The expensive stage. Batched inference → 12 score columns per (date, symbol) from one forward pass. Crash-safe: one parquet part per date, `--resume` by default. | ~5 h |
| signal | `diagnose_drift.py` | Dumps the predicted path shape per step, sampled vs greedy — the mechanism behind everything. | 5 min |
| analysis | `backtest.py` | Line-for-line port of qlib 0.9.7 `TopkDropoutStrategy`, with the execution lag asserted in code. | — |
| analysis | `costs.py` | Indian delivery cost model. Every rate a named parameter; handles the non-scale-invariance the plan's draft missed. | — |
| analysis | `controls.py` | IC with Newey–West and block-bootstrap errors, plus every control including the orthogonalised-IC test. | 2 min |
| analysis | `grid.py` | The full 96-cell grid plus the 500-seed random-signal distribution. | 4 min |
| analysis | `baseline_sweep.py` | Model-free signals (momentum and reversal at 7 horizons, the 90-bar z-score factor, and a fitted emulator of what Kronos actually computes) at **full daily resolution** — no GPU, so it can use all 756 sessions. | 8 min |
| analysis | `universe_bias.py` | Splits the equal-weight gap into an investable size tilt and pure survivorship, calibrated against real NSE size indices. | 20 s |
| analysis | `robustness.py` | The post-hoc stage the sweep needs: a **family-wise null** (the maximum over all 80 searched cells when none of them has information), sub-period stability, a capacity curve under the real cost function, an impact-based liquidity ceiling, and blends. | 70 min |
| analysis | `surface.py` | Sweeps 735 cells around the reported winner. Answers whether it is a plateau or a spike, and whether each pure arm decays smoothly in its own lookback. | 6 min |
| analysis | `benchmark_audit.py` | Prices `ew200` itself — every rebalance cadence from daily to never, free and net of real cost, on both universes. Also recovers the survivorship effect as a sign flip in the rebalancing premium. | 40 s |
| analysis | `vol_signals.py` | Tests the two Kronos scalars the directional argument does not cover: predicted path range and path asymmetry, against realized volatility, conditioned on trailing volatility. No GPU — the values were already in the signal file. | 30 s |
| extension | `fetch_us.py` | Same artifacts for an S&P 100 cross-section — is the failure India-specific? | 20 s |
| model | `train_mps.py` | Single-device finetuner. The repo's own trainer cannot run here (CUDA hardcoded, `torchrun` required, `comet_ml` imported at module scope). | 90 min |
| report | `report.py`, `make_artifact.py` | `RESULTS.md` + figures, and the standalone HTML writeup. Every number is read from a produced file, never transcribed. | 10 s |

## Design notes worth knowing before editing

- **Signals are computed once and reused everywhere.** All four repo signal definitions
  (`last`/`mean`/`max`/`min`) × three scalings (`_z` normalized, `_ret` percentage return,
  `_recon` reconstruction-space) come out of a single forward pass, so the grid costs nothing
  extra. `anchor_z` is stored alongside them so the reversal-baseline control is free.
- **Seeding is per `(date, chunk)`,** not once at start, so a resumed run reproduces an
  uninterrupted one exactly. MPS turns out to be bit-reproducible under a fixed seed.
- **Two universes.** Pass `--membership data/nse/membership_pit.parquet` to any of
  `signals.py`, `controls.py`, `grid.py` to run on point-in-time index membership instead of
  today's list back-projected. The difference between the two is the survivorship bias.
- **Two benchmarks.** `^CNX200` is the pre-registered one and is cap-weighted; `ew200` is the
  equal-weighted eligible universe. Excess over the former mixes skill with a size and
  survivorship tilt, so both are always reported.
- **The reversal that works is not the one Kronos computes.** 1-session reversal has an IC of
  +0.028 (NW t = +6.2) over the full sample; the 90-bar z-score reversal Kronos reproduces has an
  IC indistinguishable from zero. `baseline_sweep.py` is where that is measured.
- **`baseline_sweep.py`'s headline table is a search result, not a finding.** It reports the best
  of 17 arms × 4 frequencies. `robustness.py` computes what the best of 80 uninformative arms
  earns by luck (+3.9%/yr in the average draw), and `surface.py` checks whether each winner has a
  neighbourhood. Read all three together or not at all — one arm in that headline table, weekly
  5-session reversal, is an isolated spike whose four horizon-neighbours are all negative.
- **Arms added after the fact go under a separate tag.** `--with-vol-arms` exists so the
  trailing-volatility sorts can be measured without silently enlarging the family whose maximum
  set the significance bar for everything else.
- **`--dropout-mode naive`** reproduces the intuitive-but-wrong reading of TopkDropout that
  churns the book on tied scores; `qlib` (default) is the faithful port.

## Layout

```
bench/nse/          this code
bench/out/nse/      results, figures, logs, per-date signal parts
bench/out/nse/scratch/   exploratory runs kept for provenance
data/nse/           panel, calendar, membership, processed pickles, NSE press releases
data/us/            the S&P 100 comparison panel
```
