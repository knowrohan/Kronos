# Pre-registration

Written **2026-08-27**, before any IC or backtest number was computed.
Git SHA at writing: `67b630e` (+ untracked `bench/`, `data/`).

The only model output seen before this document was fixed: a 3-date smoke run used to confirm
the pipeline produces finite scores of a plausible magnitude, plus the section-4 verification
suite. No IC, no return, no backtest, no cross-sectional performance number of any kind had
been computed. The smoke run did surface two *structural* facts that shaped the control set
below and are recorded here so they are not mistaken for post-hoc rationalisation:

1. Scores are strongly negative on average (`last_ret` mean ≈ −6.9% over 10 days), because an
   autoregressive sampler on a per-window-normalized series regresses toward the window mean.
2. `score_z` is therefore, mechanically, close to `−anchor_z` where
   `anchor_z = (close_t − mean_90) / std_90`. That makes the repo's signal a candidate pure
   **short-term reversal factor** dressed up as a forecast.

Fact 2 is why control 6 (`anchor` baseline) and gate **G2b** exist. Neither was in the plan.

---

## Primary metric

**Mean cross-sectional Spearman rank IC of `last_ret`** against the *tradeable* forward
10-session return, over all test dates.

- Signal definition: `last` — the predicted terminal close of the 10-bar horizon, matching
  `predict_window=10`. This is the most literal reading of "expected move over the horizon".
- Scaling: `ret` — expected percentage return, the economically standard cross-sectional
  scaling (plan 4.4).
- Forward return: `open_{t+11} / open_{t+1} − 1`. Open-to-open with a one-session execution
  lag, so the IC measures exactly what a strategy under rule 5.1 could capture.
- Aggregation: equal-weighted mean of per-date ICs.
- Standard error: **both** Newey–West (lag = 10, matching the horizon and therefore the
  overlap-induced autocorrelation) **and** a moving-block bootstrap (block = 21 sessions,
  10 000 resamples). Daily ICs from overlapping horizons are autocorrelated; a naive t-test
  overstates significance and will not be used.
- Reported as: mean IC, NW t-stat, bootstrap 95% CI, and share of dates with IC > 0.

A **secondary** diagnostic reports the same IC against the close-to-close return
`close_{t+10} / close_t − 1`, which is the model's own training target. If the two disagree
materially, the gap is execution drag, and that is reported explicitly.

## Primary strategy result

- **Weekly** rebalance (every 5th session), `topk=50`, `n_drop=5`, `hold_thresh=5`,
  signal `last_ret`, equal weight, **30 bps round-trip** cost.
- Metric: annualised excess return vs `^CNX200`, with a moving-block bootstrap 95% CI.
- Weekly is the a-priori favourite: the signal has a 10-day horizon, so daily rebalancing on
  it is structurally overtrading (plan 4.2).

## Dividend convention

`--dividends price`: stock OHLC is split-adjusted but **not** dividend-adjusted, so the
strategy leg and the `^CNX200` / `^NSEI` price-index benchmarks are both price-return.
Direction of bias: the strategy is charged the full dividend yield it does not receive, so
reported excess return is **conservative** by roughly the universe dividend yield. The same
convention is applied to every control. The opposite choice (total-return stocks vs a price
index) would manufacture ~2%/yr of fake alpha and is not used.

## Controls — all run against cached signals

Every stochastic control runs **N = 500 seeds** and is reported as a **distribution**, not a
point estimate. The strategy result is a single draw and is positioned against that distribution.

1. **Random signal** — scores shuffled within each date. The critical control: it shows what
   top-k-dropout earns on a survivorship-biased universe with zero information.
2. **Equal-weight universe** — buy-and-hold every eligible name.
3. **Reversed signal** — negated scores.
4. **IC by horizon** — 1 / 5 / 10 / 20 sessions.
5. **Momentum baseline** — rank by trailing 10-session return. Costs nothing; Kronos costs
   ~4 GPU-hours. If Kronos does not beat it, that is the headline.
6. **Anchor / reversal baseline** — rank by `−anchor_z = −(close_t − mean_90)/std_90`, a
   one-line feature available without any model. Added because of fact 2 above.
7. **Perfect foresight** and **constant signal** — backtester sanity, both directions (plan 10.4).

## Stage gates — kill criteria, fixed in advance

| Gate | Test | Action if it fails |
|---|---|---|
| **G0** Pilot | 60 test dates × full universe. Mean rank IC at 1/5/10/20d. | If every horizon's IC CI straddles 0, stop before the 4h run and report the null. |
| **G1** Full zero-shot | Primary IC significantly > 0 (NW t > 2 **and** bootstrap CI excludes 0) | Do not finetune. Report the null. |
| **G2** Backtest | Excess return above the **95th percentile of the random-signal distribution** | The result is a universe/turnover artefact, not signal. |
| **G2b** Incremental | Kronos IC > both the momentum and the anchor/reversal baseline IC | Kronos adds nothing a one-line feature does not. This is a headline result either way. |
| **G3** Finetune | Finetuned IC > zero-shot IC **on val**, before touching test | Report that finetuning did not help; skip the second 4h signal pass. |

## Multiple-comparisons discipline

The full grid is 4 signal definitions × 3 scalings × 2 rebalance frequencies × 4 cost levels,
per model. **The full grid is reported, never the best cell.** A maximum over 96 arms is not
an estimate. Only the pre-registered cell above is called a result; everything else is
labelled secondary and interpreted as exploratory.

## Anything not listed here is secondary

If a number appears in `RESULTS.md` that is not derived from this document, it is exploratory
and is labelled as such.

---

# Addendum — deviations from this pre-registration

Appended **2026-08-27**, immediately after the G0 pilot, before the full pass was analysed.
Recording deviations rather than quietly absorbing them is the whole point of the document.

## D1. G0's kill criterion fired, and I proceeded anyway

**G0 as written:** "60 test dates × full universe. Mean rank IC at 1/5/10/20d. If every
horizon's IC CI straddles 0, **stop before the 4.7h run.** Report the null."

**What happened:** the pilot ran 63 dates strided every 12 sessions across the whole test
window (rather than the first 60 sessions, so the gate sees all three years and the parts are
reused by the full pass). Result: across **12 Kronos arms × 4 horizons = 48 cells, zero are
significant.** The largest Newey–West t on any Kronos arm is +1.69. Every CI straddles zero.
**G0 fired.**

**I proceeded to the full 756-date pass anyway.** Reasons, stated in advance of seeing its
results:

1. **The gate exists to save compute on a null, but the economic half of the null is the
   part worth having.** The finding here is not merely "IC is indistinguishable from zero" —
   it is that the signal is a mechanical reversal factor that loses to holding the universe.
   Demonstrating that requires the backtest, and the pre-registered weekly arm requires
   signals every 5th session, which the stride-12 pilot cannot supply.
2. **The full pass buys real power at short horizons.** Because the pilot strides by 12 and
   the primary horizon is 10, its 10-day ICs are already near-independent; going to 756 dates
   tightens the 10d standard error by only ~10%. But at h=1 and h=5 the observations are
   independent at every stride, so 756 dates is **2.2–3.5× tighter** there. h=1 is exactly
   where the free reversal baseline turned out to be significant, so that is where the
   comparison needs power.
3. Compute was available and the marginal cost was ~5 h of otherwise idle GPU.

**How this is handled in reporting:** the G0 outcome is reported as the pilot result it is,
the deviation is disclosed here and in `RESULTS.md`, and the full-pass numbers are *not*
treated as independent confirmation of the pilot — they are the same experiment run denser.

## D2. Two controls added after the smoke run, before any IC was computed

`anchor_reversal` and the orthogonalised-IC test (gate **G2b**) were added after a 3-date
smoke run revealed the structural fact that scores are overwhelmingly negative. No performance
number had been computed. They are additions to the control set, not to the metric set, and
they can only make the conclusion harder to reach, never easier.

## D3. An equal-weight benchmark was added

The pre-registered primary strategy metric is excess vs `^CNX200`. Measured on this window,
equal-weighting the current NIFTY 200 membership beats that cap-weighted index by a very large
margin **with no signal at all**, so excess vs `^CNX200` mixes selection skill with a
weighting-plus-survivorship tilt. Every arm is therefore additionally reported against the
equal-weighted eligible universe. The pre-registered `^CNX200` number is still reported first
and unchanged; the equal-weight column is an addition, and it makes the strategy look worse,
not better.

## D4. `n_random` for the random-signal distribution

Pre-registered at N = 500 seeds; run at 500 for the zero-shot grid and 200 for the finetuned
grid, purely for wall-clock. Both exceed the plan's N ≥ 200 floor.

---

## Addendum D2 — post-hoc analyses, declared as post-hoc (2026-08-27)

Everything in this addendum was decided **after** seeing results. None of it is pre-registered
and none of it can be treated as confirmatory. It is recorded here so the distinction survives
into the writeup.

### D2.1 Why these were added

`baseline_sweep.py` reports the best cells of a 17-arm × 4-frequency search and checks each
against a **per-cell** random-signal control. That control answers "could this one arm have
arisen by chance?", which is the wrong question for a number that was selected because it was
the largest. Three tests were added to ask the right ones.

| test | script | question | why the existing controls could not answer it |
|---|---|---|---|
| family-wise null | `robustness.py` | what does the *maximum* of 80 uninformative arms earn? | the per-cell null is a distribution for a fixed arm, not for the max over a search |
| parameter surface | `surface.py` | is the winner a plateau or a spike? | needs no null at all; a real effect decays smoothly in its own parameters |
| benchmark audit | `benchmark_audit.py` | is `ew200` a fair bar, and is any of it reachable? | `ew200` is a cost-free daily-rebalanced mean, which nobody can run |
| conditional vol IC | `vol_signals.py` | does the model forecast volatility beyond trailing volatility? | orthogonalised IC misbehaves on a product; conditioning does not |

### D2.2 Family definition, and one honest complication

The family-wise null draws **20** independent random arms per draw: the 17 arms
`baseline_sweep.py` searched, plus the 3 blends `robustness.py` itself introduced. Blends are a
second round of search over the same data and belong in the family whose maximum sets the bar.

Independent random arms overstate the spread of the maximum relative to the real family, whose
arms are heavily correlated (`momentum_20d` and `momentum_60d` share most of their ranking). The
resulting threshold is therefore **conservative in the direction that makes it harder to declare
a finding**, which is the direction to err in.

Trailing-volatility arms were tested afterwards and are deliberately **not** in this family; they
run under `--with-vol-arms` and a separate tag, and are held to the bar computed without them.

### D2.3 The max statistic is not the only bar

The maximum is the right null for a cherry-picked cell. It is the wrong null for a claim about a
whole family — "every momentum lookback from 10 to 60 sessions is positive" is a statement about
the median and the share positive, and those have a much tighter null. Both are recorded per
draw (`median_excess`, `share_positive`, and the same two per frequency) so the two kinds of
claim are never judged against each other's bar.

### D2.4 Deviation from the plan's cost treatment

The plan's cost model has no market-impact term, which makes cost per trade fall monotonically
with size and implies larger books are always cheaper. `robustness.py` adds a square-root impact
charge sized against 20-session median traded value. It is a **model**, not a measurement, with
`impact_coef = 1.0` as the usual starting calibration; it is there to locate where the edge
breaks, not to price a fill.
