# Kronos on NSE — cross-sectional ranking backtest

**Audience:** an agent with full access to this repo and no prior context.
**Status:** plan only. Nothing under `bench/nse/` has been written yet.

## How to use this document

Work top to bottom. Section 4 ("Claims to verify") comes **before** Section 8 ("Build")
deliberately — several numbers below drive the whole compute budget and at least one earlier
draft of this plan contained a methodological error that would have manufactured fake alpha
(see 4.6). Verify first, then build.

Line-number references like `finetune/qlib_test.py:277` were correct at git SHA `67b630e`.
They drift. Every reference below is paired with a `grep` anchor — prefer the anchor.

Everything you write goes in `bench/nse/`. **Do not modify `model/`, `finetune/`, `webui/`, or
`examples/`.** If you need a repo file changed, copy it into `bench/nse/` and change the copy,
and say so in the writeup. The one exception is `finetune/config.py`, which some repo modules
import at module scope — see 8.2.

---

## 1. Orientation — what this repo is

Kronos is a decoder-only foundation model for financial K-lines (OHLCV candlesticks), AAAI 2026,
arXiv 2508.02739. Two stages:

1. A **hierarchical tokenizer** (BSQ-style, two codebooks `s1`/`s2`) quantizes a normalized OHLCV
   bar into a pair of discrete tokens.
2. An **autoregressive Transformer** predicts the next token pair, conditioned on the token
   history plus time-of-day/week/month features.

Released checkpoints (from `README.md`, "Model Zoo"):

| Model | Tokenizer | Context | Params | Released |
|---|---|---|---|---|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | yes |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | yes |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | yes |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | **no** |

Use **Kronos-base** + **Kronos-Tokenizer-base**. It is the largest released model and fits
comfortably on the target machine (M-series MacBook Air, 32 GB, `mps`).

### Repo map — what matters here

| Path | What it is | Relevance |
|---|---|---|
| `model/kronos.py` | Model, tokenizer, `KronosPredictor`, `auto_regressive_inference` | Core. Read it. |
| `finetune/config.py` | Every hyperparameter, as a Python class | Source of the repo's defaults |
| `finetune/qlib_data_preprocess.py` | Qlib → pickles | Defines the data contract you must match |
| `finetune/dataset.py` | `QlibDataset` — training sampler | Reusable, with caveats (4.8) |
| `finetune/qlib_test.py` | **Inference → signal → qlib backtest** | The single most important file |
| `finetune/train_predictor.py` | DDP trainer | Cannot run on this machine (4.9) |
| `examples/`, `webui/` | Single-series demo forecasting | Not what this build targets |
| `tests/test_kronos_regression.py` | Pins HF revisions, CPU-deterministic | Copy its revision-pinning habit |

---

## 2. What already exists from prior work

A previous session built a single-series benchmark under `bench/`. It is untracked (`git status`
shows `?? bench/`, `?? data/`). Read these before writing anything — they solve problems you
would otherwise re-solve:

- `bench/fetch_data.py` — Binance + yfinance → Kronos CSV schema. Reference for the
  `timestamps,open,high,low,close,volume,amount` contract.
- `bench/kronos_eval.py` — rolling-origin harness. **Contains `predict_paths()`, the workaround
  for the ensemble-collapse bug (4.7).** Reuse this function.
- `bench/validate_wrapper.py` — proves that workaround is bit-identical to the library API.
- `data/*.csv` + `data/manifest.json` — BTC and NIFTY/RELIANCE series, 1h and 1d.
- `bench/out/RESULTS.md` — the prior benchmark's results.

`.venv/` already exists with `requirements.txt` + `yfinance` + `requests` installed.
`requirements.txt` is locally modified (pins `numpy==2.2.6`, `torch==2.11.0`) — **keep those
pins**, they are what makes `mps` work here.

---

## 3. Why this build exists

The prior benchmark scored Kronos as a **level forecaster**: predict the next N bars of one
series, compare RMSE against naive persistence (tomorrow = today). Result over 48 rolling
windows across 6 series:

- beat naive in **7/48** (15%)
- mean directional accuracy **0.520** (0.484–0.570; the two Indian daily series were the best)
- p10–p90 cone coverage **0.22** against 0.80 nominal
- predicted move magnitude **1.2×–3.0×** realised

That was the wrong target. `finetune/qlib_test.py` shows what the authors actually optimise —
each forecast is collapsed to **one scalar per stock per day**:

```python
# grep -n "last_day_close" finetune/qlib_test.py   -> ~line 276
last_day_close = x[:, -1, 3].numpy()
signals = {
    'last': preds[:, -1, 3] - last_day_close,
    'mean': np.mean(preds[:, :, 3], axis=1) - last_day_close,
    'max':  np.max(preds[:, :, 3], axis=1)  - last_day_close,
    'min':  np.min(preds[:, :, 3], axis=1)  - last_day_close,
}
```

That scalar feeds `TopkDropoutStrategy(topk=50, n_drop=5, hold_thresh=5)` over a 300-name CSI300
universe, and the only reported metric is **excess return over a benchmark index, with and
without cost** (`risk_analysis(report["return"] - report["bench"], ...)`). The price path is
thrown away.

Two consequences:

1. A uniform magnitude inflation does not change rank order. The over-extension finding is
   largely orthogonal to what the repo optimises.
2. Naive persistence is not even a valid cross-sectional competitor — it emits score 0 for every
   stock, producing no ranking at all.

**Question this build answers:** does a Kronos-derived cross-sectional ranking earn excess return
over the NIFTY 200, after realistic Indian transaction costs — and does finetuning buy anything
over zero-shot?

---

## 4. Claims to verify before building

Each of these is either (a) measured on different hardware/data than yours, (b) reasoned from
source without an empirical check, or (c) recalled rather than looked up. **Verify each and
record the result in `bench/nse/VERIFIED.md` before depending on it.** Where a claim turns out
false, the plan changes — the "if false" line says how.

### 4.1 Indian transaction cost rates — LOWEST CONFIDENCE ITEM

These were written from memory, not from a source, and Indian statutory rates change with each
Union Budget. Today is 2026-08-27; these may be years stale.

```
stt_buy        0.001      # 0.10%, delivery equity
stt_sell       0.001      # 0.10%
stamp_duty_buy 0.00015    # 0.015%, buy side only
exchange_txn   0.0000297  # NSE 0.00297%
sebi_fee       0.000001   # 0.0001%, i.e. Rs 10 per crore
brokerage      0.0003     # discount broker, or flat Rs 20/order
gst            0.18       # on (brokerage + exchange txn) only
slippage_bps   5
```
→ **≈ 0.30% round trip.**

**Verify** against a current published brokerage charge sheet (Zerodha/Groww/Upstox all publish
one) or the NSE circular list. **If false:** the numbers change but not the design — every rate
is already a named parameter, and results are reported across a cost sweep (8.4), so a wrong
central estimate degrades gracefully.

### 4.2 Turnover cost drag is ~7.6%/yr at the repo's default settings

`topk=50, n_drop=5` daily means 5 sells + 5 buys per day = **20% of capital transacted daily**.
At ~0.15% one-way: `0.20 × 0.0015 × 252 ≈ 7.6% per year`.

This is the single most important number in the plan. It is a very high hurdle — larger than
most realistic equity alphas. **Verify the arithmetic, then verify it empirically** by reading
realised turnover out of your own backtester rather than assuming it.

`hold_thresh=5` (minimum 5-day hold) does not help much: 5 drops/day out of 50 gives a mean
holding period of ~10 days, so the constraint is mostly non-binding.

**If true — and it almost certainly is — this changes the plan:** treat rebalance frequency as a
first-class experimental arm, not a fixed constant. Run `n_drop ∈ {1, 2, 5}` and
`rebalance ∈ {daily, weekly}`. Note that `predict_window=10` means the signal has a **10-day
horizon**, so daily rebalancing on it is structurally overtrading. Weekly is the more defensible
default and the a-priori favourite.

### 4.3 Throughput saturates at batch 32

Measured last session on this machine, at `lookback=90, predict=10`, Kronos-base on `mps`:
batch 1 = **3648 ms/series**, batch 32 = **112 ms/series** (~32×), flat beyond 32.

The 112 ms figure drives the entire compute budget. **Re-measure it** — sweep batch
∈ {1, 8, 16, 32, 64, 128} and record ms/series. Note an earlier session measured *no* batching
benefit at `ctx=512` and wrongly generalised; the win is config-specific, so measure at the exact
config you will run.

**If false (say it saturates at 64 or 128):** great, budget shrinks. **If it is much slower than
112 ms:** cut the test window to 1 year or drop to Kronos-small, and say which you did.

### 4.4 Signals are produced in NORMALIZED space, not price space

`auto_regressive_inference` (`grep -n "def auto_regressive_inference" model/kronos.py`) **never
denormalizes.** `KronosPredictor.predict()` denormalizes afterwards; `qlib_test.py` does not.
`QlibTestDataset.__getitem__` normalizes `x` per-window with
`(x - mean) / (std + 1e-5)` over the lookback, then `clip(±5)`.

So the repo's signal is:

```
score  =  E[close_{t+h}] - close_t   ... in units of the stock's own 90-bar close std
       ≈  (expected move) / sigma_close
```

That is a **volatility-normalized** score, not an expected return. Cross-sectionally this is a
real and consequential design choice — it systematically favours low-volatility names, and it is
*not* the same ranking you would get from expected percentage return.

**Verify empirically:** print `preds` for one window and check the values are O(1) z-scores, not
rupee prices.

**If true — this changes the plan:** compute and evaluate **both** signal scalings as separate
arms, and report both:
- `score_z` — the repo's exact definition, normalized space (replication).
- `score_ret` — denormalize with the window's `mean`/`std`, then divide by `close_t` to get an
  expected percentage return (the economically standard scaling).

Do not silently pick one. The difference between them is a testable claim about whether the
repo's headline results depend on an implicit low-vol tilt.

### 4.5 `preds` contains the reconstructed context, not just the future

`auto_regressive_inference` decodes the last `max_context` tokens of context+prediction, so the
returned array is longer than `pred_len`. `qlib_test.py` slices with
`preds = preds[:, -config['pred_len']:, :]`. **Verify the returned shape** before slicing.

Related and subtler: `last_day_close` is read from the **raw input `x`**, while `preds` comes out
of an encode→decode round trip. Subtracting one from the other folds **tokenizer reconstruction
error** into every signal. **Measure that error** (compare `preds[:, ctx-1, 3]` against
`x[:, -1, 3]` across many windows). If it is not negligible, add a third arm using
`preds[:, -1, 3] - preds[:, ctx-1, 3]`, which stays entirely inside reconstruction space.

### 4.6 Benchmark and dividend treatment — a bug in the previous draft

An earlier draft of this plan said "use `auto_adjust=True` so the series is total-return and
consistent with the `^NSEI` benchmark". **That is wrong.** `^NSEI` and `^CNX200` are **price
indices**; they exclude dividends. `auto_adjust=True` makes the stock series **total-return**.
Mixing the two hands the strategy the entire dividend yield as free "excess return".

Measured this session, `auto_adjust=True` vs `False` CAGR gap, 2021-01→2026-08:

| RELIANCE | TCS | HDFCBANK | ITC | INFY | mean |
|---|---|---|---|---|---|
| 0.39% | 2.69% | 1.26% | 4.60% | 2.67% | **2.32%/yr** |

**~2.3%/yr of fabricated alpha** — larger than most real signals. **So:**
- Use **`auto_adjust=False`** for stocks so both legs are price-return and consistent, **or**
  source a NIFTY 200 TRI (yfinance does not serve one).
- Whichever you choose, the equal-weight control must use the identical convention.
- **State the choice explicitly in `RESULTS.md`.**

**Verify** the gap on the full 200-name universe, not the 5 names sampled here.

Splits, separately, are **not** an issue: yfinance back-adjusts splits regardless of
`auto_adjust` (that flag gates *dividends* only). Confirmed on RELIANCE's 2024-10-28 2:1 split —
zero >15% daily moves in the unadjusted series, unadjusted close equals adjusted close.
**Re-verify on one split name** in your fetched panel anyway.

Benchmark ticker: the universe is NIFTY 200, so the benchmark should be **`^CNX200`**, not
`^NSEI` (NIFTY 50). Verified reachable this session: 3113 daily rows from 2014-01-01. **Verify
`^CNX200` really is the NIFTY 200** — compare its level against a published NIFTY 200 quote — and
keep `^NSEI` as a secondary reference line.

### 4.7 `sample_count > 1` destroys the ensemble

`grep -n "np.mean(preds, axis=1)" model/kronos.py` — at the end of `auto_regressive_inference`:

```python
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)     # <-- N stochastic paths collapsed to their mean
return preds
```

The shipped API cannot return a distribution. Workaround (validated bit-identical last session,
see `bench/validate_wrapper.py`): pass a **batch of N identical rows with `sample_count=1`**, so
the reshape becomes `(N, 1, L, D)` and the mean over axis 1 is a no-op. Same GPU cost.

For this build, use **`sample_count=1`** for the headline run — averaging N paths shifts every
stock's score in the same direction and barely reorders. Revisit only if IC is non-zero, and if
you do, use path *dispersion* as a separate feature rather than just the mean.

### 4.8 `QlibDataset` is reusable but has two traps

`finetune/dataset.py:QlibDataset` is device-agnostic (reads pickles, returns tensors), so it can
be reused for MPS training. But:

1. **`__getitem__` ignores `idx`.** It draws from `self.py_rng` instead. With
   `DataLoader(num_workers>0)`, every worker forks the same seeded RNG, so **all workers emit the
   same sample sequence** — silent duplication of training data. **Verify** by pulling two
   batches with `num_workers=2` and checking for identical rows. Mitigation: `num_workers=0`, or
   reseed per worker in `worker_init_fn`.
2. **It constructs `Config()` internally** via `from config import Config`, so it only imports
   with `finetune/` on `sys.path`, and it reads paths from `finetune/config.py`. "Reuse
   unchanged" is true of the class but requires config plumbing. Set the paths via a
   `Config` instance you mutate at runtime rather than editing the file in place.

Also note a real inconsistency between the two datasets, so it does not confuse you:
`QlibDataset` uses `window = lookback + predict + 1` and normalizes over the **first `lookback`
bars only** (correct, no leakage). `QlibTestDataset` in `qlib_test.py` uses
`window = lookback + predict` (no `+1`) and normalizes over the whole lookback slice. Both are
leak-free; they just differ.

### 4.9 The repo's trainer cannot run on this machine

- `grep -n 'cuda:{local_rank}' finetune/train_predictor.py` — device hardcoded to CUDA
- `grep -n 'WORLD_SIZE' finetune/train_predictor.py` — raises `RuntimeError` unless launched by
  `torchrun`
- `grep -n 'torch.cuda.set_device' finetune/utils/training_utils.py`
- `dist.all_reduce` / `dist.barrier` throughout; `import comet_ml` at module scope (line 12)

**Verify** by attempting `python finetune/train_predictor.py` and capturing the exact error.
**If it somehow runs, the plan gets simpler** — say so and use it.

### 4.10 Data sources still live

- Constituents: `https://archives.nseindia.com/content/indices/ind_nifty200list.csv` — returned
  200 rows with a `Symbol` column last session. The `nseindia.com/api/*` and `niftyindices.com`
  endpoints both failed. **Verify the archives URL still responds**; if not, fall back to a
  scrape or a checked-in snapshot, and **commit whatever list you use** so the run is reproducible.
- OHLCV: yfinance `{SYMBOL}.NS`. 15 tickers × 5y in 2.4s, zero NaNs. Expect some symbol-mapping
  misses (renames, `-EQ` suffixes, `M&M` → `M&M.NS`); **log every miss, do not drop silently**.
- **Verify whether NSE publishes *historical* constituent lists.** If yes, use them and the
  survivorship-bias limitation in Section 11 largely disappears — a major upgrade to the result's
  credibility. This is worth 20 minutes of searching.

### 4.11 Things to verify that this plan simply does not know

- **MPS sampling reproducibility.** `torch.multinomial` on `mps` under a fixed seed — is it
  bit-reproducible across runs? Test it. If not, cache signals once (you were going to anyway)
  and note that exact reproduction requires the cache, not just the seed.
- **Memory.** Kronos-base, batch 32, ctx 100 on 32 GB — expected fine, unverified. Sweep up until
  it fails and record the ceiling.
- **`hold_thresh` semantics.** This plan asserts "minimum holding days". Confirm against qlib's
  `TopkDropoutStrategy` source if qlib is installed, since your backtester clones its behaviour.
- **Whether qlib is installable at all** on this platform. The plan does not depend on it, but
  the port-to-qlib fallback does.
- **Circuit limits.** qlib's config uses `limit_threshold: 0.095` (China ±10%) to block
  untradeable fills. NSE has per-stock circuits (2/5/10/20%). Decide whether to model them; at
  minimum, check how many NIFTY 200 open-to-open moves in the test window exceed 10%, and if the
  count is negligible, say so and skip it.

---

## 5. Non-negotiable correctness rules

Violate any of these and the result is worthless-but-plausible, which is worse than no result.

1. **Execution lag.** Score for date `t` is computed from bars **up to and including `t`'s
   close**. The position is taken at **`t+1`'s open**. Portfolio return accrues open `t+1` →
   open `t+2`. This mirrors qlib's `delay_execution=True, deal_price="open"`
   (`grep -n "delay_execution" finetune/qlib_test.py`). Assert it in code, not in a comment.
2. **Point-in-time universe.** On date `t`, a name is eligible only if it has ≥ `lookback` prior
   bars, traded on `t`, and traded on `t+1`. Names that IPO'd mid-sample make the panel ragged.
   Emit the eligibility mask as a saved artifact and plot universe size over time.
3. **Never fit anything on test dates.** Normalization is per-window and backward-looking already
   (`QlibDataset` normalizes on the lookback slice only) — keep it that way. No cross-sectional
   z-scoring using full-sample moments.
4. **One convention for dividends,** applied to strategy, benchmark, and every control (4.6).
5. **No peeking at future bars to decide whether a date is scoreable.** `QlibTestDataset`
   requires `predict_window` future bars to exist
   (`num_samples = len(df) - window_size + 1`), which silently truncates the last 10 dates. Your
   `signals.py` needs only the *future timestamps* (`y_stamp`), which come from the exchange
   calendar — generate them, don't require realized bars.

---

## 6. Decisions already taken

| Choice | Value |
|---|---|
| Universe | NIFTY 200 |
| Benchmark | `^CNX200` primary, `^NSEI` secondary (4.6) |
| Rebalance | Daily signals; **daily and weekly** rebalance arms (4.2) |
| Test window | 3 years, ~750 trading days |
| Engine | Custom backtester cloning `TopkDropoutStrategy`; port to qlib only if signal appears |
| Model | Kronos-base — zero-shot first, then finetuned |
| Signal scaling | Both `score_z` and `score_ret` (4.4) |
| Controls | Random-signal, reversed-signal, equal-weight, IC-by-horizon |

### Time splits

```
train      2015-01-01 .. 2021-12-31
val        2022-01-01 .. 2023-06-30
test       2023-08-01 .. 2026-08-26     (~750 rebalances)
lookback_window = 90,  predict_window = 10        # repo defaults, finetune/config.py
```

The one-month gap between val end and test start is deliberate — it exceeds `lookback +
predict`, so no test window can touch a bar the model trained on.

---

## 7. Pre-registration — decide these before you see results

Without this, ~16 curves (4 signal defs × 2 scalings × 2 rebalance freqs) times 4 cost levels
times zero-shot/finetuned guarantees something looks good by chance. Fix the following **in
writing, in `bench/nse/PREREG.md`, before running the full pass:**

- **Primary metric:** rank IC of `score_ret` at horizon 10d (matching `predict_window`),
  averaged over test dates, with a Newey-West or block-bootstrap standard error.
- **Primary strategy result:** weekly rebalance, `topk=50`, `n_drop=5`, `score_ret`, at the
  central cost estimate — annualised excess return vs `^CNX200`, with a bootstrapped CI.
- **Everything else is secondary and must be labelled as such** in `RESULTS.md`.
- **Report the full grid, not the best cell.** A max over 16 arms is not an estimate.

### Stage gates — kill criteria

| Gate | Test | If it fails |
|---|---|---|
| **G0** Pilot | 60 test dates × full universe (~25 min). Mean rank IC at 1/5/10/20d. | If every horizon's IC CI straddles 0, **stop before the 4.7h run.** Report the null. It is a real, publishable-quality finding and costs 10h less. |
| **G1** Full zero-shot | IC significantly > 0 at any horizon | If not, do not finetune. Report null. |
| **G2** Backtest | Excess return beats the **random-signal control distribution**, not merely the index | If not, the result is universe/turnover artefact, not signal |
| **G3** Finetune | Finetuned IC > zero-shot IC on **val**, before touching test | If not, report that finetuning did not help and skip the second 4.7h signal pass |

Do the pilot. It is the highest-value 25 minutes in this plan.

---

## 8. Build

Seven scripts, `bench/nse/`. Each is independently runnable, takes `--help`, and writes a
JSON sidecar recording its arguments, git SHA, package versions, and HF model revisions
(`tests/test_kronos_regression.py` pins `MODEL_REVISION`/`TOKENIZER_REVISION` — copy that habit;
`grep -n "REVISION" tests/test_kronos_regression.py`).

### 8.1 `fetch_universe.py`

Constituent list → `{SYMBOL}.NS` → bulk yfinance daily, `period="max"`, dividend convention per
4.6. Fetch `^CNX200` and `^NSEI` separately.

**Out:**
- `data/nse/panel.parquet` — tidy long: `date, symbol, open, high, low, close, volume`
- `data/nse/bench.parquet` — `date, cnx200, nsei`
- `data/nse/universe.json` — per symbol: industry, first/last date, row count, NaN%, whether the
  yfinance mapping succeeded
- `data/nse/fetch_log.txt` — **every** symbol that failed to map or returned short history

Drop names with `< lookback + predict + 1` rows, mirroring
`grep -n "lookback_window + self.config.predict_window + 1" finetune/qlib_data_preprocess.py`.

### 8.2 `preprocess.py`

Emit the repo's **exact pickle contract** so `finetune/dataset.py:QlibDataset` loads it unchanged:
`{symbol: DataFrame}` indexed by `datetime`, columns = `feature_list` →
`train_data.pkl` / `val_data.pkl` / `test_data.pkl`.

Two contract details that will silently break things if missed:
- Column names are **`vol` and `amt`**, not `volume`/`amount`
  (`grep -n "feature_list" finetune/config.py`).
- `amt = (open + high + low + close) / 4 * vol`
  (`grep -n "symbol_df\['amt'\]" finetune/qlib_data_preprocess.py`). Use this formula exactly,
  computed from prices in your chosen dividend convention.

Point `Config().dataset_path` at `data/nse/processed/` by mutating a `Config` instance at
runtime, not by editing `finetune/config.py`.

**Out:** three pickles + `splits.json` recording exact date boundaries and per-split symbol counts.

### 8.3 `signals.py` — the expensive one

Batched inference → one score per `(date, symbol, signal_def, scaling)`.

- Reuse the four repo signal definitions verbatim (`last`, `mean`, `max`, `min`) and both
  scalings from 4.4.
- Batch **all eligible symbols for a rebalance date in one call** — batching is a 32× win (4.3).
- **Checkpoint and resume.** This is a multi-hour job; it must not lose work to a crash. Append
  to `bench/out/nse/signals_{tag}.parquet` per date, and on start resume from the last completed
  date. Verify the resume path by killing the process and restarting it.
- `sample_count=1` (4.7). Use `bench/kronos_eval.py:predict_paths` if you want paths.
- Fixed seed, recorded. Note 4.11 on MPS determinism.

**CLI:** `--start --end --model --tag --batch-size --limit-dates --resume`
(`--limit-dates` is what makes the G0 pilot cheap.)

**Out:** `bench/out/nse/signals_{tag}.parquet` — `date, symbol, signal_def, scaling, score`,
plus `signals_{tag}_meta.json` with wall-clock, ms/series, and the batch size actually used.

### 8.4 `backtest.py`

Clone of `TopkDropoutStrategy`: rank by score, hold `topk`, drop at most `n_drop`/rebalance,
respect `hold_thresh`. Defaults `topk=50, n_drop=5, hold_thresh=5`
(`grep -n "backtest_n_symbol" finetune/config.py`). Equal-weight the held names.

Execution per rule 5.1. Every cost rate a **named parameter, never a literal** (4.1). Report at
**0 / 15 / 30 / 50 bps** round-trip so cost sensitivity is visible rather than buried in one
assumption. For reference, the repo's own qlib config uses `open_cost 0.001, close_cost 0.0015,
min_cost 5` (`grep -n "open_cost" finetune/qlib_test.py`) — i.e. 25 bps round trip, close to the
Indian estimate.

**Metrics:** annualised return; annualised excess vs `^CNX200`; information ratio; max drawdown;
**realised turnover**; total cost drag in %/yr (compare against 4.2's ~7.6% prediction); hit
rate; and bootstrapped CIs on excess return, not point estimates.

**Out:** `bench/out/nse/bt_{tag}.parquet` (daily returns, positions, costs) + `bt_{tag}.json`
(metrics) + two-panel cumulative-return / cumulative-excess-return figure, matching the layout of
`qlib_test.py`'s plot.

### 8.5 `controls.py`

All four run against **cached** signals, so they cost seconds each. Run every stochastic control
over **N ≥ 200 seeds and report the distribution**, not a point estimate — the strategy result is
a single draw and must be positioned against that distribution.

1. **Random-signal** — shuffle scores within each date. *The* critical control: it shows what
   top-k-dropout earns on a survivorship-biased universe with zero information. Kronos must clear
   **this distribution**, not merely clear the index.
2. **Equal-weight universe** — buy-and-hold all eligible names. Separates "picks winners" from
   "these 200 survivors beat a cap-weighted index".
3. **Reversed signal** — negate scores. Symmetric loss implies real directional information;
   both-lose implies noise plus costs.
4. **IC by horizon** — Spearman rank correlation of score vs realised forward return at
   1/5/10/20 days, per date, then averaged with a proper standard error (daily ICs are
   autocorrelated — use block bootstrap or Newey-West, not a naive t-test).

The repo has **no IC code anywhere** — verified, zero matches for
`rank_ic|information_coefficient|spearman|pearson` across `finetune/` and `model/`. You are
writing this from scratch; get the standard errors right.

Add one control the earlier draft missed:

5. **Momentum baseline** — rank by trailing 10-day return. Kronos costs ~5 GPU-hours; a one-line
   momentum score costs nothing. If Kronos does not beat it, that is the headline result.

### 8.6 `train_mps.py`

Standalone single-device trainer. Reuses `finetune/dataset.py:QlibDataset` (with the 4.8
caveats), no DDP, no comet, `device="mps"`. Mirror the optimiser settings from
`finetune/config.py`: `tokenizer_learning_rate 2e-4`, `predictor_learning_rate 4e-5`,
AdamW β=(0.9, 0.95), weight decay 0.1.

**Size the schedule from measurement, not from the config.** The repo's `epochs=30`,
`n_train_iter = 2000 × batch_size` target a multi-GPU box. Measure steps/sec on one batch, pick
an epoch count that fits a stated wall-clock budget, and **record what was cut**. Checkpoint every
N steps; log train and val loss to a CSV.

Finetune **Kronos-base** so the comparison against the zero-shot baseline is like-for-like.
Select the checkpoint on **val** loss, and gate on G3 before spending another 4.7h of inference.

Set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Expect some ops to fall back to CPU; if throughput collapses,
profile which op and say so rather than quietly switching to CPU.

### 8.7 `report.py`

`bench/out/nse/RESULTS.md` + a consolidated figure. Must contain, in this order:

1. The pre-registered primary result with its CI.
2. The full grid of secondary arms, labelled secondary.
3. Strategy vs random-signal **distribution** (not vs a single random draw).
4. Cost sensitivity across 0/15/30/50 bps.
5. IC decay curve across horizons.
6. Zero-shot vs finetuned vs momentum baseline.
7. Realised turnover and cost drag.
8. Every limitation from Section 11, stated up front, not buried.
9. The `VERIFIED.md` outcomes — which Section 4 claims held and which did not.

---

## 9. Compute budget

| Stage | Estimate | Note |
|---|---|---|
| Data fetch | ~5 min | |
| **G0 pilot** — 200 × 60 × 112 ms | **~25 min** | gate before committing to the full run |
| Zero-shot signals — 200 × 750 × 112 ms | **~4.7 h** | resumable, checkpointed |
| Backtest + all controls | seconds | runs off cached signals |
| Finetune tokenizer + predictor | measure first, then bound | |
| Finetuned signals | **~4.7 h** | only if G3 passes |

~10h+ if every gate passes; ~30 min if G0 fails. Run long stages in the background and report at
each gate boundary rather than going dark.

---

## 10. Verification tests — write these as assertions, not prose

1. **Pickle contract:** `preprocess.py` output loads through the **unmodified**
   `finetune/dataset.py:QlibDataset`, and one sample has shape
   `(lookback + predict + 1, len(feature_list))`.
2. **Signal reproducibility:** re-run one rebalance date, confirm identical scores under a fixed
   seed. If MPS is nondeterministic (4.11), quantify the spread instead and say so.
3. **Pipeline cross-check:** generate a forecast for one symbol/date through `signals.py` and
   independently through `bench/kronos_eval.py`. They must agree. This catches batching,
   normalization, and stamp-alignment bugs that no single-path test will.
4. **Backtester sanity — both directions:**
   - Perfect-foresight signal (next-period realised return) → must produce a strongly positive
     curve. If it does not, the backtester is broken.
   - Constant signal → must produce ≈ equal-weight-of-top-50 minus costs, with **near-zero
     turnover**. Non-zero turnover on a constant signal means the ranking tie-break is churning.
   A backtester that fails either is measuring nothing.
5. **Lookahead assertions, in code:**
   - `max(train ∪ val dates) + lookback + predict < min(test dates)`
   - for every rebalance, `max(context bar date) <= signal date < execution date`
   - shifting all signals back one day must **destroy** performance. If it does not, you have
     lookahead somewhere.
6. **Cost model:** hand-check one round trip against the stated rates, and reconcile the
   backtester's total cost drag against the closed-form estimate in 4.2.
7. **Universe integrity:** plot eligible-universe size per date. Sharp cliffs mean a data bug.
8. **Split adjustment:** confirm zero >20% single-day moves in the final panel, or list each one
   and identify it as a real event.

---

## 11. Known limitations — state these in the writeup, do not bury them

- **Survivorship bias.** The constituent list is *current* membership back-projected to 2015.
  Names that fell out of the index are absent. The random-signal and equal-weight controls exist
  to quantify how much of any excess return this explains. **If historical membership lists turn
  out to be available (4.10), use them — this limitation mostly disappears.**
- **Dividend convention.** Whichever branch of 4.6 you take, name it and state its direction of
  bias.
- **No liquidity or market-impact model** beyond flat slippage. Top-50 NIFTY 200 names are liquid,
  so this is mild — but it is an assumption, and it gets worse as `topk` shrinks.
- **No circuit-limit modelling** unless 4.11 says it matters.
- **Single market, single ~3-year regime**, one of unusually strong Indian equity performance.
- **Corporate actions beyond splits/dividends** (mergers, demergers, ticker changes) are not
  handled. Log any symbol whose history has a suspicious discontinuity.
- **The signal has a 10-day horizon but the strategy may rebalance daily** — a structural
  mismatch inherited from the repo's defaults (4.2), which is exactly why the weekly arm exists.

---

## Appendix A — environment

```bash
cd /Users/rap/Documents/z-projects/Kronos
source .venv/bin/activate          # already provisioned: requirements.txt + yfinance + requests
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

- Keep the local `requirements.txt` modifications (`numpy==2.2.6`, `torch==2.11.0`). They are
  uncommitted and they are what makes `mps` work here.
- Models download from HF on first use: `NeoQuasar/Kronos-base`, `NeoQuasar/Kronos-Tokenizer-base`.
  **Pin `revision=` on both**, as `tests/test_kronos_regression.py` does.
- Target machine: M-series MacBook Air, 32 GB. Device order in `KronosPredictor` is
  cuda → mps → cpu, so `mps` is selected automatically.
- Heads-up: `examples/prediction_example.py` is broken out of the box (reads a CSV not present in
  the repo, and its `sys.path.append("../")` only resolves when run from inside `examples/`).
  Don't use it as a template — use `bench/kronos_eval.py`.
- Some repo CSVs carry a UTF-8 BOM (`finetune_csv/data/HK_ali_09988_kline_5min_all.csv`); read
  with `encoding='utf-8-sig'` or `df['timestamps']` raises `KeyError`.

## Appendix B — grep anchors

Line numbers were correct at SHA `67b630e` and will drift. Use these instead:

| What | Anchor |
|---|---|
| Signal definitions | `grep -n "last_day_close" finetune/qlib_test.py` |
| Ensemble-collapse bug | `grep -n "np.mean(preds, axis=1)" model/kronos.py` |
| Test-time normalization | `grep -n "Instance-level normalization" finetune/qlib_test.py` |
| Train-time normalization | `grep -n "past_x = x\[:past_len\]" finetune/dataset.py` |
| `amt` formula | `grep -n "symbol_df\['amt'\]" finetune/qlib_data_preprocess.py` |
| Min-length filter | `grep -n "lookback_window + self.config.predict_window + 1" finetune/qlib_data_preprocess.py` |
| Strategy params | `grep -n "backtest_n_symbol" finetune/config.py` |
| Execution semantics | `grep -n "delay_execution\|deal_price\|open_cost" finetune/qlib_test.py` |
| CUDA hardcode | `grep -n "cuda:{local_rank}" finetune/train_predictor.py` |
| torchrun guard | `grep -n "WORLD_SIZE" finetune/train_predictor.py` |
| HF revision pinning | `grep -n "REVISION" tests/test_kronos_regression.py` |
