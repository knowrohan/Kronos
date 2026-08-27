# VERIFIED.md — outcome of every claim in plan section 4

Machine-readable evidence: `bench/out/nse/verify_model.json`, `verify_pipeline.json`,
`verify_backtest.json`. Every claim below was checked against a source or measured on this
machine (Apple M5, 32 GB, `mps`, torch 2.11.0, numpy 2.2.6) on **2026-08-27**.

Verdict key: **CONFIRMED** / **REFUTED** / **PARTLY** (claim directionally right, numbers wrong).

| # | Claim | Verdict |
|---|---|---|
| 4.1 | Indian transaction cost rates | **PARTLY** — total right, three line items wrong |
| 4.2 | Turnover drag ~7.6%/yr at repo defaults | **PARTLY** — arithmetic right, realised lower |
| 4.3 | Batching saturates at 32, ~32× win | **REFUTED** — the win is 1.6×, not 32× |
| 4.4 | Signals are in normalized, not price, space | **CONFIRMED** |
| 4.5 | `preds` contains reconstructed context | **CONFIRMED** |
| 4.5b | Tokenizer reconstruction error may be negligible | **REFUTED** — 12% of signal magnitude |
| 4.6 | Dividend/benchmark convention | **CONFIRMED** (splits), benchmark ticker confirmed |
| 4.7 | `sample_count > 1` destroys the ensemble | **CONFIRMED** |
| 4.8 | `QlibDataset` has two traps | **CONFIRMED** — both, empirically |
| 4.9 | The repo trainer cannot run here | **CONFIRMED** |
| 4.10 | Data sources still live | **CONFIRMED** — 200/200 symbols mapped |
| 4.10b | Historical constituent lists may exist | **CONFIRMED and USED** — 6 PDFs rebuild the point-in-time universe to 199–200/session |
| 4.11 | MPS sampling reproducibility unknown | **REFUTED** — it *is* bit-reproducible |
| — | Plan §6 val→test gap "exceeds lookback+predict" | **REFUTED** — it was 5× too small |
| — | `tests/test_kronos_regression.py`'s `MODEL_REVISION` | **wrong repo** — it pins Kronos-*small* |

---

## 4.1 Indian transaction costs — the lowest-confidence item, and it was wrong

Verified against the NSE circular and current published broker charge sheets.

| line item | plan draft | verified current | verdict |
|---|---|---|---|
| `stt_buy` / `stt_sell` | 0.001 / 0.001 | 0.001 / 0.001 | correct (Budget 2026 hiked only F&O) |
| `stamp_duty_buy` | 0.00015, buy-side only | 0.00015, buy-side only | correct |
| `exchange_txn` | 0.0000297 | **0.0000307** | wrong — omits IPFT |
| `sebi_fee` | 0.000001 | 0.000001 | correct |
| `brokerage` | **0.0003 ad valorem** | **0 (Zerodha) or flat Rs 20/order** | badly wrong — 15× too high at Rs 10 lakh |
| `gst` base | brokerage + exchange_txn | + SEBI fee + IPFT + **DP charge** | incomplete |
| DP charge on sell | **missing** | Rs 13–20/scrip + GST | missing entirely |

Source for the exchange rate: NSE circular **NSE/FA/73061**, 27 Feb 2026, effective 1 Mar 2026 —
cash-market transaction + IPFT is **Rs 307 per crore per side (0.00307%)**. NSE moved money
between the transaction and IPFT lines with no change to the total, which breaks any model that
hardcoded "0.00297% + Rs 10/cr IPFT" as two separate items. Some broker pages (Upstox, Groww)
still display the stale 0.00297%.

**Hand-checked round trip, Rs 10 lakh delivery leg, zero brokerage** (reproduced exactly by
`costs.py`, asserted in `verify_backtest.py:10.6b`):

```
STT        0.001 x 10,00,000 x 2 legs        = 2,000.00
stamp      0.00015 x 10,00,000 (buy only)    =   150.00
exch+IPFT  0.0000307 x 10,00,000 x 2         =    61.40
SEBI       0.000001 x 10,00,000 x 2          =     2.00
brokerage                                     =     0.00
DP charge  (CDSL 3.50 + broker 9.50)         =    13.00
GST 18% on (0 + 61.40 + 2.00 + 13.00)        =    13.75
                                               ----------
statutory total                               = 2,240.15   -> 22.40 bps
+ slippage 5 bps round trip                   ->  27.40 bps
```

**The structural error matters more than the rate errors.** The plan's model is purely
ad valorem, so it returns 29.3 bps at every trade size. Real Indian delivery cost is not
scale-invariant — brokerage is capped at Rs 20 and DP is a flat ~Rs 15:

`cost_per_leg(V) = 0.0022248 V + 62.54`  →  bps falls with size.

| notional / leg | actual bps | plan's 29.3 bps | error |
|---|---|---|---|
| Rs 10,000 | 84.8 | 29.3 | **−55 bps (dangerously understated)** |
| Rs 88,700 | 29.3 | 29.3 | crossover |
| Rs 10,00,000 | 22.9 | 29.3 | +6.4 bps (28% too conservative) |
| Rs 50,00,000 | 22.4 | 29.3 | +6.9 bps |

**Effect on this build:** none on the design — every rate is a named parameter in `costs.py`
and results are reported across a 0/15/30/50 bps sweep. The plan's central 30 bps happens to
sit close to the real 27.4 bps at the Rs 10 lakh position size this build assumes, so it was
right for the wrong reasons. Below ~Rs 89,000 per position it would flip to understating cost.

## 4.2 Turnover drag

The closed-form arithmetic is right: `topk=50, n_drop=5` daily means 20% of capital traded per
session, and `0.20 × 0.0015 × 252 ≈ 7.6%/yr` at 15 bps one-way.

But realised turnover is materially lower, for two reasons that only show up empirically:

1. **`hold_thresh=5` binds harder than the plan expected.** The plan said a mean holding period
   of ~10 days makes it "mostly non-binding". With daily rebalancing, a name bought on session
   *t* cannot be sold before *t+5*, and a large share of drop candidates are blocked.
2. **qlib's `comb` guard** (see 4.12 below) means the realised drop count is often below
   `n_drop`.

Measured with a random signal (worst case for turnover), daily / `n_drop=5`:
**36.4× annual turnover → 5.46%/yr drag at 30 bps round trip**, versus the 50.4× the closed
form predicts. Weekly rebalancing: **10.4× → 1.56%/yr**. The engine's cost drag reconciles with
`turnover × one_way_rate` to 0.00 percentage points (`verify_backtest.py:10.6a`).

The plan's conclusion still stands and drove the design: rebalance frequency is treated as a
first-class arm (`daily` and `weekly`, plus `n_drop ∈ {1,2,5}`), and weekly is the pre-registered
primary.

## 4.3 Throughput — REFUTED

The plan reported batch 1 = 3648 ms/series, batch 32 = 112 ms/series, a ~32× win, and warned
that an earlier session had measured no batching benefit at `ctx=512` and wrongly generalised.
Re-measured at the exact config this build runs (`lookback=90, predict=10`, Kronos-base, mps):

| batch | ms/series |
|---|---|
| 1 | 160.6 |
| 8 | 100.4 |
| 16 | 117.1 |
| 32 | **99.9** |
| 64 | 108.8 |
| 128 | 108.7 |

**The win is 1.61×, not 32×.** Batch 1 is 160 ms, not 3648 ms — the plan's batch-1 figure was
almost certainly measured at `lookback=512`, where the per-call cost is dominated by a much
longer context. The mechanism is that `auto_regressive_inference` has **no KV cache**: every one
of the 10 decode steps re-runs the full 12-layer transformer over the whole 90–100 token window,
so the work is compute-bound and already saturating the GPU at batch 1.

**Effect on the budget:** none, by coincidence. The plan budgeted 112 ms/series and the measured
figure is ~100–130 ms/series in production, so the full 756-date × ~190-name pass lands at
**~4.2 h** as planned. Memory is a non-issue: batch 384 succeeded using 0.44 GB of MPS
allocation, so the 32 GB ceiling was never approached.

fp16 was tested as a speedup and **fails**: `Expected query, key, and value to have the same
dtype, but got query.dtype: float key.dtype: float and value.dtype: c10::Half` — some buffer
inside the attention path stays fp32 after `.half()`.

## 4.4 Signals live in normalized space — CONFIRMED

`auto_regressive_inference` never denormalizes; `KronosPredictor.predict` does it afterwards and
`qlib_test.py` does not. Measured over 8 windows of RELIANCE daily:

- predicted close, |max| = **3.39**, mean −0.05, std 1.01 → O(1) z-scores
- the same stock's raw close level averages **1331** rupees
- denormalizing with the window `mean`/`std` recovers a mean of 1367, a ratio of **1.027** to the
  raw level

So the repo's score is `(expected move) / sigma_close_90` — volatility-normalized, not an
expected return. Both scalings are therefore computed and reported as separate arms
(`_z` = repo replication, `_ret` = expected percentage return).

**One correction to the plan's reasoning.** The plan predicted that `score_z` "systematically
favours low-volatility names". Measured cross-sectionally on real signal dates, the tilt runs the
*other* way round from what that sentence implies: because Kronos's scores are overwhelmingly
negative (see below), multiplying a negative z-score by a larger `sigma/close` makes high-vol
names *more* negative, so it is `score_ret` that carries the low-vol tilt
(Spearman(score_ret, sigma/close) = **−0.24**) while `score_z` is close to vol-neutral (**+0.05**).

## 4.5 / 4.5b `preds` shape and tokenizer reconstruction error

Returned shape is **(B, 100, 6)** for `lookback=90, predict=10` — the reconstructed context is
included, exactly as claimed, and `preds[:, -pred_len:, :]` is the right slice.

The subtler claim is the important one. `last_day_close` is read from the raw input `x` while
`preds` comes out of an encode→decode round trip, so **tokenizer reconstruction error is folded
into every signal**. Measured:

| quantity | value (normalized units) |
|---|---|
| reconstruction error of the last context close, mean | −0.047 |
| … absolute mean | **0.113** |
| signal magnitude \|`preds[-1,close]` − `x[-1,close]`\| , mean | 0.925 |
| **error as a share of signal** | **12.2%** |
| full-context reconstruction RMSE (4 price columns) | 0.132 |
| full-context reconstruction correlation | 0.9915 |

**Not negligible**, so the third arm the plan asked for exists: `_recon`, defined as
`f(pred_future) − pred[lookback−1, close]`, which stays entirely inside reconstruction space.

Empirically the `_recon` arm barely reorders anything — cross-sectional rank correlation with
`_z` is **0.995** — so the 12% level error is largely a common shift that cancels in a ranking.
Worth knowing, and worth having measured rather than assumed.

## 4.6 Dividends, splits and the benchmark

**Splits: CONFIRMED not to be an issue.** yfinance back-adjusts splits regardless of
`auto_adjust`. RELIANCE has two recorded splits in the panel (2017-09-07 and 2024-10-28, both
2:1) and **zero** daily moves above 15% in the unadjusted close. Across the whole 200-name panel
there are 117 single-day moves above 20%, spread over 53 symbols, and **none is within 3 sessions
of a recorded split** — they are concentrated in IDEA (21), YESBANK (12) and ADANIENT (10), all
genuine events (`verify_backtest.py:10.8`).

**Dividends: the plan's correction is adopted, and its magnitude re-measured.** `^CNX200` and
`^NSEI` are price indices. Using `auto_adjust=True` stock series against them hands the strategy
the whole dividend yield as fake excess return. The plan estimated that gap at **+2.32%/yr** from
five hand-picked names; measured across all **194** names with a full history from 2021-01, the
mean gap is **+1.78%/yr** and the median **+0.91%/yr** (p10 +0.04%, p90 +4.53%). The distribution
is heavily right-skewed — ITC alone contributes +4.92% — so a five-name sample overstates it. The
correction still matters enormously: 1.8%/yr of fabricated alpha is larger than any real signal
found anywhere in this study. This build fetches with `auto_adjust=False, actions=True`, keeps **both** `close`
(split-adjusted only) and `adj_close` (split + dividend), and uses the **price-return**
convention everywhere — strategy, benchmark and every control. Direction of bias: the strategy is
charged a dividend yield it does not receive, so reported excess is conservative.

**Benchmark ticker.** `^CNX200` returns 3113 daily rows from 2014-01-01 with a latest close of
**14,168.05** on 2026-08-25, against `^NSEI` at 24,334.55 and `^CRSLDX` (NIFTY 500) at 23,611.60
on the same session. The level ratios are consistent with NIFTY 200 / NIFTY 50 / NIFTY 500.

**A new benchmark was added that the plan did not anticipate, and it turns out to matter more
than the choice between `^CNX200` and `^NSEI`.** The strategy is an *equal-weighted* top-50 drawn
from a 200-name list; `^CNX200` is *cap-weighted*. Over this test window those two facts alone —
with no signal at all — produce a very large return gap. So `bench_series(market, "ew200")`
computes the equal-weighted return of the eligible universe itself, and every arm is reported
against both. Excess vs `^CNX200` mixes selection skill with a size/weighting tilt; excess vs
`ew200` isolates selection.

## 4.7 `sample_count > 1` destroys the ensemble — CONFIRMED

With `sample_count=5` the returned array is **(2, 100, 6)** — three-dimensional, i.e. the five
stochastic paths have already been collapsed by `preds = np.mean(preds, axis=1)`. The shipped API
cannot return a distribution.

The workaround validated in `bench/validate_wrapper.py` works: passing **N identical rows with
`sample_count=1`** makes the reshape `(N, 1, L, D)` so the mean over axis 1 is a no-op. Eight
replicated rows produced **8 distinct terminal closes**, standard deviation **0.475** in
normalized units.

That number is itself worth noting: path noise (0.475) is roughly **half the signal magnitude**
(0.925) and about **46% of the cross-sectional score dispersion** (~1.04). With `sample_count=1`,
around 20% of the cross-sectional score variance is pure sampling noise, which attenuates IC by
roughly 11%. Averaging 5 paths would cut that to ~4%, at 5× the compute. The headline run uses
`sample_count=1` per the plan; the `--n-paths` flag exists to quantify the difference on a
strided subset, and it also emits path dispersion as a separate feature.

## 4.8 `QlibDataset` traps — both CONFIRMED empirically

1. **`__getitem__` ignores `idx`.** `ds[0]` and `ds[12345]` return **bit-identical tensors** when
   the RNG state is reset between calls, because the method draws `self.py_rng.randint(...)`
   instead of using `idx`. With `DataLoader(num_workers=2)`, two consecutive batches came back
   **bit-identical** — every worker forks the same seeded `py_rng` and emits the same sample
   stream, silently duplicating training data. `train_mps.py` therefore uses `num_workers=0`.
2. **It constructs `Config()` internally.** Confirmed: the class is only importable with
   `finetune/` on `sys.path`, and it reads `dataset_path` off a fresh `Config()`. Both
   `verify_pipeline.py` and `train_mps.py` patch `Config.__init__` at runtime rather than editing
   `finetune/config.py`, so no repo file is modified.

The pickle contract itself holds: our `preprocess.py` output loads through the **unmodified**
`finetune/dataset.py:QlibDataset` and one sample has shape **(101, 6)** =
`(lookback + predict + 1, len(feature_list))`, with 49 950 available windows across 185 val
symbols (`verify_pipeline.py:10.1`).

## 4.9 The repo trainer cannot run here — CONFIRMED

`python finetune/train_predictor.py` exits 1 with:

```
File "/Users/rap/Documents/z-projects/Kronos/finetune/train_predictor.py", line 12, in <module>
    import comet_ml
ModuleNotFoundError: No module named 'comet_ml'
```

That is the first of four independent blockers; the other three are still present in the source:
`train_predictor.py:185` `device = torch.device(f"cuda:{local_rank}")`, `train_predictor.py:240`
`if "WORLD_SIZE" not in os.environ:`, and `utils/training_utils.py:27` `torch.cuda.set_device`.
`bench/nse/train_mps.py` is the standalone single-device replacement.

## 4.10 Data sources — CONFIRMED, better than expected

- `https://archives.nseindia.com/content/indices/ind_nifty200list.csv` responds with **200 rows**
  and a `Symbol` column. The 50- and 500-name lists respond too. The fetched list is snapshotted
  to `data/nse/nifty200_constituents.csv` so the run is reproducible if NSE changes it.
- yfinance mapped **200 / 200** symbols with zero failures and zero NaNs in OHLC — better than
  the plan's warning about `-EQ` suffixes and `M&M`-style names. `data/nse/fetch_log.txt` is
  empty of MISS/SHORT lines. 538 660 rows, 2014-01-01 → 2026-08-25.
- **Historical constituent lists: AVAILABLE, and used.** The plan called this "worth 20 minutes
  of searching" because "if historical membership lists turn out to be available, the
  survivorship-bias limitation mostly disappears". They are available, and it took **six PDFs**.

  There is no ready-made free dataset (the public GitHub/Kaggle compilations that cover Indian
  indices do not include NIFTY 200; the NSE archives CSV is overwritten in place and dated
  variants 404; the Wayback Machine has one usable snapshot). But NIFTY publishes every
  semi-annual reconstitution as a text-layer PDF at
  `https://www.niftyindices.com/Press_Release/ind_prs<DDMMYYYY>.pdf`, each carrying an explicit
  "Nifty 200" excluded/included section and an effective date. Six of them span the entire
  2023-08 -> 2026-08 test window:

  | press release | effective | out | in |
  |---|---|---|---|
  | 2023-08-17 | 2023-09-29 | 7 | 8 |
  | 2024-02-28 | 2024-03-28 | 12 | 12 |
  | 2024-08-23 | 2024-09-30 | 15 | 15 |
  | 2025-02-21 | 2025-03-28 | 11 | 11 |
  | 2025-08-22 | 2025-09-30 | 9 | 9 |
  | 2026-02-23 | 2026-03-30 | 11 | 11 |

  `reconstruct_membership.py` walks backwards from today's published list
  (`members_before = (members_after − included) + excluded`) and self-checks that every regime
  holds 200 names. Restricted to names with usable price data the reconstructed universe is
  **199–200 per session (mean 199.6)** across the whole window — i.e. essentially exact.

  **260 distinct names were NIFTY 200 members at some point in the window, against 200 today.**
  Price data was fetched for 57 of the 60 that are not in today's list. Three are unavailable on
  yfinance and are the residual gap: `PEL` (Piramal Enterprises), `TATAMTRDVR` (Tata Motors DVR,
  cancelled in 2024) and `GUJGASLTD` (Gujarat Gas, merged). One apparent removal was a
  **rename**, not a removal — GMR Infrastructure -> GMR Airports — and is handled by an explicit
  alias; its price history is continuous under `GMRAIRPORT.NS`.

  **The six semi-annual reviews alone are not sufficient**, and the shortfall is instructive.
  Reconstructing from them lands on 201–203 members. NSE also publishes off-cycle changes, and
  it sometimes **revokes** a scheduled change before it takes effect. Four sourced corrections
  close the gap completely:

  | correction | source | effect |
  |---|---|---|
  | IDEA's 2024-09-30 exclusion was **revoked** after NSE recalculated its impact cost | `ind_prs25092024.pdf` | IDEA never left the index |
  | CENTRALBK's 2024-09-30 inclusion was **revoked** (a knock-on of IDEA staying) | `ind_prs25092024.pdf` | CENTRALBK never entered |
  | IREDA's 2024-03-28 inclusion was **revoked** for breaching the SEBI ETF impact-cost norm; **BSE Ltd. took the slot** | `ind_prs19032024.pdf` | IREDA entered later, on 2024-09-30 |
  | TATAMTRDVR excluded 2024-08-30 on the DVR share cancellation, **with no replacement** | `ind_prs23082024_1.pdf` | a second *security* of a company already in the index |

  Two further off-cycle events matter inside the window: JIOFIN entered on the Reliance demerger
  (2023-07-20) and was removed again on 2023-09-07 before re-entering at the March 2024 review.

  With those applied there are **zero residual inconsistencies**, and every remaining departure
  from exactly 200 is a sourced corporate action rather than an error: the index legitimately
  carried **201 securities for 200 companies** between 2023-09-29 and 2024-08-29 (the Tata Motors
  DVR line), which is also why the August 2023 review was asymmetric at 7 out and 8 in.
  Restricted to names with usable price data the rebuilt universe holds **198–200 per session,
  mean 199.28**, the shortfall being exactly the three unavailable names.

  A caveat worth stating: NSE routes demergers through transient `DUMMY*` placeholder listings
  (ITC Hotels, Siemens Energy, Tata Motors CV, Vedanta's four spun-off entities) that enter and
  leave within weeks. Those windows are not modelled here; each is a transient one-name
  distortion against a ~60-name correction.

  Results are therefore reported **both ways**: on the current-membership universe the plan
  specified, and on the point-in-time universe. The difference between them is the measured
  value of the survivorship bias.

## 4.11 Open questions the plan could not answer

**MPS sampling reproducibility — REFUTED (it is reproducible).** The plan flagged this as
unknown. `torch.multinomial` on `mps` under a fixed seed is **bit-identical across runs**: three
consecutive runs of the same window with `set_seed(999)` gave `max_abs_diff = 0.0`. Without
reseeding, the same call differs by up to 4.91 in normalized units — so the seed genuinely
controls it. Greedy decoding (`top_k=1`) is also bit-identical. `signals.py` therefore seeds per
`(date, chunk)`, which makes a resumed run reproduce an uninterrupted one exactly.

**Memory ceiling.** Batch 384 ran fine at **0.44 GB** of MPS allocation. Memory is nowhere near
binding; throughput is.

**`hold_thresh` semantics — CONFIRMED.** From the installed qlib 0.9.7 source: the docstring says
"minimum holding days", and the code checks
`current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh` before every sell,
where the counter is incremented once per completed bar by `add_count_all(bar=self.freq)`. So it
is a minimum holding period in **bars of the strategy's own frequency**. The backtester mirrors
this, including the detail that the check sits *inside* the sell loop, after `buy` was already
sized from `len(sell)` — a sell blocked by `hold_thresh` does not shrink the buy list.

**Is qlib installable?** **Yes** — `pip install pyqlib` succeeded, qlib 0.9.7, and it did not
disturb the `numpy==2.2.6` / `torch==2.11.0` pins. It is used here only to read the strategy
source, not to run the backtest.

**Circuit limits — negligible, skipped.** Across 145 161 open-to-open observations in the test
window, **0.415%** exceed ±10% and **0.032%** exceed ±20%. Modelling NSE per-stock circuits would
change essentially nothing, so it is not modelled, and this is the measurement that justifies it.

---

## 4.12 Things the plan did not know to ask

### `tests/test_kronos_regression.py`'s `MODEL_REVISION` is Kronos-*small*'s SHA

The plan says to copy the revision-pinning habit from that test. Doing so literally fails:

```
TypeError: Kronos.__init__() missing 11 required positional arguments: 's1_bits', ...
```

`MODEL_REVISION = "901c26c1..."` is the SHA of **`NeoQuasar/Kronos-small`**, which is what that
test loads. Pinning `Kronos-base` to it 404s the config, and `huggingface_hub` reports the
failure as a confusing `TypeError` rather than a missing-file error. Resolved from the HF API:

| repo | SHA |
|---|---|
| `NeoQuasar/Kronos-base` | `2b554741eca47781b64468546e77fef3e85130e6` |
| `NeoQuasar/Kronos-small` | `901c26c1332695a2a8f243eb2f37243a37bea320` |
| `NeoQuasar/Kronos-mini` | `f4e68697d9d5aed55cef5c96aabc3376bcad9f81` |
| `NeoQuasar/Kronos-Tokenizer-base` | `0e0117387f39004a9016484a186a908917e22426` |

### Plan §6's val→test gap was 5× too small — REFUTED

The plan sets `val 2022-01-01..2023-06-30`, `test 2023-08-01..`, and calls the gap "deliberate —
it exceeds `lookback + predict`". It does not. `lookback + predict` = **100 trading sessions**,
and one calendar month is ~21. Measured on the real NSE calendar the gap is **21 sessions**, so a
finetuned model would see test-window context bars it had trained on.

Corrected: `val` ends **2023-02-15**, giving a **111-session** gap while keeping the full
756-session test window and 280 val sessions. Asserted in code
(`verify_pipeline.py:10.5a_lookahead_split_gap`).

### The plan's own backtester sketch would have churned the book

"Drop at most `n_drop`/rebalance" reads as "always replace the worst `n_drop` holdings". An
implementation that does that churns **20.7% of the book per rebalance on a constant score**,
purely on the sort tie-break, and sells rank-46..50 to buy rank-51..55 even when it already holds
the exact top 50.

qlib does **not** do this. Reading `TopkDropoutStrategy.generate_trade_decision` from the
installed qlib 0.9.7:

```python
# combine(new stocks + last stocks),  we will drop stocks from this list
# In case of dropping higher score stock and buying lower score stock.
comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
sell = last[last.isin(get_last_n(comb, self.n_drop))]
buy  = today[: len(sell) + self.topk - len(last)]
```

If the book is already the global top-`k`, the bottom `n_drop` of `comb` are the *candidates*,
which are disjoint from `last` by construction — so `sell` is empty, `buy` is empty, and nothing
trades. `n_drop` is a **cap on turnover, not a quota**. The backtester ports this faithfully;
`--dropout-mode naive` keeps the broken variant so its cost can be quantified. Caught by
`verify_backtest.py:10.4b`, which is exactly the test plan §10.4 asked for.

Two further fidelity details taken from the same source and mirrored here: qlib **does not
rebalance existing positions** (sells are full liquidations, buys split free cash equally,
retained names drift), and it deploys `risk_degree = 0.95` of cash. This engine is fully
invested — a deviation that flatters the strategy and every control equally, and is stated in
`RESULTS.md`.

### `only_tradable` is a no-op on this panel

qlib defaults `only_tradable=False`. This engine selects candidates from tradable names only,
which corresponds to `True`. On this panel the distinction is exactly nothing: **once a symbol
appears, it never misses a session** (missed-session rate 0.000 across all 200 names). The 96.4%
overall tradable share is entirely mid-window IPOs, not suspensions.

### Kronos's cross-sectional score is largely a mechanical reversal factor

Not a plan claim — a finding that emerged from the first smoke run and shaped the control set.

An autoregressive sampler on a per-window-normalized series regresses toward the window mean.
Because `anchor_z = (close_t − mean_90)/std_90` is positive for a name trading above its own
90-bar mean, a prediction that regresses toward 0 produces `score_z ≈ −anchor_z` **by
construction**. Measured on real signal dates: `last_ret` mean **−6.9%** over 10 days across the
entire cross-section, `last_z` mean **−0.97** — the model predicts a decline for essentially
every stock.

So the repo's signal is a candidate **short-term reversal factor** dressed as a forecast. Two
controls were added to test that directly, neither of which is in the plan:

- **`anchor_reversal`** — rank by `−anchor_z`, a one-line feature requiring no model at all.
- **Orthogonalised IC** — within each date, regress the Kronos rank-score on the baseline
  rank-score and compute the IC of the residual. If the residual carries no IC, Kronos adds
  nothing the free feature does not.

These are pre-registered as gate **G2b** in `PREREG.md`. Outcomes are in `RESULTS.md`.

**Measured, and it is not subtle.** `diagnose_drift.py` dumps the predicted path itself. Averaged
over the cross-section on four sample dates spanning the test window, the 10-step score in sigma
units fits

```
score_sigma  =  -0.636 * anchor_z  -  0.002        R^2 = 0.987
```

— a line through the origin. The model reverts **64% of the distance back to the 90-bar window
mean** over ten steps. The per-name evidence is stronger than that four-point fit: across 63
dates and ~190 names each, the mean cross-sectional Spearman correlation between the Kronos
score and `-anchor_z` is **+0.757**.

**It is the model's mode, not sampling noise.** Re-run with greedy decoding (`top_k=1`), which
removes stochastic sampling entirely, the fit is `score_sigma = -0.538 * anchor_z` with
R^2 = 0.977, the predicted 10-day move is -4.89% against the sampler's -5.55% at the same
`anchor_z`, and the rank correlation with `-anchor_z` is still +0.667. Lowering the temperature
does not recover a forecast, because there was never a forecast to recover.

**And it is the wrong reversal horizon.** Run over all 756 test sessions, short-horizon reversal
is a genuinely significant factor in this universe — 1-session reversal has a mean rank IC of
**+0.0281 with a Newey-West t of +6.24**, and 2-session +0.0251 (t = +5.51). The 90-bar z-score
reversal that Kronos actually reproduces has an IC of -0.014 at 10 days and is **not**
significant. Kronos's rank correlation with the 1-day reversal that works is only **+0.143**,
against +0.667 with the 90-bar one that does not.
