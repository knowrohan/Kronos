# Does Kronos have an edge on BTC?

The NSE-200 study answered the cross-sectional question: no. This answers the
single-asset timing question on crypto, which is the market Kronos was largely
pretrained on and the one where its published demos live.

## Why this is not a rerun of the NSE study

NSE ranked ~200 names against each other every session. There is nothing to rank here,
so the tests change:

| NSE-200 | BTC |
|---|---|
| information coefficient across names | calibration of the model's own P(up) |
| top-50 book vs equal-weight universe | long/flat timing vs buy-and-hold |
| STT, stamp duty, impact vs ADV | exchange taker fee, slippage, 30% flat tax |

## Pipeline

| step | script | output |
|---|---|---|
| fetch | `fetch_btc.py` | `data/crypto/{btc_1d,btc_1h,eth_1d}.csv` + manifest |
| forecast | `btc_signals.py` | `signals_<tag>.parquet`, `paths_<tag>.npy` |
| analyse | `btc_analysis.py` | `analysis_<tag>.json` |
| queue | `queue.sh` | `queue.log`, `queue_stages.log` |

Runs, all `paths=30`, `top_p=1.0`:

| tag | series | lookback | horizon | stride | origins |
|---|---|---|---|---|---|
| `d90` | BTC daily | 90 | 10 | 1 | 3,197 |
| `h90` | BTC hourly | 90 | 24 | 24 | 2,407 |
| `eth90` | ETH daily | 90 | 10 | 2 | 1,600 |
| `d512` | BTC daily | 512 | 10 | 12 | 231 |

## Design notes

**`sample_count` is pinned to 1 and the batch is replicated instead.** Upstream averages
its sample paths inside `auto_regressive_inference` before returning, which destroys the
distribution a calibration test needs. Replicating the batch costs identical compute and
returns the individual paths. (Merging #395 also fixes this at the library level; the
harness does not depend on it.)

**`top_p=1.0`, not the 0.99/0.9 defaults.** Truncated sampling narrows every path, so a
cone measured under truncation is overconfident before the model has said anything. Any
miscalibration found at `top_p=1.0` is the model's own.

**Lookback 90 is deliberate, not inherited.** It keeps the normalization window identical
to the NSE and S&P 100 runs, so the reversion slope is directly comparable across all
three markets. `d512` exists to test whether more context weakens the artefact; it is
strided because lookback 512 measured at 0.014 origins/s on this machine versus 0.32 at
lookback 90.

**Annualised figures are computed from elapsed calendar time**, and carry an
`annualisation_reliable` flag that is false under one year. A bar-count annualisation on
a 24-origin pilot reported six million percent; that is arithmetically correct and
completely useless.

**The Indian tax overlay is an order-of-magnitude note, not tax advice.** Section 115BBH
charges 30% flat on crypto gains with no loss set-off and no expense deduction, which
penalises a strategy that realises every trade far more than one that realises once.
Section 194S withholds 1% on every sell; that is creditable against the 30%, so it is
reported as a cash-flow drag rather than an extra cost.
