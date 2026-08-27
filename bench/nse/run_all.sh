#!/usr/bin/env bash
# Full reproduction, in order. Every stage is independently runnable and resumable.
#
#   cd /Users/rap/Documents/z-projects/Kronos && bash bench/nse/run_all.sh
#
# GPU stages are marked with their measured wall clock on an Apple M5 / 32 GB / mps.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=bench/out/nse
mkdir -p "$OUT"

echo "=== 1. data (~15 s) ==============================================="
python bench/nse/fetch_universe.py --start 2014-01-01 | tee "$OUT/fetch_universe.log"
python bench/nse/preprocess.py --dividends price      | tee "$OUT/preprocess.log"

echo "=== 2. verification (no GPU: ~10 s; GPU: ~4 min) =================="
python bench/nse/verify_backtest.py | tee "$OUT/verify_backtest.log"
python bench/nse/verify_model.py --max-batch 384 | tee "$OUT/verify_model.log"
python bench/nse/verify_pipeline.py | tee "$OUT/verify_pipeline.log"

echo "=== 3. G0 pilot: 63 dates strided across the whole test window (~26 min) ==="
# Strided rather than the first 60 sessions, so the gate sees all three years and no
# work is wasted -- the full pass below resumes from these same part files.
python bench/nse/signals.py --tag zeroshot --stride 12 --batch-size 64 | tee "$OUT/pilot.log"
python bench/nse/controls.py --signals "$OUT/signals_zeroshot.parquet" \
    --tag pilot --skip-backtests | tee "$OUT/controls_pilot.log"

echo "=== 4. full zero-shot signal pass (~4.8 h, resumable) ============="
python bench/nse/signals.py --tag zeroshot --stride 1 --batch-size 64 | tee -a "$OUT/signals_zeroshot.log"

echo "=== 4b. point-in-time universe (removes survivorship) ============="
# Six NIFTY reconstitution PDFs rebuild actual index membership for the test window.
python bench/nse/reconstruct_membership.py | tee "$OUT/membership.log"
# The 57 names that were members but are not in today's list still need prices + signals.
python bench/nse/signals.py --tag pitextra --stride 1 --batch-size 64 \
    --panel data/nse/panel_price_pit.parquet \
    --membership data/nse/membership_extra.parquet | tee "$OUT/signals_pitextra.log"
python bench/nse/universe_bias.py | tee "$OUT/universe_bias.log"

echo "=== 5. controls + full grid (~5 min, no GPU) ======================"
python bench/nse/controls.py --signals "$OUT/signals_zeroshot.parquet" \
    --tag zeroshot --n-random 500 | tee "$OUT/controls_zeroshot.log"
python bench/nse/grid.py --signals "$OUT/signals_zeroshot.parquet" \
    --tag zeroshot --n-random 500 | tee "$OUT/grid_zeroshot.log"
# and again on the point-in-time universe -- the difference IS the survivorship bias
PITSIG="$OUT/signals_zeroshot.parquet,$OUT/signals_pitextra.parquet"
python bench/nse/controls.py --signals "$PITSIG" --tag pit \
    --membership data/nse/membership_pit.parquet --n-random 500 | tee "$OUT/controls_pit.log"
python bench/nse/grid.py --signals "$PITSIG" --tag pit \
    --membership data/nse/membership_pit.parquet --n-random 500 | tee "$OUT/grid_pit.log"

echo "=== 6. sampling sensitivity on the strided subset (~2 h) =========="
# Is the mean-reversion an artefact of one stochastic draw, or the model's conditional mode?
python bench/nse/signals.py --tag greedy  --stride 12 --top-k 1 --top-p 1.0 --T 1.0
python bench/nse/signals.py --tag T1p0    --stride 12 --T 1.0
python bench/nse/signals.py --tag paths5  --stride 12 --n-paths 5
for t in greedy T1p0 paths5; do
  python bench/nse/controls.py --signals "$OUT/signals_$t.parquet" --tag "$t" --skip-backtests
done

echo "=== 7. finetune on NSE (G3) (~90 min) ============================="
python bench/nse/train_mps.py --budget-min 90 --batch-size 16 | tee "$OUT/train_mps.log"

echo "=== 8. finetuned signals + controls (~1.3 h) ======================"
python bench/nse/signals.py --tag finetuned --stride 4 \
    --model-path "$OUT/finetune/best_predictor" | tee "$OUT/signals_finetuned.log"
python bench/nse/controls.py --signals "$OUT/signals_finetuned.parquet" \
    --tag finetuned --n-random 200
python bench/nse/grid.py --signals "$OUT/signals_finetuned.parquet" \
    --tag finetuned --n-random 200

echo "=== 9. report ====================================================="
python bench/nse/report.py --tag zeroshot --ft-tag finetuned
echo "done -> $OUT/RESULTS.md"
