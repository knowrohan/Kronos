#!/usr/bin/env bash
cd /Users/rap/Documents/z-projects/Kronos
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=bench/out/nse
step(){ echo "=== $1 START $(date +%H:%M) ==="; }
done_(){ echo "=== $1 DONE $(date +%H:%M) ==="; }

while pgrep -f "signals.py --tag greedy" >/dev/null; do sleep 20; done
done_ "greedy"

step "drift";  python bench/nse/diagnose_drift.py > "$OUT/drift.log" 2>&1; done_ "drift"

step "us";     python bench/nse/signals.py --tag us --stride 12 --batch-size 64 \
                 --panel data/us/panel_price.parquet --calendar data/us/calendar.json \
                 --start 2023-08-01 --end 2026-08-25 > "$OUT/signals_us.log" 2>&1; done_ "us"

step "full";   python bench/nse/signals.py --tag zeroshot --stride 1 --batch-size 64 \
                 > "$OUT/signals_full.log" 2>&1; done_ "full"

step "pitextra"; python bench/nse/signals.py --tag pitextra --stride 1 --batch-size 64 \
                 --panel data/nse/panel_price_pit.parquet \
                 --membership data/nse/membership_extra.parquet \
                 > "$OUT/signals_pitextra.log" 2>&1; done_ "pitextra"

step "finetune"; python bench/nse/train_mps.py --budget-min 90 --batch-size 16 \
                 > "$OUT/train_mps.log" 2>&1; done_ "finetune"

if [ -d "$OUT/finetune/best_predictor" ]; then
  step "ftsignals"; python bench/nse/signals.py --tag finetuned --stride 12 --batch-size 64 \
                 --model-path "$OUT/finetune/best_predictor" \
                 > "$OUT/signals_finetuned.log" 2>&1; done_ "ftsignals"
fi
echo "=== ALL GPU WORK DONE $(date) ==="
