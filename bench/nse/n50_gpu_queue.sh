#!/usr/bin/env bash
# Stage B of NIFTY50_PLAN.md sec.9: the representation ladder (arms A1-A6), serial.
#
# One GPU, one job at a time. Running two MPS extractions concurrently does not halve the
# wall clock -- it doubles the memory pressure and makes every ms/series in the sidecars
# incomparable, which is the number Stage E budgets from. So this is a queue, in the
# pattern of bench/nse/gpu_queue.sh: wait for whatever is already on the GPU, then run the
# steps in order, each to its own log under bench/out/nse/n50/logs/.
#
# Steps (letters are the plan's, arms are sec.5):
#   a  base  pretrained  --layers 4,8,12 --pool last,mean10   A1 + A2
#   b  small pretrained                                        A3
#   c  small reservoir --seed 7                                A3 control
#   d  mini  pretrained                                        A4
#   e  base  --tokenizer-z                                     A5
#   f  base  reservoir --seed 8 --cells n50                    A6 (F7: reservoir variance)
#   g  base  reservoir --seed 9 --cells n50                    A6
#
# Output names carry every axis that changes the array: hT_{model}_{arm}_{layer}_{pool}.npy,
# with the reservoir's seed inside {arm} (so f and g do not overwrite each other) and an
# _n50 tag on the restricted cell set. Step e produces hT_base_pretrained_final_last.npy a
# second time, on top of step a's -- the trunk state is bit-identical either way (pinned by
# tests/test_hidden.py::test_tokenizer_z_branch_does_not_move_the_trunk_state), only that
# one sidecar's argv ends up naming --tokenizer-z. The arm-A5 arrays are hT_tokz_{pool}.npy.
#
# A step that fails is logged and the queue CONTINUES: A4 depends on a Kronos-Tokenizer-2k
# download and f/g depend on data/nse/membership_pit_n50.parquet existing, and one missing
# input must not cost the other five arms their pass. The final summary lists what ran.
#
# Nothing here writes to bench/out/nse/ml/ -- the 2026-08-28 arrays stay untouched.
#
# Usage:  bench/nse/n50_gpu_queue.sh              run the queue
#         bench/nse/n50_gpu_queue.sh --dry-run    print the plan and the time estimate only
set -u
cd /Users/rap/Documents/z-projects/Kronos || exit 1
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

OUT=bench/out/nse/n50
LOGS=$OUT/logs
ML=bench/out/nse/ml
MEMBERSHIP=data/nse/membership_pit_n50.parquet
mkdir -p "$LOGS"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# ---- estimate ---------------------------------------------------------------------
# Anchored on the ms/series the record actually sustained over a full 150,527-cell pass
# (bench/out/nse/ml/hT_{pretrained,reservoir}_meta.json), not on a short burst: a 2,048-cell
# run of the same command clocks 10.1 ms/series against the full pass's 14.9, because the
# machine throttles. The per-model factors below are measured ratios at 2,048 cells, batch
# 64, on 2026-08-29 -- ratios survive the throttle, absolute rates do not. Capturing 8
# arrays instead of 1 measured 10.1 vs 10.3 ms/series, i.e. free, so depth and pooling do
# not enter the estimate.
estimate() {
python - "$ML" "$MEMBERSHIP" <<'PY'
import json, os, sys
import pandas as pd

ml, membership = sys.argv[1], sys.argv[2]
ms_pre = json.load(open(os.path.join(ml, "hT_pretrained_meta.json")))["ms_per_series"]
ms_res = json.load(open(os.path.join(ml, "hT_reservoir_meta.json")))["ms_per_series"]
idx = pd.read_parquet(os.path.join(ml, "windows_index.parquet"), columns=["date", "symbol"])
n_all = len(idx)

if os.path.exists(membership):
    mem = pd.read_parquet(membership, columns=["date", "symbol"]).drop_duplicates()
    mem["date"] = pd.to_datetime(mem["date"])
    left = pd.DataFrame({"date": pd.to_datetime(idx["date"].to_numpy()),
                         "symbol": idx["symbol"].to_numpy()})
    n_50, n50_src = len(left.merge(mem, on=["date", "symbol"], how="inner")), "measured"
else:
    n_50 = round(n_all * 50 / 200)
    n50_src = "ASSUMED 50/200 of the panel; the membership file does not exist yet"

SCALE = {"base": 1.00, "small": 0.32, "mini": 0.097}  # measured 2026-08-29, 2048 cells
rows = [("a", "base  pretrained  layers 4,8,12 x last,mean10", ms_pre, "base", n_all),
        ("b", "small pretrained  final x last",                ms_pre, "small", n_all),
        ("c", "small reservoir   seed 7",                      ms_res, "small", n_all),
        ("d", "mini  pretrained  final x last",                ms_pre, "mini", n_all),
        ("e", "base  pretrained  --tokenizer-z",               ms_pre, "base", n_all),
        ("f", "base  reservoir   seed 8, cells n50",           ms_res, "base", n_50),
        ("g", "base  reservoir   seed 9, cells n50",           ms_res, "base", n_50)]
print(f"  anchors: pretrained {ms_pre:.2f} ms/series, reservoir {ms_res:.2f} ms/series "
      f"(full {n_all:,}-cell passes, 2026-08-28)")
print(f"  cells:   all {n_all:,}   n50 {n_50:,} ({n50_src})")
total = 0.0
for tag, what, ms, model, n in rows:
    mins = ms * SCALE[model] * n / 1000 / 60
    total += mins
    print(f"  {tag}  {what:<44} {n:>7,} cells  {mins:6.1f} min")
print(f"  {'':3}{'TOTAL':<46}{'':>13}  {total:6.1f} min  ({total / 60:.1f} h)")
PY
}

echo "=== NIFTY50 Stage B representation-ladder queue ==="
estimate
echo

if [ "$DRY" = "1" ]; then
  echo "=== dry run: nothing launched ==="
  exit 0
fi

# ---- serialise against whatever is already on the GPU --------------------------------
# This script's own command line does not contain any of these names, so it cannot match
# itself; pgrep also excludes its own pid.
while pgrep -f "bench/nse/(hidden|signals|train_mps)\.py" >/dev/null; do
  echo "waiting for the GPU to free up ($(date +%H:%M))"; sleep 20
done

STATUS=""
run(){  # run <tag> <logname> <args...>
  local tag=$1 log=$2; shift 2
  echo "=== $tag START $(date +%H:%M) ==="
  if python bench/nse/hidden.py "$@" > "$LOGS/$log.log" 2>&1; then
    echo "=== $tag DONE  $(date +%H:%M) ==="; STATUS="$STATUS $tag:ok"
  else
    echo "=== $tag FAILED $(date +%H:%M) -- see $LOGS/$log.log ==="
    tail -5 "$LOGS/$log.log"; STATUS="$STATUS $tag:FAILED"
  fi
}

run a a_base_layers   --arm pretrained --model base  --layers 4,8,12 --pool last,mean10 \
                      --cells all --out-dir "$OUT"
run b b_small_pre     --arm pretrained --model small --cells all --out-dir "$OUT"
run c c_small_res7    --arm reservoir  --model small --seed 7 --cells all --out-dir "$OUT"
run d d_mini_pre      --arm pretrained --model mini  --cells all --out-dir "$OUT"
run e e_base_tokz     --arm pretrained --model base  --tokenizer-z --cells all \
                      --out-dir "$OUT"

# f and g need point-in-time NIFTY 50 membership. hidden.py already refuses to fall back to
# the 200 -- the check here only keeps the failure legible in this log instead of twice in
# two others.
if [ -f "$MEMBERSHIP" ]; then
  run f f_base_res8_n50 --arm reservoir --model base --seed 8 --cells n50 --out-dir "$OUT"
  run g g_base_res9_n50 --arm reservoir --model base --seed 9 --cells n50 --out-dir "$OUT"
else
  echo "=== f,g SKIPPED: $MEMBERSHIP does not exist (NIFTY50_PLAN sec.2.1) ==="
  STATUS="$STATUS f:skipped g:skipped"
fi

echo "=== ALL STAGE B WORK DONE $(date) ===$STATUS"
