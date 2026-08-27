#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "########## PHASE 1: temperature sweep ##########"
for T in 0.3 0.6 1.0; do
  echo "--- T=$T ---"
  python bench/kronos_eval.py --tag "tsweep_T${T}" --paths 8 --windows 4 --T "$T" \
    --only btc_1h,nifty50_1d,reliance_1d
done

echo "########## PHASE 2: rolling-origin backtest (T=1.0 default) ##########"
python bench/kronos_eval.py --tag rolling --paths 8 --windows 8

echo "########## DONE ##########"
