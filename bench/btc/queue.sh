#!/bin/bash
# Sequential so the jobs do not contend for MPS. Measured throughput on this machine:
# lookback 90 runs at ~0.31 origins/s; lookback 512 at ~0.014/s, which is why the
# long-context probe is strided rather than dense.
cd /Users/rap/Documents/z-projects/Kronos
P=.venv/bin/python
L=bench/out/btc
run () { echo "=== $1 START $(date +%H:%M) ==="; shift; $P bench/btc/btc_signals.py "$@" >> $L/queue.log 2>&1; echo "=== DONE $(date +%H:%M) ==="; }

{
run "btc daily L90"   --csv data/crypto/btc_1d.csv --tag d90   --lookback 90  --predict 10 --paths 30 --stride 1  --batch-origins 8
run "btc hourly L90"  --csv data/crypto/btc_1h.csv --tag h90   --lookback 90  --predict 24 --paths 30 --stride 24 --batch-origins 8
run "eth daily L90"   --csv data/crypto/eth_1d.csv --tag eth90 --lookback 90  --predict 10 --paths 30 --stride 2  --batch-origins 8
run "btc daily L512"  --csv data/crypto/btc_1d.csv --tag d512  --lookback 512 --predict 10 --paths 30 --stride 12 --batch-origins 4
echo "=== ALL BTC WORK DONE $(date) ==="
} >> $L/queue_stages.log 2>&1
