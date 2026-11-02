#!/bin/bash
# hw_v12_sweep.sh [cores...] -- crisdsc0: the complete v12 hardware suite on every listed core, one core at a time, each
# with a fresh flash (default: all 16, core 1 first).  v12's verified scope is one network on any core; this shows every
# test on every core.  Then one matrix: every test x every core.  Runs unattended; a hang on one core is recorded and
# the next core starts on a freshly flashed card.
K=$(cd "$(dirname "$0")" && pwd); CORES=${*:-1 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15}
sudo -v || exit 1; ( while true; do sudo -n true; sleep 50; done ) & KA=$!; trap "kill $KA 2>/dev/null" EXIT
M=$K/sweep_$(date +%m%d_%H%M); mkdir -p $M; echo "v12 per-core sweep $(date): cores $CORES" | tee $M/sweep.log
for c in $CORES; do
    echo "[$(date +%T)] core $c" | tee -a $M/sweep.log
    HW_CORE=$c $K/hw_v12.sh > $M/core$c.log 2>&1
    d=$(ls -td $K/results_core${c}_* 2>/dev/null | head -1); [ -n "$d" ] && cp $d/summary.txt $M/summary_core$c.txt
done
python3 $K/sweep_matrix.py $M $CORES | tee $M/matrix.txt
