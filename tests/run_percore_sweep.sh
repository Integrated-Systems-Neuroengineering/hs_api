#!/usr/bin/env bash
# run_percore_sweep.sh -- regression suite and biological feature tests on each listed core.
#
# Run from the hs_api checkout with the hardware environment active (tests/HARDWARE_TESTS.md):
#   bash tests/run_percore_sweep.sh <tag> <core> [<core> ...]
#   bash tests/run_percore_sweep.sh v7 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#
# Writes, per core N, into tests/:
#   pytest_<tag>_core<N>.log         test_bitstream_hardware_fast.py
#   pytest_<tag>_bio_core<N>.log     test_bio_features_hw.py
#   bio_traces_<tag>_core<N>.json    hardware and reference traces of every feature test
# and one summary line per suite in pytest_<tag>_sweep_summary.log.
set -u
if [ $# -lt 2 ]; then
  echo "usage: bash tests/run_percore_sweep.sh <tag> <core> [<core> ...]"; exit 2
fi
TAG=$1; shift
cd "$(dirname "$0")/.."
SUMMARY=tests/pytest_${TAG}_sweep_summary.log
echo "per-core sweep $(date)  tag=${TAG}  cores: $*" | tee "$SUMMARY"
result() {  # $1 = exit code, $2 = log file
  if [ "$1" -eq 124 ]; then echo "TIMED OUT (see $2)"; else tail -1 "$2"; fi
}
for c in "$@"; do
  echo "===== core $c =====" | tee -a "$SUMMARY"
  LOG=tests/pytest_${TAG}_core${c}.log
  HIAER_CORE_ID=$c timeout 600 python -m pytest tests/test_bitstream_hardware_fast.py -q > "$LOG" 2>&1
  echo "  regression  : $(result $? "$LOG")" | tee -a "$SUMMARY"
  LOG=tests/pytest_${TAG}_bio_core${c}.log
  rm -f tests/bio_traces_${TAG}_core${c}.json
  HIAER_CORE_ID=$c HIAER_BIO_TRACES=tests/bio_traces_${TAG}_core${c}.json \
    timeout 900 python -m pytest tests/test_bio_features_hw.py -q > "$LOG" 2>&1
  echo "  bio features: $(result $? "$LOG")" | tee -a "$SUMMARY"
done
echo "summary: $SUMMARY"
