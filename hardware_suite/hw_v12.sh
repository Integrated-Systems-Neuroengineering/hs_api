#!/bin/bash
# hw_v12.sh -- crisdsc0: the v12 bitstream (fixes 1-26 + plasticity) on the card, every test, one summary.
# Results: <kit>/results_<date>/.  Put round 27's heavy-spike replay (r_heavy_p0_tx.txt) in <kit>/sim/ to compare the
# card with the simulation.  Status bits 16-19 (watchdog, bypass, delay-slot overflow, Phase-2 fallback) must never set.
K=$(cd "$(dirname "$0")" && pwd); T=${HW_SW:-$(cd "$K/../.." && pwd)}
B=${HW_BIT:-/bitstreams/sixteen_core_noc_v12b_ila.bit}   # the bitstream under test (v12b: fixes 1-28 + the HBM-side slice stage)
C=${HW_CORE:-1}                  # the core under test (0-15); hw_v12_sweep.sh runs this on every core
case $C in [0-9]|1[0-5]) ;; *) echo "HW_CORE must be 0-15"; exit 1;; esac
O=$K/results_core${C}_$(date +%m%d_%H%M); mkdir -p $O; S=$O/summary.txt
say()  { echo "[$(date +%T)] $*"; }
stop() { say "STOP: $*"; exit 1; }
rec()  { printf "%-34s %s\n" "$1" "$2" | tee -a $S; }
[ -f $B ] || stop "$B missing"
cd $T/hs_api && source .venv/bin/activate || stop "hs_api venv"
export PYTHONPATH=$T/hs_bridge:$T/hs_api/tests PYTHONUNBUFFERED=1 HIAER_CORE_ID=$C
python3 $K/patch_strict_spikes.py $T/hs_api || stop "strict spike-count patch"
true  # the tests are committed in hs_api/tests
sudo -v || stop "sudo"; ( while true; do sudo -n true; sleep 50; done ) & KA=$!; trap "kill $KA 2>/dev/null" EXIT
flash() { sudo /bitstreams/scripts/flash.sh --pcie-addr 0000:81:00.0 $B > $O/flash.log 2>&1 || stop "flash failed (reboot crisdsc0, then run again)"; }
est() {  # error_status of the core under test via CMD 10 (bit 16 watchdog ended a phase, 17 exp-PSC bypass, 18 delay-slot overflow, 19 Phase-2 fallback)
python - << 'PY' 2>/dev/null
import os, numpy as np, hs_bridge.wrapped_dmadump.dmadump as d
cmd = np.array([0] * 62 + [(int(os.environ["HIAER_CORE_ID"]) & 0x1F) << 3, 10], dtype=np.uint64)
d.dma_dump_write(cmd, len(cmd), 1, 0, 0, 0, d.DmaMethodNormal)
for _ in range(2000):
    code, data = d.dma_dump_read(1, 0, 0, 0, d.DmaMethodNormal, 64)
    b = [int(x) & 0xFF for x in list(data)[:64]]
    if len(b) == 64 and b[63] == 0xFA and b[62] == 0xCE and b[61] == 0xFA and b[60] == 0xCE:
        print("0x%08x" % (b[56] | b[57] << 8 | b[58] << 16 | b[59] << 24)); break
PY
}
TO_FIRED=0
pyt() {  # tag timeout node... : runs pytest, records PASS / FAIL / NOT RUN / HANG (and reflashes after a hang)
    local tag=$1 to=$2; shift 2
    timeout -k 30 $to python -m pytest "$@" -q -s -p no:cacheprovider > $O/$tag.txt 2>&1; local rc=$?
    local p=$(grep -oE "[0-9]+ passed" $O/$tag.txt | tail -1); local f=$(grep -oE "[0-9]+ failed" $O/$tag.txt | tail -1)
    local es=""; [ $rc != 124 ] && [ $rc != 137 ] && es=$(est)
    if [ -n "$es" ] && [ $(( es >> 16 & 15 )) != 0 ]; then TO_FIRED=$((TO_FIRED+1)); rec "$tag status" "error_status $es: a bit of 16-19 set"; fi
    if [ $rc = 124 ] || [ $rc = 137 ]; then rec "$tag" "HANG (timeout ${to}s) ${p:+$p }${f}"; flash
    elif [ -z "$p" ] && [ -z "$f" ]; then rec "$tag" "NOT RUN ($(tail -1 $O/$tag.txt | cut -c1-60))"
    elif [ -z "$f" ]; then rec "$tag" "PASS ($p)"
    else rec "$tag" "FAIL ($p, $f: $(grep -E '^FAILED' $O/$tag.txt | head -3 | cut -c8-60 | tr '\n' ';'))"; fi
    return $rc; }
echo "v12 hardware tests, core $C, $(date)  bitstream $(md5sum $B | cut -c1-12)" > $S
flash
fw=$(python -c "import os; from hs_bridge.FPGA_Execution.fpga_controller import bitstream_features as f; print(hex(f(int(os.environ['HIAER_CORE_ID']), refresh=True)))" 2>/dev/null | tail -1)
[ "$fw" = "0xff" ] && rec "feature word" "PASS ($fw)" || { rec "feature word" "FAIL ($fw, expected 0xff)"; stop "wrong bitstream or no feature word on core $C"; }
pyt fast_suite 5400 tests/test_bitstream_hardware_fast.py
F=tests/test_bitstream_hardware_fast.py::TestBitStream   # the sequence that exposed the stale pointer (fix 26), one session, one card
pyt sequence_fanout_fanin_noise 1800 "$F::test_max_axonal_fanout[4000]" "$F::test_max_axonal_fan_in[8190]" "$F::test_LIF_neuron_with_noise[-16]"
pyt fanin_9000 900 tests/test_fanin_9000_hw.py   # an input-only microphase (fix 27): exactly 9,000, no status bits
pyt bio_suite 2400 tests/test_bio_features_hw.py
if [ "$C" = 1 ]; then   # the exec_run glitch repetitions (fix 16): core 1, where they were defined
n=0; for i in $(seq 16); do pyt F09_run$i 300 tests/test_bio_features_hw.py -k F09 > /dev/null && n=$((n+1)); done; rec "F09 x16" "$n of 16 passed"
n=0; for i in $(seq 8);  do pyt F10_run$i 300 tests/test_bio_features_hw.py -k F10 > /dev/null && n=$((n+1)); done; rec "F10 x8" "$n of 8 passed"
else rec "F09 x16" "core 1 only (F09/F10 also run once in the bio suite)"; rec "F10 x8" "core 1 only"; fi
pyt syn64_long_delay 600 tests/test_syn64_long_delay_hw.py
pyt isolated_firing_64 600 tests/test_syn64_isolated_hw.py::test_isolated_64
pyt isolated_firing_32 600 tests/test_syn64_isolated_hw.py::test_isolated_32
PLAST_OUT=$O/plast.json pyt plasticity_protocol 1200 tests/test_plasticity_protocol_hw.py
if [ -s $O/plast.json ]; then python3 $K/hw2tx.py $O/plast.json $O/card_plast > /dev/null && python3 $K/plast_check.py $O/card_plast $O/plast.json > $O/plast_check.txt 2>&1
    rec "plasticity weights vs reference" "$(tail -1 $O/plast_check.txt | cut -c1-80)"; fi
HEAVY_OUT=$O/heavy.json pyt heavy_spike 1500 tests/test_heavy_spike_psc_hw.py
rec "heavy-spike host count" "$(grep -m1 'host saw' $O/heavy_spike.txt)"
if [ -s $O/heavy.json ] && [ -s $K/sim/r_heavy_p0_tx.txt ]; then python3 $K/heavy_vs_sim.py $O/heavy.json $K/sim/r_heavy_p0 > $O/heavy_vs_sim.txt 2>&1
    rec "heavy-spike card vs simulation" "$(tail -1 $O/heavy_vs_sim.txt | cut -c1-80)"
else rec "heavy-spike card vs simulation" "NOT RUN (put round 27's r_heavy_p0_tx.txt in $K/sim/)"; fi
[ $TO_FIRED = 0 ] && rec "status bits 16-19" "PASS (never set in any test group)" || rec "status bits 16-19" "FAIL ($TO_FIRED test groups)"
# last, because bit 18 is sticky: 1,100 delayed entries into one slot -- the test asserts 1,023 delivered and bit 18 set
timeout -k 30 900 python -m pytest tests/test_delay_overflow_hw.py -q -s -p no:cacheprovider > $O/delay_overflow.txt 2>&1
grep -qE "[0-9]+ passed" $O/delay_overflow.txt && rec "delay slot overflow (fix 25)" "PASS (1023 delivered, bit 18 set)" \
    || rec "delay slot overflow (fix 25)" "FAIL: $(grep -m1 -E '^E  ' $O/delay_overflow.txt | sed 's/^E *//' | cut -c1-70)"
echo; echo "==== SUMMARY ($S)"; cat $S
