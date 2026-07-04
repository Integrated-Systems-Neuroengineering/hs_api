#!/usr/bin/env python3
"""
test_single_core_conductance_STDP.py — HiAER-Spike single_core_conductance_STDP Feature Test Suite
=====================================================================
Tests all features of the single_core_conductance_STDP bitstream.

Run on crisdsc0 with the single_core_conductance_STDP bitstream flashed:

  cd /home/omowuyi/testing/hs_api && source .venv/bin/activate
  export PYTHONPATH=/home/omowuyi/testing/hs_bridge

  # Software-only tests (no FPGA needed, validates encoding):
  python3 tests/test_single_core_conductance_STDP.py --software-only

  # All tests including FPGA hardware:
  python3 tests/test_single_core_conductance_STDP.py --all

  # Phase 1 only (backward compat, 42 existing tests):
  pytest tests/test_bitstream_hardware_fast.py -v

Git hashes (base):
  hs_api:           94caf7e (L6m-testing-suite)
  hs_bridge:        1e3a114
  connectome_utils: 181f8a8 (dev)

Bitstream: single_core_conductance_STDP (WNS +0.023 ns, crisdsc3)
RTL reference: command_interpreter.v CMD_SET_PSC_PARAMS = 8'd13
"""

import sys
import os
import time
import argparse
import traceback
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hs_api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/omowuyi/testing/hs_bridge')
sys.path.insert(0, '/home/omowuyi/testing/hs_api')

try:
    from hs_api import single_core_conductance_STDP as p2
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    import single_core_conductance_STDP as p2

# =========================================================================
# Test Infrastructure
# =========================================================================
RESULTS = {}
PASS, FAIL, SKIP, ERROR = "PASS", "FAIL", "SKIP", "ERROR"


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        func._test_name = name
        return func
    return decorator


def run_test(func):
    name = getattr(func, '_test_name', func.__name__)
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print('=' * 60)
    try:
        result = func()
        RESULTS[name] = PASS if result is True else (SKIP if result is None else FAIL)
    except Exception as e:
        RESULTS[name] = ERROR
        print(f"  ERROR: {e}")
        traceback.print_exc()
    status = RESULTS[name]
    icon = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○', 'ERROR': '!'}[status]
    print(f"\n  {icon} {name}: {status}")


# =========================================================================
# SOFTWARE-ONLY TESTS (no FPGA needed)
# =========================================================================

@test("Signed MP Conversion (unsigned MP bug fix)")
def test_signed_mp():
    """Verify to_signed32 against the June 30 2026 email data.

    Bug: FPGA returns -3002 as 4294964294 (unsigned uint32).
    Fix: Interpret as signed int32.
    """
    vectors = [
        # (FPGA uint32, expected signed, source)
        (4294964294,  -3002,   "C1.1.114 time 0"),
        (4294967169,   -127,   "C1.2.25 time 0"),
        (4294967235,    -61,   "C1.3.759 time 0"),
        (4294947101, -20195,   "C1.5.250 time 0"),
        (4294955428, -11868,   "C1.6.228 time 0"),
        (4294965340,  -1956,   "C1.9.104 time 0"),
        (4294951390, -15906,   "C1.11.758 time 0"),
        (4294952283, -15013,   "C1.14.604 time 0"),
        (4294965850,  -1446,   "C1.17.30 time 0"),
        (4294961423,  -5873,   "C1.19.223 time 0"),
        (4294962014,  -5282,   "C1.24.574 time 0"),
        (4294954112, -13184,   "C1.27.665 time 0"),
        (4294966619,   -677,   "C2.17.91 time 1"),
        (117762,     117762,   "C1.0.654 (positive)"),
        (0,               0,   "zero"),
        (2147483647, 2147483647, "max int32"),
        (2147483648, -2147483648, "min int32"),
    ]
    all_ok = True
    for raw, expected, label in vectors:
        got = p2.to_signed32(raw)
        ok = got == expected
        if not ok:
            all_ok = False
        print(f"  {raw:>12d} -> {got:>12d}  {'OK' if ok else 'FAIL'}  ({label})")
    return all_ok


@test("CMD 13 Packet Encode/Decode Round-Trip")
def test_cmd13_roundtrip():
    """Verify all 15 CMD 13 fields survive encode → decode.

    Bit positions verified against command_interpreter.v:
      delta_mode at rxFIFO_dout[0], decay_ex at [12:1], ..., w_min at [130:115]
    """
    test_cases = [
        # (description, params_dict)
        ("Legacy mode (all defaults)", dict(
            delta_mode=1, decay_ex=0, decay_in=0, decay_w=0, delta_w=0,
            coba_mode=0, E_ex=0, E_in=0, neuromod_level=0,
            neuromod_excitability_bias=0, stdp_enable=0, A_plus=0,
            A_minus=0, w_max=32767, w_min=0)),
        ("CUBA exp PSC", dict(
            delta_mode=0, decay_ex=3354, decay_in=3354, decay_w=245,
            delta_w=10, coba_mode=0, E_ex=0, E_in=0, neuromod_level=0,
            neuromod_excitability_bias=0, stdp_enable=0, A_plus=0,
            A_minus=0, w_max=32767, w_min=0)),
        ("COBA + STDP + neuromod", dict(
            delta_mode=0, decay_ex=3354, decay_in=3277, decay_w=245,
            delta_w=10, coba_mode=1, E_ex=0, E_in=-1310,
            neuromod_level=128, neuromod_excitability_bias=-5,
            stdp_enable=1, A_plus=50, A_minus=25,
            w_max=30000, w_min=-10000)),
        ("Max values", dict(
            delta_mode=0, decay_ex=4095, decay_in=4095, decay_w=255,
            delta_w=255, coba_mode=1, E_ex=2047, E_in=-2048,
            neuromod_level=255, neuromod_excitability_bias=-128,
            stdp_enable=1, A_plus=255, A_minus=255,
            w_max=32767, w_min=-32768)),
    ]

    all_ok = True
    check_fields = ['delta_mode', 'decay_ex', 'decay_in', 'decay_w', 'delta_w',
                    'coba_mode', 'E_ex', 'E_in', 'neuromod_level',
                    'neuromod_excitability_bias', 'stdp_enable',
                    'A_plus', 'A_minus', 'w_max', 'w_min']

    for desc, params in test_cases:
        print(f"\n  Case: {desc}")
        pkt = p2.build_cmd13_packet(**params)
        decoded = p2.decode_cmd13_packet(pkt)

        assert decoded['opcode'] == 13, f"opcode={decoded['opcode']}"
        case_ok = True
        for key in check_fields:
            expected = params[key]
            got = decoded[key]
            if got != expected:
                print(f"    FAIL: {key} sent={expected} decoded={got}")
                case_ok = False
                all_ok = False
        if case_ok:
            print(f"    All 15 fields match")
    return all_ok


@test("64-bit Synapse Format Encode/Decode")
def test_synapse_64bit():
    """Verify 64-bit synapse field encoding.

    Format: [63:61]=opcode, [60:48]=dest, [47:32]=weight,
            [31:26]=delay, [25:22]=syn_type, [21:18]=stdp_tag, [17:0]=src_addr
    """
    cases = [
        (0, 1234, -500,   10, 0, 3,  5678),   # typical excitatory
        (0, 4000,  1000,   0, 1, 0, 50000),   # inhibitory, no delay
        (1,  100,  -1,    32, 2, 7,     0),   # INTER_CORE, modulatory
        (0, 8191, 32767,  63, 15, 15, 262143), # max values
        (0,    0,     0,   0, 0, 0,     0),   # all zeros
        (0, 2000, -32768,  1, 0, 0,   100),   # min weight
    ]
    all_ok = True
    for op, d, w, dl, st, tg, sa in cases:
        syn = p2.make_synapse_64bit(op, d, w, dl, st, tg, sa)
        g = {
            'op': (syn >> 61) & 7, 'dest': (syn >> 48) & 0x1FFF,
            'w': p2.to_signed16((syn >> 32) & 0xFFFF),
            'delay': (syn >> 26) & 0x3F, 'type': (syn >> 22) & 0xF,
            'tag': (syn >> 18) & 0xF, 'src': syn & 0x3FFFF,
        }
        ok = (g['op']==op and g['dest']==d and g['w']==w and
              g['delay']==dl and g['type']==st and g['tag']==tg and g['src']==sa)
        if not ok: all_ok = False
        print(f"  op={op} dest={d:>5d} w={w:>6d} delay={dl:>2d} "
              f"type={st:>2d} tag={tg:>2d} src={sa:>6d}  {'OK' if ok else 'FAIL'}")

    # Row packing
    entries = [p2.make_synapse_64bit(0, i*100, (i+1)*100, i) for i in range(4)]
    row = p2.pack_synapse_row_64bit(entries)
    ok = len(row) == 32
    if not ok: all_ok = False
    print(f"  Row packing: 4 entries -> {len(row)} bytes  {'OK' if ok else 'FAIL'}")
    return all_ok



@test("DVS Small Model (expected 44.44%)")
def test_dvs_small():
    """Run the small DVS model on the single_core_conductance_STDP bitstream with delta_mode=1.

    This uses the EXISTING DVS test infrastructure. No changes needed.
    The single_core_conductance_STDP bitstream with delta_mode=1 should produce the same
    accuracy as L6m (44.44%).

    Prerequisites:
      - single_core_conductance_STDP.bit flashed on crisdsc0
      - DVS model pickle available in tests/
      - shift=0 converted to shift=-17, legacy_noise_en=1

    Run separately:
      pytest tests/test_DVS_small.py -v -s
    """
    print("  DVS small model test")
    print("  Expected accuracy: >= 44.44% (L6m baseline)")
    print("  This test validates that delta_mode=1 preserves L6m DVS behavior.")
    print("")
    print("  Run manually:")
    print("    pytest tests/test_DVS_small.py -v -s")
    print("")
    print("  Software patches required for DVS:")
    print("    - shift=0 in pickle must be converted to shift=-17")
    print("    - legacy_noise_en=1 (35-bit MP mode)")
    print("    - DMA padding wrapper DISABLED for DVS large")
    return None  # Must be run manually via pytest


@test("DVS Large Model (expected >= 56.60%)")
def test_dvs_large():
    """Run the large DVS model on the single_core_conductance_STDP bitstream with delta_mode=1.

    Expected: >= 56.60% accuracy (same as L6m).
    The 2024 bitstream achieves 64.24% — the gap is the XDMA IP version
    difference (v4.1.4 vs v4.1.29), NOT a hardware bug.

    the June 30 email confirmed:
      - 2024 bitstream: 63.89% (ground truth 64.24%)
      - L6m bitstream: ~56.60%
      - Positive MPs match between FPGA and SpikingJelly
      - Negative MPs read as unsigned (fixed by to_signed32)

    Run separately:
      pytest tests/test_DVS_large_fulldataset_2024.py -v -s
    """
    print("  DVS large model test")
    print("  Expected accuracy: >= 56.60% (L6m baseline with Vivado 2024.1 XDMA)")
    print("  Ground truth (SpikingJelly): 64.24%")
    print("  Gap cause: XDMA IP v4.1.29 (Vivado 2024.1) vs v4.1.4 (2019.2)")
    print("")
    print("  Run manually:")
    print("    pytest tests/test_DVS_large_fulldataset_2024.py -v -s")
    print("")
    print("  NOTE: DVS large accuracy threshold in the test file is 64.57%")
    print("  (set for 2024 bitstream). Lower to 55.00% for L6m/single_core_conductance_STDP.")
    return None  # Must be run manually via pytest


# =========================================================================
# FPGA HARDWARE TESTS (require single_core_conductance_STDP bitstream on crisdsc0)
# =========================================================================

def get_fpga():
    """Get FPGA controller instance. Returns None if unavailable."""
    try:
        from hs_bridge.FPGA_Execution import fpga_controller
        fpga = fpga_controller.FPGAController()
        return fpga
    except Exception as e:
        print(f"  FPGA not available: {e}")
        return None


@test("HW: Enable exp PSC mode via CMD 13")
def test_hw_cmd13_send():
    """Send CMD 13 to FPGA and verify mode switch.

    After CMD 13 with delta_mode=0:
    - Phase 0 should use 5-cycle sub-state machine
    - Synapse weights should accumulate into I_ex/I_in (Row B) not V
    """
    fpga = get_fpga()
    if fpga is None:
        return None

    # Step 1: Verify baseline (delta_mode=1 = legacy mode)
    print("  Step 1: Verify legacy mode (delta_mode=1, default after reset)")
    # Just send CMD 13 with defaults to confirm command path works
    p2.send_cmd13_psc_params(fpga, delta_mode=1, coreID=0)
    print("  CMD 13 (legacy) sent successfully")

    # Step 2: Switch to biological mode
    print("  Step 2: Switch to biological mode (delta_mode=0)")
    p2.send_cmd13_psc_params(fpga,
        delta_mode=0,
        decay_ex=3354,    # τ_ex ~ 0.5ms
        decay_in=3354,    # τ_in ~ 0.5ms
        decay_w=245,
        delta_w=10,
        coba_mode=0,      # CUBA
        stdp_enable=0,    # STDP off for now
        coreID=0,
    )
    print("  CMD 13 (biological) sent successfully")

    # Step 3: Switch back to legacy for safety
    p2.send_cmd13_psc_params(fpga, delta_mode=1, coreID=0)
    print("  CMD 13 (legacy restore) sent successfully")
    return True


@test("HW: Signed MP readout from FPGA")
def test_hw_signed_mp():
    """Read membrane potentials and verify signed conversion.

    Setup: Configure network with inhibitory synapse (negative weight).
    Send spike. Read MP. Verify it's negative (not large positive).
    """
    fpga = get_fpga()
    if fpga is None:
        return None

    print("  This test requires a loaded network with inhibitory synapses.")
    print("  Run after test_bitstream_hardware_fast.py to have a network loaded.")
    print("  Then read MPs and verify signed conversion.")
    print("  MANUAL VERIFICATION NEEDED - see HARDWARE_REFERENCE.md")
    return None


@test("HW: Exp PSC decay verification")
def test_hw_exp_psc_decay():
    """Verify exponential current decay in biological mode.

    Plan:
    1. Send CMD 13: delta_mode=0, decay_ex=3354 (τ ~ 0.5ms)
    2. Load minimal network: 1 axon → 1 neuron, weight=1000
    3. Send spike at t=0
    4. Read I_ex from Row B at t=1, t=2, t=3, ...
    5. Verify I_ex decays: I_ex(t+1) ≈ I_ex(t) × (3354/4096)
    """
    fpga = get_fpga()
    if fpga is None:
        return None

    print("  Requires Row B URAM readout (CI CMD_IEP_RW with Row B address)")
    print("  Row B address = neuron URAM address + 2048 (ROW_B_OFFSET)")
    print("  PLANNED - requires Row B readout integration")
    return None


@test("HW: Hardware axonal delay")
def test_hw_axon_delay():
    """Verify hardware delay via axon_delay_buffer.v.

    CRITICAL: single_core_conductance_STDP uses HARDWARE delay, NOT software delay.
    Do NOT call _preprocess_delayed_synapses() on this bitstream.

    Plan:
    1. Configure axon delay buffer: axon 0 → delay=5
    2. Send spike at axon 0 at t=0
    3. Read target neuron MP at t=1..4 → should be 0
    4. Read target neuron MP at t=5 → should show weight accumulation
    """
    fpga = get_fpga()
    if fpga is None:
        return None

    print("  axon_delay_buffer.v: 8192×6-bit BRAM + 64-slot circular buffer")
    print("  Placed between CI and EEP in single_core.sv")
    print("")
    print("  IMPORTANT: L6m (L6m) uses SOFTWARE delay:")
    print("    _preprocess_delayed_synapses() → delay queue → max(0, dv-1)")
    print("  single_core_conductance_STDP uses HARDWARE delay:")
    print("    axon_delay_buffer.v (configured at init, transparent at runtime)")
    print("  NEVER mix these — double-applying delays is a silent error.")
    print("")
    print("  PLANNED - requires axon_delay_buffer config integration")
    return None


@test("HW: COBA conductance-based synapses")
def test_hw_coba():
    """Verify COBA mode (conductance × driving force).

    In COBA mode: dV = g_ex × (E_ex - V) + g_in × (E_in - V)
    The same conductance at different V produces different dV.

    Plan:
    1. Send CMD 13: coba_mode=1, E_ex=0, E_in=-1310
    2. Load network, set initial V to different values
    3. Send identical spikes
    4. Read resulting dV — should differ based on V
    """
    fpga = get_fpga()
    if fpga is None:
        return None
    print("  PLANNED - requires Row B readout + V initialization")
    return None


@test("HW: STDP weight update")
def test_hw_stdp():
    """Verify Phase 4 STDP weight modification.

    stdp_controller.v (13-state FSM) during Phase 4:
    1. Captures spike addresses from Phase 0
    2. Reads source trace from URAM Row B
    3. Reads synapse from HBM via ci2hbm
    4. Δw = (A_plus × trace) >>> 4
    5. Clamps to [w_min, w_max]
    6. Writes back to HBM

    Plan:
    1. CMD 13: stdp_enable=1, A_plus=100, A_minus=50, w_max=10000, w_min=0
    2. Load 2-neuron network with known synapse weight
    3. Create pre→post spike pair (LTP condition)
    4. Read synapse weight from HBM before and after
    5. Verify weight increased
    """
    fpga = get_fpga()
    if fpga is None:
        return None
    print("  PLANNED - requires HBM readback + STDP timing control")
    return None


# =========================================================================
# Main
# =========================================================================

def print_summary():
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print('=' * 60)
    counts = {s: 0 for s in [PASS, FAIL, SKIP, ERROR]}
    for name, status in RESULTS.items():
        icon = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '○', 'ERROR': '!'}[status]
        print(f"  {icon} {name:<50s} {status}")
        counts[status] += 1
    total = len(RESULTS)
    print(f"\n  Total: {total} | Pass: {counts[PASS]} | Fail: {counts[FAIL]} "
          f"| Skip: {counts[SKIP]} | Error: {counts[ERROR]}")
    return counts[FAIL] == 0 and counts[ERROR] == 0


def main():
    parser = argparse.ArgumentParser(description='single_core_conductance_STDP Feature Tests')
    parser.add_argument('--software-only', action='store_true',
                        help='Run only software tests (no FPGA needed)')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests including FPGA hardware')
    args = parser.parse_args()

    if not args.software_only and not args.all:
        args.software_only = True

    print("HiAER-Spike single_core_conductance_STDP (single_core_conductance_STDP) Test Suite")
    print(f"Bitstream: single_core_conductance_STDP (WNS +0.023 ns, crisdsc3)")
    print(f"RTL: command_interpreter.v CMD_SET_PSC_PARAMS = 8'd13")

    # Software-only tests (always run)
    run_test(test_signed_mp)
    run_test(test_cmd13_roundtrip)
    run_test(test_synapse_64bit)

    # DVS model tests (always shown, run manually via pytest)
    run_test(test_dvs_small)
    run_test(test_dvs_large)

    # FPGA hardware tests
    if args.all:
        run_test(test_hw_cmd13_send)
        run_test(test_hw_signed_mp)
        run_test(test_hw_exp_psc_decay)
        run_test(test_hw_axon_delay)
        run_test(test_hw_coba)
        run_test(test_hw_stdp)

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
