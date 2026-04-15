#!/usr/bin/env python3
"""
HiAER-Spike Diagnostic — Lightweight Axon Mapping & Pointer Inspection
=======================================================================

This script does NOT write to the FPGA. It only:
  1. Loads the DVS model pickle
  2. Rebuilds the connectome data structures (NO fpga_compiler, NO HBM writes)
  3. Checks axon name → coreTypeIdx mapping
  4. Checks pointer tables and SPIKE_OUT entries in software
  5. Reads a few HBM rows back (read-only) to verify weights are present

Run on crisdsc0:
    cd /home/omowuyi/testing/hs_api
    python diagnose_lightweight.py
"""

import sys
import os
import pickle
import numpy as np
import logging
import time
from pathlib import Path
from math import ceil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ─── Config constants ──────────────────────────────────────────────────────
N_NG = 16
DATA_PER_ROW = 8
PTRS_PER_ROW = 8
AXN_BASE_ADDR = 0
NRN_BASE_ADDR = 2 ** 14
SYN_BASE_ADDR = 2 ** 15
PTR_ADDR_BITS = 23
PTR_LEN_BITS = 9
SYN_OP_BITS = 3
SYN_ADDR_BITS = 13
SYN_WEIGHT_BITS = 16
rows_per_ptr = ceil(N_NG / DATA_PER_ROW)  # = 2

# ─── DMA constants ─────────────────────────────────────────────────────────
_HDR_HBM_DATA   = (0xBB, 0xBB)
_HDR_FIFO_EMPTY = (0xFF, 0xFF)


# ═══════════════════════════════════════════════════════════════════════════
#  PART A: Software-only inspection (no FPGA access needed)
# ═══════════════════════════════════════════════════════════════════════════

def test_axon_mapping(connectome):
    """Check axon userKey → coreTypeIdx mapping consistency.
    
    In test_model.py, inputs are created as 'A{i}' where i is the tensor position.
    input_user() extracts the integer i and sets bit i in the one-hot vector.
    The hardware uses bit position i to index into the axon pointer table.
    
    So we need: axon with userKey containing integer i → coreTypeIdx == i
    """
    print("\n" + "=" * 70)
    print("TEST A: Axon Name → coreTypeIdx Mapping")
    print("=" * 70)

    axons = connectome.get_axons()
    print(f"Total axons: {len(axons)}")

    mismatches = 0
    first_20_ok = True

    for seq_idx, axon in enumerate(axons):
        user_key = axon.get_user_key()
        core_type_idx = axon.get_coreTypeIdx()

        # Extract integer from user_key
        if isinstance(user_key, str) and user_key.startswith('A'):
            try:
                name_int = int(user_key[1:])
            except ValueError:
                name_int = None
        elif isinstance(user_key, int):
            name_int = user_key
        else:
            name_int = None

        # Three things should all be equal:
        # 1. Sequential position in get_axons() list (seq_idx)
        # 2. The integer in the name (name_int)  
        # 3. coreTypeIdx (what the hardware uses)
        
        is_ok = (core_type_idx == seq_idx)
        name_matches = (name_int == seq_idx) if name_int is not None else None

        if seq_idx < 20 or not is_ok:
            status = "OK" if is_ok else "*** MISMATCH ***"
            print(f"  axon[{seq_idx:5d}]: userKey={str(user_key):10s}, "
                  f"coreTypeIdx={core_type_idx:6d}, nameInt={str(name_int):6s} {status}")
            if not is_ok:
                mismatches += 1
                if seq_idx < 20:
                    first_20_ok = False

    if mismatches > 0:
        print(f"\n*** CRITICAL: {mismatches} axon coreTypeIdx != sequential position!")
        print("*** This means the one-hot bit positions in input_user() do NOT")
        print("*** correspond to the correct axon pointer rows in HBM.")
        print("*** This WOULD cause identical outputs for all inputs.")
    else:
        print(f"\nAll {len(axons)} axons: coreTypeIdx matches sequential position. OK.")

    # Now check if input_user sees integer indices or string names
    print(f"\n--- How does test_model.py feed inputs? ---")
    print(f"  test_model.py creates: inputs.append(f'A{{i}}') for active pixels")
    print(f"  CRI_network.step(inputs) receives list of strings like ['A0', 'A5', ...]")
    
    # Check what step() does with these names
    # We need to look at CRI_network.step -> network.run_step -> input_user
    print(f"\n--- Tracing the input path ---")
    print(f"  CRI_network.step(inputs) calls self.CRI.run_step(inputs)")
    print(f"  network.run_step(inputs) calls input_user(inputs, numAxons)")
    print(f"  input_user() expects integer indices, not strings!")
    print(f"")
    print(f"  Let's check: does CRI_network.step() convert 'A5' → 5?")
    
    return mismatches


def test_cri_network_step_translation():
    """Inspect CRI_network.step() to see if it translates axon names to indices."""
    print("\n" + "=" * 70)
    print("TEST B: CRI_network.step() Input Translation")
    print("=" * 70)
    
    try:
        import inspect
        from hs_api.api import CRI_network
        
        # Get step() source
        step_src = inspect.getsource(CRI_network.step)
        print("CRI_network.step() source:")
        print("-" * 40)
        # Print with line numbers
        for i, line in enumerate(step_src.split('\n'), 1):
            print(f"  {i:3d}: {line}")
        print("-" * 40)
        
        # Key question: does step() convert string names to integer indices?
        if 'get_coreTypeIdx' in step_src or 'coreTypeIdx' in step_src:
            print("\n  step() DOES look up coreTypeIdx — names are translated to HW indices")
        elif 'run_step' in step_src:
            print("\n  step() passes inputs to run_step(). Let's check run_step()...")
            run_step_src = inspect.getsource(CRI_network.step)
            # Check if inputs are converted anywhere
            
        # Also check if there's any name→index translation
        if 'axon' in step_src.lower() or 'map' in step_src.lower() or 'translate' in step_src.lower():
            print("  Found axon/map/translate references in step()")
        else:
            print("  NO name translation found — strings may be passed directly!")
            print("  This means input_user() receives ['A0', 'A5'] not [0, 5]")
            print("  input_user() does: for axon in inputSegment: one_hot_bin[axon%256]")
            print("  But axon is a STRING 'A0', and 'A0'%256 would CRASH or give wrong result!")
            
    except Exception as e:
        print(f"  Could not inspect CRI_network.step(): {e}")
        print("  Trying to read the source file directly...")
        
        # Try to find and read the hs_api source
        api_paths = [
            "/home/omowuyi/testing/hs_api/hs_api/api.py",
            "/home/omowuyi/testing/hs_api/hs_api/cri_network.py",
        ]
        for p in api_paths:
            if os.path.exists(p):
                print(f"\n  Found: {p}")
                with open(p) as f:
                    src = f.read()
                # Find step() method
                if 'def step' in src:
                    # Extract the step method
                    lines = src.split('\n')
                    in_step = False
                    indent = 0
                    for i, line in enumerate(lines):
                        if 'def step' in line and 'def step_' not in line:
                            in_step = True
                            indent = len(line) - len(line.lstrip())
                            print(f"\n  step() found at line {i+1}:")
                        if in_step:
                            print(f"    {line}")
                            # Stop when we hit the next method at same indent
                            if i > 0 and in_step and line.strip() and not line.strip().startswith('#'):
                                curr_indent = len(line) - len(line.lstrip())
                                if curr_indent <= indent and 'def ' in line and 'def step' not in line:
                                    break
                    if not in_step:
                        print("  step() not found in this file")


def test_input_user_with_strings():
    """Check what happens when input_user receives string inputs like 'A5'."""
    print("\n" + "=" * 70)
    print("TEST C: What does input_user() do with string inputs?")
    print("=" * 70)

    # Simulate what input_user does with string vs int inputs
    test_inputs_str = ['A0', 'A5', 'A100']
    test_inputs_int = [0, 5, 100]

    print(f"String inputs: {test_inputs_str}")
    print(f"Integer inputs: {test_inputs_int}")

    # input_user (reserve=True mode) does:
    #   currInput = inputs
    #   currInput.sort()
    #   for axon in inputSegment:
    #       one_hot_bin[axon%256] = "1"
    #
    # If axon is 'A5' (string), then 'A5' % 256 → TypeError!
    # Unless Python somehow handles it... let's check

    print(f"\nTesting 'A5' % 256:")
    try:
        result = 'A5' % 256
        print(f"  Result: {result}")
    except TypeError as e:
        print(f"  TypeError: {e}")
        print("  So string inputs WOULD CRASH in input_user()...")
        print("  This means CRI_network.step() MUST convert strings to ints first!")

    print(f"\nTesting sort behavior:")
    try:
        mixed = ['A100', 'A5', 'A0']
        mixed.sort()
        print(f"  String sort: {mixed}")
        print(f"  Note: 'A100' < 'A5' in string sort (lexicographic)!")
        print(f"  This doesn't affect one-hot encoding, just ordering.")
    except Exception as e:
        print(f"  Sort error: {e}")


def test_pointer_tables(connectome, model_config):
    """Inspect the pointer tables to verify they make sense."""
    print("\n" + "=" * 70)
    print("TEST D: Pointer Table Inspection (software only)")
    print("=" * 70)

    outputs = model_config['outputs']
    
    # Get the class-ordered neuron list (this is what map_to_hbm_fpga uses)
    neurons_ordered = connectome.get_class_ordered_list()
    print(f"Total neurons (class-ordered): {len(neurons_ordered)}")
    
    # Check output neurons
    print(f"\nOutput neurons (labels {outputs}):")
    for out_label in outputs:
        # Find the neuron with this user_key
        try:
            neuron = connectome.get_neuron_by_key(out_label)
            print(f"  Output {out_label}: userKey={neuron.get_user_key()}, "
                  f"coreTypeIdx={neuron.get_coreTypeIdx()}, "
                  f"hbmIdx={neuron.get_hbmIdx()}, "
                  f"model={neuron.get_neuronModel()}")
        except Exception as e:
            print(f"  Output {out_label}: lookup failed: {e}")
    
    # Check cutoffs
    print(f"\nCutoffs: {connectome.cutoffs}")
    print(f"Models: {connectome.get_models()}")

    # Verify formatedOutputs match what test_model.py expects
    core_outputs = connectome.get_core_outputs_idx(0)
    print(f"\nCore 0 output coreTypeIdx list: {core_outputs}")
    print(f"These should be 109604..109614 for the DVS model")


def test_what_step_sends():
    """Actually trace what CRI_network.step() sends to input_user().
    
    Monkey-patch input_user to capture what it receives.
    """
    print("\n" + "=" * 70)
    print("TEST E: Capture actual inputs to input_user()")
    print("=" * 70)
    
    # We'll instrument input_user to see what it gets
    import hs_bridge.FPGA_Execution.fpga_controller as ctrl
    
    original_input_user = ctrl.input_user
    captured_inputs = []
    
    def capturing_input_user(inputs, numAxons, simDump=False, coreID=0, reserve=True, cont_flag=False):
        captured_inputs.append({
            'inputs': list(inputs)[:10],  # First 10 for brevity
            'types': [type(x).__name__ for x in inputs[:5]],
            'numAxons': numAxons,
            'count': len(inputs)
        })
        # Don't actually send to FPGA — use simDump mode
        return original_input_user(inputs, numAxons, simDump=True, coreID=coreID, 
                                    reserve=reserve, cont_flag=cont_flag)
    
    # Also capture execute to prevent actual execution
    original_execute = ctrl.execute
    def fake_execute(simDump=False, coreID=0):
        return original_execute(simDump=True, coreID=coreID)
    
    original_flush = ctrl.flush_spikes
    def fake_flush(coreID=0):
        return ([], 0, 0)  # Empty spikes
    
    # Monkey-patch
    ctrl.input_user = capturing_input_user
    ctrl.execute = fake_execute
    ctrl.flush_spikes = fake_flush
    
    try:
        # Now create a fake input and run one step through CRI_network
        from hs_api.api import CRI_network
        
        fixture_path = Path("/home/omowuyi/testing/hs_api/tests/fixtures/DVS_model_config_shift=-17.pkl")
        with open(fixture_path, "rb") as f:
            config = pickle.load(f)
        
        # We need to avoid the full init (which writes to HBM)
        # Instead, check if step() is accessible and what it does
        print("  Cannot safely call step() without full init.")
        print("  Instead, reading CRI_network source directly...\n")
        
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        # Restore
        ctrl.input_user = original_input_user
        ctrl.execute = original_execute
        ctrl.flush_spikes = original_flush
    
    # Read the CRI_network source to understand step()
    hs_api_path = Path("/home/omowuyi/testing/hs_api/hs_api/api.py")
    if hs_api_path.exists():
        with open(hs_api_path) as f:
            src = f.read()
        
        print(f"  hs_api/api.py contents ({len(src)} chars):")
        print("-" * 60)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            print(f"  {i+1:4d}: {line}")
        print("-" * 60)
    else:
        print(f"  {hs_api_path} not found!")
        # Try finding it
        import subprocess
        result = subprocess.run(['find', '/home/omowuyi/testing/hs_api', '-name', 'api.py', '-o', '-name', 'cri_network.py'], 
                              capture_output=True, text=True, timeout=5)
        print(f"  Found files: {result.stdout}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART B: HBM readback (read-only, no writes)
# ═══════════════════════════════════════════════════════════════════════════

def read_hbm_row(dmadump, row_addr, coreID=0):
    """Read a single 256-bit row from HBM (read-only)."""
    commandPrefix = [2, coreID] + [0] * 27
    rowAddress = '0' + np.binary_repr(row_addr, 23)
    addr_bytes = [int(rowAddress[:8], 2), int(rowAddress[8:16], 2), int(rowAddress[16:], 2)]
    cmd = commandPrefix + addr_bytes + [0] * 32
    finalCmd = np.flip(np.array(cmd, dtype=np.uint64))

    exitCode = dmadump.dma_dump_write(finalCmd, len(finalCmd), 1, 0, 0, 0, dmadump.DmaMethodNormal)
    if exitCode != 0:
        return None

    for _ in range(500):
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        b63, b62 = int(data[63]), int(data[62])
        if b63 == _HDR_HBM_DATA[0] and b62 == _HDR_HBM_DATA[1]:
            return data
        elif b63 == _HDR_FIFO_EMPTY[0] and b62 == _HDR_FIFO_EMPTY[1]:
            continue
        else:
            continue
    return None


def decode_synapse_row(raw_data):
    """Decode 0xBBBB packet → 8 synapse entries."""
    data = np.flip(raw_data)
    binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
    syn_bits = binData[-256:]

    entries = []
    for i in range(8):
        s = syn_bits[32*i : 32*(i+1)]
        op = int(s[:3], 2)
        addr = int(s[3:16], 2)
        wt_raw = int(s[16:], 2)
        wt = wt_raw - 65536 if wt_raw >= 32768 else wt_raw
        entries.append((op, addr, wt, wt_raw))
    return entries


def test_hbm_readback(dmadump):
    """Read-only: sample synapse rows from HBM to check weights are present."""
    print("\n" + "=" * 70)
    print("TEST F: HBM Synapse Row Readback (read-only)")
    print("=" * 70)

    # First, drain any stale packets
    print("Draining stale packets...")
    for _ in range(100):
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        b63, b62 = int(data[63]), int(data[62])
        if b63 == _HDR_FIFO_EMPTY[0] and b62 == _HDR_FIFO_EMPTY[1]:
            break
    print("Done.\n")

    test_rows = [0, 1, 10, 100, 500, 1000, 5000, 10000, 50000]
    all_zero = 0
    has_data = 0

    for row_idx in test_rows:
        raw = read_hbm_row(dmadump, SYN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  Row {row_idx:6d}: READ TIMEOUT")
            continue

        entries = decode_synapse_row(raw)
        nonzero = sum(1 for (op, addr, wt, wr) in entries if wr != 0 or op != 0)

        if nonzero == 0:
            print(f"  Row {row_idx:6d}: ALL ZEROS")
            all_zero += 1
        else:
            print(f"  Row {row_idx:6d}: {nonzero}/8 non-zero entries:")
            for j, (op, addr, wt, wr) in enumerate(entries):
                if wr != 0 or op != 0:
                    op_str = {0: "LOCAL", 4: "SPIKE_OUT"}.get(op, f"OP{op}")
                    print(f"            [{j}] {op_str} addr={addr} weight={wt}")
            has_data += 1

    print(f"\nSummary: {has_data} rows with data, {all_zero} all-zero rows")
    if has_data == 0:
        print("*** CRITICAL: No weight data found in HBM! ***")
        print("    The previous script may have interrupted synapse loading.")
        print("    Re-run test_model.py to reload weights, then run this again.")

    # Also read a couple of axon pointer rows
    print(f"\n--- Axon pointer rows ---")
    for row_idx in [0, 1]:
        raw = read_hbm_row(dmadump, AXN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  Axon ptr row {row_idx}: READ TIMEOUT")
            continue
        # Decode as pointers
        data = np.flip(raw)
        binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
        ptr_bits = binData[-256:]
        print(f"  Axon ptr row {row_idx}:")
        for i in range(8):
            p = ptr_bits[32*i : 32*(i+1)]
            length = int(p[:PTR_LEN_BITS], 2)
            addr = int(p[PTR_LEN_BITS:], 2)
            syn_offset = addr - SYN_BASE_ADDR if addr >= SYN_BASE_ADDR else addr
            if addr != 0 or length != 0:
                print(f"    [{i}] start=0x{addr:06X} (synRow={syn_offset}), len={length}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("HiAER-Spike Lightweight Diagnostic")
    print("(No HBM writes — safe to run anytime)")
    print("=" * 70)

    # ── Part A: Software-only analysis ──
    fixture_path = Path("/home/omowuyi/testing/hs_api/tests/fixtures/DVS_model_config_shift=-17.pkl")
    if not fixture_path.exists():
        log.error(f"Pickle not found: {fixture_path}")
        sys.exit(1)

    print(f"\nLoading model config...")
    with open(fixture_path, "rb") as f:
        model_config = pickle.load(f)
    print(f"  Axons: {len(model_config['axons'])}, Connections: {len(model_config['connections'])}")
    print(f"  Outputs: {model_config['outputs']}")

    # Rebuild connectome WITHOUT triggering fpga_compiler or HBM writes
    print(f"\nRebuilding connectome (software only, no HBM writes)...")
    from connectome_utils.connectome import connectome as Connectome, neuron as Neuron
    
    # Reset global counters
    Neuron.reset_count()
    
    conn = Connectome()
    
    # Add axons
    axons_dict = model_config['axons']
    connections_dict = model_config['connections']
    outputs_list = model_config['outputs']
    
    # Build axon objects
    axon_objects = {}
    for axon_name in axons_dict:
        ax = Neuron(axon_name, "axon", axonType='Uaxon')
        conn.addNeuron(ax)
        axon_objects[axon_name] = ax
    
    # Build neuron objects
    neuron_objects = {}
    for neuron_name in connections_dict:
        is_output = neuron_name in outputs_list
        n = Neuron(neuron_name, "neuron", neuronModel=2, output=is_output)
        conn.addNeuron(n)
        neuron_objects[neuron_name] = n
    
    # Add synapses
    for pre_name, post_list in axons_dict.items():
        pre = axon_objects[pre_name]
        for post_name, weight in post_list:
            if post_name in neuron_objects:
                pre.addSynapse(neuron_objects[post_name], weight)
    
    for pre_name, post_list in connections_dict.items():
        pre = neuron_objects[pre_name]
        for post_name, weight in post_list:
            if post_name in neuron_objects:
                pre.addSynapse(neuron_objects[post_name], weight)
    
    # Pad and order (like CRI_network does)
    conn.pad_models()
    
    print(f"  Axons: {len(conn.get_axons())}")
    print(f"  Neurons: {len(conn.get_neurons())}")
    print(f"  Cutoffs: {conn.cutoffs}")

    # Run software tests
    test_axon_mapping(conn)
    test_cri_network_step_translation()
    test_input_user_with_strings()
    test_pointer_tables(conn, model_config)
    test_what_step_sends()

    # ── Part B: HBM readback ──
    print("\n\n" + "=" * 70)
    print("PART B: HBM Readback (read-only)")
    print("=" * 70)
    try:
        import hs_bridge.wrapped_dmadump.dmadump as dmadump
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        if exitCode == 0:
            print("DMA communication OK")
            test_hbm_readback(dmadump)
        else:
            print(f"DMA read failed (exitCode={exitCode}), skipping HBM readback")
    except ImportError:
        print("dmadump not available, skipping HBM readback")

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()