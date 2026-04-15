#!/usr/bin/env python3
"""
HiAER-Spike Diagnostic Script — Test 1: HBM Weight Readback + Axon Mapping Check
==================================================================================

PURPOSE:
  Verify that trained synapse weights were correctly loaded into HBM memory,
  and that the axon index mapping between software and hardware is consistent.

USAGE:
  Run on crisdsc0 (where /dev/adxdma* devices exist):
    cd /home/omowuyi/testing/hs_api
    python diagnose_hbm_weights.py

WHAT THIS DOES:
  1. Loads the DVS model pickle to get the expected weights
  2. Reads several HBM synapse rows back via CMD_HBM_RW (read mode)
  3. Decodes the raw HBM data and compares against expected weights
  4. Checks axon pointer table entries
  5. Verifies SPIKE_OUT entries exist for output neurons
  6. Checks that the axon name→index mapping in test_model.py matches
     the connectome's axon ordering

PREREQUISITES:
  - Network must have been initialized (synapses loaded into HBM)
  - Run the test_model.py initialization first, then stop before inference
  - Or just run this after a full test_model.py run (HBM persists until reprogram)
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

# ─── Configuration constants (must match config.yaml / config.py) ──────────
N_NG = 16
DATA_PER_ROW = 8
PTRS_PER_ROW = 8
PTR_ADDR_BITS = 23
PTR_LEN_BITS = 9
AXN_BASE_ADDR = 0
NRN_BASE_ADDR = 2 ** 14
SYN_BASE_ADDR = 2 ** 15
SYN_OP_BITS = 3
SYN_ADDR_BITS = 13
SYN_WEIGHT_BITS = 16
rows_per_ptr = ceil(N_NG / DATA_PER_ROW)  # = 2

# ─── DMA header constants ──────────────────────────────────────────────────
_HDR_HBM_DATA   = (0xBB, 0xBB)
_HDR_MEM_POT    = (0xCC, 0xCC)
_HDR_FIFO_EMPTY = (0xFF, 0xFF)


def import_dmadump():
    """Import the DMA dump module."""
    try:
        import hs_bridge.wrapped_dmadump.dmadump as dmadump
        return dmadump
    except ImportError:
        log.error("Cannot import dmadump. Make sure hs_bridge is on PYTHONPATH.")
        log.error("Try: cd /home/omowuyi/testing/hs_bridge && pip install -e .")
        sys.exit(1)


def read_hbm_row(dmadump, row_addr, coreID=0):
    """Read a single 256-bit row from HBM at the given absolute row address.

    Sends CMD_HBM_RW (0x02) with R/W bit = 0 (read).
    Returns the raw 64-byte DMA packet, or None on error.
    """
    # Build the command: [2, coreID, 0*27] + [addr_bytes] + [0*32]
    # Address format: bit[23]=0 (read), bits[22:0]=row_addr
    commandPrefix = [2, coreID] + [0] * 27
    rowAddress = '0' + np.binary_repr(row_addr, 23)  # 24 bits: 0 + 23-bit addr
    addr_bytes = [
        int(rowAddress[:8], 2),
        int(rowAddress[8:16], 2),
        int(rowAddress[16:], 2)
    ]
    cmd = commandPrefix + addr_bytes + [0] * 32
    finalCmd = np.flip(np.array(cmd, dtype=np.uint64))

    exitCode = dmadump.dma_dump_write(finalCmd, len(finalCmd), 1, 0, 0, 0, dmadump.DmaMethodNormal)
    if exitCode != 0:
        log.error(f"DMA write failed with exitCode={exitCode} for row {row_addr}")
        return None

    # Poll for the 0xBBBB HBM readback packet
    for attempt in range(200):
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        b63, b62 = int(data[63]), int(data[62])
        if b63 == _HDR_HBM_DATA[0] and b62 == _HDR_HBM_DATA[1]:
            return data
        elif b63 == _HDR_FIFO_EMPTY[0] and b62 == _HDR_FIFO_EMPTY[1]:
            continue
        else:
            # Skip unexpected packets (leftover spikes, ACKs, etc.)
            continue

    log.error(f"Timeout waiting for HBM readback at row {row_addr}")
    return None


def decode_hbm_synapse_row(raw_data):
    """Decode a 0xBBBB HBM readback packet into 8 synapse entries.

    Each synapse entry is 32 bits: [31:29]=opcode, [28:16]=address, [15:0]=weight
    Returns list of 8 tuples: (opcode, address, weight_raw)
    """
    # Flip to MSB-first
    data = np.flip(raw_data)
    binData = ''.join([np.binary_repr(int(b), width=8) for b in data])

    # Header is in bits [511:496] (first 16 bits of MSB-first)
    # Data is in bits [255:0] (last 256 bits)
    synapse_bits = binData[-256:]

    entries = []
    for i in range(8):
        start = 32 * i
        stop = 32 * (i + 1)
        syn = synapse_bits[start:stop]
        opcode = int(syn[:3], 2)
        address = int(syn[3:16], 2)
        weight_raw = int(syn[16:], 2)
        # Weight is 16-bit, interpret as signed
        if weight_raw >= 32768:
            weight_signed = weight_raw - 65536
        else:
            weight_signed = weight_raw
        entries.append((opcode, address, weight_signed, weight_raw))

    return entries


def decode_hbm_pointer_row(raw_data):
    """Decode a 0xBBBB HBM readback packet into 8 pointer entries.

    Each pointer entry is 32 bits: [31:23]=length, [22:0]=start_addr
    The start_addr includes SYN_BASE_ADDR offset.
    Returns list of 8 tuples: (start_addr_raw, length)
    """
    data = np.flip(raw_data)
    binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
    ptr_bits = binData[-256:]

    entries = []
    for i in range(8):
        start = 32 * i
        stop = 32 * (i + 1)
        p = ptr_bits[start:stop]
        length = int(p[:PTR_LEN_BITS], 2)
        addr = int(p[PTR_LEN_BITS:], 2)
        entries.append((addr, length))

    return entries


def load_model_config(fixture_path):
    """Load the DVS model configuration pickle."""
    with open(fixture_path, "rb") as f:
        return pickle.load(f)


def rebuild_connectome_from_config(model_config):
    """Rebuild the connectome from the saved model config.

    This replicates what CRI_network does internally so we can
    inspect the axon ordering and pointer tables.
    """
    try:
        from hs_api.api import CRI_network
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']

        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )
        return network
    except Exception as e:
        log.error(f"Failed to rebuild CRI_network: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TEST 1A: Read and verify synapse weights
# ═══════════════════════════════════════════════════════════════════════════
def test_1a_synapse_readback(dmadump, network):
    """Read back several synapse rows from HBM and compare against expected."""
    print("\n" + "=" * 70)
    print("TEST 1A: HBM Synapse Weight Readback")
    print("=" * 70)

    if network is None:
        log.error("Network not available, skipping")
        return

    # Get the compiled HBM data
    hbm_data = network.network.hbm
    axon_ptrs, neuron_ptrs, synapses = hbm_data[0]

    total_syn_rows = len(synapses)
    print(f"Total synapse rows in software: {total_syn_rows}")
    print(f"SYN_BASE_ADDR: {SYN_BASE_ADDR}")

    # Sample a set of synapse rows to check
    # Pick rows from: beginning, middle, near output neurons, and end
    sample_rows = []
    if total_syn_rows > 0:
        sample_rows.append(0)  # First row
    if total_syn_rows > 10:
        sample_rows.append(5)  # Near beginning
    if total_syn_rows > 100:
        sample_rows.append(50)  # Early middle
        sample_rows.append(total_syn_rows // 2)  # Middle
    if total_syn_rows > 1000:
        sample_rows.append(total_syn_rows - 10)  # Near end
        sample_rows.append(total_syn_rows - 1)  # Last row

    print(f"\nSampling {len(sample_rows)} synapse rows: {sample_rows}")

    matches = 0
    mismatches = 0
    errors = 0

    for row_idx in sample_rows:
        print(f"\n--- Synapse row {row_idx} (HBM addr {SYN_BASE_ADDR + row_idx}) ---")

        # Expected data from software
        expected = synapses[row_idx]
        print(f"  Software expects: {expected}")

        # Read from hardware
        raw = read_hbm_row(dmadump, SYN_BASE_ADDR + row_idx)
        if raw is None:
            errors += 1
            continue

        hw_entries = decode_hbm_synapse_row(raw)

        # Compare. Note: the software stores tuples (opcode, address, weight) or
        # (1, neuronIdx) for spike entries. The hardware encoding is different.
        #
        # IMPORTANT: fpga_compiler.create_synapses() does np.flip() on the entire
        # byte array (line 319), which reverses the column order within each row.
        # The HBM readback returns data in the HBM's native order.
        # So hw_entries[0] corresponds to the LAST element written by the software
        # after the flip.
        #
        # Actually, the readback via CMD_HBM_RW returns the raw 256-bit row,
        # and our decode reads it MSB-first. Let's just print both and compare.

        print(f"  Hardware returns (8 entries, MSB-first decode):")
        for j, (op, addr, wt_s, wt_r) in enumerate(hw_entries):
            op_str = {0: "LOCAL", 4: "SPIKE_OUT"}.get(op, f"OP{op}")
            print(f"    [{j}] op={op_str}({op}), addr={addr}, weight={wt_s} (raw=0x{wt_r:04X})")

        # Check if at least some entries are non-zero (basic sanity)
        nonzero = sum(1 for (op, addr, wt_s, wt_r) in hw_entries if wt_r != 0 or op != 0)
        all_zero = (nonzero == 0)

        # Check expected
        expected_nonzero = sum(1 for e in expected if e != (0, 0, 0))

        if all_zero and expected_nonzero > 0:
            print(f"  *** MISMATCH: HBM row is ALL ZEROS but software has {expected_nonzero} non-zero entries!")
            mismatches += 1
        elif all_zero and expected_nonzero == 0:
            print(f"  OK: Both HBM and software are all zeros")
            matches += 1
        else:
            print(f"  HBM has {nonzero} non-zero entries, software has {expected_nonzero}")
            # Detailed comparison would need to account for the byte flip
            # For now, just flag if counts differ significantly
            if abs(nonzero - expected_nonzero) > 2:
                print(f"  *** WARNING: Non-zero entry count mismatch!")
                mismatches += 1
            else:
                matches += 1

    print(f"\nSynapse readback summary: {matches} OK, {mismatches} mismatches, {errors} read errors")
    return mismatches == 0 and errors == 0


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TEST 1B: Read and verify axon pointers
# ═══════════════════════════════════════════════════════════════════════════
def test_1b_axon_pointer_readback(dmadump, network):
    """Read back axon pointer rows from HBM and compare against expected."""
    print("\n" + "=" * 70)
    print("TEST 1B: HBM Axon Pointer Readback")
    print("=" * 70)

    if network is None:
        log.error("Network not available, skipping")
        return

    hbm_data = network.network.hbm
    axon_ptrs, neuron_ptrs, synapses = hbm_data[0]

    n_ptr_rows = len(axon_ptrs)
    print(f"Number of axon pointer rows: {n_ptr_rows}")
    print(f"AXN_BASE_ADDR: {AXN_BASE_ADDR}")
    print(f"First few axon_ptrs from software (flattened):")

    # axon_ptrs is a numpy array of shape (N, 8) where each element is a tuple
    flat_ptrs = axon_ptrs.flatten()
    for i in range(min(8, len(flat_ptrs))):
        print(f"  axon_ptr[{i}] = {flat_ptrs[i]}")

    # Read the first axon pointer row from HBM
    # Note: create_axon_ptrs() does np.fliplr() on axon_ptrs before writing
    # So the order in HBM is reversed column-wise compared to the software array
    print(f"\nReading first 2 axon pointer rows from HBM...")

    for row_idx in range(min(2, n_ptr_rows)):
        raw = read_hbm_row(dmadump, AXN_BASE_ADDR + row_idx)
        if raw is None:
            continue

        hw_ptrs = decode_hbm_pointer_row(raw)
        print(f"\n--- Axon pointer row {row_idx} (HBM addr {AXN_BASE_ADDR + row_idx}) ---")
        for j, (addr, length) in enumerate(hw_ptrs):
            syn_start = addr - SYN_BASE_ADDR if addr >= SYN_BASE_ADDR else addr
            print(f"    [{j}] start_addr=0x{addr:06X} (syn_row={syn_start}), length={length}")

        # Compare against software (accounting for fliplr)
        sw_row = axon_ptrs[row_idx]
        # fliplr reverses column order
        sw_row_flipped = list(reversed(sw_row))
        print(f"  Software (after fliplr): {sw_row_flipped}")


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TEST 1C: Check SPIKE_OUT entries for output neurons
# ═══════════════════════════════════════════════════════════════════════════
def test_1c_spike_out_entries(dmadump, network):
    """Verify that SPIKE_OUT entries (opcode=4) exist for output neurons in HBM."""
    print("\n" + "=" * 70)
    print("TEST 1C: SPIKE_OUT Entry Verification for Output Neurons")
    print("=" * 70)

    if network is None:
        log.error("Network not available, skipping")
        return

    hbm_data = network.network.hbm
    axon_ptrs, neuron_ptrs, synapses = hbm_data[0]

    outputs = network.network.outputs
    print(f"Output neuron coreTypeIdx list: {outputs}")
    print(f"Number of neuron pointer rows: {len(neuron_ptrs)}")

    # For each output neuron, find its pointer, then scan its synapse rows
    # for SPIKE_OUT entries
    flat_nrn_ptrs = neuron_ptrs.flatten()

    found_spike_out = 0
    missing_spike_out = 0

    for out_idx in outputs:
        # The pointer for this neuron
        if out_idx >= len(flat_nrn_ptrs):
            print(f"  Output {out_idx}: pointer index out of range! (max={len(flat_nrn_ptrs)-1})")
            missing_spike_out += 1
            continue

        ptr = flat_nrn_ptrs[out_idx]
        start_row = ptr[0]
        end_row = ptr[1]
        print(f"\n  Output neuron {out_idx}: synapse rows [{start_row}..{end_row}]")

        # Scan software synapse data for SPIKE_OUT entries
        has_spike_out = False
        for r in range(start_row, end_row + 1):
            if r < len(synapses):
                for entry in synapses[r]:
                    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 1:
                        print(f"    Found SPIKE_OUT in sw row {r}: neuronIdx={entry[1]}")
                        has_spike_out = True

        if has_spike_out:
            found_spike_out += 1
        else:
            print(f"    *** NO SPIKE_OUT found in software data!")
            missing_spike_out += 1

        # Now read the actual HBM rows to verify SPIKE_OUT is there
        print(f"    Reading HBM rows {start_row}..{min(start_row+1, end_row)} ...")
        for r in range(start_row, min(start_row + 2, end_row + 1)):
            raw = read_hbm_row(dmadump, SYN_BASE_ADDR + r)
            if raw is None:
                continue
            hw_entries = decode_hbm_synapse_row(raw)
            for j, (op, addr, wt_s, wt_r) in enumerate(hw_entries):
                if op == 4:
                    print(f"    HBM row {r}[{j}]: SPIKE_OUT addr={addr}")

    print(f"\nSPIKE_OUT summary: {found_spike_out} found, {missing_spike_out} missing")


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TEST 1D: Axon name → index mapping consistency
# ═══════════════════════════════════════════════════════════════════════════
def test_1d_axon_index_mapping(network):
    """Verify that axon names used in test_model.py map correctly to HBM indices.

    This is a CRITICAL check. In test_model.py, inputs are created as:
        inputs.append(f"A{i}")
    where 'i' is the position in the flattened input tensor.

    The input_user() function extracts the integer and sets bit 'i' in the
    one-hot axon vector. But the HBM axon pointer table is indexed by
    the axon's coreTypeIdx in the connectome.

    If A0 → coreTypeIdx 0, A1 → coreTypeIdx 1, etc., everything is fine.
    If the mapping is different, ALL INPUTS GO TO THE WRONG AXON POINTERS.
    """
    print("\n" + "=" * 70)
    print("TEST 1D: Axon Name → Index Mapping Consistency")
    print("=" * 70)

    if network is None:
        log.error("Network not available, skipping")
        return

    connectome = network.connectome
    axons = connectome.get_axons()

    print(f"Total axons in connectome: {len(axons)}")
    print(f"Checking first 20 axon mappings...")

    mismatches = 0
    for i, axon in enumerate(axons[:20]):
        user_key = axon.get_user_key()
        core_type_idx = axon.get_coreTypeIdx()

        # test_model.py creates inputs as "A{i}" where i is the tensor position
        # input_user() parses "A{i}" -> int(i) and sets bit i in the one-hot vector
        # The hardware uses this bit position to index into axon pointers
        # So bit position MUST equal coreTypeIdx

        # Extract the expected integer from the user_key
        if isinstance(user_key, str) and user_key.startswith('A'):
            try:
                name_idx = int(user_key[1:])
            except ValueError:
                name_idx = -1
        elif isinstance(user_key, int):
            name_idx = user_key
        else:
            name_idx = -1

        match = (core_type_idx == i)  # coreTypeIdx should match sequential order
        hw_match = (core_type_idx == name_idx) if name_idx >= 0 else None

        status = "OK" if match else "*** MISMATCH ***"
        print(f"  Axon[{i}]: userKey={user_key}, coreTypeIdx={core_type_idx}, "
              f"nameIdx={name_idx}, sequential={status}")

        if not match:
            mismatches += 1

    if mismatches > 0:
        print(f"\n*** CRITICAL: {mismatches} axon index mismatches detected!")
        print("This means input_user() is setting the wrong bits in the one-hot vector.")
        print("The hardware is looking up the WRONG synapse rows for input spikes.")
        print("This would explain identical outputs for all inputs.")
    else:
        print(f"\nAll checked axon indices are consistent.")

    # Also check: what does test_model.py actually send as input names?
    print(f"\nHow test_model.py creates input names:")
    print(f"  inputs.append(f'A{{i}}') where i is tensor position [0..{len(axons)-1}]")
    print(f"  input_user() extracts integer from name and sets one-hot bit")
    print(f"  Hardware uses bit position as index into axon pointer table")

    # Check if CRI_network's step() does any name translation
    print(f"\nChecking if CRI_network.step() translates axon names...")
    if hasattr(network, 'step'):
        import inspect
        try:
            src = inspect.getsource(network.step)
            if 'coreTypeIdx' in src or 'get_coreTypeIdx' in src:
                print("  YES — step() looks up coreTypeIdx for each input name")
                print("  This means the mapping goes: 'A5' → axon object → coreTypeIdx → bit position")
                print("  If coreTypeIdx != 5, the wrong bit is set!")
            else:
                print("  step() does NOT appear to translate names via coreTypeIdx")
                print("  Checking if it passes names directly to input_user()...")
                if 'input_user' in src:
                    print("  It passes to input_user(), which expects integer indices")
        except Exception as e:
            print(f"  Could not inspect step(): {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TEST 1E: Quick all-zeros check
# ═══════════════════════════════════════════════════════════════════════════
def test_1e_bulk_zero_check(dmadump):
    """Quick scan: read 10 evenly-spaced synapse rows and check if any are all zeros."""
    print("\n" + "=" * 70)
    print("TEST 1E: Bulk Zero Check (are weights actually in HBM?)")
    print("=" * 70)

    # Read 10 rows spread across the synapse region
    # For a network with ~100k+ synapse rows, sample broadly
    test_rows = [0, 100, 500, 1000, 5000, 10000, 20000, 50000, 80000, 100000]

    all_zero_count = 0
    has_data_count = 0

    for row_idx in test_rows:
        raw = read_hbm_row(dmadump, SYN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  Row {row_idx}: READ FAILED")
            continue

        hw_entries = decode_hbm_synapse_row(raw)
        nonzero = sum(1 for (op, addr, wt_s, wt_r) in hw_entries if wt_r != 0 or op != 0)

        if nonzero == 0:
            print(f"  Row {row_idx}: ALL ZEROS")
            all_zero_count += 1
        else:
            # Show first non-zero entry
            first_nz = next((op, addr, wt_s) for (op, addr, wt_s, wt_r) in hw_entries if wt_r != 0 or op != 0)
            print(f"  Row {row_idx}: {nonzero}/8 non-zero (first: op={first_nz[0]}, addr={first_nz[1]}, wt={first_nz[2]})")
            has_data_count += 1

    print(f"\nBulk check: {has_data_count} rows with data, {all_zero_count} all-zero rows")
    if has_data_count == 0:
        print("*** CRITICAL: All sampled rows are ZERO! Weights may not have been loaded!")
    elif all_zero_count > has_data_count:
        print("*** WARNING: Most sampled rows are zero. Check if synapse loading completed.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("HiAER-Spike Diagnostic: HBM Weight Readback & Axon Mapping")
    print("=" * 70)

    dmadump = import_dmadump()

    # Verify DMA communication works
    print("\n[Step 0] Verifying DMA communication...")
    exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
    print(f"  DMA read exitCode={exitCode}")
    if exitCode != 0:
        log.error("DMA read failed! Is the FPGA programmed and PCIe link up?")
        sys.exit(1)
    print("  DMA communication OK")

    # Load model config
    fixture_path = Path(__file__).parent / "tests" / "fixtures" / "DVS_model_config_shift=-17.pkl"
    if not fixture_path.exists():
        # Try alternate location
        fixture_path = Path("/home/omowuyi/testing/hs_api/tests/fixtures/DVS_model_config_shift=-17.pkl")
    if not fixture_path.exists():
        log.error(f"Model config not found at {fixture_path}")
        log.error("Please set the correct path to DVS_model_config_shift=-17.pkl")
        # Still run the basic tests
        network = None
    else:
        print(f"\n[Step 1] Loading model config from {fixture_path}...")
        model_config = load_model_config(fixture_path)
        print(f"  Keys: {list(model_config.keys())}")
        print(f"  Axons: {len(model_config['axons'])}")
        print(f"  Connections: {len(model_config['connections'])}")
        print(f"  Outputs: {model_config['outputs']}")

        print(f"\n[Step 2] Rebuilding CRI_network to get pointer tables...")
        network = rebuild_connectome_from_config(model_config)
        if network is not None:
            print(f"  Network rebuilt successfully")
            print(f"  num_inputs (axons): {network.network.num_inputs}")
            print(f"  num_outputs (neurons): {network.network.num_outputs}")
            print(f"  numNeurons: {network.network.numNeurons}")

    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING DIAGNOSTICS")
    print("=" * 70)

    # Test 1E first — quick sanity check
    test_1e_bulk_zero_check(dmadump)

    # Test 1A — detailed synapse readback
    test_1a_synapse_readback(dmadump, network)

    # Test 1B — axon pointers
    test_1b_axon_pointer_readback(dmadump, network)

    # Test 1C — SPIKE_OUT entries
    test_1c_spike_out_entries(dmadump, network)

    # Test 1D — axon name mapping (doesn't need HBM, just software)
    test_1d_axon_index_mapping(network)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)
    print("\nNext steps based on results:")
    print("  - If Test 1E shows ALL ZEROS → weights didn't load, re-run initialization")
    print("  - If Test 1A shows mismatches → byte-order issue in create_synapses()")
    print("  - If Test 1C shows missing SPIKE_OUT → output neurons won't report spikes")
    print("  - If Test 1D shows index mismatches → THIS IS LIKELY THE BUG!")
    print("    The input one-hot bits don't map to the correct axon pointer rows")


if __name__ == "__main__":
    main()