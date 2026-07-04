#!/usr/bin/env python3
"""
check_synapse_data.py — Inspect self.synapses BEFORE any bypass/stream packing.

Run on crisdsc0:
    source /home/omowuyi/testing/hs_api/.venv/bin/activate
    python check_synapse_data.py

This loads the DVS model config, runs compileNetwork + fpga_compiler.__init__
(WITHOUT writing to HBM), and inspects the resulting synapse data to see if
it's correct before any byte packing occurs.
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/home/omowuyi/testing/hs_bridge')
sys.path.insert(0, '/home/omowuyi/testing/hs_api')

def main():
    from hs_bridge.compile_network import compileNetwork
    from hs_bridge.FPGA_Execution.fpga_compiler import fpga_compiler
    from hs_bridge.config import SYN_BASE_ADDR, SYN_OP_BITS, SYN_ADDR_BITS, SYN_WEIGHT_BITS

    fixture_dir = Path('/home/omowuyi/testing/hs_api/tests/fixtures')
    config_path = fixture_dir / 'DVS_model_config_shift=-17.pkl'

    print("Loading model config...")
    with open(config_path, 'rb') as f:
        model_config = pickle.load(f)

    axons = model_config['axons']
    connections = model_config['connections']
    outputs = model_config['outputs']

    print(f"Model: {len(axons)} axons, {len(outputs)} outputs")
    print(f"Outputs: {outputs[:5]}... (first 5)")

    # Run compileNetwork to get HBM data structures
    print("\nRunning compileNetwork...")
    hbm_data, numAxon = compileNetwork(
        loadFile=False,
        connectome=None,  # We need the actual connectome - see below
        outputs=outputs
    )

    # Actually, we need to go through CRI_network to get the connectome.
    # Let's use a different approach: load an existing network object.
    print("\n--- Alternative: loading via CRI_network (skip HBM write) ---")

    # Monkey-patch to skip HBM writes
    import hs_bridge.FPGA_Execution.fpga_compiler as fpc
    original_bypass = fpc.USE_BYPASS_FOR_HBM
    fpc.USE_BYPASS_FOR_HBM = False  # Use stream path so create_script doesn't hang

    # Also patch dma_dump_write to be a no-op
    import hs_bridge.wrapped_dmadump.dmadump as dmadump
    original_write = dmadump.dma_dump_write
    dmadump.dma_dump_write = lambda *args, **kwargs: 0  # no-op

    # Patch write_parameters_simple and write_neuron_type similarly
    import hs_bridge.FPGA_Execution.fpga_controller as fpga_ctrl
    original_wps = fpga_ctrl.write_parameters_simple
    original_wnt = fpga_ctrl.write_neuron_type
    original_flush = fpga_ctrl.flush_c2h
    fpga_ctrl.write_parameters_simple = lambda *args, **kwargs: None
    fpga_ctrl.write_neuron_type = lambda *args, **kwargs: None
    fpga_ctrl.flush_c2h = lambda *args, **kwargs: None

    from hs_api.api import CRI_network

    print("Creating CRI_network (HBM writes disabled)...")
    try:
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )
    except Exception as e:
        print(f"CRI_network creation failed: {e}")
        print("Falling back to manual compileNetwork...")
        # If CRI_network fails, we can still inspect the fpga_compiler data
        # by constructing it manually from compileNetwork output
        import traceback
        traceback.print_exc()
        return

    # Restore patches
    fpc.USE_BYPASS_FOR_HBM = original_bypass
    dmadump.dma_dump_write = original_write
    fpga_ctrl.write_parameters_simple = original_wps
    fpga_ctrl.write_neuron_type = original_wnt
    fpga_ctrl.flush_c2h = original_flush

    # Now inspect the synapse data
    compiler = network.compiledNetwork
    synapses = compiler.synapses

    print(f"\n{'='*70}")
    print(f"SYNAPSE DATA INSPECTION")
    print(f"{'='*70}")
    print(f"Total synapse rows: {len(synapses)}")
    print(f"Entries per row: {len(synapses[0]) if len(synapses) > 0 else 'N/A'}")

    # Sample some rows
    total_rows = len(synapses)
    sample_indices = sorted(set([0, 1, 2, 10, 100, 1000, 10000,
                                  total_rows//4, total_rows//2,
                                  total_rows-10, total_rows-1]))
    sample_indices = [i for i in sample_indices if 0 <= i < total_rows]

    print(f"\n--- Sample rows ---")
    for idx in sample_indices:
        row = synapses[idx]
        # Count non-zero entries
        nonzero = sum(1 for e in row if e != (0, 0, 0))
        print(f"\n  Row {idx}: {nonzero}/8 non-zero")
        for j, entry in enumerate(row):
            if len(entry) == 3:
                flag, addr, wt = entry
                if flag != 0 or addr != 0 or wt != 0:
                    print(f"    [{j}] flag={flag} addr={addr} wt={wt}")
            elif len(entry) == 2:
                flag, val = entry
                print(f"    [{j}] flag={flag} val={val} (spike entry)")
            else:
                print(f"    [{j}] UNEXPECTED FORMAT: {entry}")

    # Statistics
    print(f"\n--- Global statistics ---")
    all_zero_rows = 0
    all_same_addr_rows = 0
    spike_entries = 0
    total_nonzero = 0
    addr_counter = Counter()

    for i, row in enumerate(synapses):
        nonzero = 0
        addrs = []
        for entry in row:
            if len(entry) == 3:
                flag, addr, wt = entry
                if flag != 0 or addr != 0 or wt != 0:
                    nonzero += 1
                    addr_counter[addr] += 1
                    addrs.append(addr)
            elif len(entry) == 2:
                spike_entries += 1
                nonzero += 1

        total_nonzero += nonzero
        if nonzero == 0:
            all_zero_rows += 1
        if len(set(addrs)) == 1 and len(addrs) > 1:
            all_same_addr_rows += 1

    print(f"  All-zero rows: {all_zero_rows}/{total_rows} ({100*all_zero_rows/total_rows:.1f}%)")
    print(f"  Total non-zero entries: {total_nonzero}")
    print(f"  Spike (flag=1) entries: {spike_entries}")
    print(f"  Rows where all non-zero entries have same addr: {all_same_addr_rows}")

    # Check for the specific corruption pattern
    print(f"\n--- Checking for addr=6272 pattern ---")
    rows_with_6272 = 0
    for i, row in enumerate(synapses):
        for entry in row:
            if len(entry) == 3 and entry[1] == 6272:
                rows_with_6272 += 1
                break
    print(f"  Rows containing addr=6272: {rows_with_6272}")

    # Top 10 most common addresses
    print(f"\n--- Top 10 most common synapse addresses ---")
    for addr, count in addr_counter.most_common(10):
        print(f"  addr={addr}: {count} entries")

    # Test _pack_synapse_row on a sample row
    print(f"\n--- Byte packing test ---")
    if len(synapses) > 0:
        test_row = synapses[0]
        packed = compiler._pack_synapse_row(test_row)
        print(f"  Row 0 data: {test_row}")
        print(f"  Packed bytes ({len(packed)} bytes): {list(packed)}")
        print(f"  Packed hex: {packed.hex()}")

        # Decode the packed bytes back to verify round-trip
        # After reversal, bytes are [entry7_b3, ..., entry0_b0]
        # To decode: reverse back and parse as big-endian 4-byte groups
        unpacked = list(reversed(packed))
        print(f"  Un-reversed: {unpacked}")
        for i in range(8):
            b = unpacked[i*4:(i+1)*4]
            val = (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3]
            op = val >> 29
            addr = (val >> 16) & 0x1FFF
            wt = val & 0xFFFF
            if wt >= 32768:
                wt -= 65536
            if op != 0 or addr != 0 or wt != 0:
                print(f"    Entry {i}: op={op} addr={addr} wt={wt}")

    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}")
    print("If addr=6272 appears in the source data -> compileNetwork bug")
    print("If source data looks correct -> byte packing or bypass write bug")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
