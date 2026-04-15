#!/usr/bin/env python3
"""
HiAER-Spike Lightweight Diagnostic v2
======================================
No HBM writes. Safe to run anytime.

Run on crisdsc0:
    cd /home/omowuyi/testing/hs_api
    python diagnose_v2.py
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

# ─── Config constants ──────────────────────────────────────────────────────
N_NG = 16
DATA_PER_ROW = 8
AXN_BASE_ADDR = 0
NRN_BASE_ADDR = 2 ** 14
SYN_BASE_ADDR = 2 ** 15
PTR_LEN_BITS = 9
PTR_ADDR_BITS = 23
_HDR_HBM_DATA   = (0xBB, 0xBB)
_HDR_FIFO_EMPTY = (0xFF, 0xFF)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 0: Inspect the pickle structure
# ═══════════════════════════════════════════════════════════════════════════
def inspect_pickle(model_config):
    print("\n" + "=" * 70)
    print("STEP 0: Pickle Data Structure Inspection")
    print("=" * 70)

    axons = model_config['axons']
    connections = model_config['connections']
    outputs = model_config['outputs']

    print(f"  type(axons)       = {type(axons).__name__}")
    print(f"  type(connections) = {type(connections).__name__}")
    print(f"  type(outputs)     = {type(outputs).__name__}")
    print(f"  len(axons)        = {len(axons)}")
    print(f"  len(connections)  = {len(connections)}")
    print(f"  outputs           = {outputs}")

    # Show first few axon entries
    print(f"\n  First 3 axon keys: {list(axons.keys())[:3]}")
    for i, (k, v) in enumerate(axons.items()):
        if i >= 2:
            break
        print(f"    axons[{repr(k)}] = (type={type(v).__name__}, len={len(v) if hasattr(v,'__len__') else 'N/A'})")
        if hasattr(v, '__iter__'):
            for j, entry in enumerate(v):
                if j >= 3:
                    print(f"      ... ({len(v)} total)")
                    break
                print(f"      [{j}] = {repr(entry)} (type={type(entry).__name__})")

    # Show first few connection entries
    print(f"\n  First 3 connection keys: {list(connections.keys())[:3]}")
    for i, (k, v) in enumerate(connections.items()):
        if i >= 2:
            break
        print(f"    connections[{repr(k)}] = (type={type(v).__name__}, len={len(v) if hasattr(v,'__len__') else 'N/A'})")
        if hasattr(v, '__iter__'):
            for j, entry in enumerate(v):
                if j >= 3:
                    print(f"      ... ({len(v)} total)")
                    break
                print(f"      [{j}] = {repr(entry)} (type={type(entry).__name__})")

    # Check axon key types and naming patterns
    axon_keys = list(axons.keys())
    print(f"\n  Axon key type: {type(axon_keys[0]).__name__}")
    print(f"  First 10 axon keys: {axon_keys[:10]}")
    print(f"  Last  5 axon keys:  {axon_keys[-5:]}")

    # Check connection key types
    conn_keys = list(connections.keys())
    print(f"\n  Connection key type: {type(conn_keys[0]).__name__}")
    print(f"  First 10 connection keys: {conn_keys[:10]}")
    print(f"  Last  5 connection keys:  {conn_keys[-5:]}")

    return axons, connections, outputs


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Trace CRI_network source to understand step() and input mapping
# ═══════════════════════════════════════════════════════════════════════════
def read_cri_source():
    print("\n" + "=" * 70)
    print("STEP 1: Read CRI_network source (hs_api/api.py)")
    print("=" * 70)

    search_paths = [
        "/home/omowuyi/testing/hs_api/hs_api/api.py",
        "/home/omowuyi/testing/hs_api/api.py",
    ]

    for p in search_paths:
        if os.path.exists(p):
            print(f"  Found: {p}")
            with open(p) as f:
                src = f.read()
            lines = src.split('\n')
            print(f"  Total lines: {len(lines)}")
            print("-" * 60)
            for i, line in enumerate(lines, 1):
                print(f"  {i:4d}: {line}")
            print("-" * 60)
            return src

    # Broader search
    print("  Not found in expected locations. Searching...")
    import subprocess
    result = subprocess.run(
        ['find', '/home/omowuyi/testing/hs_api', '-name', '*.py', '-path', '*/hs_api/*'],
        capture_output=True, text=True, timeout=10
    )
    print(f"  Python files in hs_api:\n{result.stdout}")

    # Also try to get source via import
    try:
        import inspect
        from hs_api.api import CRI_network
        src = inspect.getsource(CRI_network)
        print(f"\n  CRI_network class source ({len(src)} chars):")
        print("-" * 60)
        for i, line in enumerate(src.split('\n'), 1):
            print(f"  {i:4d}: {line}")
        print("-" * 60)
        return src
    except Exception as e:
        print(f"  Import inspect failed: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Check what input_user receives (string vs int analysis)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_input_path():
    print("\n" + "=" * 70)
    print("STEP 2: Input Path Analysis")
    print("=" * 70)

    # Test what happens with string modulo
    print("  What does Python do with 'A5' % 256?")
    try:
        result = 'A5' % 256
        print(f"    Result: {result}  ← Unexpected! Strings use % for formatting")
    except TypeError as e:
        print(f"    TypeError: {e}")
        print(f"    → input_user() CANNOT receive string inputs directly")

    # Test with int
    print(f"\n  What does Python do with 5 % 256?")
    print(f"    Result: {5 % 256}")

    # The critical question: what does CRI_network.step() pass to input_user()?
    # Let's look at how test_model.py creates inputs:
    print(f"\n  test_model.py creates inputs as:")
    print(f"    inputs.append(f'A{{i}}')")
    print(f"    → inputs = ['A0', 'A5', 'A100', ...]  (STRINGS)")
    print(f"")
    print(f"  These go to: network.step(inputs)")
    print(f"    → CRI_network.step(inputs)")
    print(f"    → must convert 'A5' → integer before calling input_user()")
    print(f"")
    print(f"  The question is: does step() convert 'A5' → 5 (name index)")
    print(f"    or 'A5' → axon.coreTypeIdx (which might differ from 5)?")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Build minimal connectome and check axon ordering
# ═══════════════════════════════════════════════════════════════════════════
def build_and_check_axon_mapping(axons_dict, connections_dict, outputs_list):
    print("\n" + "=" * 70)
    print("STEP 3: Axon Mapping Check (minimal connectome build)")
    print("=" * 70)

    from connectome_utils.connectome import connectome as Connectome, neuron as Neuron

    Neuron.reset_count()
    conn = Connectome()

    # Add axons (same order as the pickle dict)
    axon_objects = {}
    for axon_name in axons_dict:
        ax = Neuron(axon_name, "axon", axonType='Uaxon')
        conn.addNeuron(ax)
        axon_objects[axon_name] = ax

    # Add neurons (don't need synapses for this test)
    neuron_objects = {}
    for neuron_name in connections_dict:
        is_output = neuron_name in outputs_list
        n = Neuron(neuron_name, "neuron", neuronModel=2, output=is_output)
        conn.addNeuron(n)
        neuron_objects[neuron_name] = n

    # Now check axon ordering
    axons = conn.get_axons()
    print(f"  Total axons in connectome: {len(axons)}")

    axon_key_list = list(axons_dict.keys())

    mismatches = 0
    for i, axon in enumerate(axons):
        user_key = axon.get_user_key()
        cti = axon.get_coreTypeIdx()
        original_dict_pos = axon_key_list.index(user_key) if user_key in axon_key_list else -1

        # Parse integer from key
        if isinstance(user_key, str):
            # Could be 'A0', 'a0', or just '0'
            stripped = user_key.lstrip('AaBb')
            try:
                name_int = int(stripped) if stripped else None
            except ValueError:
                name_int = None
        elif isinstance(user_key, int):
            name_int = user_key
        else:
            name_int = None

        ok = (cti == i)
        if i < 20 or not ok:
            print(f"  axon[{i:5d}]: key={str(user_key):12s} coreTypeIdx={cti:6d} "
                  f"dictPos={original_dict_pos:6d} nameInt={str(name_int):6s} "
                  f"{'OK' if ok else '*** MISMATCH ***'}")
        if not ok:
            mismatches += 1

    # Summary
    if mismatches > 0:
        print(f"\n  *** {mismatches} MISMATCHES: coreTypeIdx != sequential position!")
    else:
        print(f"\n  All {len(axons)} axons: coreTypeIdx == sequential position. GOOD.")

    # Check: does the dict key order match what test_model.py expects?
    print(f"\n  Dict key ordering check:")
    print(f"    First 5 keys from pickle: {axon_key_list[:5]}")
    print(f"    test_model.py sends 'A{{i}}' where i = pixel position in flattened tensor")
    print(f"    If pickle keys are ['A0','A1','A2',...], axon 'A5' → position 5 → coreTypeIdx 5 → OK")
    print(f"    If pickle keys are [0,1,2,...] (ints), test sends 'A0' but connectome has int key 0")

    # Check if keys match the 'A{i}' pattern
    key_type = type(axon_key_list[0]).__name__
    if key_type == 'str':
        # Check if they're 'A0', 'A1', ...
        try:
            indices = [int(k.lstrip('AaBb')) for k in axon_key_list[:20]]
            print(f"\n    Parsed indices from first 20 keys: {indices}")
            if indices == list(range(20)):
                print(f"    Keys are sequential 'A0'..'A19' — matches test_model.py ✓")
            else:
                print(f"    Keys are NOT sequential! This could cause mapping issues.")
        except ValueError:
            print(f"    Cannot parse integer from keys")
    elif key_type == 'int':
        print(f"\n    Keys are integers: {axon_key_list[:10]}")
        print(f"    But test_model.py sends STRINGS like 'A5'")
        print(f"    → CRI_network.step() must strip 'A' and convert to int")
        print(f"    → Then look up the axon by int key to get coreTypeIdx")

    return conn, axon_objects, neuron_objects


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: HBM readback (read-only)
# ═══════════════════════════════════════════════════════════════════════════
def read_hbm_row(dmadump, row_addr, coreID=0):
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
    return None


def decode_synapse_row(raw_data):
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


def hbm_readback(dmadump):
    print("\n" + "=" * 70)
    print("STEP 4: HBM Synapse Readback (read-only)")
    print("=" * 70)

    # Drain stale packets
    print("  Draining stale packets...")
    for _ in range(200):
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        if int(data[63]) == 0xFF and int(data[62]) == 0xFF:
            break
    print("  Done.\n")

    test_rows = [0, 1, 2, 10, 100, 500, 1000, 5000, 10000, 50000]
    all_zero = 0
    has_data = 0

    for row_idx in test_rows:
        raw = read_hbm_row(dmadump, SYN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  SynRow {row_idx:6d}: READ TIMEOUT")
            continue
        entries = decode_synapse_row(raw)
        nonzero = sum(1 for (op, addr, wt, wr) in entries if wr != 0 or op != 0)
        if nonzero == 0:
            print(f"  SynRow {row_idx:6d}: ALL ZEROS")
            all_zero += 1
        else:
            print(f"  SynRow {row_idx:6d}: {nonzero}/8 entries with data:")
            for j, (op, addr, wt, wr) in enumerate(entries):
                if wr != 0 or op != 0:
                    op_s = {0:"LOCAL",4:"SPIKE_OUT"}.get(op, f"OP{op}")
                    print(f"              [{j}] {op_s:10s} addr={addr:5d} weight={wt:6d} (0x{wr:04X})")
            has_data += 1

    print(f"\n  Summary: {has_data} rows with data, {all_zero} all-zero")
    if has_data == 0:
        print("  *** CRITICAL: NO WEIGHTS IN HBM! ***")
        print("  The earlier interrupted script may have corrupted HBM.")
        print("  Re-run test_model.py to reload, then re-run this diagnostic.")

    # Axon pointer rows
    print(f"\n  --- Axon pointer rows ---")
    for row_idx in [0, 1, 2]:
        raw = read_hbm_row(dmadump, AXN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  AxnPtrRow {row_idx}: READ TIMEOUT")
            continue
        data = np.flip(raw)
        binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
        ptr_bits = binData[-256:]
        print(f"  AxnPtrRow {row_idx}:")
        for i in range(8):
            p = ptr_bits[32*i : 32*(i+1)]
            length = int(p[:PTR_LEN_BITS], 2)
            addr = int(p[PTR_LEN_BITS:], 2)
            if addr != 0 or length != 0:
                syn_off = addr - SYN_BASE_ADDR if addr >= SYN_BASE_ADDR else addr
                print(f"    [{i}] addr=0x{addr:06X} (synRow={syn_off:6d}) len={length:3d}")

    # Neuron pointer rows for output neurons (near row 109604/16 ≈ row 6850)
    print(f"\n  --- Neuron pointer rows (near output neurons) ---")
    # Output neurons are at coreTypeIdx 109604..109614
    # Pointer row = coreTypeIdx // PTRS_PER_ROW = 109604 // 8 = 13700
    nrn_ptr_row = 109604 // PTRS_PER_ROW
    for row_idx in [nrn_ptr_row, nrn_ptr_row + 1]:
        raw = read_hbm_row(dmadump, NRN_BASE_ADDR + row_idx)
        if raw is None:
            print(f"  NrnPtrRow {row_idx}: READ TIMEOUT")
            continue
        data = np.flip(raw)
        binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
        ptr_bits = binData[-256:]
        print(f"  NrnPtrRow {row_idx} (HBM addr {NRN_BASE_ADDR + row_idx}):")
        for i in range(8):
            p = ptr_bits[32*i : 32*(i+1)]
            length = int(p[:PTR_LEN_BITS], 2)
            addr = int(p[PTR_LEN_BITS:], 2)
            global_neuron_idx = row_idx * PTRS_PER_ROW + i
            label = f" ← OUTPUT" if 109604 <= global_neuron_idx <= 109614 else ""
            if addr != 0 or length != 0:
                syn_off = addr - SYN_BASE_ADDR if addr >= SYN_BASE_ADDR else addr
                print(f"    [{i}] neuron={global_neuron_idx:6d} addr=0x{addr:06X} "
                      f"(synRow={syn_off:6d}) len={length:3d}{label}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("HiAER-Spike Diagnostic v2 (no HBM writes)")
    print("=" * 70)

    fixture_path = Path("/home/omowuyi/testing/hs_api/tests/fixtures/DVS_model_config_shift=-17.pkl")
    if not fixture_path.exists():
        print(f"ERROR: Pickle not found: {fixture_path}")
        sys.exit(1)

    print(f"\nLoading {fixture_path}...")
    with open(fixture_path, "rb") as f:
        model_config = pickle.load(f)

    # Step 0: Understand the data structure
    axons, connections, outputs = inspect_pickle(model_config)

    # Step 1: Read CRI_network source
    cri_src = read_cri_source()

    # Step 2: Analyze input path
    analyze_input_path()

    # Step 3: Build minimal connectome and check axon mapping
    try:
        conn, axon_objs, neuron_objs = build_and_check_axon_mapping(axons, connections, outputs)
    except Exception as e:
        print(f"\n  Connectome build failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: HBM readback
    try:
        import hs_bridge.wrapped_dmadump.dmadump as dmadump
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        if exitCode == 0:
            print("\n  DMA OK")
            hbm_readback(dmadump)
        else:
            print(f"\n  DMA read failed (exitCode={exitCode})")
    except ImportError:
        print("\n  dmadump not available, skipping HBM readback")

    print("\n" + "=" * 70)
    print("DONE — Review output above for mismatches and mapping issues")
    print("=" * 70)


if __name__ == "__main__":
    main()