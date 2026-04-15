#!/usr/bin/env python3
"""
test_modelly.py — DVS model test with HBM readback diagnostic
Run directly: python tests/test_modelly.py
"""
import sys
import os
import pickle
import numpy as np
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---- HBM readback constants and helpers ----
_HDR_HBM_DATA   = (0xBB, 0xBB)
_HDR_FIFO_EMPTY = (0xFF, 0xFF)
AXN_BASE_ADDR = 0
NRN_BASE_ADDR = 2 ** 14
SYN_BASE_ADDR = 2 ** 15
PTR_LEN_BITS = 9
DATA_PER_ROW = 8


def _read_hbm_row(dmadump, row_addr, coreID=0):
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


def _decode_synapse_entries(raw_data):
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


def _decode_pointer_entries(raw_data):
    data = np.flip(raw_data)
    binData = ''.join([np.binary_repr(int(b), width=8) for b in data])
    ptr_bits = binData[-256:]
    entries = []
    for i in range(8):
        p = ptr_bits[32*i : 32*(i+1)]
        length = int(p[:PTR_LEN_BITS], 2)
        addr = int(p[PTR_LEN_BITS:], 2)
        entries.append((addr, length))
    return entries


def hbm_readback_diagnostic(network_obj):
    """Run HBM readback after network init, before inference."""
    import hs_bridge.wrapped_dmadump.dmadump as dmadump

    print("\n" + "=" * 70)
    print("HBM READBACK DIAGNOSTIC (post-init, pre-inference)")
    print("=" * 70)

    # Drain stale packets
    for _ in range(200):
        exitCode, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        if int(data[63]) == 0xFF and int(data[62]) == 0xFF:
            break

    # Get software reference data
    hbm_data = network_obj.CRI.hbm
    sw_axon_ptrs, sw_neuron_ptrs, sw_synapses = hbm_data[0]
    print("Software: %d axon ptr rows, %d neuron ptr rows, %d synapse rows" % (
        len(sw_axon_ptrs), len(sw_neuron_ptrs), len(sw_synapses)))

    # -- 1. Axon Pointer Readback --
    print("\n--- 1. Axon Pointer Readback (first 4 rows) ---")
    sw_axn_ptrs_flipped = np.fliplr(sw_axon_ptrs)

    for row_idx in range(min(4, len(sw_axon_ptrs))):
        raw = _read_hbm_row(dmadump, AXN_BASE_ADDR + row_idx)
        if raw is None:
            print("  Row %d: READ TIMEOUT" % row_idx)
            continue
        hw_ptrs = _decode_pointer_entries(raw)
        sw_row = sw_axn_ptrs_flipped[row_idx]

        print("  AxnPtrRow %d:" % row_idx)
        all_match = True
        for i in range(8):
            hw_addr, hw_len = hw_ptrs[i]
            sw_start = int(sw_row[i][0])
            sw_end = int(sw_row[i][1])
            sw_len_expected = sw_end - sw_start
            sw_addr_expected = sw_start + SYN_BASE_ADDR

            match = (hw_addr == sw_addr_expected and hw_len == sw_len_expected)
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            print("    [%d] HW: addr=0x%06X len=%3d  |  SW: addr=0x%06X len=%3d  %s" % (
                i, hw_addr, hw_len, sw_addr_expected, sw_len_expected, status))
        if all_match:
            print("    -> All 8 pointers match")

    # -- 2. Synapse Data Readback (sample rows) --
    print("\n--- 2. Synapse Data Readback (sample rows) ---")
    total_syn = len(sw_synapses)
    sample_rows = sorted(set([0, 1, 10, total_syn // 4, total_syn // 2,
                               total_syn - 10, total_syn - 1]))
    sample_rows = [r for r in sample_rows if 0 <= r < total_syn]

    total_match = 0
    total_mismatch = 0

    for syn_row_idx in sample_rows:
        raw = _read_hbm_row(dmadump, SYN_BASE_ADDR + syn_row_idx)
        if raw is None:
            print("  SynRow %d: READ TIMEOUT" % syn_row_idx)
            continue
        hw_entries = _decode_synapse_entries(raw)
        sw_row = sw_synapses[syn_row_idx]

        hw_nonzero = sum(1 for (op, addr, wt, wr) in hw_entries if wr != 0 or op != 0)
        sw_nonzero = sum(1 for e in sw_row if e != (0, 0, 0))

        print("\n  SynRow %d (HBM addr %d):" % (syn_row_idx, SYN_BASE_ADDR + syn_row_idx))
        print("    SW: %s" % str(sw_row))
        print("    HW: %d/8 non-zero, SW: %d/8 non-zero" % (hw_nonzero, sw_nonzero))
        for j, (op, addr, wt, wr) in enumerate(hw_entries):
            op_s = {0: "LOCAL", 4: "SPIKE_OUT"}.get(op, "OP%d" % op)
            if wr != 0 or op != 0:
                print("      HW[%d]: %s addr=%5d wt=%7d (0x%04X)" % (j, op_s, addr, wt, wr))

        # Check corruption pattern: all same addr, arithmetic weights
        addrs = [hw_entries[i][1] for i in range(8)]
        if len(set(addrs)) == 1 and addrs[0] != 0:
            wts = [hw_entries[i][2] for i in range(8)]
            diffs = [wts[k+1] - wts[k] for k in range(7)]
            if len(set(diffs)) == 1:
                print("    *** CORRUPTION: all addr=%d, weights are arithmetic (diff=%d)" % (
                    addrs[0], diffs[0]))
                total_mismatch += 1
                continue

        if abs(hw_nonzero - sw_nonzero) <= 1:
            total_match += 1
        else:
            total_mismatch += 1
            print("    *** Non-zero count differs!")

    print("\n  Synapse summary: %d plausible, %d suspicious" % (total_match, total_mismatch))

    # -- 3. Row Uniqueness Check --
    print("\n--- 3. Row Uniqueness Check ---")
    check_rows = [r for r in [0, 100, 1000, 10000, 50000] if r < total_syn]
    row_data = {}
    for r in check_rows:
        raw = _read_hbm_row(dmadump, SYN_BASE_ADDR + r)
        if raw is not None:
            fingerprint = tuple(int(raw[i]) for i in range(32))
            row_data[r] = fingerprint

    unique_fps = set(row_data.values())
    print("  Read %d rows, found %d unique patterns" % (len(row_data), len(unique_fps)))
    if len(unique_fps) == 1 and len(row_data) > 1:
        print("  *** CRITICAL: ALL ROWS IDENTICAL - HBM not properly loaded!")
    elif len(unique_fps) < len(row_data) // 2:
        print("  *** WARNING: Many duplicate rows - possible partial corruption")
    else:
        print("  Rows appear diverse - weights look properly loaded")

    # -- 4. Output Neuron SPIKE_OUT Check --
    print("\n--- 4. Output Neuron Pointer & SPIKE_OUT Check ---")
    outputs_cti = network_obj.CRI.outputs
    print("  Output coreTypeIdx: %s" % str(outputs_cti))

    sw_nrn_ptrs_flipped = np.fliplr(sw_neuron_ptrs)

    for out_cti in outputs_cti[:3]:
        ptr_row = out_cti // DATA_PER_ROW
        ptr_col_orig = out_cti % DATA_PER_ROW
        ptr_col_flipped = (DATA_PER_ROW - 1) - ptr_col_orig

        print("\n  Output neuron coreTypeIdx=%d: ptr row=%d col=%d (flipped=%d)" % (
            out_cti, ptr_row, ptr_col_orig, ptr_col_flipped))

        raw = _read_hbm_row(dmadump, NRN_BASE_ADDR + ptr_row)
        if raw is None:
            print("    HBM pointer READ TIMEOUT")
            continue

        hw_ptrs = _decode_pointer_entries(raw)
        hw_addr, hw_len = hw_ptrs[ptr_col_flipped]

        sw_entry = sw_neuron_ptrs[ptr_row][ptr_col_orig]
        sw_start = int(sw_entry[0])
        sw_end = int(sw_entry[1])

        print("    SW ptr: synapse rows [%d..%d]" % (sw_start, sw_end))
        print("    HW ptr: addr=0x%06X (synRow=%d), len=%d" % (
            hw_addr, hw_addr - SYN_BASE_ADDR, hw_len))

        print("    Scanning for SPIKE_OUT (opcode=4)...")
        found = False
        for sr in range(sw_start, min(sw_start + 4, sw_end + 1)):
            raw_syn = _read_hbm_row(dmadump, SYN_BASE_ADDR + sr)
            if raw_syn is None:
                continue
            hw_syn = _decode_synapse_entries(raw_syn)
            for j, (op, addr, wt, wr) in enumerate(hw_syn):
                if op == 4:
                    print("      SPIKE_OUT at synRow=%d[%d]: neuronAddr=%d" % (sr, j, addr))
                    found = True
        if not found:
            print("      *** NO SPIKE_OUT found!")

    # -- 5. First Axon Synapse Comparison --
    print("\n--- 5. First Axon Synapse Comparison ---")
    raw = _read_hbm_row(dmadump, AXN_BASE_ADDR + 0)
    if raw is not None:
        hw_ptrs = _decode_pointer_entries(raw)
        hw_addr, hw_len = hw_ptrs[7]  # fliplr: original col 0 -> HBM col 7
        syn_start = hw_addr - SYN_BASE_ADDR
        print("  Axon 0 pointer: synRow start=%d, len=%d" % (syn_start, hw_len))
        print("  SW axon_ptrs[0][0] = %s" % str(sw_axon_ptrs.flatten()[0]))

        for offset in range(min(2, hw_len + 1)):
            raw_syn = _read_hbm_row(dmadump, SYN_BASE_ADDR + syn_start + offset)
            if raw_syn is None:
                continue
            hw_syn = _decode_synapse_entries(raw_syn)
            sw_syn = sw_synapses[syn_start + offset] if (syn_start + offset) < len(sw_synapses) else None
            print("\n  SynRow %d:" % (syn_start + offset))
            print("    SW: %s" % str(sw_syn))
            print("    HW:")
            for j, (op, addr, wt, wr) in enumerate(hw_syn):
                if wr != 0 or op != 0:
                    op_s = {0: "LOCAL", 4: "SPIKE_OUT"}.get(op, "OP%d" % op)
                    print("      [%d] %s addr=%d wt=%d" % (j, op_s, addr, wt))

    print("\n" + "=" * 70)
    print("HBM READBACK DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("If rows are all identical -> HBM corrupted, weights not loaded")
    print("If rows diverse but wrong -> encoding/byte-order bug")
    print("If weights match SW -> problem is elsewhere")
    print("=" * 70 + "\n")


# ---- Main test logic (no pytest) ----
def main():
    import torch
    from hs_api.api import CRI_network
    import hs_bridge

    fixture_dir = Path(__file__).parent / "fixtures"
    config_path = fixture_dir / "DVS_model_config_shift=-17.pkl"
    batch_path = fixture_dir / "DVS_test_batch.pkl"

    print("Loading model config from %s ..." % config_path)
    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    print("Loading test batch from %s ..." % batch_path)
    with open(batch_path, "rb") as f:
        test_batch = pickle.load(f)

    axons = model_config['axons']
    connections = model_config['connections']
    outputs = model_config['outputs']

    print("Creating CRI_network (this loads weights into HBM ~21 min)...")
    network = CRI_network(
        axons=axons,
        connections=connections,
        outputs=outputs,
        target="CRI"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- HBM READBACK DIAGNOSTIC ----
    print("\n>>> Running HBM readback diagnostic...")
    try:
        hbm_readback_diagnostic(network)
    except Exception as e:
        print(">>> HBM diagnostic failed: %s" % str(e))
        import traceback
        traceback.print_exc()
    print(">>> HBM diagnostic complete. Proceeding to inference...\n")

    # Build output key mapping
    output_key_to_idx = {}
    for idx, out_key in enumerate(outputs):
        try:
            n = network.connectome.get_neuron_by_key(out_key)
            user_key = n.get_user_key()
            output_key_to_idx[user_key] = idx
            output_key_to_idx[out_key] = idx
        except Exception:
            output_key_to_idx[out_key] = idx
    print("Output key mapping: %s" % str(output_key_to_idx))

    TRAILING_TIMESTEPS = 20

    correct = 0
    total = len(test_batch['images'])
    for img, label in zip(test_batch['images'], test_batch['labels']):
        hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, 0)

        img = img.to(device)
        spike_counts = torch.zeros(len(outputs))
        for t in range(img.shape[0]):
            frame = img[t, :, :, :]
            inp = frame.unsqueeze(0).flatten(start_dim=1).to(torch.int16)

            inputs = []
            for i, elem in enumerate(inp[0, :]):
                if elem.item() > 0:
                    inputs.append("A%d" % i)

            results = network.read_membrane(outputs)
            print("Membrane potentials: %s" % str(results))

            hardwareSpikes, _, _ = network.step(inputs)
            print("Output spikes: %s" % str(hardwareSpikes))

            for spike in hardwareSpikes:
                if spike in output_key_to_idx:
                    spike_counts[output_key_to_idx[spike]] += 1
                else:
                    print("Non-output spike (hidden layer): %s" % str(spike))

        for i in range(TRAILING_TIMESTEPS):
            inputs = []
            hardwareSpikes, _, _ = network.step(inputs)
            if hardwareSpikes:
                print("Output spikes (trailing t=%d): %s" % (i, str(hardwareSpikes)))
            for spike in hardwareSpikes:
                if spike in output_key_to_idx:
                    spike_counts[output_key_to_idx[spike]] += 1
                else:
                    print("Non-output spike (trailing, hidden): %s" % str(spike))

        spike_counts = spike_counts / img.size(0)
        print("Spike counts: %s" % str(spike_counts))

        predicted = torch.argmax(spike_counts).item()
        print("Predicted: %d, Label: %s" % (predicted, str(label)))

        if predicted == label:
            correct += 1

        running_accuracy = 100 * correct / total
        print("Running accuracy: %.2f%%" % running_accuracy)

    accuracy = 100 * correct / total
    print("\nFinal accuracy: %.2f%%" % accuracy)
    if accuracy >= 55:
        print("PASS: accuracy >= 55%%")
    else:
        print("FAIL: expected >= 55%%, got %.2f%%" % accuracy)


if __name__ == "__main__":
    main()