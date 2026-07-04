#!/usr/bin/env python3
"""
synapse_bypass_diagnostic.py

DEFINITIVE diagnostic for the synapse bypass write corruption.

This script does FOUR things, in order, with no model loading:

1. Drains FIFO. Reads raw HBM at synapse rows 0, 1, 2, 3, 10, 1000, 100000,
   1750798, 7003000 via the bypass BAR. Logs the raw bytes.
2. Writes a UNIQUE, recognizable pattern to those exact rows (different
   from the existing addr=6272 pattern, different from verify_bypass_write
   patterns).
3. Reads them back via the bypass BAR -> must match.
4. Reads them back via the stream-path HBM read command -> must match.

If step 3 matches but step 4 doesn't, the bypass and the core are accessing
DIFFERENT physical HBM. If step 3 doesn't match, the bypass write isn't
landing. If both match, the bypass write/read mechanism is fully working
and the corruption seen in test_modelly is from self.synapses being wrong.

After this runs, look for:
  - Step 1 bytes: should be the addr=6272 corruption pattern. If they ARE
    the verify_bypass_write test pattern (0xF0, 0x8A, 0x80, 0x18, etc.)
    then we have direct evidence the corruption is leftover test patterns.
  - Step 3 result: BYPASS_OK count (should be 9/9).
  - Step 4 result: STREAM_OK count (should be 9/9).

Run on crisdsc0:
    cd /home/omowuyi/testing/hs_api
    source .venv/bin/activate
    python synapse_bypass_diagnostic.py 2>&1 | tee diag.log
"""

import sys
import os
import time

sys.path.insert(0, '/home/omowuyi/testing/hs_bridge')
sys.path.insert(0, '/home/omowuyi/testing/hs_api')

import numpy as np
import hs_bridge.wrapped_dmadump.dmadump as dmadump
from hs_bridge.config import SYN_BASE_ADDR

HDR_HBM_DATA = (0xBB, 0xBB)
HDR_FIFO_EMPTY = (0xFF, 0xFF)


def drain_fifo(n=500):
    cleared = 0
    for _ in range(n):
        try:
            ec, data = dmadump.dma_dump_read(1, 0, 0, 0,
                                             dmadump.DmaMethodNormal, 64)
            if int(data[63]) == 0xFF and int(data[62]) == 0xFF:
                cleared += 1
                if cleared >= 3:
                    return
            else:
                cleared = 0
        except Exception:
            return


def stream_read_row(row_addr, core_id=0):
    cmd = [2, core_id] + [0] * 27
    row_bin = '0' + np.binary_repr(row_addr, 23)
    cmd += [int(row_bin[0:8], 2),
            int(row_bin[8:16], 2),
            int(row_bin[16:24], 2)]
    cmd += [0] * 32
    final_cmd = np.flip(np.array(cmd, dtype=np.uint64))

    ec = dmadump.dma_dump_write(final_cmd, len(final_cmd), 1, 0, 0, 0,
                                dmadump.DmaMethodNormal)
    if ec != 0:
        return None, f"write ec={ec}"

    for _ in range(2000):
        ec, data = dmadump.dma_dump_read(1, 0, 0, 0,
                                         dmadump.DmaMethodNormal, 64)
        b63, b62 = int(data[63]), int(data[62])
        if (b63, b62) == HDR_HBM_DATA:
            return [int(data[i]) for i in range(32)], None
        if (b63, b62) == HDR_FIFO_EMPTY:
            continue
    return None, "timeout"


def bypass_read_row(row_addr):
    offset = row_addr * 32
    try:
        ec, data = dmadump.dma_bypass_read(32, offset)
    except Exception as e:
        return None, f"exc: {e}"
    if ec != 0:
        return None, f"ec={ec}"
    return [int(data[i]) for i in range(32)], None


def bypass_write_row(row_addr, byte_list):
    offset = row_addr * 32
    arr = np.array(byte_list, dtype=np.uint8)
    ec = dmadump.dma_bypass_write(arr, offset)
    return ec


def make_marker(row_addr):
    """Generate a 32-byte marker that is BLATANTLY recognizable and unique
    per row. Specifically chosen NOT to look like synapse data, NOT to look
    like pointer data, and NOT to look like the existing corruption pattern.

    Each byte i is (row_addr + i) ^ 0x5A, mod 256. Easy to verify.
    """
    return [((row_addr + i) ^ 0x5A) & 0xFF for i in range(32)]


def fmt_bytes(bs):
    if bs is None:
        return "None"
    return ' '.join(f'{b:02x}' for b in bs[:16]) + ' ...'


def main():
    test_rows = [
        SYN_BASE_ADDR + 0,
        SYN_BASE_ADDR + 1,
        SYN_BASE_ADDR + 2,
        SYN_BASE_ADDR + 3,
        SYN_BASE_ADDR + 10,
        SYN_BASE_ADDR + 1000,
        SYN_BASE_ADDR + 100000,
        SYN_BASE_ADDR + 1750798,
        SYN_BASE_ADDR + 7003000,
    ]

    print("=" * 70)
    print("SYNAPSE BYPASS DIAGNOSTIC")
    print("Test rows (HBM row addresses):", test_rows)
    print("=" * 70)

    # --- Step 1: read existing HBM contents via bypass ---
    print("\n--- STEP 1: Existing HBM at synapse rows (via bypass) ---")
    drain_fifo()
    pre_bytes = {}
    for r in test_rows:
        b, err = bypass_read_row(r)
        pre_bytes[r] = b
        if b is None:
            print(f"  Row {r}: BYPASS READ FAILED ({err})")
        else:
            print(f"  Row {r}: {fmt_bytes(b)}")
            # Check: does this look like the addr=6272 corruption?
            # That pattern has bytes [2]=0x80, [3]=0x18 in every 4-byte group.
            corruption = all(b[4*i+2] == 0x80 and b[4*i+3] == 0x18
                             for i in range(8))
            if corruption:
                print(f"           -> matches addr=6272 corruption "
                      f"(every 4-byte group ends in 0x80 0x18)")

    # --- Step 2: write unique markers via bypass ---
    print("\n--- STEP 2: Write unique markers via bypass ---")
    for r in test_rows:
        marker = make_marker(r)
        ec = bypass_write_row(r, marker)
        if ec != 0:
            print(f"  Row {r}: WRITE FAILED ec={ec}")
        else:
            print(f"  Row {r}: marker[0:8]={marker[:8]}, ec=0")

    # Tiny pause to ensure HBM commits
    time.sleep(0.1)

    # --- Step 3: read back via bypass ---
    print("\n--- STEP 3: Read back via bypass ---")
    bypass_ok = 0
    for r in test_rows:
        expected = make_marker(r)
        b, err = bypass_read_row(r)
        if b is None:
            print(f"  Row {r}: READ FAILED ({err})")
            continue
        if b == expected:
            print(f"  Row {r}: OK")
            bypass_ok += 1
        else:
            print(f"  Row {r}: MISMATCH")
            print(f"      expected: {fmt_bytes(expected)}")
            print(f"      got:      {fmt_bytes(b)}")

    # --- Step 4: read back via stream path ---
    print("\n--- STEP 4: Read back via stream path (CMD_HBM_RW read) ---")
    drain_fifo()
    stream_ok = 0
    for r in test_rows:
        expected = make_marker(r)
        b, err = stream_read_row(r)
        if b is None:
            print(f"  Row {r}: READ FAILED ({err})")
            continue
        # Stream returns 32 raw HBM bytes in data[0..31]
        if b == expected:
            print(f"  Row {r}: OK")
            stream_ok += 1
        else:
            print(f"  Row {r}: MISMATCH")
            print(f"      expected: {fmt_bytes(expected)}")
            print(f"      got:      {fmt_bytes(b)}")
            # Maybe it's flipped within the row?
            if list(reversed(b)) == expected:
                print(f"      -> matches expected if reversed!")

    # --- Step 5: bulk write at non-test offset, then verify chunks ---
    print("\n--- STEP 5: Bulk write 32 MB (crosses CHUNK boundary) ---")
    bulk_start_row = SYN_BASE_ADDR + 200000
    bulk_size = 32 * 1024 * 1024  # 32 MB - guarantees crossing the 16MB CHUNK
    bulk_offset = bulk_start_row * 32
    bulk = np.zeros(bulk_size, dtype=np.uint8)
    # Each row's first 4 bytes = row_offset_from_start as 0xCAFE | row
    for i in range(0, bulk_size, 32):
        rn = i // 32
        bulk[i+0] = rn & 0xFF
        bulk[i+1] = (rn >> 8) & 0xFF
        bulk[i+2] = 0xCA
        bulk[i+3] = 0xFE
    # Use the chunked write path (16MB chunks) just like create_synapses_bypass
    CHUNK = 16 * 1024 * 1024
    pos = 0
    print(f"  Writing 32 MB at offset 0x{bulk_offset:X} in 16 MB chunks...")
    while pos < bulk_size:
        end = min(pos + CHUNK, bulk_size)
        ec = dmadump.dma_bypass_write(bulk[pos:end], bulk_offset + pos)
        if ec != 0:
            print(f"  Chunk write FAILED at pos={pos}: ec={ec}")
            break
        pos = end
    print(f"  Bulk write complete.")

    # Spot-check: rows 0, 100, 250000 (in chunk 0), 524288 (start chunk 1),
    # 1000000, 1048575 (end of bulk = row 1048575 of bulk = total row
    # bulk_start_row + 1048575)
    spot_offsets_in_bulk = [0, 100, 250000, 524288, 524289, 1000000, 1048575]
    print("  Spot-check rows within bulk write:")
    spot_ok = 0
    for off in spot_offsets_in_bulk:
        row = bulk_start_row + off
        b, err = bypass_read_row(row)
        if b is None:
            print(f"    bulk_offset={off}: READ FAILED ({err})")
            continue
        exp_b0 = off & 0xFF
        exp_b1 = (off >> 8) & 0xFF
        ok = (b[0] == exp_b0 and b[1] == exp_b1
              and b[2] == 0xCA and b[3] == 0xFE)
        status = "OK" if ok else "MISMATCH"
        if ok:
            spot_ok += 1
        print(f"    bulk_offset={off:>7d} (row={row}): "
              f"b0={b[0]:02x} b1={b[1]:02x} b2={b[2]:02x} b3={b[3]:02x} "
              f"(expected {exp_b0:02x} {exp_b1:02x} ca fe)  {status}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Step 3 (bypass readback):   {bypass_ok}/{len(test_rows)} rows OK")
    print(f"Step 4 (stream readback):   {stream_ok}/{len(test_rows)} rows OK")
    print(f"Step 5 (bulk spot checks):  {spot_ok}/{len(spot_offsets_in_bulk)} rows OK")
    print()
    if bypass_ok == len(test_rows) and stream_ok == len(test_rows):
        print(">>> CONCLUSION: Bypass write/read works end-to-end. "
              "Both bypass and core see the SAME HBM. The corruption seen "
              "in test_modelly must therefore be in self.synapses BEFORE "
              "packing. Run check_synapse_data.py next.")
    elif bypass_ok == len(test_rows) and stream_ok < len(test_rows):
        print(">>> CONCLUSION: Bypass write succeeds and bypass-read sees "
              "the data, but the CORE (via stream path) sees something "
              "else. The bypass and the core are NOT pointing at the same "
              "HBM. Address-decode bug in hbm_xbar or HBM channel mapping.")
    elif bypass_ok < len(test_rows):
        print(">>> CONCLUSION: Bypass writes are NOT persisting. "
              "Check ADXDMA_WriteWindow flush / preferredSize / window "
              "address mapping.")
    if spot_ok < len(spot_offsets_in_bulk):
        print()
        print(">>> ALSO: chunked bulk write has problems (rows on the far "
              "side of the 16 MB chunk boundary failed). The "
              "create_synapses_bypass chunking is the suspect.")
    print("=" * 70)


if __name__ == "__main__":
    main()
