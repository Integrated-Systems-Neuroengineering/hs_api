#!/usr/bin/env python3
"""
bypass_path_diagnostic.py — Determine WHERE bypass-write lands and
WHETHER bypass-read can see anything.

This is the next-level diagnostic after synapse_bypass_diagnostic.py.
The previous run showed bypass-read returns 0xFF everywhere and
stream-read sees a different counter pattern. Three hypotheses:

  H1. Bypass-read is fundamentally broken (PCIe BAR errors → 0xFF).
  H2. Bypass writes/reads go to a different HBM region than stream
      (path divergence — different channel, different address translation).
  H3. Bypass works at small offsets (axon pointer region) but fails at
      large offsets (>some boundary), suggesting window-size limit.

This script tests all three.

Run on crisdsc0:
    cd /home/omowuyi/testing/hs_api
    python bypass_path_diagnostic.py 2>&1 | tee bypass_diag.log
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/omowuyi/testing/hs_bridge')
sys.path.insert(0, '/home/omowuyi/testing/hs_api')

import hs_bridge.wrapped_dmadump.dmadump as dmadump
from hs_bridge.config import *

# Header tags from the stream-path C2H protocol
_HDR_HBM_DATA   = (0xBB, 0xBB)
_HDR_FIFO_EMPTY = (0xFF, 0xFF)


def drain_fifo(n=200):
    for _ in range(n):
        try:
            ec, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
            if int(data[63]) == 0xFF and int(data[62]) == 0xFF:
                break
        except Exception:
            break


def stream_read_row(row_addr, coreID=0):
    """Issue a CMD_HBM_RW read command via stream path, return 32 bytes
    of HBM payload, or None on failure.
    """
    commandPrefix = [2, coreID] + [0] * 27
    rowAddress = '0' + np.binary_repr(row_addr, 23)
    addr_bytes = [int(rowAddress[:8], 2),
                  int(rowAddress[8:16], 2),
                  int(rowAddress[16:], 2)]
    cmd = commandPrefix + addr_bytes + [0] * 32
    finalCmd = np.flip(np.array(cmd, dtype=np.uint64))

    ec = dmadump.dma_dump_write(finalCmd, len(finalCmd), 1, 0, 0, 0,
                                 dmadump.DmaMethodNormal)
    if ec != 0:
        return None

    for _ in range(500):
        ec, data = dmadump.dma_dump_read(1, 0, 0, 0, dmadump.DmaMethodNormal, 64)
        b63, b62 = int(data[63]), int(data[62])
        if b63 == _HDR_HBM_DATA[0] and b62 == _HDR_HBM_DATA[1]:
            return bytes(int(data[i]) for i in range(32))
        if b63 == _HDR_FIFO_EMPTY[0] and b62 == _HDR_FIFO_EMPTY[1]:
            continue
    return None


def bypass_read_row(row_addr):
    offset = row_addr * 32
    ec, data = dmadump.dma_bypass_read(32, offset)
    if ec != 0:
        return None, ec
    return bytes(data), 0


def bypass_write_bytes(byte_offset, data_bytes):
    arr = np.array(data_bytes, dtype=np.uint8)
    return dmadump.dma_bypass_write(arr, byte_offset)


def hexdump(b, n=16):
    return ' '.join(f'{b[i]:02x}' for i in range(min(n, len(b))))


def main():
    print("=" * 72)
    print("BYPASS PATH DIAGNOSTIC")
    print("=" * 72)

    # ---- TEST A: Bypass-read at axon pointer region (offset 0..) ----
    # The handoff says pointers were verified correct via bypass at row 0..3.
    # If bypass-read works AT ALL, it should see real data here.
    print("\n--- TEST A: bypass-read of axon-pointer region (rows 0..3) ---")
    print("    These rows were written by create_axon_ptrs_bypass.")
    print("    If bypass-read works at all, we should see structured data,")
    print("    NOT all-0xFF.")
    a_all_ff = True
    for row in range(4):
        b, ec = bypass_read_row(row)
        if b is None:
            print(f"  row {row}: bypass-read FAILED (ec={ec})")
            continue
        is_ff = all(x == 0xFF for x in b)
        a_all_ff = a_all_ff and is_ff
        marker = "  *** ALL 0xFF ***" if is_ff else ""
        print(f"  row {row}: {hexdump(b, 16)} ...{marker}")

    # ---- TEST B: Stream-read of the same axon pointer region ----
    # If stream-read sees the SAME structured data the SW reference has,
    # this confirms axon pointers really are loaded.
    print("\n--- TEST B: stream-read of the same axon-pointer region ---")
    drain_fifo()
    for row in range(4):
        b = stream_read_row(row)
        if b is None:
            print(f"  row {row}: stream-read TIMEOUT")
            continue
        print(f"  row {row}: {hexdump(b, 16)} ...")

    # ---- TEST C: bypass-write/read at SMALL offsets (within first 1 MB) ----
    # Pick offsets in the axon-pointer region beyond what's currently used.
    # Axon ptrs use ~994 rows = ~31 KB. Rows 1024..1031 should be unused
    # but still in the first 1 MB of HBM.
    print("\n--- TEST C: bypass write+read at SMALL offsets (rows 1024..1031) ---")
    small_rows = list(range(1024, 1032))
    small_patterns = {}
    for row in small_rows:
        # 32-byte pattern unique to this row
        pat = bytes([(row + i * 13) & 0xFF for i in range(32)])
        small_patterns[row] = pat
        ec = bypass_write_bytes(row * 32, list(pat))
        if ec != 0:
            print(f"  row {row}: bypass-write FAILED (ec={ec})")

    # Small delay then read back
    time.sleep(0.05)
    c_match = 0
    for row in small_rows:
        b, ec = bypass_read_row(row)
        if b is None:
            print(f"  row {row}: bypass-read FAILED (ec={ec})")
            continue
        match = (b == small_patterns[row])
        if match:
            c_match += 1
        flag = "MATCH" if match else "MISMATCH"
        print(f"  row {row}: {flag}")
        if not match:
            print(f"    expected: {hexdump(small_patterns[row], 16)} ...")
            print(f"    got:      {hexdump(b, 16)} ...")

    # ---- TEST D: bypass-write at small offset, stream-read same offset ----
    # If bypass-write actually lands in HBM channel 0, stream-read MUST see it.
    # If stream-read sees something different, the bypass and stream paths
    # are accessing different physical HBM locations.
    print("\n--- TEST D: bypass-write small row, stream-read SAME row ---")
    print("    If both paths target the same physical HBM, they MUST see")
    print("    the bytes we just wrote.")
    drain_fifo()
    d_match = 0
    for row in small_rows:
        b = stream_read_row(row)
        if b is None:
            print(f"  row {row}: stream-read TIMEOUT")
            continue
        # Stream read returns the row in some byte order. The marker pattern
        # we wrote is bytes([(row + i*13) & 0xFF for i in range(32)]).
        # Bypass writes byte i at hbm_byte_addr i. Stream-read returns 32
        # bytes of the HBM row; the byte order should match what bypass wrote
        # if both paths see the same HBM (modulo any reversal the stream-read
        # protocol does).
        expected = small_patterns[row]
        if b == expected:
            d_match += 1
            print(f"  row {row}: STREAM SEES BYPASS WRITES (direct match)")
        elif bytes(reversed(b)) == expected:
            d_match += 1
            print(f"  row {row}: STREAM SEES BYPASS WRITES (byte-reversed)")
        else:
            print(f"  row {row}: MISMATCH")
            print(f"    expected:        {hexdump(expected, 16)}")
            print(f"    got (forward):   {hexdump(b, 16)}")
            print(f"    got (reversed):  {hexdump(bytes(reversed(b)), 16)}")

    # ---- TEST E: bypass-read at increasing offsets to find a boundary ----
    # If bypass-read works at small offsets but fails at large offsets,
    # we'll find the boundary here.
    print("\n--- TEST E: bypass-read at increasing offsets ---")
    print("    Looking for a boundary beyond which bypass-read returns 0xFF.")
    test_offsets_kb = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                       16384, 32768, 65536, 131072, 262144]  # KB
    for kb in test_offsets_kb:
        byte_off = kb * 1024
        row = byte_off // 32
        b, ec = bypass_read_row(row)
        if b is None:
            print(f"  offset {kb:>7} KB (row {row:>10}): READ FAILED (ec={ec})")
        else:
            is_ff = all(x == 0xFF for x in b)
            tag = "all-0xFF" if is_ff else hexdump(b, 8)
            print(f"  offset {kb:>7} KB (row {row:>10}): {tag}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if a_all_ff:
        print("TEST A: bypass-read of pointer region returned ALL 0xFF.")
        print("        → bypass-read is broken at the BAR/window level,")
        print("          OR axon-pointer bypass writes never landed either")
        print("          (and the previous 'pointers verified' was via stream).")
    else:
        print("TEST A: bypass-read of pointer region returned real data.")
        print("        → bypass-read works at small offsets.")

    print(f"TEST C: {c_match}/{len(small_rows)} small-offset round-trip matched.")
    if c_match == len(small_rows):
        print("        → bypass write+read works in the first 1 MB of HBM.")
    elif c_match == 0:
        print("        → bypass writes do NOT persist anywhere bypass-read can see.")
    else:
        print("        → partial — some writes land, some don't.")

    print(f"TEST D: {d_match}/{len(small_rows)} bypass-writes visible to stream-read.")
    if d_match == len(small_rows):
        print("        → bypass and stream see the SAME physical HBM. Path is OK.")
    elif d_match == 0:
        print("        → bypass and stream are accessing DIFFERENT physical HBM!")
        print("          (different channel, or different address translation)")
    else:
        print("        → partial.")

    print()
    print("INTERPRETATION:")
    print("  TEST A=clean + TEST D=match → bypass works for small offsets;")
    print("    issue is a write-size or chunking boundary at large offsets.")
    print("  TEST A=all-FF + TEST C=fail → bypass-read is broken regardless;")
    print("    likely BAR window mapping or ADXDMA driver issue.")
    print("  TEST D=mismatch → bypass and core access different physical HBM;")
    print("    likely PCIe-to-AXI translation in xdma IP is non-zero, or")
    print("    bypass goes to a different HBM channel than the core.")
    print("=" * 72)


if __name__ == "__main__":
    main()
