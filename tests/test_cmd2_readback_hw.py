"""
test_cmd2_readback_hw.py -- CMD 2 HBM write followed by read-back of the same rows.

Writes five rows (low and high addresses) with distinct 256-bit patterns through CMD 2, then
reads each row back through CMD 2 and compares the returned row (bytes 0..31 of the 0xBBBB
response) with what was written.

Packet layout (command_interpreter.v / hbm_processor.v):
  byte 63 = opcode 2, byte 62 = core << 3
  bytes 34..32 = {rw, row[22:0]}  (rw = 1 write, 0 read);  bytes 31..0 = the 256-bit row
  response: bytes 63,62 = BB BB, bytes 31..0 = the row read
Run: HIAER_CORE_ID=1 python -m pytest tests/test_cmd2_readback_hw.py -v -s
"""
import os
import numpy as np
import hs_bridge.wrapped_dmadump.dmadump as d

CORE = int(os.environ.get("HIAER_CORE_ID", 0))
ROWS = [50000, 123457, 0x400000, 0x400001, 0x400002]


def _cmd(op):
    b = np.zeros(64, dtype=np.uint64); b[63] = op; b[62] = (CORE & 0x1F) << 3
    return b


def _hbm(row, write, data=None):
    b = _cmd(2); a = ((1 << 23) if write else 0) | (row & 0x7FFFFF)
    b[34], b[33], b[32] = (a >> 16) & 0xFF, (a >> 8) & 0xFF, a & 0xFF
    if data is not None:
        b[0:32] = data
    return b


def _pattern(k):
    return np.array([(k * 37 + i * 11 + 5) & 0xFF for i in range(32)], dtype=np.uint64)


def _send(pk):
    pk = pk.astype(np.uint64)
    return d.dma_dump_write(pk, len(pk), 1, 0, 0, 0, d.DmaMethodNormal)


def _read():
    ec, data = d.dma_dump_read(1, 0, 0, 0, d.DmaMethodNormal, 64)
    return [int(x) & 0xFF for x in list(data)[:64]]


def _poll_bb(maxreads=4000):
    for _ in range(maxreads):
        b = _read()
        if b[63] == 0xBB and b[62] == 0xBB:
            return b
    return None


def test_cmd2_readback():
    for _ in range(64):                     # drain anything already queued
        _read()
    for k, row in enumerate(ROWS):
        _send(_hbm(row, True, _pattern(k)))
    bad = []
    for k, row in enumerate(ROWS):
        _send(_hbm(row, False))
        b = _poll_bb()
        exp = [int(x) for x in _pattern(k)]
        if b is None:
            print("CMD2 row %d: NO RESPONSE" % row); bad.append(row); continue
        got = b[0:32]
        ok = got == exp
        print("CMD2 row %d: %s%s" % (row, "MATCH" if ok else "DIFF",
              "" if ok else "  got=%s exp=%s" % (" ".join("%02X" % x for x in got), " ".join("%02X" % x for x in exp))))
        if not ok:
            bad.append(row)
    assert not bad, "CMD 2 read-back differs on rows %s" % bad
