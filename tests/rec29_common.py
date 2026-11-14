"""Shared by the round-29 recording tests: read core status with CMD 10 (the same raw access the hardware kit uses), so a
recording carries the status request and its replay can be checked."""
import numpy as np
def read_status(core):
    import hs_bridge.wrapped_dmadump.dmadump as d
    cmd = np.array([0] * 62 + [(core & 0x1F) << 3, 10], dtype=np.uint64)
    d.dma_dump_write(cmd, len(cmd), 1, 0, 0, 0, d.DmaMethodNormal)
    for _ in range(2000):
        code, data = d.dma_dump_read(1, 0, 0, 0, d.DmaMethodNormal, 64)
        b = [int(x) & 0xFF for x in list(data)[:64]]
        if len(b) == 64 and b[63] == 0xFA and b[62] == 0xCE and b[61] == 0xFA and b[60] == 0xCE:
            return b[56] | b[57] << 8 | b[58] << 16 | b[59] << 24, b[52] | b[53] << 8 | b[54] << 16 | b[55] << 24
    return None, None
