#!/usr/bin/env python3
"""hw2tx.py PROTOCOL_JSON PREFIX -- the card's saved spikes (per step, hardware indices) and read-back rows, written in
the replay stream format (<PREFIX>_tx.txt / _log.txt), so plast_check.py judges the card exactly as it judged the
simulation."""
import sys, json
P = json.load(open(sys.argv[1])); pre = sys.argv[2]
if "hw_spikes" not in P or "hw_rows" not in P: sys.exit("protocol has no card results (was it run in recording mode?)")
def pkt(hdr, val):
    b = bytearray(val.to_bytes(64, "little")); b[63], b[62] = hdr; return bytes(b).hex()
L = []
for st in P["hw_spikes"]:
    words = [(1 << 23) | a for a in st]; ch = [words[i:i + 14] for i in range(0, len(words), 14)] or [[]]
    for j, c in enumerate(ch):
        L.append(pkt((0xAB, 0xCD) if j == len(ch) - 1 else (0xEE, 0xEE), sum(w << (32 + 32 * k) for k, w in enumerate(c))))
L += [pkt((0xBB, 0xBB), int(h, 16)) for h in P["hw_rows"]]
open(pre + "_tx.txt", "w").write("".join("%d.0 %s\n" % (i, l) for i, l in enumerate(L)))
open(pre + "_log.txt", "w").write("card results\n")
print("%d steps, %d spikes, %d rows written to %s_tx.txt" % (len(P["hw_spikes"]), sum(map(len, P["hw_spikes"])), len(P["hw_rows"]), pre))
