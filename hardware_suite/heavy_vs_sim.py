#!/usr/bin/env python3
"""heavy_vs_sim.py CARD_JSON SIM_PREFIX -- the card's spikes per step for the heavy-spike test must equal the
simulation's (round 16b replay of the same commands on the same design)."""
import sys, json
C = json.load(open(sys.argv[1])); steps, cur = [], []
for l in open(sys.argv[2] + "_tx.txt"):
    b = bytes.fromhex(l.split()[1]); v = int.from_bytes(b, "little")
    if (b[63], b[62]) in ((0xEE, 0xEE), (0xAB, 0xCD)):
        for k in range(14):
            s = (v >> (32 + 32 * k)) & 0xFFFFFFFF
            if (s >> 23) & 1: cur.append(s & 0x7FFFF)
        if (b[63], b[62]) == (0xAB, 0xCD): steps.append(sorted(cur)); cur = []
card = [sorted(s) for s in C["spikes"]]
print("card: %d steps, %d spikes | simulation: %d steps, %d spikes" % (len(card), sum(map(len, card)), len(steps), sum(map(len, steps))))
diff = [i for i in range(min(len(card), len(steps))) if card[i] != steps[i]]
if len(card) != len(steps) or diff:
    i = diff[0] if diff else None
    print("FAIL card differs from simulation%s" % ("" if i is None else " first at step %d: card %d spikes, simulation %d; card-only %s, simulation-only %s" % (
        i, len(card[i]), len(steps[i]), sorted(set(card[i]) - set(steps[i]))[:6], sorted(set(steps[i]) - set(card[i]))[:6])))
    sys.exit(1)
print("PASS card identical to simulation at every step"); sys.exit(0)
