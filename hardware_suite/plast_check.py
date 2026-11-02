#!/usr/bin/env python3
"""plast_check.py PREFIX PROTOCOL_JSON [REPLAY_STIM] -- plasticity-on check.  Post and neuron-pre spike times from the replay's spike
reports, pre-axon times from the recorded schedule; the expected weight of every plastic synapse from stdp_ref.py (v3) at
the hardware's absolute steps (clear at t = 0, step s runs at t = s + 1, a reward after step s applies at t = s + 1);
compared bit-exactly with the weights read back over CMD 2 (entries identified by (src, dest row))."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stdp_ref as R
from stdp_ref_v2 import NeuronStamps, AxonStamps
import plasticity as PL
prefix, P = sys.argv[1], json.load(open(sys.argv[2]))
steps, cur, bb = [], [], []
for l in open(prefix + "_tx.txt"):
    b = bytes.fromhex(l.split()[1]); v = int.from_bytes(b, "little")
    if (b[63], b[62]) in ((0xEE, 0xEE), (0xAB, 0xCD)):
        for k in range(14):
            s = (v >> (32 + 32 * k)) & 0xFFFFFFFF
            if (s >> 23) & 1: cur.append(s & 0x7FFFF)
        if (b[63], b[62]) == (0xAB, 0xCD): steps.append(set(cur)); cur = []
    elif (b[63], b[62]) == (0xBB, 0xBB):
        bb.append(v & ((1 << 256) - 1))
fail = []
if len(steps) != P["steps"]: fail.append("%d steps completed, expected %d" % (len(steps), P["steps"]))
R_ = len(P["rows_read"])
if len(sys.argv) > 3 and sys.argv[3] == "--reads-last":
    # DIAGNOSTIC ONLY (design without fix 18): Phase-4 responses also reach the host as BB copies, so the read
    # replies are taken as the last R BB packets -- the reads run after the last Phase 4.  Not a pass criterion.
    print("   DIAGNOSTIC mapping: last %d of %d BB replies taken as the reads" % (R_, len(bb)))
    if len(bb) < R_: fail.append("only %d BB replies" % len(bb))
elif len(sys.argv) > 3:
    # CMD 2 writes are not answered (measured: no BB before the first step); every CMD 2 read is answered by exactly
    # one BB reply.  So the replay must hold exactly as many BB replies as the recording has reads.
    nw = nr = 0
    for l in open(sys.argv[3]):
        h = l.strip()
        if len(h) == 128 and int(h[126:128], 16) == 2:
            if (int.from_bytes(bytes.fromhex(h), "little") >> 279) & 1: nw += 1
            else: nr += 1
    print("   recording: %d CMD 2 writes, %d CMD 2 reads; replay: %d BB replies" % (nw, nr, len(bb)))
    if nr != R_: fail.append("recording has %d CMD 2 reads, protocol read %d rows" % (nr, R_))
    if len(bb) != nr: fail.append("%d BB replies, expected %d (one per CMD 2 read; extra BB packets reached the host)" % (len(bb), nr))
elif len(bb) != R_:
    fail.append("%d BB replies, expected %d (pass the recording to account for CMD 2 writes)" % (len(bb), R_))
if fail: print("FAIL " + "; ".join(fail)); sys.exit(1)
bb = bb[-R_:]                                      # the reads, in the order the protocol issued them
tabs = {int(t): (PL.trace_table(tau) if k == "trace" else PL.decay_table(tau, sh)) for t, (k, tau, sh) in P["tables"].items()}
luts = [tabs.get(i, [0] * 4096) for i in range(8)]
rules = {}
for slot, f in P["rules"].items():
    rl = R.Rule(f["A2p"], f["A3p"], f["A2m"], f["w_min"], f["w_max"], soft=f["soft"], A3m=f["A3m"], A0=f["A0"], s2p=f["s2p"],
                s3p=f["s3p"], s2m=f["s2m"], s3m=f["s3m"], s0=f["s0"], elig=f["elig"], eta=f["eta"], elig_reset=f["elig_reset"],
                e_shift=f["e_shift"], lut_p=f["lut_p"], lut_m=f["lut_m"], lut_y=f["lut_y"], lut_x=f["lut_x"], lut_e=f["lut_e"])
    assert rl.inv_range == f["inv_range"], "rule %s: reciprocal differs between host and reference" % slot
    rules[int(slot)] = rl
rewards = {int(k): v for k, v in P["rewards"].items()}
hw = P["hw"]; g = P["g"]
known = set(hw.values()); seen = set().union(*steps) if steps else set()
outside = sorted(seen - known)
if outside:
    print("FAIL spike addresses outside the network: %s (all reported addresses: %s)" % (outside[:10], sorted(seen)[:10])); sys.exit(1)
print("   spike addresses: %d distinct, all inside the network; %d spikes in %d steps" % (len(seen), sum(len(s) for s in steps), len(steps)))
# every post has its own driver axon (weight 1000, threshold 300): a post that received drive and never spiked means
# its spikes are not being reported under its own address (round 14: every 64-bit spike reported as neuron 0)
silent = [k for k in hw if k.startswith("N1.") and hw[k] not in seen and any(("D" + k[3:]) in st for st in P["schedule"])]
if silent:
    print("FAIL %d driven post neurons never reported a spike: %s" % (len(silent), silent[:8])); sys.exit(1)
def expected(pre, post, rule, w0):
    rl = rules[rule]
    axon = pre.startswith("P")
    PRE = {s for s in range(P["steps"]) if pre in P["schedule"][s]} if axon else {s for s in range(P["steps"]) if hw[pre] in steps[s]}
    POST = {s for s in range(P["steps"]) if hw[post] in steps[s]}
    a = AxonStamps(1) if axon else NeuronStamps(1); q = NeuronStamps(1); a.clear(); q.clear()
    w, e, te = w0, 0, 0
    for s in range(P["steps"]):
        a.advance(); q.advance(); t = q.t
        fp, fq = s in PRE, s in POST
        if fp: a.spike(0)
        if fq: q.spike(0)
        for ltp, fired, d, isi in ((False, fp, q.d(0), a.isi[0]), (True, fq, a.d(0), q.isi[0])):
            if not fired: continue
            p = R.event_p(rl, luts, ltp, d, isi)
            if p is None: continue
            if rl.elig: e, te = R.elig_accumulate(rl, luts, e, te, t, p)
            else: w = R.apply_p(rl, p, w, g)
        if rl.elig and s in rewards:
            w, e, te = R.reward_apply(rl, luts, e, te, t, rewards[s], w)
    return w, len(PRE), len(POST)
want = {}
for sy in P["synapses"]:
    src = ((1 << 17) | hw[sy["pre"]]) if sy["pre"].startswith("P") else hw[sy["pre"]]
    key = (src, hw[sy["post"]] // 16)
    assert key not in want, "protocol has two plastic synapses with the same (pre, dest row)"
    want[key] = (sy,) + expected(sy["pre"], sy["post"], sy["rule"], sy["w0"])
got = {}
for row, val in zip(P["rows_read"], bb):
    for s in range(4):
        e = (val >> (64 * s)) & ((1 << 64) - 1)
        if (e >> 61) == 0 and (e >> 18) & 0xF:
            key = (e & 0x3FFFF, (e >> 48) & 0x1FFF); w = (e >> 32) & 0xFFFF; w = w - (1 << 16) if w >= 1 << 15 else w
            got.setdefault(key, []).append((w, (e >> 18) & 0xF, row, s))
bad = []; changed = 0
for key, (sy, w, npre, npost) in want.items():
    g_ = got.get(key, [])
    if len(g_) != 1: bad.append("%s->%s: found %d times in the read-back" % (sy["pre"], sy["post"], len(g_))); continue
    wr, tag, row, s = g_[0]
    changed += (w != sy["w0"])
    if tag != sy["rule"] or wr != w:
        bad.append("%s->%s rule %d (pre %d spikes, post %d): read %d (tag %d, row %d slot %d), expected %d" % (sy["pre"], sy["post"], sy["rule"], npre, npost, wr, tag, row, s, w))
extra = set(got) - set(want)
if extra: bad.append("%d plastic entries in the read-back not in the protocol" % len(extra))
print("%s plasticity-on: %d steps, %d synapses (%d changed from w0), %d rewards; %d mismatches%s" % (
    "PASS" if not bad else "FAIL", len(steps), len(want), changed, len(rewards), len(bad), "" if not bad else ":"))
for x in bad[:12]: print("   " + x)
sys.exit(0 if not bad else 1)
