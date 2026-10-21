"""
test_noc_hw.py -- hardware tests of spike delivery between cores over the NoC.

    L1         source core -> destination core in the same 4-core cluster
    L2         source core -> destination core in another cluster
    MULTICAST  one source neuron -> three destination cores
    RING       a spike chain through every listed core and back to the first

A source neuron S0 is driven every step and fires every step.  It connects with weight 50
to destination neurons that only accumulate (threshold out of reach).  Each destination
must end with V = 50 * k, k >= 1 and close to S0's spike count; a control neuron on a core
S0 does not connect to must still read V = 0.  The ring passes when every ring neuron fires,
in ring order.  Networks are partitioned with CRI_network(membership=..., n_cores=16), which
loads every core's NoC routing table.

These tests run only on the NoC test bitstream and are skipped unless HIAER_NOC_TESTS=1.
Cores are chosen with environment variables (defaults in brackets):
    HIAER_NOC_SRC [0]  HIAER_NOC_L1 [1]  HIAER_NOC_L2 [12]  HIAER_NOC_CTRL [5]
    HIAER_NOC_MCAST [1,2,3]  HIAER_NOC_RING [0-15]  HIAER_NOC_STEPS [10]

    HIAER_NOC_TESTS=1 python -m pytest tests/test_noc_hw.py -v -s
See tests/HARDWARE_TESTS.md.
"""
import os

import pytest

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

pytestmark = pytest.mark.skipif(
    os.environ.get("HIAER_NOC_TESTS") != "1",
    reason="NoC tests run only on the NoC test bitstream; set HIAER_NOC_TESTS=1 after flashing it")


def _cores(name, default):
    spec = os.environ.get(name, default)
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


SRC = int(os.environ.get("HIAER_NOC_SRC", 0))
L1 = int(os.environ.get("HIAER_NOC_L1", 1))
L2 = int(os.environ.get("HIAER_NOC_L2", 12))
CTRL = int(os.environ.get("HIAER_NOC_CTRL", 5))
MCAST = _cores("HIAER_NOC_MCAST", "1,2,3")
RING = _cores("HIAER_NOC_RING", "0-15")
STEPS = int(os.environ.get("HIAER_NOC_STEPS", 10))

BIG = 1_000_000
W_DRIVE, W_SYN = 200, 50


def _src_model():
    return LIF_neuron(threshold=100, shift=-17, leak=63)


def _dst_model():
    return LIF_neuron(threshold=BIG, shift=-17, leak=63)


def _to_s32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _spikes(res):
    return list(res[0]) if isinstance(res, tuple) else list(res)


def _delivery(src, dsts, ctrl):
    axons = {"A0": [("S0", W_DRIVE)]}
    conns = {"S0": ([("D%d" % i, W_SYN) for i in range(len(dsts))], _src_model()),
             "CTRL": ([], _dst_model())}
    member = {"A0": src, "S0": src, "CTRL": ctrl}
    for i, c in enumerate(dsts):
        conns["D%d" % i] = ([], _dst_model())
        member["D%d" % i] = c
    outs = ["S0", "CTRL"] + ["D%d" % i for i in range(len(dsts))]
    net = CRI_network(axons=axons, connections=conns, outputs=outs, target="CRI",
                      membership=member, n_cores=16)
    s0 = sum(1 for _ in range(STEPS) if "S0" in _spikes(net.step(["A0"])))
    mp = {k: _to_s32(v) for k, v in net.read_membrane(outs)}
    print("\n  S0 (core %d) spikes=%d  CTRL (core %d) V=%d" % (src, s0, ctrl, mp["CTRL"]))
    assert s0 > 0, "source neuron on core %d never fired" % src
    assert mp["CTRL"] == 0, "control neuron on core %d received V=%d" % (ctrl, mp["CTRL"])
    for i, c in enumerate(dsts):
        v = mp["D%d" % i]
        k = v // W_SYN if v % W_SYN == 0 else None
        print("  D%d (core %d) V=%d -> %s spikes delivered" % (i, c, v, k))
        assert k is not None and k >= 1 and s0 - 2 <= k <= s0, (
            "core %d -> core %d: destination V=%d, source fired %d times" % (src, c, v, s0))


def test_noc_l1_same_cluster():
    _delivery(SRC, [L1], CTRL)


def test_noc_l2_cross_cluster():
    _delivery(SRC, [L2], CTRL)


def test_noc_multicast():
    _delivery(SRC, MCAST, CTRL)


def test_noc_ring():
    n = len(RING)
    axons = {"A0": [("R0", W_DRIVE)]}
    conns, member = {}, {"A0": RING[0]}
    for i, c in enumerate(RING):
        conns["R%d" % i] = ([("R%d" % ((i + 1) % n), W_DRIVE)], _src_model())
        member["R%d" % i] = c
    net = CRI_network(axons=axons, connections=conns, outputs=["R%d" % i for i in range(n)],
                      target="CRI", membership=member, n_cores=16)
    first = {}
    for t in range(3 * n + 5):
        for k in _spikes(net.step(["A0"] if t == 0 else [])):
            first.setdefault(k, t)
    times = [first.get("R%d" % i) for i in range(n)]
    print("\n  ring cores %s\n  first spike step per ring neuron %s" % (RING, times))
    assert all(x is not None for x in times), "ring neurons that never fired: %s" % (
        [RING[i] for i, x in enumerate(times) if x is None])
    assert all(times[i] < times[i + 1] for i in range(n - 1)), "spikes out of ring order: %s" % times
