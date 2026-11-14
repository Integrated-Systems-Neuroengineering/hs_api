"""
test_relay_net_hw.py -- 64-axon network used to test relayed (inter-core) spikes in simulation.

A_i -> N_i (weight 100), i = 0..63, delta mode, threshold out of reach: each delivered input adds 100
to its neuron, and it stays.  8 steps; the only host input is A2 at step 3.  After every step the
test reads N0 N1 N2 N3 N17 N33 N48 N63.  On the card: N2 steps to 100, every other neuron stays 0.
In simulation, relayed spikes are injected at the core's relay port (round3_kit/sim_round3.sh) and
each neuron's step to 100 must land on the step the rules give.
Run: HIAER_CORE_ID=1 python -m pytest tests/test_relay_net_hw.py -v -s
"""
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
from hs_api import single_core_conductance_STDP as p2

from .test_bio_features_hw import BIG, _CORE_ID, _to_s32

READ = [0, 1, 2, 3, 17, 33, 48, 63]


def test_relaynet():
    neuron = LIF_neuron(threshold=BIG + 31, shift=-17, leak=63)
    axons = {"A%d" % i: [("N%d" % i, 100)] for i in range(64)}
    conns = {"N%d" % i: ([], neuron) for i in range(64)}
    names = ["N%d" % i for i in READ]
    net = CRI_network(axons=axons, connections=conns, outputs=names, target="CRI", coreID=_CORE_ID)
    try:
        p2.send_cmd13_psc_params(coreID=_CORE_ID, delta_mode=1)
        rows = []
        for t in range(8):
            net.step(["A2"] if t == 3 else [])
            mp = dict(net.read_membrane(names))
            rows.append([_to_s32(mp[n]) for n in names])
    finally:
        p2.send_cmd13_psc_params(coreID=_CORE_ID)
    print("RELAYNET hw = %s" % rows)
    n2 = [r[2] for r in rows]
    others = sorted(set(r[k] for r in rows for k in range(len(READ)) if k != 2))
    assert others == [0], "a neuron other than N2 changed on the card: %s" % rows
    assert n2 == sorted(n2) and set(n2) == {0, 100}, "N2 did not step once from 0 to 100: %s" % n2
