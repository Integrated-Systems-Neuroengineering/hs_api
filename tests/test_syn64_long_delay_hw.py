"""test_syn64_long_delay_hw.py -- 64-bit synapses with delay 40 (delay[5] set) in output lists (fix 19).
16 neurons N1.0-15 are driven once at step 0 (weight 1000, threshold 300); each also has a synapse (weight 1, delay 40)
to a partner N1.16-31 that can never fire.  Exactly N1.0-15 must spike, once each, all at the same step, no earlier
than the delay; no other address may ever be reported (v10 reported every 64-bit spike as neuron 0, and the even
32-bit half of a delay >= 32 synapse as a false spike)."""
import os
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron
import hs_api.single_core_conductance_STDP as p2
CORE = int(os.environ.get("HIAER_CORE_ID", 1))
DELAY, STEPS = 40, 60
def test_syn64_long_delay():
    model = ANN_neuron(300, shift=-17)
    connections = {f"N1.{i}": ([(f"N1.{16 + i}", 1)] if i < 16 else [], model) for i in range(32)}
    axons = {f"A{i}": [(f"N1.{i}", 1000)] for i in range(16)}
    net = CRI_network(axons=axons, connections=connections, outputs=[f"N1.{i}" for i in range(32)], target="CRI",
                      coreID=CORE, syn_64bit=True, syn_delay=DELAY)
    p2.send_cmd13_psc_params(coreID=CORE, delta_mode=1, syn_64bit_en=1)
    fired = {}
    for s in range(STEPS):
        out = net.step([f"A{i}" for i in range(16)] if s == 0 else [])
        for k in (out[0] if isinstance(out, tuple) else out):
            fired.setdefault(k, []).append(s)
    p2.send_cmd13_psc_params(coreID=CORE)
    print("spikes by neuron:", fired)
    assert set(fired) == {f"N1.{i}" for i in range(16)}, "unexpected spiking set: %s" % sorted(fired)
    assert all(len(v) == 1 for v in fired.values()), "a neuron spiked more than once: %s" % fired
    steps = {v[0] for v in fired.values()}
    assert len(steps) == 1, "the 16 neurons did not spike in the same step: %s" % fired
    assert steps.pop() >= DELAY, "spikes arrived before the %d-step delay: %s" % (DELAY, fired)
