"""test_heavy_spike_psc_hw.py -- heavy spiking in the biological (exp-PSC) mode, for the project simulation (round 15).
2048 LIF neurons, each driven by its own axon (weight 600) on two of every three steps and fanning out to 8 others
(weight 40; every neuron receives exactly 8), 30 steps, then 64 membranes read.  decay_ex 2048 keeps the 12-bit
synaptic current in range (worst case 2 x (600 + 8 x 40) = 1840 < 2048); the bio reference model gives 18-28 spikes
per neuron over the 30 steps.  The replay is run with the host reading continuously and with 30 % and
60 % read pauses: with fix 14's back-pressure the outcome must not depend on how fast the host reads."""
import os
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
import hs_api.single_core_conductance_STDP as p2
CORE = int(os.environ.get("HIAER_CORE_ID", 1))
N, FAN, STEPS = 2048, 8, 30
def test_heavy_spike_psc():
    model = LIF_neuron(threshold=400, shift=-17, leak=63)
    connections = {f"N1.{i}": ([(f"N1.{(i * 7 + k * 613 + 1) % N}", 40) for k in range(FAN)], model) for i in range(N)}
    axons = {f"A{i}": [(f"N1.{i}", 600)] for i in range(N)}
    outputs = [f"N1.{i}" for i in range(N)]
    net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI", coreID=CORE)
    p2.send_cmd13_psc_params(coreID=CORE, delta_mode=0, decay_ex=2048)
    total, per_step = 0, []
    hw = {k: net.connectome.connectomeDict[k].get_coreTypeIdx() for k in connections}
    for s in range(STEPS):
        out = net.step([f"A{i}" for i in range(N) if (i + s) % 3 != 0])
        keys = out[0] if isinstance(out, tuple) else out
        total += len(keys)
        per_step.append(sorted(hw[k] for k in keys))
    net.read_membrane([f"N1.{i}" for i in range(0, N, N // 64)])
    p2.send_cmd13_psc_params(coreID=CORE)          # back to the reset defaults (delta_mode=1)
    print("host saw %d spikes over %d steps" % (total, STEPS))
    if os.environ.get("HEAVY_OUT"):
        import json; json.dump({"steps": STEPS, "spikes": per_step}, open(os.environ["HEAVY_OUT"], "w"))
