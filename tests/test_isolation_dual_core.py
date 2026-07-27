"""
Isolation test: N0 and N1 both on core 5 (known-working config from
test_core5_network.py), PLUS one unconnected dummy neuron on core 6 just to
force _active_cores = [5, 6] -- with NO real cross-core synapse involved.

If N0 still fails to accumulate here, the bug is triggered by merely having
a second active core (any dual-core init), not specifically by the
INTER_CORE/relay-axon mechanism.
"""
import hs_bridge
import hs_bridge.FPGA_Execution.fpga_controller as fc
from hs_api.api import CRI_network
from hs_api.neuron_models import IF_neuron

THRESH = 1000
WEIGHT = 2000

axons = {"A0": [("N0", WEIGHT)]}
connections = {
    "N0": ([("N1", WEIGHT)], IF_neuron(THRESH, nu=0), 5),
    "N1": ([], IF_neuron(THRESH, nu=0), 5),
    "Dummy": ([], IF_neuron(THRESH, nu=0), 6),  # unconnected, forces core 6 active
}
outputs = ["N0", "N1"]

net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI", coreID=None)
print(f"BEFORE PATCH: _active_cores={net.CRI._active_cores}")
net.CRI._active_cores = [5]  # force subsequent step() calls to only touch core 5
print(f"AFTER PATCH: _active_cores={net.CRI._active_cores}")

n0 = net.connectome.get_neuron_by_key("N0")
n1 = net.connectome.get_neuron_by_key("N1")
print(f"N0: core={n0.get_core()}, hbmIdx={n0.get_hbmIdx()}")
print(f"N1: core={n1.get_core()}, hbmIdx={n1.get_hbmIdx()}")
print(f"axon cores: {set(a.get_core() for a in net.connectome.get_axons())}")
print(f"neuron cores: {set(nn.get_core() for nn in net.connectome.get_neurons())}")

for step in range(4):
    inputs = ["A0"] if step == 0 else []
    spikes = net.step(inputs)
    n0_mp = fc.readSelect([n0.get_hbmIdx()], coreID=5)
    n1_mp = fc.readSelect([n1.get_hbmIdx()], coreID=5)
    print(
        f"step {step}: inputs={inputs} "
        f"N0_MP={n0_mp[0][3] if n0_mp else 'N/A'} "
        f"N1_MP={n1_mp[0][3] if n1_mp else 'N/A'} "
        f"spikes={spikes[0] if isinstance(spikes, tuple) else spikes}"
    )
