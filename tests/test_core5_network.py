"""Whole tiny network on core 5. Validates multicore: element-62 routing,
per-neuron+axon core assignment, compile, execute, spike readout on a non-zero core."""
import hs_bridge
from hs_api.api import CRI_network
from hs_api.neuron_models import IF_neuron
import hs_bridge.FPGA_Execution.fpga_controller as fc

TARGET_CORE = 5
THRESH = 1000

axons = {"A0": [("N0", 2000)]}
connections = {
    "N0": ([("N1", 2000)], IF_neuron(THRESH, nu=0), TARGET_CORE),
    "N1": ([], IF_neuron(THRESH, nu=0), TARGET_CORE),
}
outputs = ["N0", "N1"]

net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI", coreID=None)

print("axon cores:", set(a.get_core() for a in net.connectome.get_axons()))
print("neuron cores:", set(n.get_core() for n in net.connectome.get_neurons()))
# Explicitly write routing table entry 0 (covers neurons 0-511, includes N0=0, N1=32)
# as LOCAL for core 5 — testing whether spikes need this explicitly loaded even
# for host-only visibility.
fc.write_route_table_entry(core_id=TARGET_CORE, entry_addr=0, level=0b01)
print("wrote routing table entry for core 5")

for step in range(4):
    inputs = ["A0"] if step == 0 else []
    spikes = net.step(inputs)
    mps = fc.readSelect([n.get_hbmIdx() for n in net.connectome.get_neurons() if n.get_user_key() in ("N0","N1")], coreID=5)
    print(f"step {step}: inputs={inputs} spikes={spikes[0]} MPs(core5)={[(m[0],m[3]) for m in mps]}")