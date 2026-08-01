"""
Quick test for simple_sim.clear() -- confirms it resets membrane potentials
to zero after some activity, and that stepNum resets too.
"""
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

axons = {"A0": [("N0", 2000)]}
connections = {
    "N0": ([], LIF_neuron(theta=1000000, nu=0, Lambda=63)),  # high threshold so it never spikes/resets on its own
}
outputs = ["N0"]

network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

# Step a few times with input so MP accumulates
for _ in range(3):
    network.step(["A0"])

mp_before_clear = network.read_membrane(outputs)
print(f"MP before clear: {mp_before_clear}")
assert any(int(x) != 0 for x in mp_before_clear), "Expected non-zero MP before clear -- test setup issue?"

# Call clear() directly on the underlying simple_sim object
network.simpleSim.clear()

mp_after_clear = network.read_membrane(outputs)
print(f"MP after clear: {mp_after_clear}")
assert all(int(x) == 0 for x in mp_after_clear), f"FAIL: expected all-zero MP after clear, got {mp_after_clear}"

print(f"stepNum after clear: {network.simpleSim.stepNum}")
assert network.simpleSim.stepNum == 0, "FAIL: expected stepNum to reset to 0"

print("PASS: clear() correctly reset membrane potentials and stepNum")
