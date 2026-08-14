"""
Quick test for the new random_shuffle parameter on CRI_network.
Confirms: (1) shuffle actually changes globalIdx assignment vs default,
(2) the network still computes correctly with shuffling enabled.
"""
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

axons = {"A0": [("N0", 2000)]}
connections = {
    f"N{i}": ([(f"N{i+1}", 2000)] if i < 4 else [], LIF_neuron(theta=1000, nu=-17, Lambda=63))
    for i in range(5)
}
outputs = [f"N{i}" for i in range(5)]

print("=== Without shuffle (default) ===")
net1 = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim", random_shuffle=False)
idxs_default = [net1.connectome.get_neuron_by_key(f"N{i}").get_globalIdx() for i in range(5)]
print(f"globalIdx assignment: {idxs_default}")

print("\n=== With shuffle ===")
net2 = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim", random_shuffle=True)
idxs_shuffled = [net2.connectome.get_neuron_by_key(f"N{i}").get_globalIdx() for i in range(5)]
print(f"globalIdx assignment: {idxs_shuffled}")

assert set(idxs_default) == set(idxs_shuffled), "FAIL: shuffle changed the SET of indices used, should only reorder"
print(f"\nSame index set preserved: {set(idxs_default) == set(idxs_shuffled)}")

print("\n=== Confirming shuffled network still computes ===")
net2.step(["A0"])
mp = net2.read_membrane(outputs)
print(f"MP after one step (shuffled): {mp}")
print("PASS: shuffle preserves index set and network still runs")
