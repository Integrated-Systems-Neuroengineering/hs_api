import sys
sys.path.insert(0, '/home/prpandit/hs_api')
sys.path.insert(0, '/home/omowuyi/testing/hs_api/.venv/lib/python3.10/site-packages')

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

theta = 10
weight = 7
steps = 10

axons = {"A0": [("N0", weight)]}
neuron = LIF_neuron(threshold=theta, shift=-17, leak=63, soft_reset_en=True)
connections = {"N0": ([], neuron)}

net = CRI_network(axons=axons, connections=connections, outputs=["N0"], target="simpleSim")

hw = []
for t in range(steps):
    net.step(["A0"])
    mp = net.read_membrane(["N0"])
    hw.append(int(mp[0]))

expected = [7, 14, 11, 8, 15, 12, 9, 16, 13, 10]

print("hw      =", hw)
print("expected=", expected)
if hw == expected:
    print("PASS")
else:
    first = next(t for t in range(steps) if hw[t] != expected[t])
    print(f"FAIL at step {first}: hw={hw[first]}, expected={expected[first]}")
