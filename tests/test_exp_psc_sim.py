import sys
sys.path.insert(0, '/home/prpandit/hs_api')
sys.path.insert(0, '/home/omowuyi/testing/hs_api/.venv/lib/python3.10/site-packages')

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

# F01 settings from test_bio_features_hw.py
theta = 1000001
weight = 1000
decay_ex = 3354
steps = 12

axons = {"A0": [("N0", weight)]}
neuron = LIF_neuron(threshold=theta, shift=-17, leak=63, decay_ex=decay_ex)
connections = {"N0": ([], neuron)}

net = CRI_network(axons=axons, connections=connections, outputs=["N0"], target="simpleSim")

hw = []
for t in range(steps):
    net.step(["A0"] if t == 0 else [])
    mp = net.read_membrane(["N0"])
    hw.append(int(mp[0]))

model = [0, 818, 1487, 2034, 2481, 2847, 3146, 3390, 3589, 3751, 3883, 3991]

print("hw   =", hw)
print("model=", model)

first_diff = next((t for t in range(steps) if hw[t] != model[t]), None)
if first_diff is None:
    print("PASS")
else:
    print(f"FAIL at step {first_diff}: hw={hw[first_diff]}, model={model[first_diff]}")