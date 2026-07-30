"""
Simulator (target="simpleSim") version of NoiseTest_hardware.py.

Same test as the FPGA version -- characterizes membrane-potential noise
magnitude across nu (shift) values from -17 to +17, for both legacy_noise_en
modes -- but run on the software simulator instead of hardware.

Note: the simulator's read_membrane() returns a positionally-indexed array
(not (key, value) pairs like the hardware path), so this script zips it with
`outputs` instead of calling dict() directly on it.
"""
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

legacy_noise_en = 0  # CHANGE THIS to 0 or 1 for each run

shift_values = list(range(-17, 18))
mp_diff_sums = {}
for shift in shift_values:
    mp_diff_sums[shift] = 0

for i in range(0, len(shift_values), 5):
    axons = {}
    connections = {}
    outputs = []
    axons["A0"] = []
    shifts = shift_values[i:i+5]
    for shift in shifts:
        neuron_model = LIF_neuron(theta=1000000, nu=shift, Lambda=63, legacy_noise_en=legacy_noise_en)
        neuron_name = shift
        axons["A0"].append((neuron_name, 0))
        connections[neuron_name] = ([], neuron_model)
        outputs.append(neuron_name)

    print(len(connections))
    network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

    network.step([])
    prev_mp = dict(zip(outputs, [int(x) for x in network.read_membrane(outputs)]))

    for _ in range(1, 10000):
        network.step([])
        curr_mp = dict(zip(outputs, [int(x) for x in network.read_membrane(outputs)]))
        for neuron_name in outputs:
            mp_diff_sums[neuron_name] += abs(curr_mp[neuron_name] - prev_mp[neuron_name])
        prev_mp = curr_mp

print("Absolute MP differences over 10000 steps:")
for shift in shift_values:
    print(f"{shift}: {mp_diff_sums[shift]}")

print("\nNeighboring neuron MP quotients:")
for shift in range(-17, 17):
    prev_sum = mp_diff_sums[shift]
    current_sum = mp_diff_sums[shift + 1]
    quotient = current_sum / prev_sum if prev_sum != 0 else float('inf')
    print(f"{shift + 1} / {shift}: {quotient}")