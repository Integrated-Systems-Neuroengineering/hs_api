"""
Variant of NoiseTest_simulator.py that calls clear() after every timestep,
per Leif's suggestion, to test whether resetting MP each step reduces
overflow/saturation for high positive shifts (since noise wouldn't compound
on top of an already-large accumulated value).
"""
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

timesteps = 10000

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

    # Read the first membrane potential after the first step
    network.step([])
    prev_mp = [int(x) for x in network.read_membrane(outputs)]

    for _ in range(1, timesteps):
        network.step([])
        curr_mp = [int(x) for x in network.read_membrane(outputs)]

        for index, neuron_name in enumerate(outputs):
            mp_diff_sums[neuron_name] += abs(curr_mp[index] - prev_mp[index])

        # Reset MP to zero after every timestep, per Leif's suggestion
        network.simpleSim.clear()
        prev_mp = [0] * len(outputs)

print("Absolute MP differences over " + str(timesteps) + " steps (with clear() after every step):")
for shift in shift_values:
    print(f"{shift}: {mp_diff_sums[shift]}")

print("\nNeighboring neuron MP quotients:")
for shift in range(-17, 17):
    prev_sum = mp_diff_sums[shift]
    current_sum = mp_diff_sums[shift + 1]
    quotient = current_sum / prev_sum if prev_sum != 0 else float('inf')
    print(f"{shift + 1} / {shift}: {quotient}")
