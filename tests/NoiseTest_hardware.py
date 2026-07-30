from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

shift_values = list(range(-17, 18))  # Shifts from -17 to +17
mp_diff_sums = {} # To accumulate absolute MP differences for each neuron
for shift in shift_values:
    mp_diff_sums[shift] = 0

for i in range(0, len(shift_values), 5):
    axons = {}
    connections = {}
    outputs = []
    axons["A0"] = []
    shifts = shift_values[i:i+5]  # Get 5 shifts at a time
    for shift in shifts:
        neuron_model = LIF_neuron(theta=0, nu=shift, Lambda=63)
        neuron_name = shift  # Naming neurons based on shift
        axons["A0"].append((neuron_name, 0))  # Connect A0 to each neuron with weight=0
        connections[neuron_name] = ([], neuron_model)  # No incoming connections, just the neuron model
        outputs.append(neuron_name)

    print(len(connections))  # Should print 5 neurons
    network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI")

    # Read the first membrane potential after the first step
    network.step([])  # No input
    prev_mp = dict(network.read_membrane(outputs))

    # remaining 9999 steps and accumulate differences
    for _ in range(1, 10000):
        network.step([])  # No input
        curr_mp = dict(network.read_membrane(outputs))

        for neuron_name in outputs:
            mp_diff_sums[neuron_name] += abs(curr_mp[neuron_name] - prev_mp[neuron_name])

        prev_mp = curr_mp

print("Absolute MP differences over 10000 steps:")
for shift in shift_values:
    print(f"{shift}: {mp_diff_sums[shift]}")

#calculate neighboring neuron MP quotients
print("\nNeighboring neuron MP quotients:")

for shift in range(-17, 17): 
    prev_sum = mp_diff_sums[shift]
    current_sum = mp_diff_sums[shift + 1]
    if prev_sum != 0:
        quotient = current_sum / prev_sum 
    else:
        quotient = float('inf')
    print(f"{shift + 1} / {shift}: {quotient}")

