from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

timesteps = 1000

shift_values = list(range(-17, 20))  # Shifts from -17 to +17
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
        neuron_model = LIF_neuron(threshold=0, shift=shift, leak=63)
        neuron_name = shift  # Naming neurons based on shift
        axons["A0"].append((neuron_name, 0))  # Connect A0 to each neuron with weight=0
        connections[neuron_name] = ([], neuron_model)  # No incoming connections, just the neuron model
        outputs.append(neuron_name)

    print(len(connections))  # Should print 5 neurons
    network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

    # Read the first membrane potential after the first step
    network.step([])  # No input
    prev_mp = network.read_membrane(outputs)

    # remaining timesteps and accumulate differences
    for _ in range(1, timesteps):
        network.step([])  # No input
        curr_mp = network.read_membrane(outputs)

        for index, neuron_name in enumerate(outputs):
            diff = abs(curr_mp[index]() - prev_mp[index]())
            mp_diff_sums[neuron_name] += float(diff)
        prev_mp = curr_mp

print("Absolute MP differences over " + str(timesteps) + " steps:")
for shift in shift_values:
    print(f"{shift}: {mp_diff_sums[shift]}")

#calculate neighboring neuron MP quotients
print("\nNeighboring neuron MP quotients:")

for shift in range(-17, 19): 
    prev_sum = mp_diff_sums[shift]
    current_sum = mp_diff_sums[shift + 1]
    if prev_sum != 0:
        quotient = current_sum / prev_sum 
    else:
        quotient = float('inf')
    print(f"{shift + 1} / {shift}: {quotient}")

