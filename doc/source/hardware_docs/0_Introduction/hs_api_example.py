"""
hs_api Example Network
======================

A simple network demonstrating hs_api usage:
- 5 axons (a0-a4) fully connected to 5 hidden neurons (h0-h4)
- 5 hidden neurons fully connected to 5 output neurons (o0-o4)
- All synaptic weights = 1000
- Input: First 3 axons (a0, a1, a2) fire at each timestep
- Simulation: 10 timesteps using software simulation
"""

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

# Define the neuron model
# v_thr: threshold voltage (neuron spikes when membrane potential >= v_thr)
# perturbation: random noise added to threshold (0 = deterministic)
# leak: leak rate (63 = 2^6-1 = IF neuron with no leak)
lif_model = LIF_neuron(v_thr=2000, perturbation=0, leak=63)

# Define axons: each axon connects to all 5 hidden neurons with weight 1000
# Format: axon_name -> [(target_neuron, weight), ...]
axons = {
    'a0': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a1': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a2': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a3': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)],
    'a4': [('h0', 1000), ('h1', 1000), ('h2', 1000), ('h3', 1000), ('h4', 1000)]
}

# Define neuron connections: each hidden neuron connects to all 5 output neurons
# Format: neuron_name -> ([(target_neuron, weight), ...], neuron_model)
connections = {
    'h0': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h1': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h2': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h3': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model),
    'h4': ([('o0', 1000), ('o1', 1000), ('o2', 1000), ('o3', 1000), ('o4', 1000)], lif_model)
}

# Define network configuration
config = {
    'neuron_type': 'LIF',
    'global_neuron_params': {
        'v_thr': 2000  # Global threshold (used by hardware, but neuron model has its own)
    }
}

# Define which neurons to record spikes from (output layer)
outputs = ['o0', 'o1', 'o2', 'o3', 'o4']

# Create the network with software simulation target
network = CRI_network(
    axons=axons,
    connections=connections,
    config=config,
    outputs=outputs,
    target='simpleSim'  # Use software simulation instead of hardware
)

# Run simulation for 10 timesteps
# Input: First 3 axons (a0, a1, a2) fire at each timestep
print("Running 10-timestep simulation with axons a0, a1, a2 firing\n")
print("=" * 60)

inputs = ['a0', 'a1', 'a2']  # First 3 axons fire each timestep

for timestep in range(10):
    # Step the network forward by one timestep
    spikes = network.step(inputs)

    print(f"Timestep {timestep:2d}: Spikes = {spikes}")

print("=" * 60)
print("\nSimulation complete!")
