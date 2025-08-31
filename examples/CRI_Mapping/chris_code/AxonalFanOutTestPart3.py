from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron
import concurrent.futures as cf

#parameters
threshold = 1
N = ANN_neuron(threshold, shift = 0)
numberAxons = 1
number_blocks = 4096  # Number of blocks to test
numberN1 = 1
#weight = math.ceil(threshold/numberAxons)
weightAxon_N1 = 2
weightN1_N2 = 2
#define dictionaries
axons = {}
connections = {}

#creating inputs, axons, and ONE input axon that branches to all N1 neurons
inputs = []
axonToNeuron = []  #this will contain connections to all N1 neurons

#create all N1 neurons
for i in range(number_blocks):
    connections[f"N.1.{i}"] = ([], N)  #create N1 neuron for each block
    connectingNeuron = (f"N.1.{i}", weightAxon_N1) 
    axonToNeuron.append(connectingNeuron)  #add connection to the ONE axon

#add the ONE axon with connections to all N1 neurons
axons["A0"] = axonToNeuron
inputs.append("A0")  #just one input

#creating the shared N2 neuron
connections[f"N.2.0"] = ([], N)  #create single N2 neuron

#connect all N1 neurons to the same N2 neuron
for i in range(number_blocks):
    connections[f"N.1.{i}"][0].append((f"N.2.0", weightN1_N2)) #connect N1 --> N2

#creating output neurons
outputs = []
outputs.append("N.2.0")  #add just the N2 neuron to output list

#create list of neurons we want to read the MPs of 
neurons_to_read = []
neurons_to_read.append("N.1.0")  #first N1 neuron for reference
neurons_to_read.append("N.2.0")  #the shared N2 neuron

network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")
currSpikes1 = network.step(inputs) #1st time step - single axon spikes, activating all N1 neurons
results1 = network.read_membrane(neurons_to_read)
currSpikes2 = network.step([]) #2nd time step - N1 neurons should spike, activating N2
results2 = network.read_membrane(neurons_to_read)
currSpikes3 = network.step([]) #3rd time step - should be quiet
results3 = network.read_membrane(neurons_to_read)

# Get N2 membrane potential and check if it's what we expect
n2_potential = results2[1][1]
expected_potential = 2 * number_blocks

print(f"\nRunning synaptic fan-in test (2nd part) with {number_blocks} blocks")
print(f"N2 membrane potential: {n2_potential}")
print(f"Expected potential: {expected_potential}")
print(f"Accurate? {n2_potential == expected_potential}")

print(f"\nSpikes1: {currSpikes1}")
print(f"Spikes2: {currSpikes2}")
print(f"Spikes3: {currSpikes3}")
print(f"MP1: {results1}")
print(f"MP2: {results2}")
print(f"MP3: {results3}")