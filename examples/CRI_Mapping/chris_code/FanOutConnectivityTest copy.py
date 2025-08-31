from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron
import concurrent.futures as cf

#parameters
threshold = 1
N = ANN_neuron(threshold, shift = 0)
numberAxons = 1
numberN1 = 4096
numberN2 = 1
#weight = math.ceil(threshold/numberAxons)
weightAxon_N1 = 2
weightN1_N2 = 2
#define dictionaries
axons = {}
connections = {}

#creating inputs, axons, and N1 neurons
inputs = []
for i in range(numberAxons): #connect each axon with each neuron
    axonToNeuron = []
    for j in range (numberN1):  
        connections[f"N.1.{j}"] = ([], N)  #create N1 neuron
        connectingNeuron = (f"N.1.{j}", weightAxon_N1) 
        axonToNeuron.append(connectingNeuron)
    axons[f"A{i}"] = axonToNeuron
    inputs.append(f"A{i}")

#creating all N1 neurons and connect to N2 neurons
for i in range(numberN1):
    for j in range(numberN2):
        connections[f"N.2.{j}"] = ([], N)  #create N2 neuron
        connections[f"N.1.{i}"][0].append((f"N.2.{j}", weightN1_N2)) #connect N1 --> N2

#creating output neurons
outputs = []
for i in range(numberN2):    #add N2 neurons to the output list
    outputs.append(f"N.2.{i}")

#create list of neurons we want to read the MPs of 
neurons_to_read = []
#for i in range(numberN1):    #add N1 neurons to the list
    #neurons_to_read.append(f"N.1.{i}")
neurons_to_read.append("N.1.0")
neurons_to_read.append("N.1.1")
neurons_to_read.append("N.1.2")
for i in range(numberN2):    #add N2 neurons to the list
    neurons_to_read.append(f"N.2.{i}")



network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")
currSpikes1 = network.step(inputs) #1st time step
results1 = network.read_membrane(neurons_to_read)
currSpikes2 = network.step([]) #2nd time step
results2 = network.read_membrane(neurons_to_read)
currSpikes3 = network.step([]) #3rd time step
results3 = network.read_membrane(neurons_to_read)

print(f"Spikes1: {currSpikes1}")
print(f"Spikes2: {currSpikes2}")
print(f"Spikes3: {currSpikes3}")
print(f"MP1: {results1}")
print(f"MP2: {results2}")
print(f"MP3: {results3}")
#print(f"Axons: {axons}")
#print(f"Connections: {connections}")
#print(f"Outputs: {outputs}")



