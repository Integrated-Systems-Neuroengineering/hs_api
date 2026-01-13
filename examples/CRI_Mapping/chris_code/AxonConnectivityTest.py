from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron
import concurrent.futures as cf

def test(numberAxons, numberNeurons, weight):
    #define dictionaries
    axons = {}
    connections = {}

    #creating axons and inputs
    inputs = []
    for i in range(numberAxons): #connect each axon with each neuron
        axonToNeuron = []
        for j in range (numberNeurons):  
            connectingNeuron = (f"N{j}", weight)
            axonToNeuron.append(connectingNeuron)
        axons[f"A{i}"] = axonToNeuron
        inputs.append(f"A{i}")

    #creating output neurons
    outputs = []
    for i in range(numberNeurons):    #add each neuron to the connections dictionary
        connections[f"N{i}"] = ([], N)
        outputs.append(f"N{i}")

    network = CRI_network(axons=axons,connections=connections,outputs=outputs, target = "CRI")
    currSpikes = network.step([], membranePotential=True) #1st time step
    currSpikes = network.step(inputs, membranePotential=True) #2nd time step
    currSpikes = network.step([], membranePotential=True) #3rd time step

    return currSpikes

#parameters
threshold = 1
N = ANN_neuron(threshold, shift = 0)
numberAxons = 16000
numberNeurons = 1
#weight = math.ceil(threshold/numberAxons)
weight = 2

failure_count = 0
failed_axon_number = []
counter = 1

while True:
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(test, numberAxons=numberAxons, numberNeurons=numberNeurons, weight=weight)
        try:
            spikes = future.result(timeout=60)   # 60-s limit
        except cf.TimeoutError:
            print("system crash")
            print(f"Axon count before system crash: {numberAxons}")
            print(f'Failed Axons: {failed_axon_number}')
            spikes = None
    
    print(f"Axons: {numberAxons}")

    if spikes[1][0] == []:
        failure_count += 1
        failed_axon_number.append(numberAxons)
        print(spikes)
    else:
        failure_count = 0

    if failure_count == 3:  #testing stops only if there are three consecutive fails
        print(f"Max Number of Axons before three trial failure: {numberAxons}")
        break

    if counter % 1000 == 0:  #insert breakpoint every 1000 iterations to check failed axons
        print(f'Failed Axons: {failed_axon_number}')
        breakpoint()

    numberAxons += 1
    counter += 1
    
print(f'Failed Axons: {failed_axon_number}')