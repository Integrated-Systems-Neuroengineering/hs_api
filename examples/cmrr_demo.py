from l2s.api import CRI_network
import sys
import subprocess
import time
#Define a configuration dictionary
config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 9

############################
# Let's try a simple network
############################

#Define an inputs dictionary
inputs = {}
for i in range(100):
    #if (i%3 == 0):
    inputs[i] = ['alpha']
    #elif(i%3 == 1):
    #    inputs[i] = ['beta']
    #else:
    #    inputs[i] = ['alpha', 'beta']

print(inputs)

#Define an axons dictionary
axons = {'alpha': [('a', 1.0),('b', 2.0),('c', 3.0),('d', 4.0),('e',5.0)]}

#Define a connections dictionary
connections = {'01': [],
               '02': [],
               '03': [],
               '04': [],
               '05': [],
               '06': [],
               '07': [],
               '08': [],
               '09': [],
               '10': [],
               '11': [],
               '12': [],
               '13': [],
               '14': [],
               '15': [],
               '16': [],
               'a': [('z',4)],
               'b': [('y',4)],
               'c': [('x',4)],
               'd': [('w',4)],
               'e': [('v',4)],
               'f': [],
               'g': [],
               'h': [],
               'i': [],
               'j': [],
               'k': [],
               'l': [],
               'm': [],
               'n': [],
               'o': [],
               'p': [],
               'q': [],
               'r': [],
               's': [],
               't': [],
               'u': [],
               'v': [],
               'w': [],
               'x': [],
               'y': [],
               'z': []}

#Initialize a CRI_network object for interacting with the hardware and the software
hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='CRI', outputs = connections.keys())
softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, outputs = connections.keys(), target='simpleSim')

#hardwareNetwork.write_synapse('alpha', 'a', -3)
#softwareNetwork.write_synapse('alpha', 'a', -3)

#Execute the network stepwise in the hardware and the simulator
for i in range(100):
    start = time.time()
    hwResult, hwSpike = hardwareNetwork.step(inputs[i], membranePotential=True)
    end = time.time()
    #print(end - start)
    start = time.time()
    swResult, swSpike = softwareNetwork.step(inputs[i], membranePotential=True)
    end = time.time()
    #print(end - start)
    print("timestep: "+str(i)+":")
    print("hardware result: ")
    #print(synthSpike)
    print(hwSpike)
    print(hwResult)
    print("software result: ")
    print(swSpike)
    print(swResult)
    #Verify that the outputs match
    #for idx in range(len(swResult)):
    #    if(swResult[idx][1] != hwResult[idx][1][3]):
    #        print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
