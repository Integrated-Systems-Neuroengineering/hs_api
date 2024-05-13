from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
import sys
import subprocess
import time
#Define a configuration dictionary
config = {}
config['neuron_type'] = "LI&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 2**19

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

N1 = LIF_neuron(3,-16,2**5)
N2 = LIF_neuron(6,-16,2**5)

#Define an axons dictionary
axons = {'alpha': [('01', 1.0),('02', 2.0),('03', 3.0)]}

#Define a connections dictionary
connections = {'01': (N1,[('02',6),('03',8) ]),
               '02': (N1, [('01',2)]),
               '03': (N2,[])}

#Initialize a CRI_network object for interacting with the hardware and the software
breakpoint()
hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='simpleSim', outputs = connections.keys(),simDump = False)
#hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='CRI', outputs = connections.keys())

# hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config, target='CRI')
#softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, outputs = connections.keys(), target='simpleSim')
# softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, target='simpleSim')

#hardwareNetwork.write_synapse('alpha', 'a', -3)
#softwareNetwork.write_synapse('alpha', 'a', -3)

#Execute the network stepwise in the hardware and the simulator
for i in range(20):
    #start = time.time()
    hwResult = hardwareNetwork.step(['alpha'],membranePotential = True)
    #print(inputs[i])
    #end = time.time()
    #print(end - start)
    #start = time.time()
    #swResult, swSpike = softwareNetwork.step(inputs[i], membranePotential=True)
    #end = time.time()
    #print(end - start)
    #print("timestep: "+str(i)+":")
    print("hardware result: ")
    #print(synthSpike)
    #print(hwSpike)
    print(hwResult)
    #print("software result: ")
    #print(swSpike)
    #print(swResult)
    #Verify that the outputs match
    #for idx in range(len(swResult)):
    #    if(swResult[idx][1] != hwResult[idx][1][3]):
    #        print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
#hardwareNetwork.sim_flush('jul13.txt')    

    
