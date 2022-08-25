from l2s.api import CRI_network
import sys
import subprocess
import time
import pickle
#Define a configuration dictionary
config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 100

############################
# Let's try a simple network
############################

#Define an inputs dictionary
inputs = {}
steps = 100
for i in range(steps):
    #if (i%3 == 0):
    inputs[i] = ['alpha']
    #elif(i%3 == 1):
    #    inputs[i] = ['beta']
    #else:
    #    inputs[i] = ['alpha', 'beta']

print(inputs)

#Define an axons dictionary
axonsDict = {
#         'alpha': [('a', 1.0),('b', 2.0),('c', 3.0),('d', 4.0),('e',5.0)],
}

#Define a connections dictionary
neuronsDict = {'0': [],
               '1': [],
               '2': [],
               '3': [],
               '4': [],
               '5': [],
               '6': [],
               '7': [],
               '8': [],
               '9': [],
               '10': [],
               '11': [],
               '12': [],
               '13': [],
               '14': [],
               '15': [],
               '16': [],
               '17': [('43',4)],
               '18': [('42',4)],
               '19': [('41',4)],
               '20': [('40',4)],
               '21': [('39',4)],
               '22': [],
               '23': [],
               '24': [],
               '25': [],
               '26': [],
               '27': [],
               '28': [],
               '29': [],
               '30': [],
               '31': [],
               '32': [],
               '33': [],
               '34': [],
               '35': [],
               '36': [],
               '37': [],
               '38': [],
               '39': [],
               '40': [],
               '41': [],
               '42': [],
               '43': []}
outputs=neuronsDict.keys()

#for i in range (20000-len(neuronsDict)):
#    neuronsDict['q'+str(i)] = []

for i in range (257-len(axonsDict)):
    axonsDict['a'+str(i)] = [] #[(str(i%44),4)]

axonsDict = pickle.load(open( "axonsDict.p", "rb" ) )
#axonsDict = {'a781' : axonsDict['a781'] }
#print(axonsDict)
#breakpoint()
neuronsDict = pickle.load(open( "neuronsDict.p", "rb" ) )
outputs = pickle.load(open( "outputs.p", "rb" ) )
pickleInput = pickle.load(open("input.p", "rb"))
#print(pickleInput)
#breakpoint()
#Initialize a CRI_network object for interacting with the hardware and the software
hardwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config,target='CRI', outputs = outputs)
softwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config, outputs = outputs, target='simpleSim')

#hardwareNetwork.write_synapse('alpha', 'a', -3)
#softwareNetwork.write_synapse('alpha', 'a', -3)

#Execute the network stepwise in the hardware and the simulator
for i in range(steps):
    #start = time.time()
    hwSpike, hwResult = hardwareNetwork.step(pickleInput[0][i], membranePotential=True)
    #end = time.time()
    #print(end - start)
    #start = time.time()
    swSpike, swResult = softwareNetwork.step(pickleInput[0][i], membranePotential=True)
    #end = time.time()
    #print(end - start)
    print("timestep: "+str(i)+":")
    print("hardware result: ")
    #print(synthSpike)
    print(hwSpike)
    print(hwResult)
    print("timestep: "+str(i)+" end")
#    print("software result: ")
#    print(swSpike)
#    print(swResult)
    #Verify that the outputs match
    for idx in range(len(swResult)):
        if(swResult[idx][1] != hwResult[idx][1]):
            print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
#hardwareNetwork.sim_flush('jul26Dump257Axon2.txt')
