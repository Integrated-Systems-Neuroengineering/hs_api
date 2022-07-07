from l2s.api import CRI_network
import sys
import subprocess

#Define a configuration dictionary
config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 10

############################
# Let's try a simple network
############################

#Define an inputs dictionary
inputs = {0: ['alpha','beta'],
          1: ['alpha','beta'],
          2: ['alpha','beta'],
          3: ['alpha','beta'],
          4: ['alpha','beta'],
          5: ['alpha','beta'],
          6: ['alpha','beta'],
          7: ['alpha','beta']}
for i in range(100):
    inputs[i+7] = ['alpha','beta']


#Define an axons dictionary
axons = {'alpha': [('a', 3.0)],
         'beta': [('t', 3.0)]}

#Define a connections dictionary
connections = {'a': [('b', 1.0)],
               'b': [],
               'c': [],
               'd': [],
               'e': [],
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
               't': [('d', 1.0)],
               'u': []}

#Initialize a CRI_network object for interacting with the hardware and the software
hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='CRI')
softwareNetwork = CRI_network(axons=axons,connections=connections,config=config)

hardwareNetwork.write_synapse('a', 'b', 2)
#softwareNetwork.write_synapse('a', 'b', 2)

#Execute the network stepwise in the hardware and the simulator
for i in range(100):
    hwResult = hardwareNetwork.step(inputs[0])
    swResult = softwareNetwork.step(inputs[0])
    print("timestep: "+str(i)+":")
    print("hardware result: ")
    print(hwResult)
    print("software result: ")
    print(swResult)
    #Verify that the outputs match
    for idx in range(len(swResult)):
        if(swResult[idx][1] != hwResult[idx][1][3]):
            print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
#print("last flush")
#print(subprocess.run(['sudo', 'adxdma_dmadump', 'rb', '0', '0' ,'0x40'], stdout=subprocess.PIPE, check=True).stdout.decode('utf-8'))

#hardwareNetwork.sim_flush('Jun11simDump.txt')

#sys.exit()

############################
# Let's try a complex network
############################
"""
#Define an inputs dictionary
inputs = {0: [1, 3, 4],
          1: [4],
          2: [0, 3, 4],
          3: [4],
          4: [0, 2, 3, 4],
          5: [1, 3, 4],
          6: [4],
          7: [0, 3, 4],
          8: [4],
          9: [0, 2, 3, 4]}

#Define an axons dictionary
axons = {'a': [(1, 1.0), (2, 1.0)],
         'b': [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
         'c': [(0, 1.0), (3, 1.0)],
         'd': [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
         'e': [(0, -1.0), (1, -1.0), (2, -1.0), (3, -1.0), (4, -7.0), (5, -7.0), (6, -7.0), (7, -7.0), (8, -7.0), (9, -7.0), (10, -7.0), (11, -7.0), (12, -7.0), (13, -7.0), (14, -7.0), (15, -7.0), (28, -3.0), (29, -3.0), (30, -3.0), (31, -3.0), (32, -3.0), (33, -3.0), (34, -3.0), (35, -3.0), (36, -3.0), (37, -3.0), (38, -3.0), (39, -3.0)]}

#Define a connections dictionary
connections = {0: [(4, 8.0), (6, 8.0)],
               1: [(7, 8.0), (8, 8.0), (9, 8.0)],
               2: [(10, 8.0), (11, 8.0), (12, 8.0)],
               3: [(13, 8.0), (14, 8.0), (15, 8.0)],
               4: [(28, 1.0), (30, 1.0), (32, 1.0), (33, 1.0), (34, 1.0), (36, 1.0), (38, 1.0), (39, 1.0)],
               5: [(28, 1.0), (29, 1.0), (37, 1.0), (39, 1.0)],
               6: [(30, 1.0), (33, 1.0), (34, 1.0), (37, 1.0), (38, 1.0), (39, 1.0)],
               7: [(31, 1.0), (33, 1.0), (34, 1.0), (35, 1.0), (36, 1.0), (37, 1.0), (39, 1.0)],
               8: [(28, 1.0), (29, 1.0), (31, 1.0), (37, 1.0), (38, 1.0), (39, 1.0)],
               9: [(29, 1.0), (32, 1.0), (33, 1.0), (34, 1.0), (36, 1.0)],
               10: [(30, 1.0), (32, 1.0), (33, 1.0), (34, 1.0), (35, 1.0)],
               11: [(29, 1.0), (30, 1.0), (31, 1.0), (32, 1.0), (35, 1.0), (36, 1.0), (37, 1.0), (38, 1.0), (39, 1.0)],
               12: [(28, 1.0), (29, 1.0), (30, 1.0), (31, 1.0), (32, 1.0), (33, 1.0), (34, 1.0), (35, 1.0), (36, 1.0), (39, 1.0)],
               13: [(28, 1.0), (29, 1.0), (30, 1.0), (31, 1.0), (35, 1.0), (36, 1.0), (38, 1.0)],
               14: [(28, 1.0), (30, 1.0), (31, 1.0), (32, 1.0), (33, 1.0), (34, 1.0), (35, 1.0), (37, 1.0), (38, 1.0)],
               15: [(28, 1.0), (29, 1.0), (31, 1.0), (32, 1.0), (35, 1.0), (36, 1.0), (37, 1.0), (38, 1.0)],
               16: [(4, 4.0), (5, -2.0), (6, -2.0)],
               17: [(4, -2.0), (5, 4.0), (6, -2.0)],
               18: [(4, -2.0), (5, -2.0), (6, 4.0)],
               19: [(7, 4.0), (8, -2.0), (9, -2.0)],
               20: [(7, -2.0), (8, 4.0), (9, -2.0)],
               21: [(7, -2.0), (8, -2.0), (9, 4.0)],
               22: [(10, 4.0), (11, -2.0), (12, -2.0)],
               23: [(10, -2.0), (11, 4.0), (12, -2.0)],
               24: [(10, -2.0), (11, -2.0), (12, 4.0)],
               25: [(13, 4.0), (14, -2.0), (15, -2.0)],
               26: [(13, -2.0), (14, 4.0), (15, -2.0)],
               27: [(13, -2.0), (14, -2.0), (15, 4.0)],
               28: [(16, 1.0)], 29: [(17, 1.0)],
               30: [(18, 1.0)], 31: [(19, 1.0)],
               32: [(20, 1.0)], 33: [(21, 1.0)],
               34: [(22, 1.0)], 35: [(23, 1.0)],
               36: [(24, 1.0)], 37: [(25, 1.0)],
               38: [(26, 1.0)], 39: [(27, 1.0)]}

#Initialize a CRI_network object for interacting with the hardware and the software
hardwareNetwork = CRI_network(axons=axons,connections=connections, config=config, inputs = inputs, target='CRI')
softwareNetwork = CRI_network(axons=axons,connections=connections, config=config,inputs = inputs)

#Execute the network stepwise in the hardware and the simulator
for i in range(len(inputs)):
    hwResult = hardwareNetwork.step(inputs[i])
    swResult = softwareNetwork.step(inputs[i])
    print("timestep: "+str(i)+":")
    print("hardware result: ")
    print(hwResult)
    print("software result: ")
    print(swResult)
    
#hardwareNetwork.sim_flush('simDumpFull.txt')
    #print('dumped')
    #Verify that the outputs match
    for idx in range(len(swResult)):
        if(swResult[idx][1] != hwResult[idx][1][3]):
            print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
"""
