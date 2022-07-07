from l2s.api import CRI_network
import sys
import subprocess

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
    if (i%3 == 0):
        inputs[i] = ['alpha']
    elif(i%3 == 1):
        inputs[i] = ['beta']
    else:
        inputs[i] = ['alpha', 'beta']

#print(inputs)

#Define an axons dictionary
axons = {'alpha': [('a', 1.0),('b', 2.0),('c', 3.0),('d', 4.0),('e',5.0)],
         'beta' : [('a', 2.0)]}

#Define a connections dictionary
connections = {'a': [('z',4)],
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



Network = CRI_network(axons=axons,connections=connections,config=config, outputs=['a','b','c'])




for i in range(100):
    print("executing a timestep: ")
    Result, Spike = Network.step(inputs[i], membranePotential=True)
    print("result: ")
    print(Spike)
    print(Result)
