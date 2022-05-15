from cri_simulations.config import *
from cri_simulations.api import *
from cri_simulations.compile_network import load_network
from l2s.api import CRI_network
#config = {"neuronType" : 3, "voltageThresh" : 0} 

config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 1
axons, connections, inputs, outputs, n_cores = load_network()
#myNet = network(axons,connections,inputs,outputs,config)
#myNet.initalize_network()
#myNet.run_step()

#print(myNet)

print("done")

testNetwork = CRI_network(axons=axons,connections=connections,inputs=inputs,config=config,target='CRI')
result = testNetwork.step()
print('my result:')
print(result)
testNetwork = CRI_network(axons=axons,connections=connections,inputs=inputs,config=config)
result = testNetwork.step()

