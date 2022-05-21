from l2s._simple_sim import simple_sim, map_neuron_type_to_int
from cri_simulations import network

class CRI_network:

    def __init__(self,axons,connections,inputs,config,target = 'simpleSim'):
        self.axons = axons
        self.connections = connections
        self.inputs = inputs
        self.config = config
        #self.neuronModel = _simple_sim.map_neuron_type_to_int(neuronModel)
        #self.threshold = threshold
        self.simpleSim = None
        self.target = target
        #we may need more attributes to track hardware implementation
        if(self.target == 'CRI'):
            print('Initilizing to run on hardware')
            self.CRI = network(self.axons, self.connections, self.inputs, {}, self.config)
            self.CRI.initalize_network()
        elif(self.target == "simpleSim"):
            self.simpleSim = simple_sim(map_neuron_type_to_int(self.config['neuron_type']), self.config['global_neuron_params']['v_thr'], self.axons, self.connections, self.inputs)

    def write_synapse(self,preIndex, postIndex, weight, axonFlag = False):
        if (self.target == "simpleSim"):
            self.simpleSim.write_synapse(preIndex, postIndex, weight, axonFlag)
        elif (self.target == "CRI"):
            self.CRI.write_synapse(preIndex, postIndex, weight, axonFlag)
        else:
            raise Exception("Invalid Target")

    def read_synapse(self,preIndex, postIndex, axonFlag = False):
        if (self.target == "simpleSim"):
            print("moo")
        elif (self.target == "CRI"):
            return self.CRI.read_synapse(preIndex, postIndex, axonFlag)
        else:
            raise Exception("Invalid Target")

    def step(self,target="simpleSim"):
        if (self.target == "simpleSim"):
            return self.simpleSim.step_run()
        elif (self.target == "CRI"):
            return self.CRI.run_step()
        else:
            raise Exception("Invalid Target")

