import _simple_sim

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

    def update_synapse(self):
        pass

    def freeRun(self):
        """Run Network countinuously

        """
        if (self.target == "simpleSim"):
            if(not self.simpleSim):
                _simple_sim.simple_sim(_simple_sim.map_neuron_type_to_int(self.config['neuron_type']), self.config['threshold'], self.axons, self.connections, self.inputs)

            self.simpleSim.free_run()
        else:
            raise Exception("Invalid Target")

    def step(self,target="simpleSim"):
        if (self.target == "simpleSim"):
            if(not self.simpleSim):
                _simple_sim.simple_sim(_simple_sim.map_neuron_type_to_int(self.config['neuron_type']), self.config['threshold'], self.axons, self.connections, self.inputs)

            _simple_sim.step_run()
        else:
            raise Exception("Invalid Target")
