import _simple_sim

class CRI_network:

    def __init__(self,axons,connections,inputs,neuronModel,threshold):
        self.axons = axons
        self.connections = connections
        self.inputs = inputs
        self.neuronModel = _simple_sim.map_neuron_type_to_int(neuronModel)
        self.threshold = threshold
        self.simpleSim = None
        #we may need more attributes to track hardware implementation

    def update_synapse(self):
        pass

    def freeRun(self,target="simpleSim"):
        """Run Network countinuously

        """
        if (target == "simpleSim"):
            if(not self.simpleSim):
                _simple_sim.simple_sim(self.neuronModel, self.threshold, self.axons, self.connections, self.inputs)

            self.simpleSim.free_run()
        else:
            raise Exception("Invalid Target")

    def step(self,target="simpleSim"):
        if (target == "simpleSim"):
            if(not self.simpleSim):
                _simple_sim.simple_sim(self.neuronModel, self.threshold, self.axons, self.connections, self.inputs)

            _simple_sim.step_run()
        else:
            raise Exception("Invalid Target")
