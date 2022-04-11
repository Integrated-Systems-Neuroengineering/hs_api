import _simple_sim

class CRI_network:

    def __init__(self,axons,connections,inputs,neuronModel,threshold):
        self.axons = axons
        self.connections = connections
        self.inputs = inputs
        self.neuronModel = _simple_sim.map_neuron_type_to_int(neuronModel)
        self.threshold = threshold
        #we may need more attributes to track hardware implementation

    def update_synapse(self):
        pass

    def freeRun(self,target="simpleSim"):
        """Run Network countinuously

        """
        if (target == "simpleSim"):
            _simple_sim.simulate(neuronModel,threshold,self.axons,self.connections,self.inputs)
        else:
            raise Exception("Invalid Target")

    def step(self,target="simpleSim"):
        if (target == "simpleSim"):
            _simple_sim.simulate(neuronModel,threshold,self.axons,self.connections,self.inputs)
        else:
            raise Exception("Invalid Target")
