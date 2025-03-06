import random



############################
# Let's try a simple network
############################
class synthnet:

    def __init__(self,numAxons, numNeurons, numOutputs, minWeight, maxWeight, maxFan):
        self.numAxons = numAxons
        self.numNeurons = numNeurons
        self.numOutputs = numOutputs
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.maxFan = maxFan
        self.axonsDict = {}
        self.neuronsDict = {}

        if self.maxFan > self.numNeurons:
            raise Exception("maxFan can't be greater than the number of neurons")

        self.gen_axon_dict()
        self.gen_neuron_dict()

        self.gen_outputs()

    def gen_axon_name(self,idx):
        return 'a'+str(idx)

    def gen_neuron_name(self,idx):

        return 'n'+str(idx)
    '''
    def gen_synapse(self):
        return (self.draw_neuron(),self.draw_weight())

    def draw_neuron(self):
        idx = random.randrange(0,self.numNeurons)
        return self.gen_neuron_name(idx)

    def draw_weight(self):
        #breakpoint()
        return random.randrange(self.minWeight,self.maxWeight)
    '''
    def roll_axon(self):
        fan = random.randrange(0,self.maxFan)
        neurons = random.sample(range(0,self.numAxons), k=fan)
        neurons = [str(neuron) for neuron in neurons]
        breakpoint()
        weights = random.choices(range(self.minWeight,self.maxWeight), k=fan)
        return list(zip(neurons, weights))


    def roll_neuron(self):
        fan = random.randrange(0,self.maxFan)
        neurons = random.sample(range(0,self.numNeurons), k=fan)
        neurons = [str(neuron) for neuron in neurons]
        synapses = random.choices(range(self.minWeight,self.maxWeight), k=fan)
        return list(zip(neurons, synapses))
    def gen_axon_dict(self):
        for i in range(self.numAxons):
            self.axonsDict[self.gen_axon_name(i)] = self.roll_axon()

    def gen_neuron_dict(self):
        for i in range(self.numNeurons):
            self.neuronsDict[self.gen_neuron_name(i)] = self.roll_neuron()

    def gen_outputs(self):
        neurons = random.sample(range(0, self.numNeurons), k=self.numOutputs)
        neurons_keys = [str(neu) for neu in neurons]
        self.outputNeurons = neurons_keys

    def gen_inputs(self):
         numInputs = random.randrange(0,self.numAxons)
         return [self.gen_axon_name(axonIdx) for axonIdx in random.sample(range(0, self.numAxons), numInputs)]
