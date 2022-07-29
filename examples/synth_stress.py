from l2s.api import CRI_network
import sys
import subprocess
import time
import pickle
import random
#Define a configuration dictionary
config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 100

############################
# Let's try a simple network
############################
class synthnet:

    def __init__(self,numAxons, numNeurons, minWeight, maxWeight, maxFan):
        self.numAxons = numAxons
        self.numNeurons = numNeurons
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.maxFan = maxFan
        self.axonsDict = {}
        self.neuronsDict = {}
        self.gen_axon_dict()
        self.gen_neuron_dict()

    def gen_axon_name(self,idx):
        return 'a'+str(idx)

    def gen_neuron_name(self,idx):
        return 'n'+str(idx)

    def gen_synapse(self):
        return (self.draw_neuron(),self.draw_weight())

    def draw_neuron(self):
        idx = random.randrange(0,self.numNeurons)
        return self.gen_neuron_name(idx)

    def draw_weight(self):
        #breakpoint()
        return random.randrange(self.minWeight,self.maxWeight)

    def roll_axon(self):
        fan = random.randrange(0,self.maxFan)
        return [self.gen_synapse() for i in range(fan)]

    def roll_neuron(self):
        fan = random.randrange(0,self.maxFan)
        return [self.gen_synapse() for i in range(fan)]

    def gen_axon_dict(self):
        for i in range(self.numAxons):
            self.axonsDict[self.gen_axon_name(i)] = self.roll_axon()

    def gen_neuron_dict(self):
         for i in range(self.numNeurons):
            self.neuronsDict[self.gen_neuron_name(i)] = self.roll_neuron()

    def gen_inputs(self):
         numInputs = random.randrange(0,self.numAxons)
         return [self.gen_axon_name(axonIdx) for axonIdx in random.sample(range(0, self.numAxons), numInputs)]

synth = synthnet(1000,10000,-10,10,1000)
#breakpoint()
#Initialize a CRI_network object for interacting with the hardware and the software
hardwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config,target='CRI', outputs = synth.neuronsDict.keys())
softwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config, outputs = synth.neuronsDict.keys(), target='simpleSim')

#Execute the network stepwise in the hardware and the simulator
steps = 100
for i in range(steps):
    currInput = synth.gen_inputs()
    hwResult, hwSpike = hardwareNetwork.step(currInput, membranePotential=True)
    swResult, swSpike = softwareNetwork.step(currInput, membranePotential=True)
    print("timestep: "+str(i)+":")
    print("hardware result: ")
    print(hwSpike)
    print(hwResult)
    print("timestep: "+str(i)+" end")
    for idx in range(len(swResult)):
        if(swResult[idx][1] != hwResult[idx][1]):
            print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
            breakpoint()
