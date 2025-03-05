from hs_api.api import CRI_network
import sys
import subprocess
import time
import pickle
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
            self.neuronsDict[self.gen_neuron_name(i)] = self.roll_neuron(i)

    def gen_outputs(self):
        neurons = random.sample(range(0, self.numNeurons), k=self.numOutputs)
        neurons_keys = [str(neu) for neu in neurons]
        self.outputNeurons = neurons_keys

    def gen_inputs(self):
         numInputs = random.randrange(0,self.numAxons)
         return [self.gen_axon_name(axonIdx) for axonIdx in random.sample(range(0, self.numAxons), numInputs)]

swTest = True
membranePotential = True
fixedSeed = True
if fixedSeed:
    random.seed(237)

synth = synthnet(3,3,-2,10,2)
breakpoint()
#continuous execution is more or less deprecated at the moment
cont_exec = False
sim_dump = True
#Initialize a CRI_network object for interacting with the hardware and the software

#hardwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config,target='CRI', outputs = synth.neuronsDict.keys(),coreID=1, perturbMag = 0, simDump = sim_dump)
softwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config, outputs = synth.neuronsDict.keys(), target='simpleSim', perturbMag = 16)

#a = synth.gen_inputs()
#b = synth.gen_inputs()

#curInputs = [a,b]

#Execute the network stepwise in the hardware and the simulator
steps = 9
stepInputs = []
stepSpikes = []
if membranePotential:
    stepPotential = []
for i in range(steps):
    currInput = synth.gen_inputs()
    stepInputs.append(currInput)
    if swTest and not sim_dump:
    #    spikes = hardwareNetwork.run_cont(curInputs)
    #    breakpoint()
        #hwSpike = hardwareNetwork.step(currInput, membranePotential=False)
        if membranePotential:
            #breakpoint()
            swMem, swSpike = softwareNetwork.step(currInput, membranePotential=True)
            stepPotential = stepPotential+[(i,membrane[0],membrane[1]) for membrane in swMem]
        else:
            swSpike = softwareNetwork.step(currInput, membranePotential=False)
        stepSpikes= stepSpikes+[(i,spike) for spike in swSpike]
        print("timestep: "+str(i)+":")
        #print("hardware result: ")
        #print(hwSpike)
        #format: (timestep, fired neuron)
        print(stepSpikes)

cont_exec = False
if cont_exec:
    breakpoint()
    spikes, latency, access = hardwareNetwork.run_cont(stepInputs)
    breakpoint()
    print('latency: '+str(latency))
    print('access: '+str(access))
else:
    spikes=[]
    if membranePotential:
        potential = []
    i = 0
    for currInput in stepInputs:
        #breakpoint()
        if membranePotential and not sim_dump:
            hwMem, spikeResult = hardwareNetwork.step(currInput, membranePotential=True)
            spike, latency, hbmAcc = spikeResult
            potential= potential+[(i,membrane[0],membrane[1]) for membrane in hwMem]
        elif sim_dump:
            hardwareNetwork.step(currInput, membranePotential = False)
        else:
            spike, latency, hbmAcc = hardwareNetwork.step(currInput, membranePotential=False)

        if not sim_dump:
            spikes= spikes+[(i,currSpike) for currSpike in spike]
            breakpoint()
            print("timestep: "+str(i)+":")
            print("Latency: "+str(latency))
            print("hbmAcc: "+str(hbmAcc))
            #print("hardware result: ")
            #print(hwSpike)
            #breakpoint()
            print(spikes)
            i = i+1

#print(synth.axonsDict)
print(spikes)
breakpoint()

if sim_dump:
    hardwareNetwork.sim_flush('seed.txt')

if swTest and not sim_dump:

    breakpoint()
    #continuous execution is more or less deprecated at the moment
    cont_exec = False
    sim_dump = True
    #Initialize a CRI_network object for interacting with the hardware and the software

    #hardwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config,target='CRI', outputs = synth.neuronsDict.keys(),coreID=1, perturbMag = 0, simDump = sim_dump)
    softwareNetwork = CRI_network(axons=synth.axonsDict,connections=synth.neuronsDict,config=config, outputs = synth.neuronsDict.keys(), target='simpleSim', perturbMag = 16)

    #a = synth.gen_inputs()
    #b = synth.gen_inputs()

    #curInputs = [a,b]

    #Execute the network stepwise in the hardware and the simulator
    steps = 9
    stepInputs = []
    stepSpikes = []
    if membranePotential:
        stepPotential = []
    for i in range(steps):
        currInput = synth.gen_inputs()
        stepInputs.append(currInput)
        if swTest and not sim_dump:
        #    spikes = hardwareNetwork.run_cont(curInputs)
        #    breakpoint()
            #hwSpike = hardwareNetwork.step(currInput, membranePotential=False)
            if membranePotential:
                #breakpoint()
                swMem, swSpike = softwareNetwork.step(currInput, membranePotential=True)
                stepPotential = stepPotential+[(i,membrane[0],membrane[1]) for membrane in swMem]
            else:
                swSpike = softwareNetwork.step(currInput, membranePotential=False)
            stepSpikes= stepSpikes+[(i,spike) for spike in swSpike]
            print("timestep: "+str(i)+":")
            #print("hardware result: ")
            #print(hwSpike)
            #format: (timestep, fired neuron)
            print(stepSpikes)

    cont_exec = False
    if cont_exec:
        breakpoint()
        spikes, latency, access = hardwareNetwork.run_cont(stepInputs)
        breakpoint()
        print('latency: '+str(latency))
        print('access: '+str(access))
    else:
        spikes=[]
        if membranePotential:
            potential = []
        i = 0
        for currInput in stepInputs:
            #breakpoint()
            if membranePotential and not sim_dump:
                hwMem, spikeResult = hardwareNetwork.step(currInput, membranePotential=True)
                spike, latency, hbmAcc = spikeResult
                potential= potential+[(i,membrane[0],membrane[1]) for membrane in hwMem]
            elif sim_dump:
                hardwareNetwork.step(currInput, membranePotential = False)
            else:
                spike, latency, hbmAcc = hardwareNetwork.step(currInput, membranePotential=False)

            if not sim_dump:
                spikes= spikes+[(i,currSpike) for currSpike in spike]
                breakpoint()
                print("timestep: "+str(i)+":")
                print("Latency: "+str(latency))
                print("hbmAcc: "+str(hbmAcc))
                #print("hardware result: ")
                #print(hwSpike)
                #breakpoint()
                print(spikes)
                i = i+1

    #print(synth.axonsDict)
    print(spikes)
    breakpoint()

    if sim_dump:
        hardwareNetwork.sim_flush('seed.txt')

    if swTest and not sim_dump:
        breakpoint()
        print('results _____________________-')

        print(set(spikes)==set(stepSpikes))
        print(set(stepSpikes)-set(spikes))
        spikes.sort()
        stepSpikes.sort()
        #print(stepSpikes)
        #print(spikes)
        print(len(stepSpikes))
        #print(stepSpikes)
        print(len(spikes))

        print(len(set(stepSpikes)-set(spikes)))
        print(len(set(spikes)-set(stepSpikes)))
        #print(spikes)
        if membranePotential:
            print(len(set(stepPotential)-set(potential)))
        #print('up to match: ')
        #smallStep = filter(lambda spike:spike[0]<8, stepSpikes)
        #smallCont = filter(lambda spike:spike[0]<8, spikes)
        #print(set(smallCont)==set(smallStep))
        #print(synth.axonsDict)

        #print(hwResult)
        #print("timestep: "+str(i)+" end")
        #magicBreak = False
        #if (set(hwSpike) != set(swSpike)):
        #    print("Incongruent Spike Results Detected")
            #breakpoint()
        #    magicBreak = True
        #else:
        #    print("Spike Results match simulator")
        #potentialFlag = False
        #for idx in range(len(swResult)):
        #    if(swResult[idx][1] != hwResult[idx][1]):
        #        print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
        #        potentialFlag = True
        #        magicBreak = True
        #if potentialFlag:
        #    print("Incongruent Membrane Potential Results Detected")
        #else:
        #    print("Membrane Potentials Match")
        #if magicBreak:
        #    breakpoint()

if __name__ == '__main__':
    main()
