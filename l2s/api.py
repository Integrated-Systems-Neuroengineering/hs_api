from l2s._simple_sim import simple_sim, map_neuron_type_to_int
from cri_simulations import network
from cri_simulations.utils import *
from bidict import bidict
import copy

class CRI_network:

    # TODO: remove inputs
    # TODO: move target config.yaml
    def __init__(self,axons,connections,config, inputs, target = 'simpleSim', simDump = False, coreID=0):
        self.userAxons = copy.deepcopy(axons)
        self.userConnections = copy.deepcopy(connections)
        self.axons, self.connections, self.symbol2index = self.__format_input(copy.deepcopy(axons),copy.deepcopy(connections))
        self.inputs = inputs #This may later be settable via a function for continuous running networks
        self.config = config
        self.simpleSim = None
        self.target = target
        self.key2index = {}
        self.simDump = simDump
        self.connectome = None
        self.gen_connectome()
        if(self.target == 'CRI'):
            print('Initilizing to run on hardware')
            self.CRI = network(self.connectome, self.inputs, {}, self.config, simDump = simDump, coreOveride = coreID)
            self.CRI.initalize_network()
        elif(self.target == "simpleSim"):
            self.simpleSim = simple_sim(map_neuron_type_to_int(self.config['neuron_type']), self.config['global_neuron_params']['v_thr'], self.axons, self.connections, self.inputs)


    def gen_connectome(self):
        neuron.reset_count() #reset static variables for neuron class
        self.connectome = connectome()
        
        #add neurons/axons to connectome
        for axonKey in self.userAxons:
            self.connectome.addNeuron(neuron(axonKey,"axon"))
        for neuronKey in self.userConnections:
            self.connectome.addNeuron(neuron(neuronKey,"neuron"))


        #assign synapses to neurons in connectome
        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                self.connectome.connectomeDict[axonKey].addSynapse(postsynapticNeuron,weight)

        for neuronKey in self.userConnections:
            synapses = self.userConnections[neuronKey]
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                self.connectome.connectomeDict[neuronKey].addSynapse(postsynapticNeuron,weight)
        print("moo")
        

    def __format_input(self,axons,connections):
        #breakpoint()
        axonKeys =  axons.keys()
        connectionKeys = connections.keys()
        #ensure keys in axon and neuron dicts are mutually exclusive
        if (set(axonKeys) & set(connectionKeys)):
            raise Exception("Axon and Connection Keys must be mutually exclusive")
        #map those keys to indicies
        mapDict = {} #holds maping from symbols to indicies

        axonIndexDict = {}
        #construct axon dictionary with ordinal numbers as keys
        for idx, symbol in enumerate(axonKeys):
            axonIndexDict[idx] = axons[symbol]
            mapDict[symbol] = (idx,'axons')
        connectionIndexDict = {}
        #construct connections dicitonary with ordinal numbers as keys
        for idx, symbol in enumerate(connectionKeys):
            connectionIndexDict[idx] = connections[symbol]
            mapDict[symbol] = (idx,'connections')

        symbol2index = bidict(mapDict)
        
        #go through and change symbol based postsynaptic neuron values to corresponding index
        for idx in axonIndexDict:
            for listIdx in range(len(axonIndexDict[idx])):
                oldTuple = axonIndexDict[idx][listIdx]
                newTuple = (symbol2index[oldTuple[0]][0],oldTuple[1])
                axonIndexDict[idx][listIdx] = newTuple

        for idx in connectionIndexDict:
            for listIdx in range(len(connectionIndexDict[idx])):
                oldTuple = connectionIndexDict[idx][listIdx]
                newTuple = (symbol2index[oldTuple[0]][0],oldTuple[1])
                connectionIndexDict[idx][listIdx] = newTuple


        return axonIndexDict, connectionIndexDict, symbol2index
                    

    #wrap with a function to accept list input/output
    def write_synapse(self,preIndex, postIndex, weight):
        #TODO: you must update the connectome!!!
        #convert user defined symbols to indicies
        preIndex, synapseType = self.symbol2index[preIndex]
        if (synapseType == 'axons'):
            axonFlag = True
        else:
            axonFlag = False
        postIndex = self.symbol2index[postIndex][0]

        if (self.target == "simpleSim"):
            self.simpleSim.write_synapse(preIndex, postIndex, weight, axonFlag)
        elif (self.target == "CRI"):
            self.CRI.write_synapse(preIndex, postIndex, weight, axonFlag)
        else:
            raise Exception("Invalid Target")

    def read_synapse(self,preIndex, postIndex):
        #convert user defined symbols to indicies
        preIndex, synapseType = self.symbol2index[preIndex]
        if (synapseType == 'axons'):
            axonFlag = True
        else:
            axonFlag = False
        postIndex = self.symbol2index[postIndex][0]

        if (self.target == "simpleSim"):
            return self.simpleSim.read_synapse(preIndex, postIndex, axonFlag)
        elif (self.target == "CRI"):
            return self.CRI.read_synapse(preIndex, postIndex, axonFlag)
        else:
            raise Exception("Invalid Target")

    def sim_flush(self,file):
        if (self.target == "simpleSim"):
            raise Exception("sim_flush not available for simpleSim")
        elif (self.target == "CRI"):
            return self.CRI.sim_flush(file)
        else:
            raise Exception("Invalid Target")
    



    def step(self,inputs,target="simpleSim"):
        formated_inputs = [self.symbol2index[symbol][0] for symbol in inputs] #convert symbols to internal indicies 
        if (self.target == "simpleSim"):
            output = self.simpleSim.step_run(formated_inputs)
            #breakpoint()
            output = [(self.symbol2index.inverse[(idx,'connections')], potential) for idx,potential in enumerate(output)]
        elif (self.target == "CRI"):
            output = self.CRI.run_step(formated_inputs)
            if(not self.simDump):
                numNeurons = len(self.connections)
                output = [(self.symbol2index.inverse[(idx,'connections')], potential) for idx,potential in enumerate(output[:numNeurons])] #because the number of neurons will always be a perfect multiple of 16 there will be extraneous neurons at the end so we slice the output array just to get the numNerons valid neurons, due to the way we construct networks the valid neurons will be first
        else:
            raise Exception("Invalid Target")
        return output

