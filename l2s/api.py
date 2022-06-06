from l2s._simple_sim import simple_sim, map_neuron_type_to_int
from cri_simulations import network
from bidict import bidict
import copy

class CRI_network:

    # TODO: remove inputs
    # TODO: move target config.yaml
    def __init__(self,axons,connections,config, inputs, target = 'simpleSim'):
        self.userAxons = copy.deepcopy(axons)
        self.userConnections = copy.deepcopy(connections)
        self.axons, self.connections, self.symbol2index = self.__format_input(self.userAxons,self.userConnections)
        self.inputs = inputs #This may later be settable via a function for continuous running networks
        self.config = config
        self.simpleSim = None
        self.target = target
        self.key2index = {}
        if(self.target == 'CRI'):
            print('Initilizing to run on hardware')
            self.CRI = network(self.axons, self.connections, self.inputs, {}, self.config)
            self.CRI.initalize_network()
        elif(self.target == "simpleSim"):
            self.simpleSim = simple_sim(map_neuron_type_to_int(self.config['neuron_type']), self.config['global_neuron_params']['v_thr'], self.axons, self.connections, self.inputs)

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



    def step(self,inputs,target="simpleSim"):
        if (self.target == "simpleSim"):
            output = self.simpleSim.step_run(inputs)
            #breakpoint()
            output = [(self.symbol2index.inverse[(idx,'connections')], potential) for idx,potential in enumerate(output)]
        elif (self.target == "CRI"):
            output = self.CRI.run_step(inputs)
            numNeurons = len(self.connections)
            output = [(self.symbol2index.inverse[(idx,'connections')], potential) for idx,potential in enumerate(output[:numNeurons])] #because the number of neurons will always be a perfect multiple of 16 there will be extraneous neurons at the end so we slice the output array just to get the numNerons valid neurons, due to the way we construct networks the valid neurons will be first
        else:
            raise Exception("Invalid Target")
        return output

