from l2s._simple_sim import simple_sim, map_neuron_type_to_int
from cri_simulations import network
from cri_simulations.utils import *
from bidict import bidict
import os
import copy
import logging
class CRI_network:
     
    # TODO: remove inputs
    # TODO: move target config.yaml
    def __init__(self,axons,connections,config, outputs, target = None, simDump = False, coreID=0):
        #return
        if (target): #check if user provides an override for target
            self.target = target
        else:
            if (self.checkHw()): #if not check for the magic file and set to run on hardware if the magic file exists
                self.target = 'CRI'
            else:
                self.target = 'simpleSim'

        self.outputs = outputs #outputs is a list

        #Checking for the axon type and synapse length
        if (type(axons)==dict):
            for keys in axons:
                for values in axons[keys]:
                    if(not((type(values)==tuple) and (len(values)==2))): 
                        logging.error('Each synapse should only consists of 2 elements: neuron, weight')
        else:
            logging.error('Axons should be a dictionary')
        self.userAxons = copy.deepcopy(axons)
        #Checking for the connection type and synapse length
        if (type(connections)==dict):
            for keys in connections:
                for values in connections[keys]:
                    if(not((type(values)==tuple) and (len(values)==2))):
                        logging.error('Each synapse should only consists of 2 elements: neuron, weight')
        else:
            logging.error('Connections should be a dictionary')
        self.userConnections = copy.deepcopy(connections)
        


        #Checking for config type and keys
        if (type(config)==dict):
            if ('neuron_type' and 'global_neuron_params') in config:
                self.config = config
            else: 
                logging.error('config does not contain neuron type or global neuron params')
        else:
            logging.error('config should be a dictionary')
        

        self.simpleSim = None
        self.key2index = {}
        self.simDump = simDump
        self.connectome = None
        self.gen_connectome()
        self.axons, self.connections = self.__format_input(copy.deepcopy(axons),copy.deepcopy(connections))
         
        if(self.target == 'CRI'):
            logging.info('Initilizing to run on hardware')
            formatedOutputs = self.connectome.get_core_outputs_idx(coreID)
            self.CRI = network(self.connectome, formatedOutputs, self.config, simDump = simDump, coreOveride = coreID)
            self.CRI.initalize_network()
        elif(self.target == "simpleSim"):
            formatedOutputs = self.connectome.get_outputs_idx()
            self.simpleSim = simple_sim(map_neuron_type_to_int(self.config['neuron_type']), self.config['global_neuron_params']['v_thr'], self.axons, self.connections, outputs = formatedOutputs)
        #breakpoint()
        print("initialized")

    def checkHw(self):
        """check if the magic file exists to demark that were running on a system with CRI hardware accessible
        """
        pathToFile = os.path.join(os.path.dirname(__file__), "magic.txt")
        return os.path.exists(pathToFile)

    def gen_connectome(self):
        neuron.reset_count() #reset static variables for neuron class
        self.connectome = connectome()
        
        #add neurons/axons to connectome
        for axonKey in self.userAxons:
            self.connectome.addNeuron(neuron(axonKey,"axon"))
        print("added axons to connectome")
        for neuronKey in self.userConnections:
            self.connectome.addNeuron(neuron(neuronKey,"neuron", output = neuronKey in self.outputs ))
        print("added neurons to connectome")

        #assign synapses to neurons in connectome
        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                self.connectome.connectomeDict[axonKey].addSynapse(postsynapticNeuron,weight)
        print("added axon synpases")
        for neuronKey in self.userConnections:
            synapses = self.userConnections[neuronKey]
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                self.connectome.connectomeDict[neuronKey].addSynapse(postsynapticNeuron,weight)
        print("added neuron synapses")
        
        print("generated Connectome")

    def __format_input(self,axons,connections):
        #breakpoint()
        axonKeys =  axons.keys()
        connectionKeys = connections.keys()
        #ensure keys in axon and neuron dicts are mutually exclusive
        if (set(axonKeys) & set(connectionKeys)):
            raise Exception("Axon and Connection Keys must be mutually exclusive")

        axonIndexDict = {}
        #construct axon dictionary with ordinal numbers as keys
        for idx, symbol in enumerate(axonKeys):
            axonIndexDict[idx] = axons[symbol]
        connectionIndexDict = {}
        #construct connections dicitonary with ordinal numbers as keys
        for idx, symbol in enumerate(connectionKeys):
            connectionIndexDict[idx] = connections[symbol]
        
        #go through and change symbol based postsynaptic neuron values to corresponding index
        for idx in axonIndexDict:
            for listIdx in range(len(axonIndexDict[idx])):
                oldTuple = axonIndexDict[idx][listIdx]
                newTuple = (self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),oldTuple[1])
                axonIndexDict[idx][listIdx] = newTuple

        for idx in connectionIndexDict:
            for listIdx in range(len(connectionIndexDict[idx])):
                oldTuple = connectionIndexDict[idx][listIdx]
                newTuple = (self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),oldTuple[1])
                connectionIndexDict[idx][listIdx] = newTuple
        return axonIndexDict, connectionIndexDict

    #wrap with a function to accept list input/output
    def write_synapse(self,preKey, postKey, weight):
        #breakpoint()
        self.connectome.get_neuron_by_key(preKey).get_synapse(postKey).set_weight(weight) #update synapse weight in the connectome
        #TODO: you must update the connectome!!!
        #convert user defined symbols to indicies
        
        preIndex = self.connectome.get_neuron_by_key(preKey).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preKey).get_neuron_type()
        breakpoint()
        if (synapseType == 'axon'):
            axonFlag = True
        else:
            axonFlag = False

        postIndex = self.connectome.get_neuron_by_key(postKey).get_coreTypeIdx()

        if (self.target == "simpleSim"):
            self.simpleSim.write_synapse(preIndex, postIndex, weight, axonFlag)
        elif (self.target == "CRI"):
            self.CRI.write_synapse(preIndex, postIndex, weight, axonFlag)
        else:
            raise Exception("Invalid Target")
    
    #Update a list of synapses
    def write_listofSynapses(self, preKeys, postKeys, weights):
        for i in range(len(preKeys)):
            self.write_synapse(preKeys[i],postKeys[i],weights[i])

    def read_synapse(self,preIndex, postIndex):
        #convert user defined symbols to indicies
        preIndex = self.connectome.get_neuron_by_key(preIndex).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preIndex).get_neuron_type()
        if (synapseType == 'axons'):
            axonFlag = True
        else:
            axonFlag = False
        postIndex = self.connectome.get_neuron_by_key(postIndex).get_coreTypeIdx()

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
    
    def step(self,inputs,target="simpleSim",membranePotential=False):
        #formated_inputs = [self.symbol2index[symbol][0] for symbol in inputs] #convert symbols to internal indicies
        formated_inputs = [self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx() for symbol in inputs] #convert symbols to internal indicies 
        if (self.target == "simpleSim"):
            output, spikeOutput = self.simpleSim.step_run(formated_inputs)
            output = [(self.connectome.get_neuron_by_idx(idx).get_user_key(), potential) for idx,potential in enumerate(output)]
            spikeOutput = [self.connectome.get_neuron_by_idx(spike).get_user_key() for spike in spikeOutput]
            if (membranePotential == True):
                return output, spikeOutput
            else:
                return spikeOutput

        elif (self.target == "CRI"):
            
            if(self.simDump):
                return self.CRI.run_step(formated_inputs)
            else:
                if (membranePotential == True):
                    output, spikeList = self.CRI.run_step(formated_inputs, membranePotential)
                    #we currently ignore the run execution counter
                    spikeList = [self.connectome.get_neuron_by_idx(spike[1]).get_user_key() for spike in spikeList]
                    numNeurons = len(self.connections)
                    #we currently only print the membrane potential, not the other contents of the spike packet
                    output = [(self.connectome.get_neuron_by_idx(idx).get_user_key(), data[3]) for idx,data in enumerate(output[:numNeurons])] #because the number of neurons will always be a perfect multiple of 16 there will be extraneous neurons at the end so we slice the output array just to get the numNerons valid neurons, due to the way we construct networks the valid neurons will be first
                    return output, spikeList
                else: 
                    spikeList = self.CRI.run_step(formated_inputs, membranePotential)
                    spikeList = [self.connectome.get_neuron_by_idx(spike[1]).get_user_key() for spike in spikeList]
                    return spikeList
        else:
            raise Exception("Invalid Target")

        

