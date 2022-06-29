import copy
import logging

class synapse:
    #postsynapticNeuron = None
    #weight = None
    #synapseType = None
    def __init__(self,presynapticNeuron, postsynapticNeuron, weight):
        self.presynapticNeuron = presynapticNeuron
        self.postsynapticNeuron = postsynapticNeuron
        self.weight = weight
        self.synapseType = None
    def get_postsynapticNeuron(self):
        return self.postsynapticNeuron
    def set_synapseType(self):
        #TODO: check for None
        if (self.postsynapticNeuron and self.presynapticNeuron):
            if(self.postsynapticNeuron.get_core() != self.presynapticNeuron.get_core()):
                self.synapseType = "hetero"
            else:
                self.synapseType = "homo"
        else:
            raise Exception("core is None")
    def get_weight(self):
        return self.weight
    def set_weight(self, newWeight):
        self.weight = newWeight
    def __repr__(self):
        return self.obj2string()
    def __str__(self):
        return self.obj2string()
    def obj2string(self):
        return "("+str(self.postsynapticNeuron.get_user_key())+", "+str(self.weight)+", "+str(self.synapseType)+")"

class neuron:
    globalAxonCount = 0 #STATIC total number of axons
    globalNeuronCount = 0 #STATIC total number of nonAxon neurons
    globalCount = 0 #STATIC total number of both axons and neurons

    #globalIdx = None #Index unique among all neurons/axons
    #globalTypeIdx = None #Index unique among neurons of the same type
    #userKey = None
    #coreIdx = None ##Index unique among all neurons/axons in local core
    #coreTypeIdx = None #Index unique among neurons of the same type in local core
    #core = None
    #neuronType = None
    #synapses = None
    def __init__(self, userKey, neuronType):
        self.userKey = userKey
        self.neuronType = neuronType
        if (neuronType == 'axon'):
            self.globalTypeIdx = copy.deepcopy(neuron.globalAxonCount)
            neuron.globalAxonCount += 1
        if (neuronType == 'neuron'):
            self.globalTypeIdx = copy.deepcopy(neuron.globalNeuronCount)
            neuron.globalNeuronCount += 1
        self.globalIdx = copy.deepcopy(neuron.globalCount)
        neuron.globalCount += 1
        self.core = 0
        self.coreTypeIdx = self.globalTypeIdx
        self.coreIdx = self.globalIdx
        self.synapses = []
    @classmethod
    def reset_count(cls):
        #resets counting variables for neuron class
        #breakpoint()
        cls.globalAxonCount = 0
        cls.globalNeuronCount = 0
        cls.globalCount = 0
    def addSynapse(self,postsynapticNeuron,weight):
        newSynapse = synapse(self,postsynapticNeuron,weight)
        self.synapses.append(newSynapse)
    def get_user_key(self):
        return self.userKey
    def get_synapses(self):
        return self.synapses
    def get_neuron_type(self):
        return self.neuronType
    def set_core(self,core):
        self.core = core
    def get_core(self):
        return self.core
    def set_coreIdx(self,coreIdx):
        self.coreIdx = coreIdx
    def get_coreIdx(self):
        return self.coreIdx
    def set_coreTypeIdx(self,coreTypeIdx):
        self.coreTypeIdx = coreTypeIdx
    def get_coreTypeIdx(self):
        return self.coreTypeIdx
    def set_globalIdx(self,globalIdx):
        self.globalIdx = globalIdx
    def get_globalIdx(self):
        return self.globalIdx
    def get_synapse(self, postsynapticKey):
        canidateSynapses = [currSynapse for currSynapse in self.synapses if currSynapse.get_postsynapticNeuron().get_user_key()==postsynapticKey]
        if (len(canidateSynapses) == 0):
            logging.error("Neuron has no synapse to specified postsynaptic neuron")
        else:
            if (len(canidateSynapses) != 1):
                logging.error("Neuron has two synapses to the same postsynaptic neuron")
            else:
                return canidateSynapses[0]
    def set_synapseTypes(self):
        for synapse in self.synapses:
            synapse.set_synapseType()
    def __repr__(self):
        return self.obj2string()
    def __str__(self):
        return self.obj2string()
    def obj2string(self):
        string =  "userID: "+str(self.userKey)+", GlobalID: "+str(self.globalIdx)+", Core Assignment: "+str(self.core)+", Synapses: "
        for currSynapse in self.synapses:
            string = string + str(currSynapse)
        return string

class connectome:
    #connectomeDict = None
    def __init__(self):
        self.connectomeDict = {}
    def addNeuron(self, neuron):
        self.connectomeDict[neuron.userKey] = neuron
    def __repr__(self):
        return self.obj2string()
    def __str__(self):
        return self.obj2string()
    def get_neuron_by_key(self, neuronKey):
        return self.connectomeDict[neuronKey]
    def obj2string(self):
        string = ""
        for key in self.connectomeDict:
            string = string + str(self.connectomeDict[key])+"\n"
        return string
    def get_axons(self):
        axons = {}
        for key in self.connectomeDict:
            if (self.connectomeDict[key].get_neuron_type() == 'axon'):
                axons[key] = self.connectomeDict[key]
        return axons
    def get_neurons(self):
        neurons = {}
        for key in self.connectomeDict:
            if (self.connectomeDict[key].get_neuron_type() == 'neuron'):
                neurons[key] = self.connectomeDict[key]
        return neurons
    def get_merged_neurons(self):
        mergedNeurons = {}
        for key in self.connectomeDict:
            mergedNeurons[self.connectomeDict[key].get_globalIdx()] = self.connectomeDict[key]
        return mergedNeurons

    def get_part_format(self):
        mergedNeurons = self.get_merged_neurons()
        networkConnectivity = []
        for key in mergedNeurons.keys(): #Nishant's partitioning code expects a specific format so we convert to that format
            networkConnectivity.append([copy.deepcopy(synapse.get_postsynapticNeuron().get_globalIdx()) for synapse in mergedNeurons[key].get_synapses()])
        return networkConnectivity

    def apply_partition(self,membership):
        #breakpoint()
        mergedNeurons = self.get_merged_neurons()
        for key in membership.keys():
            mergedNeurons[key].set_core(membership[key])
        #breakpoint()
        for neuronKey in self.connectomeDict.keys():
            self.connectomeDict[neuronKey].set_synapseTypes()

       

        
