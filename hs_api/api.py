from hs_api._simple_sim import simple_sim, map_neuron_type_to_int
#from cri_simulations import network
#from cri_simulations.utils import *
from connectome_utils.connectome import *
from bidict import bidict
import os
import copy
import logging

#handle the ordering of the elements of an entry in the neurons dictionary
synapseIdx = 0
modelIdx = 1
class perturbMagError(ValueError):
    pass


class CRI_network:
    '''
    This class represents a CRI network which initializes the network, checks hardware, 
    generates connectome, formats input, reads and writes synapse, and runs simulation steps.

    Attributes
    ----------
    userAxons : dict
        A copy of the axons dictionary provided by the user.
    userConnections : dict
        A copy of the connections dictionary provided by the user.
    config : dict
        The configuration parameters for the network.
    simpleSim : str
        A string representing the simple simulation.
    key2index : dict
        A dictionary mapping keys to indices.
    simDump : bool
        A boolean value indicating whether to dump simulation results.
    connectome : str
        A string representing the connectome of the network.
    axons : dict
        The formatted axons dictionary.
    connections : dict
        The formatted connections dictionary.
    '''

    def __init__(
        self,
        axons,
        connections,
        config,
        outputs,
        target=None,
        simDump=False,
        coreID=0
    ):
        '''
        Initialize the CRI network, validate inputs, and set up the simulation target.

        Args:
            axons (dict): Dictionary defining axon synapses.
            connections (dict): Dictionary defining neuron connections and models.
            config (dict): Global configuration including neuron types and params.
            outputs (list): List of neuron keys designated as output neurons.
            target (str, optional): Explicit override for simulation target ('CRI' or 'simpleSim').
            simDump (bool, optional): Flag to enable hardware simulation dumping. Defaults to False.
            coreID (int, optional): Hardware core index for the FPGA. Defaults to 0.
        '''
        if target:  # check if user provides an override for target
            self.target = target
        else:
            if (
                self.checkHw()
            ):  # if not check for the magic file and set to run on hardware if the magic file exists
                self.target = "CRI"
            else:
                self.target = "simpleSim"

        if self.target == "CRI":
            from hs_bridge import network

        self.outputs = outputs  # outputs is a list
        # Checking for the axon type and synapse length
        if type(axons) == dict:
            for keys in axons:
                for values in axons[keys]:
                    if not ((type(values) == tuple) and (len(values) == 2)):
                        logging.error(
                            "Each synapse should only consists of 2 elements: neuron, weight"
                        )
        else:
            logging.error("Axons should be a dictionary")
        self.userAxons = copy.deepcopy(axons)        # Checking for the connection type and synapse length
        if type(connections) == dict:
            for keys in connections:
                #print(keys)
                if connections[keys]:
                    for values in connections[keys][synapseIdx]: #synapse list is first element in tuple
                        if not ((type(values) == tuple) and (len(values) == 2)):
                            logging.error(
                                "Each synapse should only consists of 2 elements: neuron, weight"
                            )
        else:
            logging.error("Connections should be a dictionary")
        self.userConnections = copy.deepcopy(connections)

        # Checking for config type and keys
        if type(config) == dict:
            if ("neuron_type" and "global_neuron_params") in config:
                self.config = config
            else:
                logging.error(
                    "config does not contain neuron type or global neuron params"
                )
        else:
            logging.error("config should be a dictionary")

        self.simpleSim = None
        self.key2index = {}
        self.simDump = simDump
        self.connectome = None
        self.gen_connectome()
        #breakpoint()
        self.axons, self.connections = self.__format_input(
            copy.deepcopy(axons), copy.deepcopy(connections)
        )

        if self.target == "CRI":
            logging.info("Initilizing to run on hardware")
            self.connectome.pad_models()
            formatedOutputs = self.connectome.get_outputs_idx()
            print("formatedOutputs:", formatedOutputs)
            self.CRI = network(
                self.connectome,
                formatedOutputs,
                self.config,
                simDump=simDump,
                coreOveride=coreID,
            )
            self.CRI.initalize_network()
        elif self.target == "simpleSim":
            formatedOutputs = self.connectome.get_outputs_idx()
            self.simpleSim = simple_sim(
                self.config["global_neuron_params"]["v_thr"],
                self.axons,
                self.connections,
                outputs=formatedOutputs,
            )

    def set_perturbMag(self,perturbMag):
        '''
        Set the magnitude of perturbation for the current simulation target.

        Args:
            perturbMag (int): The magnitude value to apply.
        '''
        if self.target == 'simpleSim':
            self.simpleSim .set_perturbMag(perturbMag)
        elif self.target == 'CRI':
            self.CRI.set_perturbMag(perturbMag)
        else:
            logging.error('invalid target') 

    def checkHw(self):
        '''
        Checks if the magic file exists to demark that we're running on a system 
        with CRI hardware accessible.

        Returns:
            bool: True if the hardware marker exists, False otherwise.
        '''
        pathToFile = os.path.join(os.path.dirname(__file__), "magic.txt")
        return os.path.exists(pathToFile)

    def gen_connectome(self):
        '''
        Generates a connectome for the CRI network by parsing user-provided 
        axon and connection dictionaries into neuron objects.
        '''
        neuron.reset_count()  # reset static variables for neuron class
        self.connectome = connectome()

        # add axons to connectome
        for axonKey in self.userAxons:
            self.connectome.addNeuron(neuron(axonKey, "axon", axonType = "Uaxon"))
        # add neurons to connectome
        for neuronKey in self.userConnections:
            neuron_model = self.userConnections[neuronKey][modelIdx]
            self.connectome.addNeuron(
                neuron(neuronKey, "neuron", output=neuronKey in self.outputs, neuronModel=neuron_model)
            )

        # assign synapses to neurons in connectome
        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                self.connectome.connectomeDict[axonKey].addSynapse(
                    postsynapticNeuron, weight
                )
        for neuronKey in self.userConnections:
            synapses = self.userConnections[neuronKey][synapseIdx]
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                self.connectome.connectomeDict[neuronKey].addSynapse(
                    postsynapticNeuron, weight
                )

    def __format_input(self, axons, connections):
        '''
        Map symbol-based keys in axons and connections to hardware-compatible 
        integer indices.

        Args:
            axons (dict): Symbol-based axon dictionary.
            connections (dict): Symbol-based connection dictionary.

        Returns:
            tuple: (axonIndexDict, connectionIndexDict) containing formatted mapping.
        
        Raises:
            Exception: If axon and connection keys are not mutually exclusive.
        '''
        axonKeys = axons.keys()
        connectionKeys = connections.keys()
        if set(axonKeys) & set(connectionKeys):
            raise Exception("Axon and Connection Keys must be mutually exclusive")

        axonIndexDict = {}
        for idx, symbol in enumerate(axonKeys):
            axonIndexDict[idx] = axons[symbol]
        connectionIndexDict = {}
        for idx, symbol in enumerate(connectionKeys):
            connectionIndexDict[idx] = connections[symbol]

        for idx in axonIndexDict:
            for listIdx in range(len(axonIndexDict[idx])):
                oldTuple = axonIndexDict[idx][listIdx]
                newTuple = (
                    self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),
                    oldTuple[1],
                )
                axonIndexDict[idx][listIdx] = newTuple

        for idx in connectionIndexDict:
            for listIdx in range(len(connectionIndexDict[idx][0])):
                oldTuple = connectionIndexDict[idx][0][listIdx]
                newTuple = (
                    self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),
                    oldTuple[1],
                )
                connectionIndexDict[idx][0][listIdx] = newTuple
        return axonIndexDict, connectionIndexDict

    def write_synapse(self, preKey, postKey, weight):
        '''
        Updates a specific synapse weight in the connectome and synchronizes 
        it with the simulation target (hardware or software).

        Args:
            preKey (str): Key of the presynaptic neuron/axon.
            postKey (str): Key of the postsynaptic neuron.
            weight (int): New weight value to write.
        '''
        self.connectome.get_neuron_by_key(preKey).get_synapse(postKey).set_weight(
            weight
        )  
        preIndex = self.connectome.get_neuron_by_key(preKey).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preKey).get_neuron_type()

        axonFlag = True if synapseType == "axon" else False
        postIndex = self.connectome.get_neuron_by_key(postKey).get_coreTypeIdx()
        index = (
            self.connectome.get_neuron_by_key(preKey).get_synapse(postKey).get_index()
        )

        if self.target == "simpleSim":
            self.simpleSim.write_synapse(preIndex, postIndex, weight, axonFlag)
        elif self.target == "CRI":
            self.CRI.write_synapse(preIndex, index, weight, axonFlag)
        else:
            raise Exception("Invalid Target")

    def write_listofSynapses(self, preKeys, postKeys, weights):
        '''
        Writes multiple synapse weights sequentially.

        Args:
            preKeys (list): List of presynaptic keys.
            postKeys (list): List of postsynaptic keys.
            weights (list): List of corresponding weight values.
        '''
        for i in range(len(preKeys)):
            self.write_synapse(preKeys[i], postKeys[i], weights[i])

    def read_synapse(self, preKey, postKey):
        '''
        Reads the weight of a specific synapse from the current simulation target.

        Args:
            preKey (str): Key of the presynaptic neuron/axon.
            postKey (str): Key of the postsynaptic neuron.

        Returns:
            int: The current weight value of the synapse.
        '''
        preIndex = self.connectome.get_neuron_by_key(preKey).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preKey).get_neuron_type()

        axonFlag = True if synapseType == "axon" else False
        postIndex = self.connectome.get_neuron_by_key(postKey).get_coreTypeIdx()
        index = (
            self.connectome.get_neuron_by_key(preKey).get_synapse(postKey).get_index()
        )

        if self.target == "simpleSim":
            return self.simpleSim.read_synapse(preIndex, postIndex, axonFlag)
        elif self.target == "CRI":
            return self.CRI.read_synapse(preIndex, index, axonFlag)
        else:
            raise Exception("Invalid Target")

    def sim_flush(self, file):
        '''
        Flushes the hardware simulation results to a specified file. 
        Only available when the target is 'CRI'.

        Args:
            file (str): The destination file path.
        '''
        if self.target == "simpleSim":
            raise Exception("sim_flush not available for simpleSim")
        elif self.target == "CRI":
            return self.CRI.sim_flush(file)
        else:
            raise Exception("Invalid Target")

    def step(self, inputs, target="simpleSim", membranePotential=False):
        '''
        Execute a single simulation time step.

        Args:
            inputs (list): List of symbol keys for axons that are spiking.
            target (str, optional): Ignored; the class uses self.target.
            membranePotential (bool, optional): If True, returns membrane potentials 
                alongside spikes. Defaults to False.

        Returns:
            list or tuple: Spiking neuron keys, or a tuple of (potentials, spikes) 
                if membranePotential is True.
        '''
        formated_inputs = [
            self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx()
            for symbol in inputs
        ]  
        if self.target == "simpleSim":
            output, spikeOutput = self.simpleSim.step_run(formated_inputs)
            spikeOutput = [
                self.connectome.get_neuron_by_idx(spike).get_user_key()
                for spike in spikeOutput
            ]
            if membranePotential == True:
                output = [
                    (self.connectome.get_neuron_by_idx(idx).get_user_key(), potential)
                    for idx, potential in enumerate(output)
                ]
                return output, spikeOutput
            else:
                return spikeOutput

        elif self.target == "CRI":
            if self.simDump:
                return self.CRI.run_step(formated_inputs)
            else:
                if membranePotential == True:
                    output, spikeResult = self.CRI.run_step(
                        formated_inputs, membranePotential
                    )
                    spikeList = spikeResult[0]
                    spikeList = [
                        self.connectome.get_neuron_by_idx(spike[1]).get_user_key()
                        for spike in spikeList
                    ]
                    output = [
                        (self.connectome.get_neuron_by_idx(idx).get_user_key(), data[3])
                        for idx, data in enumerate(output)
                    ]  
                    return output, (spikeList, spikeResult[1], spikeResult[2])
                else:
                    spikeResult = self.CRI.run_step(formated_inputs, membranePotential)
                    spikeList = spikeResult[0]
                    spikeList = [
                        self.connectome.get_neuron_by_idx(spike[1]).get_user_key()
                        for spike in spikeList
                    ]
                    return (spikeList, spikeResult[1], spikeResult[2])
        else:
            raise Exception("Invalid Target")

    def run_cont(self, inputs):
        '''
        Run the simulation continuously over a batch of input spikes.

        Args:
            inputs (list of list): A sequence of spike inputs for multiple steps.

        Returns:
            tuple: (spikeList, breakOccurred, executionCounter) containing 
                logged spikes and status.
        '''
        formated_inputs = []
        for curInputs in inputs:
            formated_inputs.append(
                [
                    self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx()
                    for symbol in curInputs
                ]
            )  

        result = self.CRI.run_cont(formated_inputs)
        spikeList = result[0]
        if self.simDump == False:
            spikeList = [
                (spike[0], self.connectome.get_neuron_by_idx(spike[1]).get_user_key())
                for spike in spikeList
            ]
            return (spikeList, result[1], result[2])

def main():
    '''
    Test execution block for the CRI_network compiler.
    '''
    with open('test.pkl', 'rb') as f:
        data = pickle.load(f)

    compiler = CRI_network(data, 6271, outputs=[])
    compiler.create_script('test_config')

if __name__ == '__main__':
    main()