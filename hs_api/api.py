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
    """
    This class represents a CRI network which initializes the network, checks hardware, generates connectome,
    formats input, reads and writes synapse, and runs simulation steps.

    Attributes
    ----------
    userAxons : dict
        A copy of the axons dictionary provided by the user.
    userConnections : dict
        A copy of the connections dictionary provided by the user.
    config : dict
        The configuration parameters for the network.
    perturb : bool
        A boolean value indicating whether to perturb the network.
    perturbMag : int
        The magnitude of perturbation.
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


    Examples
    --------
    >>> axons = {'axon1': [('neuron1', 1), ('neuron2', 2)]}
    >>> connections = {'neuron1': [('axon1', 1)], 'neuron2': [('axon1', 2)]}
    >>> config = {'neuron_type': 'type1', 'global_neuron_params': {'v_thr': 1.0}}
    >>> outputs = ['output1', 'output2']
    >>> network = CRI_network(axons, connections, config, outputs)
    """

    # TODO: remove inputs
    # TODO: move target config.yaml
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
        # return
        #breakpoint()
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

        #self.perturb = perturb
        #if perturbMag > 35 or perturbMag < 0 or not isinstance(perturbMag, int):
        #    raise perturbMagError('bad perturbMag')
        #self.perturbMag = perturbMag
        #if perturbMag:
        #    if perturbMag > 16:
        #        logging.error("perturbMag must be less than 16")
        #self.leak = leak
        #if leak > 2**6:
        #    logging.error("Leak must be less than two to the sixth")
        #self.perturbMag = perturbMag
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
            ##neurons are default to core ID 0, need to be fixed in the connectome to assign correct coreIdx to neurons
            # formatedOutputs = self.connectome.get_core_outputs_idx(coreID)
            formatedOutputs = self.connectome.get_outputs_idx()
            print("formatedOutputs:", formatedOutputs)
            self.CRI = network(
                self.connectome,
                formatedOutputs,
                self.config,
                simDump=simDump,
                coreOveride=coreID,
            )
            #breakpoint()
            self.CRI.initalize_network()
        elif self.target == "simpleSim":
            formatedOutputs = self.connectome.get_outputs_idx()
            self.simpleSim = simple_sim(
                self.config["global_neuron_params"]["v_thr"],
                self.axons,
                self.connections,
                outputs=formatedOutputs,
                #perturbMag=self.perturbMag,
                #leak=self.leak,
            )
        # breakpoint()
        # print("initialized")

    def set_perturbMag(self,perturbMag):
        if self.target == 'simpleSim':
            self.simpleSim .set_perturbMag(perturbMag)
        elif self.target == 'CRI':
            self.CRI.set_perturbMag(perturbMag)
        else:
            logging.error('invalid target') 

    def checkHw(self):
        """
        Checks if the magic file exists to demark that we're running on a system with CRI hardware accessible.

        Returns
        -------
        bool
            True if the magic file exists, False otherwise.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.checkHw()
        True
        """
        pathToFile = os.path.join(os.path.dirname(__file__), "magic.txt")
        return os.path.exists(pathToFile)

    def gen_connectome(self):
        """
        Generates a connectome for the CRI network.

        Notes
        -----
        The function resets the count of neurons and creates a new connectome. It then adds neurons/axons and assigns synapses to them.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.gen_connectome()
        """

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
        # print("added neurons to connectome")

        # assign synapses to neurons in connectome
        for axonKey in self.userAxons:
            synapses = self.userAxons[axonKey]
            for axonSynapse in synapses:
                weight = axonSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[axonSynapse[0]]
                self.connectome.connectomeDict[axonKey].addSynapse(
                    postsynapticNeuron, weight
                )
        # print("added axon synpases")
        for neuronKey in self.userConnections:
            #breakpoint()
            synapses = self.userConnections[neuronKey][synapseIdx]
            #breakpoint()
            for neuronSynapse in synapses:
                weight = neuronSynapse[1]
                postsynapticNeuron = self.connectome.connectomeDict[neuronSynapse[0]]
                self.connectome.connectomeDict[neuronKey].addSynapse(
                    postsynapticNeuron, weight
                )
        # print("added neuron synapses")

        # print("generated Connectome")

    def __format_input(self, axons, connections):
        """
        Formats the input axons and connections.

        Parameters
        ----------
        axons : dict
            A dictionary containing axon data.
        connections : dict
            A dictionary containing connection data.

        Returns
        -------
        tuple
            A tuple containing two dictionaries for formatted axons and connections.

        Raises
        ------
        Exception
            If Axon and Connection Keys are not mutually exclusive.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> axons = {'axon1': [('neuron1', 1), ('neuron2', 2)]}
        >>> connections = {'neuron1': [('axon1', 1)], 'neuron2': [('axon1', 2)]}
        >>> network.__format_input(axons, connections)
        """

        # breakpoint()
        axonKeys = axons.keys()
        connectionKeys = connections.keys()
        # ensure keys in axon and neuron dicts are mutually exclusive
        if set(axonKeys) & set(connectionKeys):
            raise Exception("Axon and Connection Keys must be mutually exclusive")

        axonIndexDict = {}
        # construct axon dictionary with ordinal numbers as keys
        for idx, symbol in enumerate(axonKeys):
            axonIndexDict[idx] = axons[symbol]
        connectionIndexDict = {}
        # construct connections dicitonary with ordinal numbers as keys
        for idx, symbol in enumerate(connectionKeys):
            connectionIndexDict[idx] = connections[symbol]

        # go through and change symbol based postsynaptic neuron values to corresponding index
        for idx in axonIndexDict:
            for listIdx in range(len(axonIndexDict[idx])):
                oldTuple = axonIndexDict[idx][listIdx]
                newTuple = (
                    self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),
                    oldTuple[1],
                )
                axonIndexDict[idx][listIdx] = newTuple

        for idx in connectionIndexDict:
            #breakpoint()
            for listIdx in range(len(connectionIndexDict[idx][0])):
                oldTuple = connectionIndexDict[idx][0][listIdx]
                newTuple = (
                    self.connectome.get_neuron_by_key(oldTuple[0]).get_coreTypeIdx(),
                    oldTuple[1],
                )
                connectionIndexDict[idx][0][listIdx] = newTuple
        #breakpoint()
        return axonIndexDict, connectionIndexDict

    # wrap with a function to accept list input/output
    def write_synapse(self, preKey, postKey, weight):
        """
        Writes a synapse to the connectome.

        Parameters
        ----------
        preKey : str
            A string representing the key of the presynaptic neuron.
        postKey : str
            A string representing the key of the postsynaptic neuron.
        weight : int
            An integer representing the weight of the synapse.

        Raises
        ------
        Exception
            If the target is not valid ("simpleSim" or "CRI").

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.write_synapse('axon1', 'neuron1', 1)
        """

        self.connectome.get_neuron_by_key(preKey).get_synapse(postKey).set_weight(
            weight
        )  # update synapse weight in the connectome
        # TODO: you must update the connectome!!!
        # convert user defined symbols to indicies
        preIndex = self.connectome.get_neuron_by_key(preKey).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preKey).get_neuron_type()

        if synapseType == "axon":
            axonFlag = True
        else:
            axonFlag = False

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

    # Update a list of synapses
    def write_listofSynapses(self, preKeys, postKeys, weights):
        """
        Writes a list of synapses to the connectome.

        Parameters
        ----------
        preKeys : list of str
            A list of strings representing the keys of the presynaptic neurons.
        postKeys : list of str
            A list of strings representing the keys of the postsynaptic neurons.
        weights : list of int
            A list of integers representing the weights of the synapses.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> preKeys = ['axon1', 'axon2']
        >>> postKeys = ['neuron1', 'neuron2']
        >>> weights = [1, 2]
        >>> network.write_listofSynapses(preKeys, postKeys, weights)
        """
        for i in range(len(preKeys)):
            self.write_synapse(preKeys[i], postKeys[i], weights[i])

    def read_synapse(self, preKey, postKey):
        """
        Reads a synapse from the connectome.

        Parameters
        ----------
        preKey : str
            A string representing the key of the presynaptic neuron.
        postKey : str
            A string representing the key of the postsynaptic neuron.

        Returns
        -------
        int
            The weight of the synapse.

        Raises
        ------
        Exception
            If the target is not valid ("simpleSim" or "CRI").

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.read_synapse('axon1', 'neuron1')
        1
        """
        # convert user defined symbols to indicies
        preIndex = self.connectome.get_neuron_by_key(preKey).get_coreTypeIdx()
        synapseType = self.connectome.get_neuron_by_key(preKey).get_neuron_type()

        if synapseType == "axon":
            axonFlag = True
        else:
            axonFlag = False

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
        """
        Flushes the simulation results to a file.

        Parameters
        ----------
        file : str
            A string representing the file to which to flush the simulation results.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the target is not "CRI" or if the target is invalid.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.sim_flush('results.txt')
        """
        if self.target == "simpleSim":
            raise Exception("sim_flush not available for simpleSim")
        elif self.target == "CRI":
            return self.CRI.sim_flush(file)
        else:
            raise Exception("Invalid Target")

    def step(self, inputs, target="simpleSim", membranePotential=False):
        """
        Runs a step of the simulation.

        Parameters
        ----------
        inputs : list
            A list of inputs for the simulation.
        target : str, optional
            A string representing the target for the simulation. Default is "simpleSim".
        membranePotential : bool, optional
            A boolean value indicating whether to return the membrane potential. Default is False.

        Returns
        -------
        list or tuple
            The simulation outputs or a tuple containing the simulation outputs and spike outputs.

        Raises
        ------
        Exception
            If the target is invalid.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.step(['input1', 'input2'])
        """
        #breakpoint()
        # formated_inputs = [self.symbol2index[symbol][0] for symbol in inputs] #convert symbols to internal indicies
        formated_inputs = [
            self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx()
            for symbol in inputs
        ]  # convert symbols to internal indicies
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
                    #breakpoint()
                    spikeList = spikeResult[0]
                    # we currently ignore the run execution counter
                    spikeList = [
                        self.connectome.get_neuron_by_idx(spike[1]).get_user_key()
                        for spike in spikeList
                    ]
                    numNeurons = len(self.connections)
                    # we currently only print the membrane potential, not the other contents of the spike packet
                    output = [
                        (self.connectome.get_neuron_by_idx(idx).get_user_key(), data[3])
                        for idx, data in enumerate(output)
                    ]  # because the number of neurons will always be a perfect multiple of 16 there will be extraneous neurons at the end so we slice the output array just to get the numNerons valid neurons, due to the way we construct networks the valid neurons will be first
                    return output, (spikeList, spikeResult[1], spikeResult[2])
                else:
                    #breakpoint()
                    spikeResult = self.CRI.run_step(formated_inputs, membranePotential)
                    #breakpoint()
                    spikeList = spikeResult[0]
                    spikeList = [
                        self.connectome.get_neuron_by_idx(spike[1]).get_user_key()
                        for spike in spikeList
                    ]
                    return (spikeList, spikeResult[1], spikeResult[2])
        else:
            raise Exception("Invalid Target")

    def run_cont(self, inputs):
        """
        Runs a continuous simulation.

        Parameters
        ----------
        inputs : list of list
            A list of lists of inputs for the simulation.

        Returns
        -------
        tuple
            A tuple containing the spike list, a boolean indicating whether a break occurred, and the execution counter.

        Raises
        ------
        Exception
            If the simulation dump flag is False and an error occurs during the conversion of neuron indices to user keys.

        Examples
        --------
        >>> network = CRI_network(axons, connections, config, outputs)
        >>> network.run_cont([['input1', 'input2'], ['input3', 'input4']])
        """
        # formated_inputs = [self.symbol2index[symbol][0] for symbol in inputs] #convert symbols to internal indicies
        formated_inputs = []
        for curInputs in inputs:
            formated_inputs.append(
                [
                    self.connectome.get_neuron_by_key(symbol).get_coreTypeIdx()
                    for symbol in curInputs
                ]
            )  # convert symbols to internal indicies

        result = self.CRI.run_cont(formated_inputs)
        spikeList = result[0]
        #breakpoint()
        if self.simDump == False:
            spikeList = [
                (spike[0], self.connectome.get_neuron_by_idx(spike[1]).get_user_key())
                for spike in spikeList
            ]
            return (spikeList, result[1], result[2])
