import numpy as np
from scipy.sparse import dok_array, csr_matrix
from fxpmath import Fxp
from fxpmath.functions import leftshiftArr, rightshiftArr


def map_neuron_type_to_int(neuron_type):
    """maps neuron type strings to integer

    Parameters
    ----------
    neuron_type : str
        neuron type specifier: I&F for integrate and fire, LI&F for leaky integrate and fire, ANN for memoryless neuron

    Returns
    -------
    neuron_int : int
        Integer specifier for neuon type, I&F:3 LI&F:2 ANN:0

    Raises
    ------
    Exception
        If neuron_type doesn't match one of the specified types

    """
    mapping = {"I&F": 3, "LI&F": 2, "ANN": 0}
    try:
        neuron_int = mapping[neuron_type]
        return neuron_int
    except:
        raise Exception("Invalid neuron type")


class simple_sim:
    """Software simulator for spiking neural networks.

    Simulates the behavior of CRI hardware in software using fixed-point arithmetic.
    Supports leaky integrate-and-fire (LIF) and memoryless (ANN) neuron models.

    Attributes
    ----------
    connectome : connectome
        The network topology containing neurons, axons, and synapses.
    weights : Fxp
        Sparse weight matrix (CSR format) mapping presynaptic spikes to
        postsynaptic membrane potential updates. Shape: (numNeurons, nTotal).
    membranePotentials : Fxp
        Current membrane potential for each neuron.
    outputs : list
        Indices of output neurons in pureNeuronArr space.
    numNeurons : int
        Number of neurons (excluding axons).
    numAxons : int
        Number of axons (input sources).
    stepNum : int
        Current simulation timestep.
    """

    def __init__(self, connectome):
        """Initialize the simulator with a connectome.

        Parameters
        ----------
        connectome : connectome
            Network topology object containing neurons, axons, and their synapses.
        """
        self.stepNum = 0
        self.formatDict = {
            "membrane_potential": "fxp-s32/0",
            "synapse_weights": "fxp-s16/0",
            "voltage_threshold": "fxp-s32/0",
            "perturbation": "fxp-s17/0",
            "shift": "fxp-s6/0",
        }
        self.connectome = connectome
        self.outputs = connectome.get_outputs_idx()
        self.numNeurons = len(connectome.get_neurons())
        self.numAxons = len(connectome.get_axons())
        self.gen_weights()
        self.initialize_sim_vars(self.numNeurons)


    def get_membranePotentials(self):
        """Return the current membrane potentials for all neurons."""
        return self.membranePotentials

    def set_perturbMag(self, perturbMag):
        """Set the perturbation magnitude for stochastic noise injection."""
        self.perturbMag = perturbMag

    def initialize_sim_vars(self, numNeurons):
        """Reset simulation state variables.

        Parameters
        ----------
        numNeurons : int
            Number of neurons to initialize membrane potentials for.
        """
        self.membranePotentials = Fxp(
            np.zeros(numNeurons), dtype=self.formatDict["membrane_potential"]
        )
        self.refractoryCounters = np.zeros(numNeurons, dtype=np.int8)
        self.refractory_maxes = np.array(
            [n.get_neuronModel().get_refractory_max() for n in self.connectome.get_neurons()]
        )
        self.firedNeurons = []

    def gen_weights(self):
        """Build the sparse weight matrix from the connectome.

        Constructs a single weight matrix W where W[post, pre] contains the
        synaptic weight from presynaptic neuron/axon to postsynaptic neuron.

        The presynaptic index is in neuronArr space (all axons and neurons),
        while the postsynaptic index is in pureNeuronArr space (neurons only).
        """
        nTotal = len(self.connectome.neuronArr)

        W = dok_array((nTotal, self.numNeurons), dtype=np.float32)
        for preNeuron in self.connectome.neuronArr:
            preIdx = self.connectome.connectomeDict[preNeuron.get_user_key()]
            for synapse in preNeuron.get_synapses():
                postKey = synapse.get_postsynapticNeuron().get_user_key()
                postIdx = self.connectome.get_pureNeuron_idx(postKey)
                W[preIdx, postIdx] = synapse.get_weight()

        self.weights = Fxp(
            csr_matrix(W.transpose()), dtype=self.formatDict["synapse_weights"]
        )

    def write_synapse(self, preIndex, postIndex, weight):
        """Update a synapse weight in the weight matrix.

        Parameters
        ----------
        preIndex : int
            Presynaptic index in neuronArr space.
        postIndex : int
            Postsynaptic index in pureNeuronArr space.
        weight : float
            New synaptic weight value.
        """
        self.weights[postIndex, preIndex] = weight

    def read_synapse(self, preIndex, postIndex):
        """Read a synapse weight from the weight matrix.

        Parameters
        ----------
        preIndex : int
            Presynaptic index in neuronArr space.
        postIndex : int
            Postsynaptic index in pureNeuronArr space.

        Returns
        -------
        float
            The synaptic weight value.
        """
        return self.weights[postIndex, preIndex]()

    def get_perturbMag(self):
        """Return the perturbation shift values for all neurons."""
        return [n.get_neuronModel().get_shift() for n in self.connectome.get_neurons()]

    def get_threshold(self):
        """Return the spike threshold values for all neurons."""
        return [n.get_neuronModel().get_threshold() for n in self.connectome.get_neurons()]

    def get_leak(self):
        """Return the leak values for all neurons."""
        return [n.get_neuronModel().get_leak() for n in self.connectome.get_neurons()]

    def get_model(self):
        """Return the neuron model type codes for all neurons.

        Returns
        -------
        list
            Model codes: 0=memoryless (ANN), 2=leaky I&F, 3=non-leaky I&F.
        """
        return [n.get_neuronModel().get_neuronModel() for n in self.connectome.get_neurons()]

    def step_run(self, inputs):
        """Execute one simulation timestep.

        Performs the following operations in order:
        1. Generate perturbation noise
        2. Detect spikes (neurons exceeding threshold)
        3. Reset spiked neurons and apply leak to LIF neurons
        4. Reset memoryless neurons to zero
        5. Propagate spikes through weight matrix to update membrane potentials

        Parameters
        ----------
        inputs : list of int
            List of active input indices in neuronArr space (axon indices).

        Returns
        -------
        tuple
            (membrane_potentials, output_spikes) where membrane_potentials is
            an array of current potentials and output_spikes is a list of
            output neuron indices that fired this timestep.
        """
        leaks = np.array(self.get_leak())
        threshs = self.get_threshold()
        perturbs = self.get_perturbMag()
        lifNeurons = np.where(np.array(self.get_model()) == 2)[0]
        memLessNeurons = np.where(np.array(self.get_model()) == 0)[0]

        nNeurons = self.numNeurons
        perturbBits = 17
        perturbation = Fxp(
            np.random.randint(0, 2 ** perturbBits, size=nNeurons),
            dtype=self.formatDict["membrane_potential"],
        )
        perturbation(perturbation | Fxp(1, dtype="fxp-u32/0"))
        perturbation = leftshiftArr(perturbation, perturbs, np.greater(perturbs, 0))
        perturbation = rightshiftArr(
            perturbation, np.absolute(perturbs), np.less(perturbs, 0)
        )
        perturbation[np.equal(perturbs, 0)] = 0

        # Spike detection — neurons with refractory counter > 0 cannot spike.
        # Spiked neurons have their counter loaded with refractory_max this step
        # and are not decremented until the next step, so they are blocked for
        # exactly refractory_max timesteps.
        eligible = self.refractoryCounters == 0
        spiked_inds = np.nonzero((self.membranePotentials() > threshs) & eligible)
        self.membranePotentials[spiked_inds] = 0
        self.firedNeurons = np.transpose(spiked_inds).flatten().tolist()

        # Load counter for spiked neurons; decrement all others that are active
        spiked_mask = np.zeros(self.numNeurons, dtype=bool)
        spiked_mask[spiked_inds] = True
        self.refractoryCounters[spiked_mask] = self.refractory_maxes[spiked_mask]
        self.refractoryCounters[~spiked_mask] = np.maximum(
            0, self.refractoryCounters[~spiked_mask] - 1
        )

        # Update LIF neurons
        if lifNeurons.size > 0:
            self.membranePotentials[lifNeurons] = self.membranePotentials[lifNeurons] - (self.membranePotentials[lifNeurons] // np.power(2, leaks[lifNeurons]))

        # Update ANN neurons
        if memLessNeurons.size > 0:
            self.membranePotentials[memLessNeurons] = 0

        # Build combined spike vector (neuronArr indices)
        nTotal = len(self.connectome.neuronArr)
        spikeVec = np.zeros(nTotal)
        spikeVec[inputs] = 1
        for neuronIdx in spiked_inds[0]:
            neuronArrIdx = self.connectome.pureNeuronArr[neuronIdx]
            spikeVec[neuronArrIdx] = 1
        spikeVec = csr_matrix(np.atleast_2d(spikeVec).T)

        # Update membrane potentials
        membraneUpdates = self.weights.get_val() @ spikeVec
        membraneUpdates = Fxp(
            membraneUpdates, dtype=self.formatDict["membrane_potential"]
        )
        membranePotentials = self.membranePotentials + membraneUpdates.transpose()
        membranePotentials = membranePotentials.flatten()
        self.membranePotentials(membranePotentials)

        # Apply perturbation noise to LIF neurons
        if lifNeurons.size > 0:
            self.membranePotentials[lifNeurons] = self.membranePotentials[lifNeurons] + perturbation[lifNeurons]

        self.stepNum = self.stepNum + 1
        outputSpikes = [i for i in self.firedNeurons if i in self.outputs]
        return self.membranePotentials(), outputSpikes
