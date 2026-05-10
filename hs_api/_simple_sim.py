import numpy as np
from ast import literal_eval
from scipy.sparse import dok_array, csr_matrix
from fxpmath import Fxp

def load_network(input_file, connex_file, output_file):
    """Loads the network specification."""
    axons = {}
    connections = {}
    inputs = {}
    outputs = {}

    with open(connex_file, "r") as f:
        ax = False
        for line in f:
            if not line.startswith("#") and line.strip():
                if "axons" in line.lower():
                    ax = True
                elif "neurons" in line.lower():
                    ax = False
                else:
                    pre, post = line.split(":")
                    weights = literal_eval(post.strip())
                    weights = [(int(i[0]), float(i[1])) for i in weights]
                    if ax:
                        axons[int(pre.strip())] = weights
                    else:
                        connections[int(pre.strip())] = weights

    with open(input_file, "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                pre, post = line.split(":")
                inputs[int(pre.strip())] = literal_eval(post.strip())

    with open(output_file, "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                pre, post = line.split(":")
                outputs[int(pre.strip())] = literal_eval(post.strip())

    return axons, connections, inputs, outputs


class simple_sim:
    def __init__(self, axons, connections, outputs):
        self.stepNum = 0
        self.formatDict = {
            "membrane_potential": "fxp-s35/0",
            "synapse_weights": "fxp-s16/0",
            "voltage_threshold": "fxp-s35/0",
            "perturbation": "fxp-s17/0",
            "shift": "fxp-s6/0",
        }
        self.axons = axons
        self.connections = connections
        self.outputs = outputs
        self.numNeurons = len(connections)
        self.gen_weights()
        self.initialize_sim_vars(self.numNeurons)

    def get_membranePotentials(self):
        return self.membranePotentials

    def set_perturbMag(self, perturbMag):
        self.perturbMag = perturbMag

    def initialize_sim_vars(self, numNeurons):
        self.membranePotentials = Fxp(
            np.zeros(numNeurons), dtype=self.formatDict["membrane_potential"]
        )
        self.firedNeurons = []

    def gen_weights(self):
        nNeurons = len(self.connections)
        nAxons = len(self.axons)
        S = dok_array((nNeurons, nNeurons), dtype=np.float32)
        for key, value in self.connections.items():
            for synapse in value[0]:
                presynapticIdx = key
                postsynapticIdx, weight = synapse
                S[presynapticIdx, postsynapticIdx] = weight

        A = dok_array((nAxons, nNeurons), dtype=np.float32)
        for key, value in self.axons.items():
            for synapse in value:
                presynapticIdx = key
                postsynapticIdx, weight = synapse
                A[presynapticIdx, postsynapticIdx] = weight

        self.neuronWeights = Fxp(
            csr_matrix(S.transpose()), dtype=self.formatDict["synapse_weights"]
        )
        self.axonWeights = Fxp(
            csr_matrix(A.transpose()), dtype=self.formatDict["synapse_weights"]
        )

    def write_synapse(self, preIndex, postIndex, weight, axonFlag=False):
        if axonFlag:
            self.axonWeights[postIndex, preIndex] = weight
        else:
            self.neuronWeights[postIndex, preIndex] = weight

    def read_synapse(self, preIndex, postIndex, axonFlag=False):
        if axonFlag:
            return self.axonWeights[postIndex, preIndex]()
        else:
            return self.neuronWeights[postIndex, preIndex]()

    def get_perturbMag(self):
        return [self.connections[key][1].get_shift() for key in self.connections.keys()]

    def get_threshold(self):
        return [self.connections[key][1].get_threshold() for key in self.connections.keys()]

    def get_leak(self):
        return [self.connections[key][1].get_leak() for key in self.connections.keys()]

    def get_model(self):
        return [self.connections[key][1].get_neuronModel() for key in self.connections.keys()]

    def step(self, inputs):
        """Entry point for the CRI_network API."""
        return self.step_run(inputs)

    def step_run(self, inputs):
        leaks = np.array(self.get_leak())
        threshs = np.array(self.get_threshold())
        perturbs = np.array(self.get_perturbMag())
        models = np.array(self.get_model())
        
        lif_mask = (models == 2)
        ann_mask = (models == 0)
        nNeurons = self.numNeurons
        nAxons = len(self.axons)
        
        # 1. SYNAPTIC UPDATES
        a = np.zeros(nAxons)
        a[inputs] = 1
        a_sparse = csr_matrix(np.atleast_2d(a).transpose())
        
        # Spikes from PREVIOUS state
        all_spiked_inds = np.nonzero(self.membranePotentials() > threshs)[0]
        spikeVec = np.zeros(nNeurons)
        spikeVec[all_spiked_inds] = 1
        spikeVec_sparse = csr_matrix(np.atleast_2d(spikeVec).transpose())

        upd_axon = (self.axonWeights.get_val() @ a_sparse).toarray().flatten()
        upd_neuron = (self.neuronWeights.get_val() @ spikeVec_sparse).toarray().flatten()
        
        # Intermediate math
        current_mem = self.membranePotentials().astype(np.float64)
        current_mem += (upd_axon + upd_neuron)

        # 2. HOUSEKEEPING
        current_mem[all_spiked_inds] = 0
        if np.any(lif_mask):
            current_mem[lif_mask] -= (current_mem[lif_mask] // np.power(2, leaks[lif_mask]))
        if np.any(ann_mask):
            current_mem[ann_mask] = 0

        # 3. NOISE
        noise = np.random.randint(-2**16, 2**16, size=nNeurons).astype(np.float64)
        for i in range(nNeurons):
            s = int(perturbs[i])
            if s > 0: noise[i] = int(noise[i]) << s
            elif s < 0: noise[i] = int(noise[i]) >> abs(s)
        
        current_mem += noise

        # 4. SYNC BACK
        self.membranePotentials(current_mem)
        self.firedNeurons = [i for i in all_spiked_inds if i in self.outputs]
        self.stepNum += 1
        
        return self.membranePotentials(), self.firedNeurons, None

def map_neuron_type_to_int(neuron_type):
    mapping = {"I&F": 3, "LI&F": 2, "ANN": 0}
    try:
        return mapping[neuron_type]
    except KeyError:
        raise Exception(f"Invalid neuron type: {neuron_type}")