import numpy as np
from fxpmath import Fxp
from fxpmath.functions import leftshiftArr, rightshiftArr
from scipy.sparse import csr_matrix, dok_array

def map_neuron_type_to_int(neuron_type):
    """
    Maps neuron type specifier strings to integers for the hardware/simulator.
    I&F: 3, LI&F: 2, ANN: 0
    """
    mapping = { "I&F": 3, "LI&F": 2, "ANN": 0 }
    try:
        return mapping[neuron_type]
    except KeyError:
        return 3

class simple_sim:
    def __init__(self, axons, connections, outputs, perturbMag=18, leak=0):
        self.stepNum = 0
        self.formatDict = {
            "membrane_potential": 'fxp-s35/0',
            "synapse_weights": 'fxp-s16/0',
            "voltage_threshold": 'fxp-s35/0',
            "perturbation": 'fxp-s17/0',
            "shift": 'fxp-s6/0'
        }
        self.axons = axons
        self.connections = connections
        self.outputs = outputs
        self.perturbMag = perturbMag
        self.leak = leak
        self.numNeurons = len(connections)
        
        self.gen_weights()
        self.initialize_sim_vars(self.numNeurons)

    def initialize_sim_vars(self, numNeurons):
        self.membranePotentials = Fxp(np.zeros(numNeurons), dtype=self.formatDict['membrane_potential'])
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
        
        self.neuronWeights = Fxp(csr_matrix(S.transpose()), dtype=self.formatDict['synapse_weights'])
        self.axonWeights = Fxp(csr_matrix(A.transpose()), dtype=self.formatDict['synapse_weights'])

    def get_perturbMag(self):
        return [self.connections[key][1].get_shift() for key in self.connections.keys()]

    def get_threshold(self):
        return [self.connections[key][1].get_threshold() for key in self.connections.keys()]

    def get_leak(self):
        return [self.connections[key][1].get_leak() for key in self.connections.keys()]

    def step_run(self, inputs):
        leaks = np.array(self.get_leak())
        threshs = np.array(self.get_threshold())
        perturbs = np.array(self.get_perturbMag())
        
        nNeurons = self.numNeurons
        nAxons = len(self.axons)
        perturbBits = 17

        # 1. NOISE
        perturbation = Fxp(np.random.randint(-1*2**(perturbBits-1), 2**(perturbBits-1), size=nNeurons), 
                           dtype=self.formatDict['membrane_potential'])
        perturbation(perturbation | Fxp(1, dtype='fxp-u35/0'))
        perturbation = leftshiftArr(perturbation, perturbs, np.greater(perturbs, 0))
        perturbation = rightshiftArr(perturbation, np.absolute(perturbs), np.less(perturbs, 0))

        if any(a != -17 for a in perturbs):
            self.membranePotentials(self.membranePotentials + perturbation)

        # 2. FIRING
        spiked_inds = np.nonzero(self.membranePotentials() >= threshs)
        self.membranePotentials[spiked_inds] = 0
        self.firedNeurons = np.transpose(spiked_inds).flatten().tolist()

        # 3. LEAK
        self.membranePotentials(self.membranePotentials() - (self.membranePotentials() // np.power(2, leaks)))

        # 4. INTEGRATION (FIXED)
        # Create sparse input vector
        a_vec = np.zeros(nAxons)
        a_vec[inputs] = 1
        a_sparse = csr_matrix(a_vec).transpose()

        # Create sparse spike vector
        s_vec = np.zeros(nNeurons)
        s_vec[spiked_inds] = 1
        s_sparse = csr_matrix(s_vec).transpose()

        # Perform multiplication and immediately convert to dense NumPy arrays
        # This prevents the 'csc_matrix has no attribute flatten' error
        upd_axon = (self.axonWeights.get_val() @ a_sparse).toarray().flatten()
        upd_neuron = (self.neuronWeights.get_val() @ s_sparse).toarray().flatten()

        combinedUpdates = Fxp(upd_axon + upd_neuron, dtype=self.formatDict['membrane_potential'])
        
        self.membranePotentials(self.membranePotentials + combinedUpdates)

        self.stepNum += 1
        outputSpikes = [i for i in self.firedNeurons if i in self.outputs]
        
        # Note: Your api.py expects 3 return values based on the traceback
        return self.membranePotentials(), outputSpikes, None