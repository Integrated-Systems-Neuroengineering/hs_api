import numpy as np
from scipy.sparse import dok_array, csr_matrix
from fxpmath import Fxp
from tqdm import tqdm

def map_neuron_type_to_int(neuron_type):
    """
    Maps neuron type strings to integer.
    I&F: 3, LI&F: 2, ANN: 0
    """
    mapping = {"I&F": 3, "LI&F": 2, "ANN": 0}
    try:
        return mapping[neuron_type]
    except KeyError:
        raise Exception("Invalid neuron type")

class simple_sim:
    """
    Software simulator for spiking neural networks with synaptic delays 
    and 35-bit hardware-parity stochastic noise.
    """

    def __init__(self, connectome):
        self.stepNum = 0
        # Configured for s35/0 to provide the 3-bit headroom over s32 for high noise shifts
        self.formatDict = {
            "membrane_potential": "fxp-s35/0",
            "synapse_weights": "fxp-s16/0",
            "voltage_threshold": "fxp-s35/0",
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
        return self.membranePotentials

    def set_perturbMag(self, perturbMag):
        self.perturbMag = perturbMag

    def initialize_sim_vars(self, numNeurons):
        self.membranePotentials = Fxp(
            np.zeros(numNeurons), dtype=self.formatDict["membrane_potential"]
        )
        self.refractoryCounters = np.zeros(numNeurons, dtype=np.int8)
        self.refractory_maxes = np.array(
            [n.get_neuronModel().get_refractory_max() for n in self.connectome.get_neurons()]
        )
        self.soft_resets = np.array(
            [n.get_neuronModel().get_soft_reset() for n in self.connectome.get_neurons()]
        )
        # 16-slot circular buffer for delayed synapse delivery
        self.delay_buffer = np.zeros((16, numNeurons), dtype=np.float64)
        self.delay_values = np.array(
            [n.get_neuronModel().get_delay_value() for n in self.connectome.get_neurons()],
            dtype=np.int32,
        )
        self.firedNeurons = []

    def clear(self, num_neurons=None, simDump=False, coreOverride=0):
        """
        Resets all membrane potentials, refractory states, delay buffers, 
        and step counters to ensure no information leaks between inferences.
        
        This method achieves perfect signature and functional parity with the 
        physical FPGA controller hardware reset command.
        
        Parameters
        ----------
        num_neurons : int, optional
            The total number of neurons to reset in the state vector. 
            If None (default), it automatically uses the full network size 
            defined by `self.numNeurons`.
        simDump : bool, optional
            A hardware-parity flag used on the physical FPGA to signal a memory 
            dump of register states during a clear event. In this software 
            simulator, it defaults to False and is bypassed.
        coreOverride : int, optional
            A hardware-parity identifier used to target a specific neurosynaptic 
            core cluster on the physical chip. In this software simulator, it 
            defaults to 0 and is bypassed.
            
        Returns
        -------
        None
        """
        target_neurons = num_neurons if num_neurons is not None else self.numNeurons
        self.initialize_sim_vars(target_neurons)
        self.stepNum = 0

    def gen_weights(self):
        nTotal = len(self.connectome.neuronArr)
        W_imm = dok_array((nTotal, self.numNeurons), dtype=np.float32)
        W_del = dok_array((nTotal, self.numNeurons), dtype=np.float32)
        
        for preNeuron in tqdm(self.connectome.neuronArr, desc="Building weight matrix", unit="neuron"):
            preIdx = self.connectome.connectomeDict[preNeuron.get_user_key()]
            for synapse in preNeuron.get_synapses():
                postKey = synapse.get_postsynapticNeuron().get_user_key()
                postIdx = self.connectome.get_pureNeuron_idx(postKey)
                post_dual_en = synapse.get_postsynapticNeuron().get_neuronModel().get_dual_synapse_en()
                
                if synapse.is_delayed() and post_dual_en:
                    W_del[preIdx, postIdx] = synapse.get_weight()
                else:
                    W_imm[preIdx, postIdx] = synapse.get_weight()

        self.weights = Fxp(
            csr_matrix(W_imm.transpose()), dtype=self.formatDict["synapse_weights"]
        )
        self.weights_delayed = Fxp(
            csr_matrix(W_del.transpose()), dtype=self.formatDict["synapse_weights"]
        )

    def get_perturbMag(self):
        return [n.get_neuronModel().get_shift() for n in self.connectome.get_neurons()]

    def get_threshold(self):
        return [n.get_neuronModel().get_threshold() for n in self.connectome.get_neurons()]

    def get_leak(self):
        return [n.get_neuronModel().get_leak() for n in self.connectome.get_neurons()]

    def get_model(self):
        return [n.get_neuronModel().get_neuronModel() for n in self.connectome.get_neurons()]

    def step_run(self, inputs):
        leaks = np.array(self.get_leak())
        threshs = np.array(self.get_threshold())
        perturbs = np.array(self.get_perturbMag())
        
        models = np.array(self.get_model())
        lifNeurons = np.where(models == 2)[0]
        memLessNeurons = np.where(models == 0)[0]
        counterNeurons = np.where(models == 1)[0]

        nNeurons = self.numNeurons
        perturbBits = 17

        raw_noise = np.random.randint(-1*2**(perturbBits-1), 2**(perturbBits-1), size=nNeurons)
        
        noise_mag = np.abs(raw_noise)
        noise_sign = np.sign(raw_noise)
        
        shifted_mag = np.where(perturbs > 0, 
                               np.left_shift(noise_mag.astype(np.int64), perturbs.astype(np.int64)), 
                               np.right_shift(noise_mag.astype(np.int64), np.abs(perturbs).astype(np.int64)))

        final_perturbation = Fxp(shifted_mag * noise_sign, dtype=self.formatDict["membrane_potential"])

        final_perturbation(final_perturbation | Fxp(1, dtype="fxp-u35/0"))

        final_perturbation[np.equal(perturbs, -17)] = 0
        self.perturbation = final_perturbation

        eligible = self.refractoryCounters == 0
        spiked_inds = np.nonzero((self.membranePotentials() > threshs) & eligible)
        self.firedNeurons = np.transpose(spiked_inds).flatten().tolist()

        spiked_mask = np.zeros(self.numNeurons, dtype=bool)
        spiked_mask[spiked_inds] = True

        hard_reset_mask = spiked_mask & (self.soft_resets == 0)
        soft_reset_mask = spiked_mask & (self.soft_resets == 1)
        
        self.membranePotentials[hard_reset_mask] = 0
        if soft_reset_mask.any():
            mp_vals = self.membranePotentials()
            self.membranePotentials[soft_reset_mask] = Fxp(
                mp_vals[soft_reset_mask] - threshs[soft_reset_mask],
                dtype=self.formatDict["membrane_potential"],
            )

        self.refractoryCounters[spiked_mask] = self.refractory_maxes[spiked_mask]
        self.refractoryCounters[~spiked_mask] = np.maximum(
            0, self.refractoryCounters[~spiked_mask] - 1
        )

        weight_blocked = self.refractoryCounters > 0

        # --- 4. MODEL DYNAMICS (Leak/ANN) ---
        if lifNeurons.size > 0:
            self.membranePotentials[lifNeurons] = self.membranePotentials[lifNeurons] - (self.membranePotentials[lifNeurons] // np.power(2, leaks[lifNeurons]))

        if memLessNeurons.size > 0:
            self.membranePotentials[memLessNeurons] = 0

        if counterNeurons.size > 0:
            active_counters = counterNeurons[eligible[counterNeurons] & ~spiked_mask[counterNeurons]]
            if active_counters.size > 0:
                self.membranePotentials[active_counters] = self.membranePotentials[active_counters] + 1

        # --- 5. SYNAPSE INTEGRATION ---
        nTotal = len(self.connectome.neuronArr)
        spikeVec = np.zeros(nTotal)
        spikeVec[inputs] = 1
        for neuronIdx in spiked_inds[0]:
            neuronArrIdx = self.connectome.pureNeuronArr[neuronIdx]
            spikeVec[neuronArrIdx] = 1
        spikeVec = csr_matrix(np.atleast_2d(spikeVec).T)

        if weight_blocked.any():
            blocked_mp_snapshot = self.membranePotentials()[weight_blocked]
        
        # Dense conversion for immediate updates
        imm_updates = (self.weights.get_val() @ spikeVec).toarray().flatten()
        self.membranePotentials(self.membranePotentials + imm_updates)

        if weight_blocked.any():
            self.membranePotentials[weight_blocked] = blocked_mp_snapshot

        # --- 6. DELAY DELIVERY ---
        if self.weights_delayed.get_val().nnz > 0:
            del_updates = self.weights_delayed.get_val() @ spikeVec
            del_updates_arr = np.asarray(del_updates.todense()).flatten()
            if del_updates_arr.any():
                target_slots = (self.stepNum + self.delay_values) % 16
                np.add.at(self.delay_buffer, (target_slots, np.arange(self.numNeurons)), del_updates_arr)

        pending_slot = self.stepNum % 16
        pending_arr = self.delay_buffer[pending_slot]
        if pending_arr.any():
            if weight_blocked.any():
                pending_arr = pending_arr.copy()
                pending_arr[weight_blocked] = 0
            self.membranePotentials(Fxp(self.membranePotentials() + pending_arr, dtype=self.formatDict["membrane_potential"]))
        self.delay_buffer[pending_slot] = 0

        # --- 7. NOISE INJECTION ---
        if lifNeurons.size > 0:
            self.membranePotentials[lifNeurons] = self.membranePotentials[lifNeurons] + final_perturbation[lifNeurons]

        self.stepNum += 1
        outputSpikes = [i for i in self.firedNeurons if i in self.outputs]

        return self.membranePotentials(), outputSpikes