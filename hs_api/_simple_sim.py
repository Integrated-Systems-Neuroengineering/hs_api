#from numba import jit
import numpy as np
import yaml
from ast import literal_eval
import copy
from scipy.sparse import dok_array, csr_matrix
from fxpmath import Fxp
from fxpmath.functions import leftshiftArr, rightshiftArr

def load_network(input, connex, output):
    axons = {}
    connections = {}
    inputs = {}
    outputs = {}

    ax = None
    with open (connex, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                if 'axons' in line.lower():
                    ax = True
                elif 'neurons' in line.lower():
                    ax = False
                else:
                    pre, post = line.split(':')
                    weights = literal_eval(post.strip())
                    weights = [(int(i[0]), float(i[1])) for i in weights]
                    if ax:
                        axons[int(pre.strip())] = weights
                    else:
                        connections[int(pre.strip())] = weights

    with open(input, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                pre, post = line.split(':')
                inputs[int(pre.strip())] = literal_eval(post.strip())

    with open(output, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                pre, post = line.split(':')
                outputs[int(pre.strip())] = literal_eval(post.strip())

    assert(len(connections.keys())-1 in connections.keys())
    return axons, connections, inputs, outputs

def phase_one(neuronModel, threshold, membranePotentials, firedNeurons):
  for neuron, potential in enumerate(membranePotentials):
    if potential > threshold:
      membranePotentials[neuron] = 0 
      firedNeurons.append(neuron) 
    else:
          if neuronModel == 0:
              membranePotentials[neuron] = 0
          elif neuronModel == 2:
              membranePotentials[neuron] = membranePotentials[neuron] - (membranePotentials[neuron] // (2 ** 3))
          elif neuronModel == 3:
              membranePotentials[neuron] = membranePotentials[neuron]
          else:
              raise Exception('Invaled Neuron model supplied, note neuron model 1 not supported')
  return membranePotentials, firedNeurons

def phase_two(firedNeurons, currentInputs, membranePotentials, axons, connections ):
  for input in currentInputs:
    synapses = axons[input]
    for synapse in synapses:
      membranePotentials[synapse[0]] = membranePotentials[synapse[0]] + synapse[1]
  
  for spike in firedNeurons:
    synapses = connections[spike]
    for synapse in synapses:
      membranePotentials[synapse[0]] = membranePotentials[synapse[0]] + synapse[1]

  return membranePotentials

def simulate(neuronModel,threshold, axons, connections, inputs):
  timesteps = range(len(inputs)) 
  numNeurons = len(connections)
  membranePotentials = np.zeros(numNeurons)
  firedNeurons = [] 
  for time in timesteps:
    currentInputs = np.array(inputs[time])
    membranePotentials, firedNeurons = phase_one(neuronModel, threshold, membranePotentials, firedNeurons)
    membranePotentials = phase_two(firedNeurons, currentInputs, membranePotentials, axons, connections)
    firedNeurons = [] 

def read_config(config_path):
  with open(config_path) as f:
      config = yaml.safe_load(f)
  return config

def map_neuron_type_to_int(neuron_type):
  mapping = { "I&F": 3, "LI&F": 2, "ANN":0 }
  try:
    neuron_int = mapping[neuron_type]
    return neuron_int
  except:
    raise Exception("Invalid neuron type")

class simple_sim:
    def __init__(self, axons, connections, outputs, threshold=0, perturbMag=18, leak=0):
          self.stepNum = 0
          self.formatDict = {
                "membrane_potential" : 'fxp-s35/0',
                "synapse_weights" : 'fxp-s16/0',
                "voltage_threshold" : 'fxp-s35/0',
                "perturbation" : 'fxp-s17/0',
                "shift" : 'fxp-s6/0'
            }
          self.axons = axons
          self.connections = connections
          self.outputs = outputs
          self.perturbMag = perturbMag
          self.leak = leak
          self.numNeurons = len(connections)
          self.gen_weights()
          self.initialize_sim_vars(self.numNeurons)

    def set_perturbMag(self, perturbMag):
        self.perturbMag = perturbMag

    def initialize_sim_vars(self, numNeurons):
          self.membranePotentials = Fxp(np.zeros(numNeurons),dtype=self.formatDict['membrane_potential'])
          self.firedNeurons = [] 

    def gen_weights(self):
        nNeurons = len(self.connections)
        nAxons = len(self.axons)
        S = dok_array((nNeurons,nNeurons), dtype=np.float32)
        for key, value in self.connections.items():
            for synapse in value[0]:
                presynapticIdx = key
                postsynapticIdx,weight = synapse
                S[presynapticIdx,postsynapticIdx] = weight

        A = dok_array((nAxons,nNeurons), dtype=np.float32)
        for key, value in self.axons.items():
            for synapse in value:
                presynapticIdx = key
                postsynapticIdx,weight = synapse
                A[presynapticIdx, postsynapticIdx] = weight

        self.neuronWeights = Fxp( csr_matrix(S.transpose()) , dtype=self.formatDict['synapse_weights'])
        self.axonWeights = Fxp( csr_matrix(A.transpose()) , dtype=self.formatDict['synapse_weights'])

    def write_synapse(self,preIndex, postIndex, weight, axonFlag = False):
        if axonFlag:
             self.axonWeights[postIndex, preIndex] = weight
        else:
            self.neuronWeights[postIndex, preIndex] = weight

    def read_synapse(self,preIndex, postIndex, axonFlag = False):
        if axonFlag:
            return self.axonWeights[postIndex, preIndex]()
        else:
            return self.neuronWeights[postIndex, preIndex]()

    def get_perturbMag(self):
        perturbs = [self.connections[key][1].get_shift() for key in self.connections.keys()] 
        return perturbs

    def get_threshold(self):
        threshs = [self.connections[key][1].get_threshold() for key in self.connections.keys()] 
        return threshs

    def get_leak(self):
        leaks = [self.connections[key][1].get_leak() for key in self.connections.keys()] 
        return leaks

    def step_run(self,inputs):
        leaks = self.get_leak()
        threshs = self.get_threshold()
        perturbs = self.get_perturbMag()

        if False: 
            print("Reinitializing simulation to timestep zero")
            initialize_sim_vars()
            self.stepNum == 0
        else:
            nNeurons = len(self.connections)
            nAxons = len(self.axons)
            perturbBits = 17
            
            # 1. Generate the raw pseudorandom perturbation
            perturbation = Fxp(np.random.randint(-1*2**(perturbBits-1),2**(perturbBits-1),size=nNeurons),dtype=self.formatDict['membrane_potential'])
            perturbation( perturbation | Fxp(1,dtype='fxp-u35/0') )
            
            # 2. Apply Shift logic:
            # If shift > 0, we left shift (increase magnitude)
            perturbation = leftshiftArr(perturbation, perturbs, np.greater(perturbs,0)) 
            # If shift < 0, we right shift (decrease magnitude)
            perturbation = rightshiftArr(perturbation, np.absolute(perturbs), np.less(perturbs,0)) 
            
            # 3. Corrected Noise Condition: 
            # We now allow noise even if shift (nu) is 0. 
            # The only time we disable it is if shift is explicitly the 'off' value (-16).
            if any(a != -16 for a in perturbs):
                self.membranePotentials(self.membranePotentials+perturbation)

            spiked_inds = np.nonzero(self.membranePotentials() >= threshs)
            self.membranePotentials[spiked_inds] = 0
            self.firedNeurons = np.transpose(spiked_inds).flatten().tolist()

            self.membranePotentials(self.membranePotentials() - (self.membranePotentials() // np.power(2,leaks)))
            
            a = np.zeros(nAxons)
            a[inputs] = 1
            a = np.atleast_2d(a)
            a = csr_matrix(np.transpose(a))
            
            spikeVec = np.zeros(nNeurons)
            spikeVec[spiked_inds] = 1
            spikeVec = np.atleast_2d(spikeVec)
            spikeVec = csr_matrix(np.transpose(spikeVec))

            membraneUpdatesAxon = self.axonWeights.get_val() @ a
            membraneUpdates = self.neuronWeights.get_val() @ spikeVec
            
            membraneUpdatesAxon = Fxp(membraneUpdatesAxon,dtype=self.formatDict['membrane_potential'])
            membraneUpdates = Fxp(membraneUpdates,dtype=self.formatDict['membrane_potential'])

            combinedUpdates = membraneUpdates + membraneUpdatesAxon
            membranePotentials = self.membranePotentials + combinedUpdates.transpose()
            membranePotentials = membranePotentials.flatten()
            self.membranePotentials(membranePotentials)
            self.stepNum = self.stepNum+1
            outputSpikes = [ i for i in self.firedNeurons if i in self.outputs]
            return self.membranePotentials(), outputSpikes, None