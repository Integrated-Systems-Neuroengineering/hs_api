from numba import jit
import numpy as np
import yaml
from ast import literal_eval

def load_network(input, connex, output):
    """Loads the network specification.

    This function loads the inputs and connections specified for the network.
    Also determines the number of FPGA cores to be used.

    Parameters
    ----------
    input : str, optional
        Path to file specifying network inputs. (the default is the path in config.yaml)
    connex : str, optional
        Path to file specifying network connections. (the default is the path in config.yaml)

    Returns
    -------
    axons : dict
        Dictionary specifying axons in the network. Key: axon number Value: Synapse Weights
    connections : dict
        Dictionary specifying neurons in the network. Key: Neuron Number Value: Synapse Weights
    inputs : dict
        Dictionary specifying inputs to the network. Key, Time Step Value, axon
    outputs : dict
        TODO: I'm not sure what the outputs are for. I belive it's unused
    ncores : int
        The number of cores peresent in the CRI system

    """
    axons = {}
    connections = {}
    inputs = {}
    outputs = {}

    #Load in the connectivity file
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

    # Load in the inputs file
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

    ## Get the number of cores to map to
    #n_cores = 0
    #for fpga_cluster_num, fpga_cluster in enumerate(ARCH):
    #    for fpga_num, fpga in enumerate(fpga_cluster['FPGA']):
    #        for core_cluster_num, core_cluster, in enumerate(fpga['Core_cluster']):
    #            for core_num in range(int(core_cluster['Cores'])):
    #                n_cores += 1
    #
    #print(n_cores)

    #assignment = partition(connex,n_cores)

    assert(len(connections.keys())-1 in connections.keys())

    return axons, connections, inputs, outputs#, n_cores


#@jit(nopython=True)
def phase_one(neuronModel, threshold, membranePotentials, firedNeurons):
  """updates neuron membrane potentials and looks for spikes

  This function scans through the neurons in the network, checks if any neurons have spiked, and updates membrane potenntialsu

  Parameters
  ----------
  neuronModel : int
      The neuron model that defines the membrane potential update. 
  threshold : int
      The membrane potential above which the neuron will spike.
  membranePotentials : numpy array
      A numpy array containing the current membrane potential of each neuron in the network. Each index is the membrane potential of the neuorn at the
      corresponding index.
  firedNeurons : list
      A list to enter fired neurons into.

  Returns
  -------
  membranePotentials : numpy array
      A numpy array containing the updated membrane potential of each neuron in the network. Each index is the membrane potential of the neuorn at the
      corresponding index.
  firedNeurons : list
      A list containing the indicies of neurons that fired.

  Notes
  -----
  (Neuron Model = 0) ==> memory-less neuron
  (Neuron Model = 1) ==> Incremental I&F Neuron (We can leave out this neuron type in the simulator as this needs the abstraction of the neuron group).
  (Neuron Model = 2) ==> Leaky I&F Neuron
  (Neuron Model = 3) ==> Non-leaky I&F Neuron

  """
  for neuron, potential in enumerate(membranePotentials):
    if potential > threshold:
      membranePotentials[neuron] = 0
      firedNeurons.append(neuron) #np.append(firedNeurons, neuron)
    else:
          #match neuronModel:
          if neuronModel == 0:
              membranePotentials[neuron] = 0
          elif neuronModel == 2:
              membranePotentials[neuron] = membranePotentials[neuron] - (membranePotentials[neuron] // (2 ** 3))
          elif neuronModel == 3:
              membranePotentials[neuron] = membranePotentials[neuron]
          else:
              raise Exception('Invaled Neuron model supplied, note neuron model 1 not supported')
  return membranePotentials, firedNeurons


#@jit(nopython=True) # can't pass dict
def phase_two(firedNeurons, currentInputs, membranePotentials, axons, connections ):
  """Processes spikes

  This function scans through the spiked neurons and axons in the network, and updates the membrane potential of all immediate postsynaptic neurons

  Parameters
  ----------
  firedNeurons : list
      A list containing the indicies of neurons that fired.
  currentInputs : list
      A list containing the indicies of spiked axons.
  membranePotentials : numpy array
      A numpy array containing the current membrane potential of each neuron in the network. Each index is the membrane potential of the neuorn at the
      corresponding index.
  axons : dict
      Dictionary specifying axons in the network. Key: axon number Value: Synapse Weights
  connections : dict
      Dictionary specifying neurons in the network. Key: Neuron Number Value: Synapse Weights

  Returns
  -------
  membranePotentials : numpy array
      A numpy array containing the updated membrane potential of each neuron in the network. Each index is the membrane potential of the neuorn at the
      corresponding index.

  """
  for input in currentInputs:
    synapses = axons[input]
    for synapse in synapses:
      #synapse[0] = neuron index, synapse[1] = synapse weight
      membranePotentials[synapse[0]] = membranePotentials[synapse[0]] + synapse[1]
      #go through all neurons the axon has synapses to and update the membrane potentials
  
  for spike in firedNeurons:
    synapses = connections[spike]
    for synapse in synapses:
      membranePotentials[synapse[0]] = membranePotentials[synapse[0]] + synapse[1]
      #go through all neurons the fired neuron has synapses to and update the membrane potentials

  return membranePotentials

def simulate(neuronModel,threshold, axons, connections, inputs):
  """
  Simulates the network

  This function simulates the SNN as processed on the CRI hardware in python and prints spiked neurons/ membrane potentials to terminal for 
  each timestep

  Parameters
  ----------
  neuronModel : int
      The neuron model that defines the membrane potential update. 
  threshold : int
      The membrane potential above which the neuron will spike.
  axons : dict
        Dictionary specifying axons in the network. Key: axon number Value: Synapse Weights
  connections : dict
        Dictionary specifying neurons in the network. Key: Neuron Number Value: Synapse Weights
  inputs : dict
        Dictionary specifying inputs to the network. Key, Time Step Value, axon

  Notes
  -----
  (Neuron Model = 0) ==> memory-less neuron
  (Neuron Model = 1) ==> Incremental I&F Neuron (We can leave out this neuron type in the simulator as this needs the abstraction of the neuron group).
  (Neuron Model = 2) ==> Leaky I&F Neuron
  (Neuron Model = 3) ==> Non-leaky I&F Neuron

  In the below equations, curr_potential is what we read from the memory and next_potential is what we write into the memory for updates.

  For Phase 1:- 

  For all neurons:-

    if (curr_potential  > threshold)           next_potential = 36'd0;

    else    if (neuron_model==2'd0)  next_potential  = 36'd0;

                else if (neuron_model==2'd1)  next_potential   =  curr_potential  + i+1;  //where i is the neuron group index.

                else if (neuron_model==2'd2)  next_potential   =   curr_potential    - ( curr_potential  >>> 3);
                else if (neuron_model==2'd3)  next_potential   = curr_potential  ;

  For Phase 2:- 

              for all synapse connections of active inputs and fired neurons:- 

                            next_potential  =  curr_potential  + synapse weight; 
  """
  # = load_network()
  timesteps = range(len(inputs)) #TODO What if not every timestep is enumerated in inputs
  numNeurons = len(connections)
  membranePotentials = np.zeros(numNeurons)
  firedNeurons = [] #np.array([], dtype=np.single)
  for time in timesteps:
    currentInputs = np.array(inputs[time])
    #do phase one
    membranePotentials, firedNeurons = phase_one(neuronModel, threshold, membranePotentials, firedNeurons)
    # phase_one(threshold,membranePotentials,firedNeurons)#look for any spiked neurons

    #do phase two
    print(time, firedNeurons)
    membranePotentials = phase_two(firedNeurons, currentInputs, membranePotentials, axons, connections)#update the membrane potentials
    
    print(time, 'Vmem', membranePotentials)
    
    firedNeurons = [] #np.array([])

def read_config(config_path):
  """reads in the network configuaration file

  Parameters
  ----------
  config_path : str
      The path to the yaml file specifying the network parameters

  Returns
  -------
  config : dict
      A dicitonary containing the configuration parameters from the yaml file. Key: parameter name Value: parameter value

  """
  with open(config_path) as f:
      config = yaml.safe_load(f)

  return config

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
  mapping = { "I&F": 3, "LI&F": 2, "ANN":0 }
  try:
    neuron_int = mapping[neuron_type]
    return neuron_int
  except:
    raise Exception("Invalid neuron type")

def run_sim():
  """Reads in netwokr parameters from config file and runs network simulation
  Reads in network parameters from config.yaml and simulates the network. At each timestep fired neurons and post firing membrane potentials are printed to the terminal.

  """
  config_path = r'./config.yaml'
  config = read_config(config_path)

  input = config['inputs_file']
  connex = config['connectivity_file']
  output = config['outputs_file']
  neuron_model = map_neuron_type_to_int(config['neuron_type'])
  threshold = config['global_neuron_params']['v_thr']

  axons, connections, inputs, outputs = load_network(
    input= input, 
             connex=connex, 
             output=output)

  simulate(neuron_model, threshold, axons, connections, inputs)

class simple_sim:
    def __init__(self, neuronModel, threshold, axons, connections, inputs):
          self.stepNum = 0
          self.neuronModel = neuronModel
          self.threshold = threshold
          self.axons = axons
          self.connections = connections
          self.inputs = inputs
          self.timesteps = range(len(inputs)) #TODO What if not every timestep is enumerated in inputs
          self.numNeurons = len(connections)
          self.initialize_sim_vars()
    def initialize_sim_vars(self):
          self.membranePotentials = np.zeros(numNeurons)
          self.firedNeurons = [] #np.array([], dtype=np.single)

    def free_run(self):
        for time in self.timesteps:
            currentInputs = np.array(self.inputs[time])
            #do phase one
            self.membranePotentials, self.firedNeurons = phase_one(self.neuronModel, self.threshold, self.membranePotentials, self.firedNeurons)
            # phase_one(threshold,membranePotentials,firedNeurons)#look for any spiked neurons

            #do phase two
            print(time, self.firedNeurons)
            self.membranePotentials = phase_two(self.firedNeurons, currentInputs, self.membranePotentials, self.axons, self.connections)#update the membrane potentials

            print(time, 'Vmem', self.membranePotentials)

            self.firedNeurons = [] #np.array([])
            #
    def step_run(self):
        if (self.stepNum == self.timesteps):
            print("Reinitializing simulation to timestep zero")
            initialize_sim_vars()
            self.stepNum == 0
        else:
            time = self.stepNum
            currentInputs = np.array(self.inputs[time])
            #do phase one
            self.membranePotentials, self.firedNeurons = phase_one(self.neuronModel, self.threshold, self.membranePotentials, self.firedNeurons)
            # phase_one(threshold,membranePotentials,firedNeurons)#look for any spiked neurons

            #do phase two
            print(time, self.firedNeurons)
            self.membranePotentials = phase_two(self.firedNeurons, currentInputs, self.membranePotentials, self.axons, self.connections)#update the membrane potentials

            print(time, 'Vmem', self.membranePotentials)

            self.firedNeurons = [] #np.array([])
            self.stepNum = self.stepNum+1
