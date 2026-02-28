#!/usr/bin/env python3


"""
Creating a Network
==================

This example walks through how to create a simple network and run it in the software simulator
"""


# %%
# Importing the necessary libraries
# ---------------------------------
# First import the CRI network class from the the hs_api library.

from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron

# %%
# Defining a neuron model
# ------------------------
# HiAER-Spike supports specifying different models for different neurons. A set of model classes are provided. Currently only variants of the integrate and fire model provided by the LIF_neuron class are supported
# LIF_neuron models have 3 parameters
#  * threshold(:math:`T`): determines the membrane potential at which the neuron spikes and the potential is reset to zero
#  * shift(:math:`P`): controls the magnitude of random noise added to the membrane potential at each step.
#  * leak(:math:`L`): controls the voltage leakage that occurs during each timestep. :math:`v=v-v/2^{leak}`

N1 = LIF_neuron(threshold = 3, shift = 2, leak = 1)

# %%
# Defining the axons dictionary
# -----------------------------
# Axons represent incoming synapses to the network. Each axon has one or more postsynaptic neurons. Users can manually send spikes over axons at each timestep

axons = {'alpha': [('a', 3),('c', 2)],
             'beta': [('b', 3)]}
# %%
# Defining the connections dictionary
# -----------------------------------
# The connections defines the neurons in the network and the synapses between them. Each neuron may have synapses to zero, one, or many postsynaptic neurons and must have a model specified by providing a neuron model object. Keys in the connections and axons dictionaries must be mutually exclusive

connections = {'a': ([('b', 1), ('d', 2)], N1),
                   'b': ([], N1),
                   'c': ([], N1),
                   'd': ([('c', 1)], N1)}

# %%
# Defining the outputs List
# -------------------------
# The outputs list defines the neurons in the network that the user wishes to monitor for spikes. Each element in the list is the key of a neuron in the connections dicitonary

outputs = ['a', 'b']

# %%
# Initializing a Network
# ----------------------
# A CRI network object must be contructed using the prevoisly defined dictionaries and list.

network = CRI_network(axons=axons,connections=connections,outputs=outputs)

# %%
# Running a timestep
# ----------------------
# The step method executes a single timestep of the network. Inputs may be provided for each timestep in the form of a list of axons to send spikes over dat the given timestep. A list of spikes is always returned. Optionally the membrane potential parameter can be set to true to return membrane potentials for all neurons in the network.

inputs = ['alpha','beta']
currSpikes = network.step(inputs)
# Alternative
# potentials, currSpikes = network.step(inputs, membranePotential=True)
print(currSpikes)

# %%
# Updating Synapses
# ----------------------
# Network topologies are frozen at object creation, but the weights of existing synapses may be altered.

currWeight = network.read_synapse('a', 'b')
network.write_synapse('a', 'b', 2)
