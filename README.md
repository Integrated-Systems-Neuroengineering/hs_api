
# Table of Contents

1.  [Introduction](#org9104c25)
2.  [Installation](#org65da685)
    1.  [Simple Installation](#org57dc672)
    2.  [Development Installation](#org06a26fa)
3.  [Usage](#org03f671f)
    1.  [Running on the Simulator](#org7b994c3)
        1.  [Defining a Network](#org2ab8c59)
        2.  [Initializing a network](#org1fd5547)
        3.  [Running a Timestep](#org800bf7a)
        4.  [Updating Synapse Weights](#org933dbeb)
    2.  [Submitting Jobs to Run on the Hardware](#org5e85c59)
4.  [Python libraries present on the CRI servers](#org870bc5a)



<a id="org9104c25"></a>

# Introduction

This repository contains a Python library for interacting with the ISN HiAER Spike project hosted at SDSC. This project aims to make massive scale simulations of spiking neural networks easily accessible to the research community, and in particular researches interested in neuromorphic computing for artificial intelligence and neuroscience researchers. This library allows a user to define a spiking neural network and execute it on one of two backends: the HiAER Spike neuromorphic hardware or if the hardware is not available a python simulation of the hardware.


<a id="org65da685"></a>

# Installation


<a id="org57dc672"></a>

## Simple Installation

    pip install hs_api


<a id="org06a26fa"></a>

## Development Installation

-   First install [Poetry](https://python-poetry.org/)
    -   If Poetry installs in may be necessary to install an alternative Python distribution such as [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
-   Then clone this repository

    git clone https://github.com/Integrated-Systems-Neuroengineering/hs_api.git

-   cd into the hs<sub>api</sub> repo you cloned and install the needed dependencies. Resolving dependencies may take a while.
    -   Some Python dependencies may require a compiler supporting C++11 to be installed on your system, such as a recent version of GCC

    cd hs_api
    poetry install

-   finally activate the development environment

    poetry shell


<a id="org03f671f"></a>

# Usage


<a id="org7b994c3"></a>

## Running on the Simulator

On your local machine you can run networks using the Python based simulator of the CRI hardware.


<a id="org2ab8c59"></a>

### Defining a Network

Users are expected to provide three data structures in order to define a network

1.  Defining the Configuration Dictionary

    The configuration dictionary specifies a few properties that are shared by every neuron in the network
    
    -   neuron<sub>type</sub> specifies the type of neuron model used to calculate membrane potentials
    -   global<sub>neuron</sub><sub>params</sub> is a sub-dictionary of the configuration dictionary
        -   v<sub>thr</sub> is an entry in the global<sub>neuron</sub><sub>params</sub> dictionary, it sets the membrane potential threshold for all neurons in the network
    
        configuration = {}
            configuration['neuron_type'] = "I&F"
            configuration['global_neuron_params'] = {}
            configuration['global_neuron_params']['v_thr'] = 4

2.  Defining the Axons Dictionary

    The axons dictionary configures inputs to the network. Axons are synapses connected to neurons in the network that the user can manually send spikes over at a given timestep. Each key in the dictionary is the name of an axon. Each value is a list of two element tuples. Each tuple defines an in-going synapse to a neuron. The first element is the name of a neuron in the network and the second element is the weight of the synaptic connection. Synapse weights must be integers, but they may be positive or negative.
    \#+BEGIN<sub>SRC</sub> python
    axons = {&rsquo;alpha&rsquo;: [(&rsquo;a&rsquo;, 3)],
                 &rsquo;beta&rsquo;: [(&rsquo;d&rsquo;, 3)]}
    \#+END<sub>SRC</sub> python

3.  Defining the Connections Dictionary

    The connections dictionary defines the neurons in the network and the connections between them. Each key in the dictionary is the name of a neuron. Of note the names of neurons in the connections dictionary and the names of axons in the axons dictionary must be mutually exclusive. Each value is a list of two element tuples. Each tuple defines a synapse between neurons in the network. The first element is the name of the postsynaptic neuron and the the second element is the weight of the synapse. Synapse weights must be integers but they may be positive or negative. If a neuron has no outgoing synapses it&rsquo;s synapse list may be left empty.
    
        connections = {'a': [('b', 1)],
                           'b': [],
                           'c': [],
                           'd': [('c', 1)]}

4.  Defining the Outputs List

    The outputs list defines the neurons in the network the user wishes to receive spikes from. Each element in the list is the key of a neuron in the connections dictionary.
    
        outputs = ['a', 'b']


<a id="org1fd5547"></a>

### Initializing a network

Once we&rsquo;ve defined the above dictionaries and list we must pass them to the CRI<sub>network</sub> constructor to create a CRI<sub>network</sub> object.

    network = CRI_network(axons=axons,connections=connections,config=config, outputs=outputs)


<a id="org800bf7a"></a>

### Running a Timestep

Once we&rsquo;ve constructed an CRI<sub>network</sub> object we can run a timestep. We do so by calling the step() method of CRI<sub>network</sub>. This method expects a single input called inputs. Inputs defines the inputs to the network at the current timestep, in particular it is a list of names of axons that you wish to carry spikes into the network at the current timestep. Normally network.step() returns a list of the keys that correspond to neurons that spiked during the given timestep, however the membranePotential parameter can be set to True to additionally output the membranePotentials for all neurons in the network.

    inputs = ['alpha','beta']
    spikes = network.step(inputs)
    #Alternative
    potentials, spikes = network.step(inputs, membranePotential=True)

This method will return a list of membrane potentials for all neurons in the network after the current timestep has elapsed.


<a id="org933dbeb"></a>

### Updating Synapse Weights

Once the CRI<sub>network</sub> class the topology of the network is fixed, that is what axon and neurons are in the network and how they are connected via synapses may not be changed. However it is possible to update the weight of preexisting synapses in the network. This can be done by calling the write<sub>synapse</sub>() method of CRI<sub>network</sub>. write<sub>synapse</sub>() takes three arguments, the presynaptic neuron name, the postsynaptic neuron name, and the new synapse weight.

    network.write_synapse('a', 'b', 2)


<a id="org5e85c59"></a>

## Submitting Jobs to Run on the Hardware

The same Python scripts you&rsquo;ve developed and run on your local machine can be deployed to the CRI servers to run on the actual CRI hardware. Just make sure all the libraries you import in your script are [available on the CRI servers](#org870bc5a). The CRI hardware is hosted in the San Diego Supercomputing Center and jobs may be submitted to run on the hardware via the [Neuroscience Gateway](https://www.nsgportal.org/index.html). First you must register an account with Neuroscience Gateway in order to submit jobs. Perform the following steps to submit a task to NSG:

-   Put your CRI Python script in a folder of any name, then zip the folder
-   Log into NSG.
-   Create a task folder if there is none listed on the upper left.  It&rsquo;s a place to hold related jobs.
-   Click on data, and save the previously created zip file as the data.  Here &rsquo;data&rsquo; is ambiguous - it is the job and its data.
-   Click on task.
-   Create a new task if needed (or clone an old one).
-   Assign the zip you just uploaded as data as the input to the task.
-   Select **Python for CRI** as the software to run.
-   Set parameters for the task:
    -   Set execution &rsquo;wall time&rsquo;, cores, and GB of DRAM if you wish. Please be consideret to others and only request the hardware you need.
    -   Enter the name of your.py python scrip as the &ldquo;input&rdquo; using the same name as is in the zip folder.
    -   Enter a name for the &ldquo;output&rdquo; (optional)
-   Click save parameters
-   Click **save and run** to run the task.
-   Click **OK** on the popup or the job will not start.
-   Click on task again in your folder at the upper left if the task list is not present.
-   View status if desired, refresh as needed, or just watch for the task done email.
-   When it is done select the &rsquo;view output&rsquo; for that task on the task list.
-   Download outputs and decompress.  Job &rsquo;inputs&rsquo; is displayed as garbage.


<a id="org870bc5a"></a>

# Python libraries present on the CRI servers

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<tbody>
<tr>
<td class="org-left">absl-py</td>
<td class="org-right">1.1.0</td>
</tr>


<tr>
<td class="org-left">bidict</td>
<td class="org-right">0.22.0</td>
</tr>


<tr>
<td class="org-left">brotlipy</td>
<td class="org-right">0.7.0</td>
</tr>


<tr>
<td class="org-left">certifi</td>
<td class="org-right">2021.10.8</td>
</tr>


<tr>
<td class="org-left">cffi</td>
<td class="org-right">1.15.0</td>
</tr>


<tr>
<td class="org-left">charset-normalizer</td>
<td class="org-right">2.0.4</td>
</tr>


<tr>
<td class="org-left">click</td>
<td class="org-right">8.1.3</td>
</tr>


<tr>
<td class="org-left">colorama</td>
<td class="org-right">0.4.4</td>
</tr>


<tr>
<td class="org-left">conda</td>
<td class="org-right">4.12.0</td>
</tr>


<tr>
<td class="org-left">conda-content-trust</td>
<td class="org-right">0+unknown</td>
</tr>


<tr>
<td class="org-left">conda-package-handling</td>
<td class="org-right">1.8.1</td>
</tr>


<tr>
<td class="org-left">confuse</td>
<td class="org-right">1.7.0</td>
</tr>


<tr>
<td class="org-left">cri-simulations</td>
<td class="org-right">0.1.2</td>
</tr>


<tr>
<td class="org-left">cryptography</td>
<td class="org-right">36.0.0</td>
</tr>


<tr>
<td class="org-left">cycler</td>
<td class="org-right">0.11.0</td>
</tr>


<tr>
<td class="org-left">fbpca</td>
<td class="org-right">1.0</td>
</tr>


<tr>
<td class="org-left">fonttools</td>
<td class="org-right">4.33.3</td>
</tr>


<tr>
<td class="org-left">idna</td>
<td class="org-right">3.3</td>
</tr>


<tr>
<td class="org-left">joblib</td>
<td class="org-right">1.1.0</td>
</tr>


<tr>
<td class="org-left">k-means-constrained</td>
<td class="org-right">0.7.1</td>
</tr>


<tr>
<td class="org-left">kiwisolver</td>
<td class="org-right">1.4.3</td>
</tr>


<tr>
<td class="org-left">hs<sub>api</sub></td>
<td class="org-right">0.1.3</td>
</tr>


<tr>
<td class="org-left">llvmlite</td>
<td class="org-right">0.38.1</td>
</tr>


<tr>
<td class="org-left">matplotlib</td>
<td class="org-right">3.5.2</td>
</tr>


<tr>
<td class="org-left">metis</td>
<td class="org-right">0.2a5</td>
</tr>


<tr>
<td class="org-left">networkx</td>
<td class="org-right">2.8.4</td>
</tr>


<tr>
<td class="org-left">numba</td>
<td class="org-right">0.55.2</td>
</tr>


<tr>
<td class="org-left">numpy</td>
<td class="org-right">1.22.4</td>
</tr>


<tr>
<td class="org-left">ortools</td>
<td class="org-right">9.3.10497</td>
</tr>


<tr>
<td class="org-left">packaging</td>
<td class="org-right">21.3</td>
</tr>


<tr>
<td class="org-left">Pillow</td>
<td class="org-right">9.1.1</td>
</tr>


<tr>
<td class="org-left">pip</td>
<td class="org-right">21.2.4</td>
</tr>


<tr>
<td class="org-left">protobuf</td>
<td class="org-right">4.21.1</td>
</tr>


<tr>
<td class="org-left">pycosat</td>
<td class="org-right">0.6.3</td>
</tr>


<tr>
<td class="org-left">pycparser</td>
<td class="org-right">2.21</td>
</tr>


<tr>
<td class="org-left">PyMetis</td>
<td class="org-right">2020.1</td>
</tr>


<tr>
<td class="org-left">pyOpenSSL</td>
<td class="org-right">22.0.0</td>
</tr>


<tr>
<td class="org-left">pyparsing</td>
<td class="org-right">3.0.9</td>
</tr>


<tr>
<td class="org-left">PySocks</td>
<td class="org-right">1.7.1</td>
</tr>


<tr>
<td class="org-left">python-dateutil</td>
<td class="org-right">2.8.2</td>
</tr>


<tr>
<td class="org-left">PyYAML</td>
<td class="org-right">6.0</td>
</tr>


<tr>
<td class="org-left">requests</td>
<td class="org-right">2.27.1</td>
</tr>


<tr>
<td class="org-left">ruamel-yaml-conda</td>
<td class="org-right">0.15.100</td>
</tr>


<tr>
<td class="org-left">scikit-learn</td>
<td class="org-right">1.1.1</td>
</tr>


<tr>
<td class="org-left">scipy</td>
<td class="org-right">1.8.1</td>
</tr>


<tr>
<td class="org-left">setuptools</td>
<td class="org-right">61.2.0</td>
</tr>


<tr>
<td class="org-left">six</td>
<td class="org-right">1.16.0</td>
</tr>


<tr>
<td class="org-left">sklearn</td>
<td class="org-right">0.0</td>
</tr>


<tr>
<td class="org-left">threadpoolctl</td>
<td class="org-right">3.1.0</td>
</tr>


<tr>
<td class="org-left">tqdm</td>
<td class="org-right">4.63.0</td>
</tr>


<tr>
<td class="org-left">urllib3</td>
<td class="org-right">1.26.8</td>
</tr>


<tr>
<td class="org-left">wheel</td>
<td class="org-right">0.37.1</td>
</tr>
</tbody>
</table>

