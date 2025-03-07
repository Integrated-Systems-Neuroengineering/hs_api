from hs_api.api import CRI_network
import sys
import subprocess
import time

# Define a configuration dictionary
config = {}
config["neuron_type"] = "LI&F"
config["global_neuron_params"] = {}
config["global_neuron_params"]["v_thr"] = 2**19

############################
# Let's try a simple network
############################

# Define an inputs dictionary
inputs = {}
for i in range(100):
    # if (i%3 == 0):
    inputs[i] = ["alpha"]
    # elif(i%3 == 1):
    #    inputs[i] = ['beta']
    # else:
    #    inputs[i] = ['alpha', 'beta']

print(inputs)

# Define an axons dictionary
axons = {"alpha": [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0), ("e", 5.0)]}

# Define a connections dictionary
connections = {
    "01": [("02", 5)],
    "02": [],
    "03": [],
    "04": [],
    "05": [],
    "06": [],
    "07": [],
    "08": [],
    "09": [],
    "10": [],
    "11": [],
    "12": [],
    "13": [],
    "14": [],
    "15": [],
    "16": [],
    "a": [],
    "b": [],
    "c": [],
    "d": [],
    "e": [],
    "f": [],
    "g": [],
    "h": [],
    "i": [],
    "j": [],
    "k": [],
    "l": [],
    "m": [],
    "n": [],
    "o": [],
    "p": [],
    "q": [],
    "r": [],
    "s": [],
    "t": [],
    "u": [],
    "v": [],
    "w": [],
    "x": [],
    "y": [],
    "z": [],
}

# Initialize a CRI_network object for interacting with the hardware and the software
# hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='CRI', outputs = connections.keys(),perturbMag=30, leak=2**5,simDump = False)
# hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config,target='CRI', outputs = connections.keys())
softwareNetwork = CRI_network(
    axons=axons,
    connections=connections,
    config=config,
    target="simpleSim",
    outputs=connections.keys(),
    perturbMag=0,
    leak=2**5,
    simDump=False,
)


# hardwareNetwork = CRI_network(axons=axons,connections=connections,config=config, target='CRI')
# softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, outputs = connections.keys(), target='simpleSim')
# softwareNetwork = CRI_network(axons=axons,connections=connections,config=config, target='simpleSim')

# hardwareNetwork.write_synapse('alpha', 'a', -3)
# softwareNetwork.write_synapse('alpha', 'a', -3)

# Execute the network stepwise in the hardware and the simulator
for i in range(20):
    # start = time.time()
    if i == 10:
        print("update pertMag")
        breakpoint()
        softwareNetwork.set_perturbMag(0)
    swResult = softwareNetwork.step(["alpha"], membranePotential=True)
    # print(inputs[i])
    # end = time.time()
    # print(end - start)
    # start = time.time()
    # swResult, swSpike = softwareNetwork.step(inputs[i], membranePotential=True)
    # end = time.time()
    # print(end - start)
    # print("timestep: "+str(i)+":")
    print("hardware result: ")
    # print(synthSpike)
    # print(hwSpike)
    print(swResult)
    # print("software result: ")
    # print(swSpike)
    # print(swResult)
    # Verify that the outputs match
    # for idx in range(len(swResult)):
    #    if(swResult[idx][1] != hwResult[idx][1][3]):
    #        print("Error: potential mismatch! sim: "+str(swResult[idx])+", hw: "+str(hwResult[idx]))
# hardwareNetwork.sim_flush('jul13.txt')
