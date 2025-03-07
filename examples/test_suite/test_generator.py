import numpy as np
import random as rnd
import logging
from l2s.api import CRI_network


def generateNetwork(
    timeSteps=100,
    numAxons=8,
    numNeurons=64,
    maxWeight=10,
    minAxonsActivated=0,
    maxAxonsActivated=3,
    minNeuronConnections=0,
    maxNeuronConnections=4,
):
    """
    Parameters
    ----------
    timeSteps : int
        The number of time steps the network runs for. The default is 100, and the maximum is 1000.
    numAxons : int
        The number of axons in the network. The default is 8.
    numNeurons : int
        The number of neurons in the network. The default is 64.
    maxWeight : int or float
        The maximum weight of any synapse or axon connection. The default is 10.
    minAxonsActivated : int
        The minimum number of axons activated at each time step. The default is 0.
    maxAxonsActivated : int
        The maximum number of axons activated at each time step. The default is 3.

    Returns
    -------

    """

    if type(timeSteps) == int & timeSteps > 0:
        if timeSteps <= 1000:
            if type(numAxons) == int & type(numNeurons) == int:
                axons = {}
                connections = {}
                neurons = []
                digits = len(str(numNeurons))
                for i in range(numNeurons):
                    neurons.append(str(i + 1).zfill(digits))

                if (
                    minAxonsActivated
                    >= 0 & maxAxonsActivated
                    <= numAxons & type(minAxonsActivated)
                    == int & type(maxAxonsActivated)
                    == int
                ):
                    numAxonsActivated = [
                        rnd.randint(minAxonsActivated, maxAxonsActivated)
                        for t in range(timeSteps)
                    ]
                    numNeuronsConnected = [
                        rnd.randint(minNeuronConnections, maxNeuronConnections)
                        for t in range(numNeurons)
                    ]
                else:
                    logging.error(
                        "Axons activated per time step should be an integer between 0 and the number of axons."
                    )
            else:
                logging.error("The number of axons and neurons should be integers.")
        else:
            logging.error("Let's not make this too long.")
    else:
        logging.error("The number of time steps should be a positive integer.")
