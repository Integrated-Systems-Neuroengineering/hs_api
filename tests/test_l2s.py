import unittest
from l2s.api import CRI_network
import numpy as np
import random as rnd
import logging

class testL2s(unittest.TestCase):

    def setUp(self):
        self.network = self.generateNetwork()

    def tearDown(self):
        pass
    
    def test_write_synapse(self):
        #TODO:checking the write function
        self.assertEqual(self.network, self.generateNetwork()) #placeholder 

    def generateNetwork(timeSteps = 100, numAxons = 8, numNeurons = 64, maxWeight = 10, 
    minAxonsActivated = 0, maxAxonsActivated = 3, minNeuronConnections = 0, maxNeuronConnections = 4):
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

        axons = {}
        connections = {}
        neurons = []
        digits = len(str(numNeurons))
        for i in range(numNeurons):
            #generate keys for each neuron e.g. 00001, 00002,..., 0000n
            neurons.append(str(i+1).zfill(digits)) 

            #selecting a random number activated axon for each time step
            numAxonsActivated = [rnd.randint(minAxonsActivated,maxAxonsActivated) for t in range(timeSteps)] 
            numNeuronsConnected = [rnd.randint(minNeuronConnections,maxNeuronConnections) for t in range(numNeurons)]

        #TODO: creating random axons dictionary

if __name__ == '__main__':
    unittest.main()