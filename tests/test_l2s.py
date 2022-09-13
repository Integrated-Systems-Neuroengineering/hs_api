from ctypes import sizeof
from sys import maxunicode
import unittest
from l2s.api import CRI_network
import numpy as np
import random as rnd
import os
import logging

class testL2s(unittest.TestCase):

    def setUp(self):
        self.generateNetworks(10,100,10,-10,10)
        # self.synth = synthnet(100,1000,-10,10,1000)
        # self.hardware = CRI_network(axons=self.synth.axonsDict,connections=self.synth.neuronsDict,config=self.config,target='CRI', outputs = self.synth.neuronsDict.keys())
        # self.software = CRI_network(axons=self.synth.axonsDict,connections=self.synth.neuronsDict,config=self.config,target='simpleSim', outputs = self.synth.neuronsDict.keys())
        
    def tearDown(self):
        pass

    def generateNetworks(self, numAxons=100, numNeurons=1000, maxSpan=100, minWeight = -10, maxWeight = 10, simDump = False):
        """
        Parameters
        ----------
        numAxons : int
            The number of axons in the network. The default is 100.
        numNeurons : int
            The number of neurons in the network. The default is 1000.
        minWeight : int or float 
            The maximum weight of any synapse or axon connection. The default is -10.
        maxWeight : int or float
            The maximum weight of any synapse or axon connection. The default is 10.
        maxSpan : 
            THen maximum number of synapses any neuron can have. The default is 100. 
        Returns
        -------
        
        """
        config = {}
        config['neuron_type'] = "I&F"
        config['global_neuron_params'] = {}
        config['global_neuron_params']['v_thr'] = 9

        axons = {}
        connections = {}

        self.numAxons = numAxons
        self.numNeurons = numNeurons
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.maxSpan = maxSpan

        for i in range(numNeurons):
            name = 'N' + str(i)
            connections[name] = []
            numSynapse = rnd.sample(range(0,numNeurons),rnd.randrange(0,maxSpan))
            for num in numSynapse: 
                connections[name].append(('N' + str(num), rnd.randrange(minWeight,maxWeight)))
        
        for i in range(numAxons):
            name = 'A' + str(i)
            axons[name] = []
            numSynapse = rnd.sample(range(0,numNeurons),rnd.randrange(0,maxSpan))
            for num in numSynapse: 
                axons[name].append(('N' + str(num), rnd.randrange(minWeight,maxWeight)))
            
        self.software = CRI_network(axons=axons,connections=connections,config=config, target='simpleSim', outputs = connections.keys())
        self.hardware = CRI_network(axons=axons,connections=connections,config=config, target='CRI', outputs = connections.keys(), simDump=simDump)
    
    def test_write(self):
        lenSynapses = 0
        while(lenSynapses < 1):
            idx = rnd.randrange(0,self.numAxons-1)
            axon = 'A'+ str(idx)
            hardware_synapses = self.hardware.connectome.get_axon_by_idx(idx).get_synapses()
            software_synapses = self.software.connectome.get_axon_by_idx(idx).get_synapses()
            lenSynapses = len(hardware_synapses)
        if (lenSynapses == 1):
            synapseIdx = 0
        else:
            synapseIdx = rnd.randrange(0,len(hardware_synapses)-1)
        synapseKey = hardware_synapses[synapseIdx].get_postsynapticNeuron().get_user_key()
        newWeight = rnd.randrange(self.minWeight, self.maxWeight)
        self.hardware.write_synapse(axon, synapseKey, newWeight)
        self.software.write_synapse(axon, synapseKey, newWeight)
        for synap in hardware_synapses:
            if synap.get_postsynapticNeuron().get_user_key() == synapseKey:
                synap.set_weight(newWeight)
        self.assertCountEqual(hardware_synapses,self.hardware.connectome.get_axon_by_idx(idx).get_synapses())
        self.assertCountEqual(software_synapses,self.software.connectome.get_axon_by_idx(idx).get_synapses())

    def test_read(self):
        lenSynapses = 0
        while(lenSynapses < 1):
            idx = rnd.randrange(0,self.numAxons-1)
            axon = 'A'+ str(idx)
            hardware_synapses = self.hardware.connectome.get_axon_by_idx(idx).get_synapses()
            software_synapses = self.software.connectome.get_axon_by_idx(idx).get_synapses()
            lenSynapses = len(hardware_synapses)
        if (lenSynapses == 1):
            synapseIdx = 0
        else:
            synapseIdx = rnd.randrange(0,len(hardware_synapses)-1)
        synapseKey = hardware_synapses[synapseIdx].get_postsynapticNeuron().get_user_key()
        weight = hardware_synapses[synapseIdx].get_weight()
        hardware_result_synapse = self.hardware.read_synapse(axon, synapseKey)
        software_result_synapse = self.software.read_synapse(axon, synapseKey)
        CORE_ID = 0
        N_NG = 16
        postIdx = int(synapseKey[1:])
        rowIdx = np.floor(postIdx / N_NG)
        try: 
            self.assertEqual(CORE_ID,hardware_result_synapse[0])
            self.assertEqual(rowIdx,hardware_result_synapse[1])
            self.assertEqual(weight,hardware_result_synapse[2])
            self.assertEqual(postIdx,software_result_synapse[0])
            self.assertEqual(weight,software_result_synapse[1])
        except AssertionError:
            print("axon: ")
            print(self.hardware.connectome.get_axon_by_idx(idx))
            print("synapses: ")
            print(hardware_synapses)
            print("synapseKey: ")
            print(synapseKey)
            print("Hardware Result: ")
            print(hardware_result_synapse)
            print("Software Result: ")
            print(software_result_synapse)

    def test_match_step(self):
        steps = 10
        self.inputs = {}
        for i in range(steps):
            self.inputs[i] = []
            inputAxons = rnd.sample(range(0, self.numAxons), rnd.randrange(0,self.numAxons))
            for num in inputAxons: 
                self.inputs[i].append('A' + str(num))
            hwResult, hwSpike = self.hardware.step(self.inputs[i], membranePotential=True)
            swResult, swSpike = self.software.step(self.inputs[i], membranePotential=True)
            try:
                self.assertCountEqual(hwSpike,swSpike)
                self.assertCountEqual(hwResult,swResult)
            except AssertionError:
                print("timestep: "+str(i)+":")
                print("hardware result: ")
                print(hwResult)
                print(hwSpike)
                print("software result: ")
                print(swResult)
                print(swSpike)
    
    def test_sim_flush(self):
        self.generateNetworks(10,100,10,-10,10,True)
        file = os.getcwd() + "/sim_flush.txt"
        print(file)
        self.hardware.sim_flush(file)
        self.assertEqual(True, os.stat(file).st_size != 0)

    def test_write_list_synapse(self):
        idices = rnd.sample(range(0, self.numAxons-1), rnd.randrange(0,self.numAxons))
        lenSynapses = [len(self.hardware.connectome.get_axon_by_idx(idx).get_synapses()) for idx in idices ]
        for i, length in enumerate(lenSynapses):
            if(length == 0):
                del idices[i]
                del lenSynapses[i]
        axons = [ 'A' + str(idx) for idx in idices]
        hardware_synapses = [self.hardware.connectome.get_axon_by_idx(idx).get_synapses() for idx in idices]
        software_synapses = [self.software.connectome.get_axon_by_idx(idx).get_synapses() for idx in idices]
        synapseIdicies = [0]*len(axons)
        for i, length in enumerate(lenSynapses):
            if length > 1: synapseIdicies[i] = rnd.randrange(0, length-1) 
        synapseKeys = []
        for i,synapseIdx in enumerate(synapseIdicies):
            synapseKeys.append(synapses[i][synapseIdx].get_postsynapticNeuron().get_user_key())
        newWeights = [ rnd.randrange(self.minWeight, self.maxWeight) for i in axons]
        self.hardware.write_listofSynapses(axons, synapseKeys, newWeights)
        self.software.write_listofSynapses(axons, synapseKeys, newWeights)

        for i, synapse in enumerate(synapses):
            for synap in synapse:
                if synap.get_postsynapticNeuron().get_user_key() == synapseKeys[i]:
                    synap.set_weight(newWeights[i])
            try:
                self.assertCountEqual(hardware_synapses[i],self.hardware.connectome.get_axon_by_idx(idx).get_synapses())
                self.assertCountEqual(software_synapses[i],self.software.connectome.get_axon_by_idx(idx).get_synapses())
            except AssertionError:
                print(hardware_synapses[i])
                print(self.hardware.connectome.get_axon_by_idx(idx).get_synapses())


if __name__ == '__main__':
    unittest.main()