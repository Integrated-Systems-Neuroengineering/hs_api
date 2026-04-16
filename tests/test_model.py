# tests/test_cifar10_inference.py
import hs_bridge
import pytest
import pickle
from hs_api.api import CRI_network
import hs_bridge
import torch

class TestDVSInference:
    """Test DVS model inference on hardware"""
    
    @pytest.fixture
    def model_config(self):
        """Load saved model configuration"""
        with open('fixtures/DVS_model_config.pkl', 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        """Load saved test batch"""
        with open('fixtures/DVS_test_batch.pkl', 'rb') as f:
            return pickle.load(f)
    
    def test_dvs_accuracy(self, model_config, test_batch,shuffle_mode):
        '''
        Validates that the DVS classification model achieves the required accuracy on CRI hardware.
        
        This test performs an end-to-end inference run by:
        1. Configuring the CRI_network with axons, connections, and output mappings.
        2. Iterating through test images and injecting spike inputs into the FPGA.
        3. Reading membrane potentials and aggregating spike counts for classification.
        4. Calculating accuracy against ground truth labels.
        
        Args:
            model_config (dict): Configuration data for network topology.
            test_batch (dict): Input images and corresponding ground truth labels.
            shuffle_mode (bool): If True, enables random shuffling of neuron-to-HBM mapping.
        
        Raises:
            AssertionError: If the calculated accuracy falls below the 55% threshold.
        '''
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']
        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI",
            random_shuffle=shuffle_mode #shuffle mode is --shuffle
        )
        print("\n" + "="*30)
        print("VERIFYING NEURON GROUP MAPPING")
        neuron_keys = list(connections.keys())
        for symbol in neuron_keys[:10]: # Check first 10 neurons
            hw_idx = network.key2idx_map[symbol]
            core_group = hw_idx // 16 
            print(f"Neuron: {symbol:<10} | HBM Address: {hw_idx:>5} | Core: {core_group}")
        print("="*30 + "\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #test model
        correct = 0
        total = len(test_batch['images'])
        for img, label in zip(test_batch['images'], test_batch['labels']):
            #reset membrane potnetials before each image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                            len(connections), False, 0
                        )

            
            img = img.to(device) #shape [T, C, H, W]
            spike_counts = torch.zeros(len(outputs))  #to count spikes over all frames
            for t in range(img.shape[0]):  #iterate through all frames
                frame = img[t,:,:,:] #shape [C, H, W]

                #convert from [C, H, W] to [1, C*H*W]
                input = frame.unsqueeze(0)  #add dimension
                #input = encoder(input)
                input = input.flatten(start_dim=1)
                input = input.to(torch.int16)        #change input from FP32 to INT16
                
                #create input list
                inputs = []
                #print all unique input values
                #print(f"Unique input values: {input.unique()}")
                for i, elem in enumerate(input[0, :]):
                    if elem.item() > 0: #changed from == 1 for MNIST
                        inputs.append(f"A{i}")
                
                results = network.read_membrane(outputs)
                print(f"Membrane potentials: {results}")

                hardwareSpikes, _, _ = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

                for spike in hardwareSpikes:
                    if spike in outputs:
                        spike_counts[spike] += 1
                    else:
                        print(f"Error: invalid output spike {spike}")

            #add 6 extra timesteps after lastinput frame to allow it to propogate through network
            for i in range(6):
                inputs = []  #no input spikes
                hardwareSpikes, _, _ = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

                for spike in hardwareSpikes:
                    if spike in outputs:
                        spike_counts[spike] += 1
                    else:
                        print(f"Error: invalid output spike {spike}")
            

            spike_counts = spike_counts / img.size(0)  #average spike counts(spike rate)
            print(f"Spike counts: {spike_counts}")

            predicted = torch.argmax(spike_counts).item()
            print(f"Predicted: {predicted}, Label: {label}")
            
            if predicted == label:
                correct += 1
            
            running_accuracy = 100 * correct / total
            print(f"Running accuracy : {running_accuracy:.2f} %")

        accuracy = 100 * correct / total
        assert accuracy >= 55, f"Expected accuracy is at least 55%, but got {accuracy:.2f}%"