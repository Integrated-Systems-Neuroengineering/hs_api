import hs_bridge
import pytest
import pickle
import torch
import numpy as np
import sys
import os

# Ensure local hs_api is prioritized over site-packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hs_api.api import CRI_network

class TestDVSInference:
    """Test DVS model inference on hardware using simpleSim"""
    
    @pytest.fixture
    def model_config(self):
        """Load saved model configuration using absolute path"""
        path = '/home/prpandit/hs_api/tests/fixtures/DVS_model_small_config_shift=-17.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        """Load saved test batch using absolute path"""
        path = '/home/prpandit/hs_api/tests/fixtures/DVS_test_batch.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def test_dvs_accuracy(self, model_config, test_batch):
        """Test that DVS model achieves expected accuracy on hardware."""
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']

        # Initialize network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="simpleSim"
        )
        
        import hs_api
        print(f"\n[DEBUG] Using hs_api from: {hs_api.__file__}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        correct = 0
        total = len(test_batch['images'])
        neuron_count = len(connections)

        print(f"--- Starting Inference on {total} samples ---")

        for idx, (img, label) in enumerate(zip(test_batch['images'], test_batch['labels'])):
            
            # --- STATE RESET ---
            network.simpleSim.membranePotentials[:] = np.zeros(neuron_count)
            network.simpleSim.firedNeurons = []
            hs_bridge.FPGA_Execution.fpga_controller.clear(neuron_count, False, 0)

            img = img.to(device) 
            spike_counts = torch.zeros(len(outputs))  
            
            # 1. Processing Frames
            for t in range(img.shape[0]): 
                frame = img[t,:,:,:] 
                input_tensor = frame.unsqueeze(0).flatten(start_dim=1).to(torch.int16)
                
                # Convert active pixels to Axon names (A0, A1, ...)
                inputs = [f"A{i}" for i, elem in enumerate(input_tensor[0, :]) if elem.item() > 0]
                
                # Execute Step
                # We handle the return as a list (spikes only)
                spikes = network.step(inputs)

                # Robust spike extraction
                if isinstance(spikes, tuple):
                    spikes = spikes[0]

                for spike in spikes:
                    if spike in outputs:
                        out_idx = outputs.index(spike)
                        spike_counts[out_idx] += 1

            # 2. Propagation Steps
            for i in range(4):
                spikes = network.step([])
                if isinstance(spikes, tuple):
                    spikes = spikes[0]
                for spike in spikes:
                    if spike in outputs:
                        out_idx = outputs.index(spike)
                        spike_counts[out_idx] += 1
            
            # 3. Prediction Logic
            # If no spikes occurred, predicted will default to 0. 
            # If noise is too high, it may favor the same index repeatedly.
            predicted = torch.argmax(spike_counts).item()
            
            if predicted == label:
                correct += 1
            
            print(f"Sample {idx+1}/{total} | Predicted: {predicted}, Label: {label} | Spikes: {int(spike_counts.sum())}")

        accuracy = 100 * correct / total
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        
        assert accuracy >= 44, f"Expected accuracy >= 44%, but got {accuracy:.2f}%"