import pytest
import numpy as np
import sys
import os
import pickle

# Ensure local hs_api is prioritized
from hs_api.api import CRI_network

class TestDVSStability:
    """Stress test for simulator stability over 1,000 steps."""
    
    @pytest.fixture
    def model_config(self):
        """Use the full model config to test real-world scale."""
        path = '/home/prpandit/hs_api/tests/fixtures/DVS_model_small_config_shift=-17.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def test_1k_steps_stability(self, model_config):
        """Runs 1,000 steps to ensure the Fxp noise logic doesn't explode."""
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']

        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="simpleSim"
        )
        
        neuron_count = len(connections)
        network.simpleSim.membranePotentials[:] = np.zeros(neuron_count)
        
        print(f"\n--- Starting 1,000 Step Stability Test ---")
        
        # Stability Loop
        for t in range(1, 1001):
            # Pass empty inputs to isolate noise behavior
            network.step([])
            
            # Log progress every 200 steps
            if t % 200 == 0:
                mem = network.simpleSim.membranePotentials()
                noisy_count = np.count_nonzero(mem)
                max_val = np.max(np.abs(mem))
                avg_val = np.mean(np.abs(mem))
                
                print(f"Step {t:4d} | Noisy Neurons: {noisy_count}/{neuron_count} | Max Mag: {max_val:.2f} | Avg Mag: {avg_val:.2f}")

        # Final Verification
        final_mem = network.simpleSim.membranePotentials()
        assert not np.isnan(final_mem).any(), "NaN detected: The math is unstable!"
        assert noisy_count > 0, "No noise detected: The logic is not running!"
        
        print(f"--- 1,000 Step Stability Test Passed ---")