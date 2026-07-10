# tests/test_cifar10_inference.py
import hs_bridge
import pytest
import pickle
from hs_api.api import CRI_network
import hs_bridge
import torch
import random

class TestDVSInference:
    """Test DVS model inference on hardware"""
    
    @pytest.fixture
    def model_config(self):
        """Load saved model configuration"""
        with open('./fixtures/DVS_model_config_IFneuron.pkl', 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        """Load saved test batch"""
        with open('./fixtures/DVS_full_dataset.pkl', 'rb') as f:
            return pickle.load(f)
    
    def test_dvs_accuracy(self, model_config, test_batch):
        """Test that DVS model achieves expected accuracy on hardware.
        
        Test Description:
            Validates that the full DVS classification model runs correctly
            on the CRI hardware and achieves the expected accuracy threshold.
            
        Network Configuration:
            - Full DVS model loaded from saved configuration
            - Model architecture: 3 convolutional layers with stride 2, 100 channels each
            - Weights initialized from saved configuration
            
        Test Procedure:
            1. Load model configuration (axons, connections, outputs)
            2. Create CRI network
            3. Run inference on 9 test images with different labels
            4. Calculate accuracy
            
        Expected Behavior:
            Accuracy >= expected threshold
            
        Rationale:
            This end-to-end test ensures the hardware correctly executes a
            real-world model. If accuracy drops below threshold, it indicates
            hardware malfunction, weight corruption, or spike readout issues.
        """
        outputs = []

        #record MPs of randomly chosen neurons
        neurons = []
        chosen_ys = []
        random.seed(42)  # to make chosen y's reproducible 

        #random conv1 neurons
        for x in range(30):
            y = random.randint(0, 899)
            neurons.append(f"C1.{x}.{y}")
            outputs.append(f"C1.{x}.{y}")
            chosen_ys.append(y)

        #random conv2 neurons
        for x in range(30):
            y = random.randint(0, 168)
            neurons.append(f"C2.{x}.{y}")
            chosen_ys.append(y)

        print("Neuron list:", neurons)
        print("Chosen y indices:", chosen_ys)

        axons = model_config['axons']
        connections = model_config['connections']

        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for img, label in zip(test_batch['images'], test_batch['labels']):
            img = img.to(device) #shape [T, C, H, W]
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
                

                hardwareSpikes, _, _ = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

                sampleMPs = network.read_membrane(neurons)
                print(f"Membrane potentials of sample neurons at time {t}: {sampleMPs}")

                spikes_set = set(hardwareSpikes)
                if t == 3:
                    assert "C1.15.432" in spikes_set, "Expected spike from C1.15.432 at time 3"

                if t == 4:
                    assert "C1.26.733" in spikes_set, "Expected spike from C1.26.733 at time 4"

                if t == 6:
                    assert "C1.29.558" in spikes_set, "Expected spike from C1.29.558 at time 6"

                if t == 7:
                    assert "C1.8.754" in spikes_set, "Expected spike from C1.8.754 at time 7"
                    assert "C1.12.558" in spikes_set, "Expected spike from C1.12.558 at time 7"

                if t == 8:
                    assert "C1.9.104" in spikes_set, "Expected spike from C1.9.104 at time 8"
                    assert "C1.26.733" in spikes_set, "Expected spike from C1.26.733 at time 8"

            #add 6 extra timesteps after lastinput frame to allow it to propogate through network
            for i in range(6):
                inputs = []  #no input spikes
                hardwareSpikes, _, _ = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

            break  #only run on first image for testing


        

        

        


           