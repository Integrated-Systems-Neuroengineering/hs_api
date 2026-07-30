#Implemented with new software that utilizes theta, nu, and Lambda

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
        with open('./fixtures/DVS_model_ch=10_conv=2_IFneuron.pkl', 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        """Load saved test batch"""
        with open('./fixtures/DVS_full_dataset.pkl', 'rb') as f:
            return pickle.load(f)
    
    def test_dvs_accuracy(self, model_config, test_batch):
        """Test that spikes from DVS model are recorded.
        
        Test Description:
            Validates that the DVS model runs correctly
            on the CRI hardware and reads all spikes from conv1 neurons for the first DVS instance
            
        Network Configuration:
            - Full DVS model loaded from saved configuration
            - Model architecture: 2 convolutional layers with stride 2, 10 channels each
            - Weights initialized from saved configuration
            
        Test Procedure:
            1. Load model configuration (axons, connections, outputs)
            2. Create CRI network
            3. Run inference on 1 test image
            4. Validates that conv1 neurons above threshold should spike
            
        Expected Behavior:
            At each timestep, conv1 neurons above threshold of 32676 should spike
        """
        outputs = []

        #record MPs of randomly chosen neurons
        chosen_ys = []
        random.seed(42)  # to make chosen y's reproducible 

        #random conv1 neurons
        for x in range(10):
            y = random.randint(0, 899)
            output.append(f"C1.{x}.{y}")
            chosen_ys.append(y)

        print("Neuron list:", outputs)
        print("Chosen y indices:", chosen_ys)

        axons = model_config['axons']
        connections = model_config['connections']

        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="simpleSim"
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
                

                hardwareSpikes = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

                sampleMPs = network.read_membrane(outputs)
                print(f"Membrane potentials of sample neurons at time {t}: {sampleMPs}")

                spikes_set = set(hardwareSpikes)
                if t == 2:
                    assert "C1.4.281" in spikes_set, "Expected spike from C1.4.281 at time 2"

                if t == 3:
                    assert "C1.8.754" in spikes_set, "Expected spike from C1.8.754 at time 3"
                    assert "C1.9.104" in spikes_set, "Expected spike from C1.9.104 at time 3"

                if t == 6:
                    assert "C1.5.250" in spikes_set, "Expected spike from C1.5.250 at time 6"

                if t == 7:
                    assert "C1.0.654" in spikes_set, "Expected spike from C1.0.654 at time 7"
                    
                if t == 8:
                    assert "C1.6.228" in spikes_set, "Expected spike from C1.6.228 at time 8"

                if t == 9:
                    assert "C1.7.142" in spikes_set, "Expected spike from C1.7.142 at time 9"
                    assert "C1.8.754" in spikes_set, "Expected spike from C1.8.754 at time 9"

            #add 5 extra timesteps after lastinput frame to allow it to propogate through network
            for i in range(5):
                inputs = []  #no input spikes
                hardwareSpikes = network.step(inputs)
                print(f"Output spikes: {hardwareSpikes}")

            break  #only run on first image for testing


        

        

        


           