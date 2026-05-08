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
        with open('./fixtures/DVS_model_small_config_shift=-17.pkl', 'rb') as f:
            return pickle.load(f)
    
    @pytest.fixture
    def test_batch(self):
        """Load saved test batch"""
        with open('./fixtures/DVS_test_batch.pkl', 'rb') as f:
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
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = model_config['outputs']

        breakpoint()
        # Create network
        network = CRI_network(
            axons=axons,
            connections=connections,
            outputs=outputs,
            target="CRI"
        )
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
            for i in range(4):
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
        assert accuracy >= 44, f"Expected accuracy is at least 44%, but got {accuracy:.2f}%"