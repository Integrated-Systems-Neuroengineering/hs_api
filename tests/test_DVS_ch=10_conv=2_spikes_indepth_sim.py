import hs_bridge
import pytest
import pickle
from hs_api.api import CRI_network
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

    @staticmethod
    def find_incoming_connections(neurons, target, neurons_only=False):
        """
        Given a target neuron, find all neurons that have an outgoing
        synapse to it, along with the connection weight.

        Returns a list of (source_neuron, weight) tuples.
        """
        incoming = []
        for source, (synapses, _neuron_type) in neurons.items():
            for dest, weight in synapses:
                if dest == target:
                    if neurons_only:
                        incoming.append(source)
                    else:
                        incoming.append((source, weight))
        return incoming
    
    def test_dvs_accuracy(self, model_config, test_batch):
        """Test that spikes from DVS model are recorded.
        
        Test Description:
            The MPs of a random sample of 10 conv2 neurons were recorded. As of 8/1/2026, the MPs of these conv2 neurons deviate
            from their expected values at timestep 2. The goal of this test is to validate new implementations of the simulator
            that tries to fix this issue. There are 5 conv2 neurons with nonzero MPs at timestep 2, and the test checks that all 
            presynaptic conv1 neurons that should fire to these conv2 neurons actually fired. The test records what presynaptic 
            conv1 neurons fired and calculates manually what the expected MP of the conv2 neurons should be at timestep 2. The test 
            only passes if all presynaptic conv1 neurons that should fire to the conv2 neurons actually fired, and the calculated MPs
            of the conv2 neurons match the ground truth MPs at timestep 2.
            
        Network Configuration:
            - Full DVS model loaded from saved configuration
            - Model architecture: 2 convolutional layers with stride 2, 10 channels each
            - Weights initialized from saved configuration
            
        Test Procedure:
            1. Load model configuration (axons, connections, outputs)
            2. Create CRI network
            3. Run inference on 1 test image
            4. Validates that conv2 neurons have correct MPs at timestep 2 by checking that all presynaptic conv1 neurons that should fire to the 
            conv2 neurons actually fired, and calculating the expected MPs of the conv2 neurons based on the firing of the presynaptic conv1 neurons.
        """
        axons = model_config['axons']
        connections = model_config['connections']
        outputs = []

        #firing conv1 neurons to postsynaptic neuron C2.0.139 at timestep 2
        firing_neurons1 = ['C1.0.619', 'C1.0.620', 'C1.1.618', 'C1.5.620', 'C1.6.619', 'C1.6.620', 'C1.8.619', 'C1.9.620', 'C1.6.622', 'C1.0.649', 'C1.2.649', 'C1.6.649', 'C1.6.650', 'C1.8.649', 'C1.9.650', 'C1.0.652', 'C1.2.651', 'C1.9.651', 'C1.9.652', 'C1.0.678', 'C1.0.679', 'C1.2.679', 'C1.5.678', 'C1.9.678', 'C1.0.680', 'C1.5.680', 'C1.6.680', 'C1.8.680', 'C1.0.682', 'C1.5.682', 'C1.9.682', 'C1.2.708', 'C1.5.708', 'C1.6.708', 'C1.2.709', 'C1.5.709', 'C1.6.709', 'C1.9.709', 'C1.0.710', 'C1.5.710', 'C1.6.710', 'C1.9.710', 'C1.0.711', 'C1.5.711', 'C1.6.711', 'C1.8.711', 'C1.9.711', 'C1.0.712', 'C1.5.712', 'C1.6.712', 'C1.7.712', 'C1.9.712', 'C1.0.738', 'C1.2.738', 'C1.5.738', 'C1.6.738', 'C1.8.738', 'C1.9.738', 'C1.5.739', 'C1.6.739', 'C1.9.739', 'C1.0.740', 'C1.2.741', 'C1.5.740', 'C1.5.741', 'C1.6.740', 'C1.9.741', 'C1.0.742', 'C1.2.742', 'C1.5.742', 'C1.6.742']

        #firing conv1 neurons to postsynaptic neuron C2.2.151 at timestep 2
        firing_neurons2 = ['C1.2.676', 'C1.2.677', 'C1.6.676', 'C1.9.676', 'C1.9.677', 'C1.0.678', 'C1.0.679', 'C1.2.679', 'C1.5.678', 'C1.9.678', 'C1.0.680', 'C1.5.680', 'C1.6.680', 'C1.8.680', 'C1.0.706', 'C1.2.706', 'C1.5.706', 'C1.6.706', 'C1.8.706', 'C1.9.706', 'C1.2.707', 'C1.5.707', 'C1.9.707', 'C1.2.708', 'C1.5.708', 'C1.6.708', 'C1.2.709', 'C1.5.709', 'C1.6.709', 'C1.9.709', 'C1.0.710', 'C1.5.710', 'C1.6.710', 'C1.9.710', 'C1.0.736', 'C1.2.736', 'C1.5.736', 'C1.6.736', 'C1.8.736', 'C1.9.736', 'C1.0.737', 'C1.2.737', 'C1.5.737', 'C1.6.737', 'C1.8.737', 'C1.9.737', 'C1.0.738', 'C1.2.738', 'C1.5.738', 'C1.6.738', 'C1.8.738', 'C1.9.738', 'C1.5.739', 'C1.6.739', 'C1.9.739', 'C1.0.740', 'C1.5.740', 'C1.6.740', 'C1.0.766', 'C1.2.766', 'C1.5.766', 'C1.6.766', 'C1.8.766', 'C1.0.767', 'C1.6.767', 'C1.8.767', 'C1.9.767', 'C1.0.768', 'C1.5.768', 'C1.8.768', 'C1.0.769', 'C1.5.769', 'C1.6.769', 'C1.8.769', 'C1.5.770', 'C1.6.770', 'C1.8.770', 'C1.1.796', 'C1.5.796', 'C1.6.796', 'C1.8.796', 'C1.0.797', 'C1.5.797', 'C1.6.797', 'C1.0.798', 'C1.1.798', 'C1.6.798', 'C1.6.799', 'C1.8.799', 'C1.1.800', 'C1.7.800']

        #firing conv1 neurons to postsynaptic neuron C2.3.108 at timestep 2
        firing_neurons3 = ['C1.5.492', 'C1.0.490', 'C1.5.490', 'C1.0.519', 'C1.1.548', 'C1.8.548', 'C1.0.549', 'C1.5.549', 'C1.0.578', 'C1.8.578', 'C1.1.580', 'C1.5.581', 'C1.0.608', 'C1.4.608', 'C1.9.611']

        #firing conv1 neurons to postsynaptic neuron C2.7.55 at timestep 2
        firing_neurons4 = ['C1.1.248', 'C1.6.278', 'C1.0.309', 'C1.7.309', 'C1.7.310']

        #firing conv1 neurons to postsynaptic neuron C2.8.59 at timestep 2
        firing_neurons5 = ['C1.7.256', 'C1.9.285', 'C1.0.314', 'C1.9.314', 'C1.9.315', 'C1.2.316', 'C1.5.317', 'C1.9.287', 'C1.4.344', 'C1.6.344', 'C1.8.344', 'C1.0.346', 'C1.2.374', 'C1.5.374', 'C1.5.375', 'C1.5.377', 'C1.4.348']

        outputs = []
        seen = set()

        firing_neurons = firing_neurons1 + firing_neurons2 + firing_neurons3 + firing_neurons4 + firing_neurons5
        print(f"Total number of firing neurons: {len(firing_neurons)} (duplicates included)")

        for neuron in firing_neurons:
            if neuron not in seen:
                outputs.append(neuron)
                seen.add(neuron)

        print(f"Length of outputs: {len(outputs)} (unique neurons)")

        firing_neurons_dict = {}
        firing_neurons_dict["C2.0.139"] = firing_neurons1
        firing_neurons_dict["C2.2.151"] = firing_neurons2
        firing_neurons_dict["C2.3.108"] = firing_neurons3
        firing_neurons_dict["C2.7.55"] = firing_neurons4
        firing_neurons_dict["C2.8.59"] = firing_neurons5

        target_neurons = ["C2.0.139", "C2.2.151", "C2.3.108", "C2.7.55", "C2.8.59"]
        target_neuron_MP_t1 = { "C2.0.139": -3516, "C2.2.151": 761, "C2.3.108": -1872, "C2.7.55": 0, "C2.8.59": 2926}
        target_neuron_ground_truth_MP_t2 = { "C2.0.139": 19825, "C2.2.151": 38817, "C2.3.108": -9977, "C2.7.55": -6581, "C2.8.59": -9541}
        spikes_set = None

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

                if t == 2:
                    spikes_set = set(hardwareSpikes)
                    sampleMPs = network.read_membrane(target_neurons)
                    print(f"Membrane potentials of target neurons at time {t}: {sampleMPs}")
                    break 
            break  #only process the first image for testing

        calculated_MPs = {}
        for neuron in target_neurons:
            neurons_dict = dict(self.find_incoming_connections(connections, neuron, neurons_only=False))

            target_neuron_MP = target_neuron_MP_t1[neuron]  #initialize with MP at timestep 1

            for pre_neuron in firing_neurons_dict[neuron]:   #iterate through all presynaptic neurons that should fire to the target neuron
                if pre_neuron not in neurons_dict:
                    print(f"Warning: Presynaptic neuron {pre_neuron} not found in connections for target neuron {neuron}")
                    raise ValueError(f"Presynaptic neuron {pre_neuron} not found in connections for target neuron {neuron}")
                if pre_neuron in spikes_set:  #check if the presynaptic neuron actually fired
                    print(f"Presynaptic neuron {pre_neuron} fired to target neuron {neuron} with weight {neurons_dict[pre_neuron]}")
                    target_neuron_MP += neurons_dict[pre_neuron]

            print(f"Total calculated MP for target neuron {neuron}: {target_neuron_MP}")
            calculated_MPs[neuron] = target_neuron_MP

        #check that all presynaptic neurons that should fire to the target neurons actually fired
        for neuron in outputs:
            if neuron in spikes_set:
                print(f"Neuron {neuron} fired as expected.")
            else:
                print(f"Warning: Neuron {neuron} did not fire as expected.")
                raise AssertionError(f"Neuron {neuron} did not fire as expected.")

        #check that calculated MPs match the ground truth MPs
        for neuron in target_neurons:
            if calculated_MPs[neuron] == target_neuron_ground_truth_MP_t2[neuron]:
                print(f"Calculated MP for neuron {neuron} matches ground truth.")
            else:
                print(f"Warning: Calculated MP for neuron {neuron} does not match ground truth.")
                raise AssertionError(f"Calculated MP for neuron {neuron} does not match ground truth.")
                    