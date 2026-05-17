import pickle
import torch
import hs_bridge
from hs_api.api import CRI_network

def load_model_config():
    """Load saved model configuration"""
    with open('./fixtures/CIFAR10_model_config_shift=-17.pkl', 'rb') as f:
        return pickle.load(f)

def load_test_batch():
    """Load saved test batch"""
    with open('./fixtures/cifar10_test_batch.pkl', 'rb') as f:
        return pickle.load(f)

def test_cifar_accuracy():
    """Test that CIFAR model achieves expected accuracy on hardware.
    
    Test Description:
        Validates that the full CIFAR classification model runs correctly
        on the CRI hardware and achieves the expected accuracy threshold.
        
    Network Configuration:
        - Full CIFAR model loaded from saved configuration
        - Model architecture: 3 convolutional layers with stride 2, 100 channels each
        - Weights initialized from saved configuration
        
    Test Procedure:
        1. Load model configuration (axons, connections, outputs)
        2. Create CRI network
        3. Run inference on 9 test images with different labels
            Image 0: shape=torch.Size([15, 32, 32]), label=3
            Image 30: shape=torch.Size([15, 32, 32]), label=6
            Image 60: shape=torch.Size([15, 32, 32]), label=7
            Image 80: shape=torch.Size([15, 32, 32]), label=8
            Image 100: shape=torch.Size([15, 32, 32]), label=4
            Image 3000: shape=torch.Size([15, 32, 32]), label=5
            Image 2000: shape=torch.Size([15, 32, 32]), label=1
            Image 10: shape=torch.Size([15, 32, 32]), label=0
            Image 8000: shape=torch.Size([15, 32, 32]), label=9
        4. Calculate accuracy
        
    Expected Behavior:
        Accuracy >= expected threshold
        
    Rationale:
        This end-to-end test ensures the hardware correctly executes a
        real-world model. If accuracy drops below threshold, it indicates
        hardware malfunction, weight corruption, or spike readout issues.
    """
    print("Loading model configuration...")
    model_config = load_model_config()
    
    print("Loading test batch...")
    test_batch = load_test_batch()
    
    axons = model_config['axons']
    connections = model_config['connections']
    outputs = model_config['outputs']
    
    #counting synapses of network
    number_synapses = 0
    max_synapses_per_neuron = 0
    max_synapses_per_axon = 0
    max_neuronal_fanin = 0
    max_fan_in_per_axon = 0
    axonal_fanin_count = {}
    fan_in_count = {}
    for key in connections:
        number_synapses += len(connections[key][0])
        if len(connections[key][0]) > max_synapses_per_neuron:
            max_synapses_per_neuron = len(connections[key][0])
        for conn in connections[key][0]:
            target_neuron = conn[0]
            if target_neuron not in fan_in_count:
                fan_in_count[target_neuron] = 0
            fan_in_count[target_neuron] += 1
            if fan_in_count[target_neuron] > max_neuronal_fanin:
                max_neuronal_fanin = fan_in_count[target_neuron]

    for key in axons:
        number_synapses += len(axons[key])
        if len(axons[key]) > max_synapses_per_axon:
            max_synapses_per_axon = len(axons[key])
        for conn in axons[key]:
            target_neuron = conn[0]
            if target_neuron not in axonal_fanin_count:
                axonal_fanin_count[target_neuron] = 0
            axonal_fanin_count[target_neuron] += 1
            if axonal_fanin_count[target_neuron] > max_fan_in_per_axon:
                max_fan_in_per_axon = axonal_fanin_count[target_neuron]

    print(f"Number of LIF neurons: {len(connections)}")
    print(f"Number of axons: {len(axons)}")
    print(f"Number of synapses: {number_synapses}")
    print(f"Max Fan Out per neuron: {max_synapses_per_neuron}")
    print(f"Max Fan Out per axon: {max_synapses_per_axon}")
    print(f"Max Fan In per neuron (from x neurons): {max_neuronal_fanin}")
    print(f"Max Fan In per neuron (from x axons): {max_fan_in_per_axon}")

    print("Creating CRI network...")
    # Create network
    network = CRI_network(
        axons=axons,
        connections=connections,
        outputs=outputs,
        target="CRI"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test model
    correct = 0
    total = len(test_batch['images'])
    T=30
    
    print(f"\nStarting inference on {total} images...")
    print("=" * 60)
    
    #test model
    correct = 0
    total = len(test_batch['images'])
    for img, label in zip(test_batch['images'], test_batch['labels']):
        #reset membrane potentials before each image
        hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, 0) 

        img = img.to(device) #shape [C, H, W]
        img = img.unsqueeze(0)  #add batch dimension -> shape [1, C, H, W]
        img = img.flatten(start_dim=1)  #flatten to shape [1, 3*32*32]
        spike_counts = torch.zeros(len(outputs))  #to count spikes over all frames
        for t in range(T):
            inputs = [] #list of input spikes for current frame

            for i, elem in enumerate(img[0, :]):
                if elem.item() > 0:  
                    inputs.append(f"A{i}")

            hardwareSpikes, _, _ = network.step(inputs)
            #print(f"Output spikes: {hardwareSpikes}")

            for spike in hardwareSpikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        #add 5 extra timesteps after lastinput frame to allow it to propogate through network
        for i in range(5):
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
    assert accuracy >= 33, f"Expected accuracy is at least 33%, but got {accuracy:.2f}%"