from hs_api.api import CRI_network
import hs_bridge
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import CIFAR10_bitslicing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
T=30
extra_timesteps = 5

def load_model_config():
    """Load saved model configuration"""
    with open('/home/ckdeng/GitHub_repo/hs_api/tests/fixtures/CIFAR10_model_config.pkl', 'rb') as f:
        return pickle.load(f)

# deterministic per-channel binarization: ToTensor -> threshold
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.th = threshold
    def __call__(self, x):
        # x is a tensor in [C,H,W] with values in [0,1]
        return (x > self.th).float()
    
#reads specific MPs from a list of specified neurons
def membrane_potential_reader(results: tuple, neurons: list):
    membrane_potentials = []
    membrane_potentials_dict = dict(results[0]) # convert to dict for easy look up
    for label in neurons:       #iterate through all output neurons
        membrane_potentials.append(membrane_potentials_dict[label])
    
    return membrane_potentials  #return output neuron with greatest membrane potential

def main():
    # deterministic per-channel binarization: ToTensor -> threshold
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.PILToTensor(),
        CIFAR10_bitslicing.cifar10_to_15channel_binary,  
        Binarize(threshold=0.5),
    ])
        
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                    train = False,
                                                    transform = transform,
                                                    download=True)
    
    C, H, W = test_dataset[0][0].shape
    print(f"Input shape: {(C, H, W)}")
    print(f"Test samples: {len(test_dataset)}")

    print("Loading model configuration...")
    model_config = load_model_config()

    axons = model_config['axons']
    connections = model_config['connections']
    outputs = model_config['outputs']

    print("Creating CRI network...")
    # Create network
    network = CRI_network(
        axons=axons,
        connections=connections,
        outputs=outputs,
        target="CRI")

    print("Running inference on test set...")
    
    #run testing
    correct = 0
    total = 0
    for img, labels in test_dataset:

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

            # Forward pass through SNN with time steps
            hardwareSpikes, _, _ = network.step(inputs)

            for spike in hardwareSpikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        #add 5 extra timesteps after last input frame to allow it to propogate through network
        for i in range(extra_timesteps):
            hardwareSpikes, _, _ = network.step([])

            for spike in hardwareSpikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        spike_counts = spike_counts / T  #average spike counts(spike rate)
        print(f"Spike counts: {spike_counts}")

        predicted = torch.argmax(spike_counts).item()
        print(f"Predicted: {predicted}, Label: {labels}")

        total += 1
        if predicted == labels:
            correct += 1
        
        running_accuracy = 100 * correct / total
        print(f"Running accuracy : {running_accuracy:.2f} %")


    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')


if __name__ == "__main__":
    main()
