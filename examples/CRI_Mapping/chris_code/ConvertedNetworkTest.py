from hs_api.api import CRI_network
import ComplexLinearModel
import torch
import torchvision
import torchvision.transforms as transforms
from hs_api.neuron_models import ANN_neuron

N = ANN_neuron(threshold = 1, shift = 0)
input_size = 784

#binarize MNIST image
class Binarize(object):
    """Convert a tensor with values in [0,1] to {0,1} by thresholding."""
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is C×H×W, already float32 after ToTensor
        return (tensor > self.thresh).float()
    

#load test dataset
test_dataset = torchvision.datasets.MNIST(root = './data',
                                              train = False,
                                              transform = transforms.Compose([
                                                      transforms.Resize((28,28)),
                                                      transforms.ToTensor(),
                                                      Binarize(0.5)]),
                                              download=True)

#convert fp32 weights in model into int16
def fp32_to_int16_state_dict(model: torch.nn.Module):
    """Return two dicts:
       1. int16 weights   2. per‑tensor scale factors (float32)"""
    int16_sd, scales = {}, {}
    for name, tensor in model.state_dict().items():
        max_val = tensor.abs().max()
        if max_val == 0:
            max_val = 1 #avoid divide-by-zero
        scale   = (2**15 - 1) / max_val
        int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
        scales[name]   = scale.item()
    return int16_sd, scales

#determines the max membrane potential from the output neurons
'''
def max_membrane_potential(object: tuple, outputs: list):
    output_membrane_potentials = []
    max = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(object[0]) # convert to dict for easy look up
    for label in outputs:       #iterate through all output neurons
        output_membrane_potentials.append(membrane_potentials_dict[label])
        if membrane_potentials_dict[label] > max:
            max = membrane_potentials_dict[label]
            max_label = label
    
    return max_label, output_membrane_potentials  #return output neuron with greatest membrane potential
'''
#determines the max membrane potential from the output neurons
def max_membrane_potential(outputs: list):
    max = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(outputs) # convert to dict for easy look up
    for key in membrane_potentials_dict:       #iterate through all output neurons
        if membrane_potentials_dict[key] > max:
            max = membrane_potentials_dict[key]
            max_label = key
    
    return max_label  #return output neuron with greatest membrane potential

PATH = "/home/ckdeng/myprojects/FCLinearModel/ComplexLinearModel_weights"
model = ComplexLinearModel.LinearModel(10)
model.load_state_dict(torch.load(PATH))

int16_sd, scales = fp32_to_int16_state_dict(model)

axons = {}
connections = {}

#creating axons
for i in range(input_size):   
    axonToNeuron = []
    for j, weight in enumerate(int16_sd["fc1.weight"][:, i]):
        connectingNeuron = (f"N1.{j}", weight.item())
        axonToNeuron.append(connectingNeuron)
    axons[f"A.{i}"] = axonToNeuron

#creating connections fc1
for col in range(int16_sd["fc2.weight"].shape[1]):  #x.shape[1] == number of col
    allConnections = []
    for i, elem in enumerate(int16_sd["fc2.weight"][:, col]):     #iterate over element in a col
        connectingNeuron = (i, elem.item())
        allConnections.append(connectingNeuron)
    connections[f"N1.{col}"] = (allConnections, N)

#print(connections["N1.0"])

#creating output neurons
outputs = []
for x in range(10):
    connections[x] = ([], N)
    outputs.append(x)

#create network
network = CRI_network(axons=axons,connections=connections,outputs=outputs, target="CRI")

#run testing
correct = 0
total = 0
for img, labels in test_dataset:
    input = img.reshape(img.size(0), -1) #flatten input to [1, 36]
    input = input.to(torch.int16)        #change input from FP32 to INT16

    #create input list
    inputs = []
    for i, elem in enumerate(input[0, :]):
        if elem.item() == 1:
            inputs.append(f"A.{i}")

    #running for 2 timsteps
    currSpikes = network.step(inputs) #1st time step
    currSpikes = network.step([])     #2nd time step
    results = network.read_membrane(outputs)

    #compare predicted with ground truth
    predicted = max_membrane_potential(results) #index of max membrane potential == predicted
    total += 1
    if predicted == labels:
        correct += 1

    running_accuracy = 100 * correct / total
    print(f"Running accuracy : {running_accuracy:.2f} %")

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')





    



        
    








