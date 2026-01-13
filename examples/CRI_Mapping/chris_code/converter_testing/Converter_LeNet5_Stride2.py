from hs_api.converter import CRI_Converter, Quantize_Network
from hs_api.api import CRI_network
import QAT_LeNet5_Stride2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_tensor
from hs_api.neuron_models import ANN_neuron
import torch.nn.functional as F
import numpy as np

'''
Adapted from /LeNet5/LeNet5_Converter.py
Implements clock cycle and hbmaccesses recording
'''

N = ANN_neuron(threshold = 0, shift = 0)   #neuron mode
kernel_size = 5      #kernel size of convolutional layers
stride = 2           #stride of convolutional layers
input_res = 28       #resolution of input MNIST image
conv1_output_res = 12     #resolution of output feature maps from conv1
conv2_output_res = 4      #resolution of output feature maps from conv2

#binarize MNIST image
class Binarize(object):
    """Convert a tensor with values in [0,1] to {0,1} by thresholding."""
    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is C×H×W, already float32 after ToTensor
        return (tensor > self.thresh).float()

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

#reads specific MPs from a list of specified neurons
def membrane_potential_reader(results: tuple, neurons: list):
    membrane_potentials = []
    membrane_potentials_dict = dict(results[0]) # convert to dict for easy look up
    for label in neurons:       #iterate through all output neurons
        membrane_potentials.append(membrane_potentials_dict[label])
    
    return membrane_potentials  #return output neuron with greatest membrane potential

#determines the max membrane potential from the output neurons
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

#determines the max membrane potential from the output neurons
def max_membrane_potential2(outputs: list):
    max = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(outputs) # convert to dict for easy look up
    for key in membrane_potentials_dict:       #iterate through all output neurons
        if membrane_potentials_dict[key] > max:
            max = membrane_potentials_dict[key]
            max_label = key
    
    return max_label  #return output neuron with greatest membrane potential

#load test dataset
test_dataset = torchvision.datasets.MNIST(root = './data',
                                              train = False,
                                              transform = transforms.Compose([
                                                      transforms.Resize((input_res,input_res)),
                                                      transforms.ToTensor(),
                                                      Binarize(0.5)]),
                                              download=True)

#Loading the dataset and preprocessing
full_train_dataset = torchvision.datasets.MNIST(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                      transforms.Resize((input_res,input_res)),
                                                      transforms.ToTensor(),
                                                      Binarize(0.5)]),
                                               download = True)

#split full_train_dataset into training set and validation set
train_size = int(0.83 * len(full_train_dataset)) #50k for training
val_size = len(full_train_dataset) - train_size #10k for validation
train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

#create a separate validation dataset with the test transforms
val_dataset = torchvision.datasets.MNIST(root = './data',
                                               train = True,
                                               transform = transforms.Compose([
                                                      transforms.Resize((input_res,input_res)),
                                                      transforms.ToTensor(),
                                                      Binarize(0.5)]),
                                               download = True)

#subset only the remaining 10% of the data for validation
_, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])

#load model architecture and model weights
PATH = "/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/QAT_LeNet5_weights_Stride2"
model = QAT_LeNet5_Stride2.LeNet5(10)
model.load_state_dict(torch.load(PATH))

#convert FP32 weights to INT16
int16_sd, scales = fp32_to_int16_state_dict(model)

#defining dictionaries and input/output lists
axons = {}
connections = {}
inputs = []
outputs = []

#create axon for every pixel in image
for i in range(input_res * input_res):
    key = f"A{i}"
    axons[key] = []

#creating axonMap to identify which axons connect to which pixel/neuron of the feature map in conv1
axonMap = torch.arange(input_res ** 2, dtype=torch.float32).reshape(1, 1, input_res, input_res)          
patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size, stride=stride)   # padding=0, dilation=1 by default

#patch_rows is a tensor where #rows = resolution of feature maps.
#Each row contains the indices of the axons corresponding to each pixel in the feature map
patch_rows = patchTensor.transpose(1, 2).squeeze(0) 
patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor

# iterate through every weight kernel in first convolutional layer and map axons → (neuron, weight)
#print("conv1 weight: ", int16_sd["conv1.weight"].shape)
for feature_map, kernel in enumerate(int16_sd["conv1.weight"]):
    flat_kernel = kernel.flatten()
    #print(kernel.shape)
    #print(patch_rows.shape)
    for index , row in enumerate(patch_rows):
        neuronName = f"C1.{feature_map}.{index}"  #each row corresponds to one pixel in feature map. Create neuron entry C1.{feature map#}.{index}
        connections[neuronName] = ([], N)
    
        for i, elem in enumerate(row):   #each elem in row is the axon index
            axon_id = int(elem.item())
            key = f"A{axon_id}"
            weight = flat_kernel[i].item()
            axons[key].append((neuronName, weight))

#creating C1Map to identify which C1 neurons connect to which pixel/neuron of the feature map in conv2
C1Map = torch.arange(conv1_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv1_output_res, conv1_output_res)          
patchTensor = F.unfold(input=C1Map, kernel_size=kernel_size, stride=stride)   # padding=0, dilation=1 by default

#patch_rows is a tensor where #rows = resolution of feature maps.
#Each row contains the indices of the axons corresponding to each pixel in the feature map
patch_rows = patchTensor.transpose(1, 2).squeeze(0)
patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor


#connecting C1 neurons in conv1 to C2 neurons in conv2
#outer loop: iterate over output channels in conv2
print("conv2 weight shape: ", int16_sd["conv2.weight"].shape)
for output_idx, output_channel in enumerate(int16_sd["conv2.weight"]): 
    #print(output_idx, output_channel.shape)
    #inner loop: iterate over input-channel kernels with index for this output channel
    for feature_map, kernel in enumerate(output_channel):  
        flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C1 -> C2
        #print("path_rows shape: ", patch_rows.shape)
        #inner loop 2: iterate through each patch row. #rows = resolution of output feature map 
        print(kernel.shape)
        print(patch_rows.shape)
        for j, row in enumerate(patch_rows):
            neuronName = f"C2.{output_idx}.{j}"  #each row corresponds to one pixel in feature map. Create neuron entry C2.{feature map#}.{index}
            connections[neuronName] = ([], N)
            print(neuronName)
    
            #inner loop 3: iterate through each elem in row. Each elem is index of C1 -> C2 
            for i, elem in enumerate(row):  
                index = int(elem.item())
                key = f"C1.{feature_map}.{index}"
                weight = flat_kernel[i].item()
                connections[key][0].append((neuronName, weight))

#connecting conv2 to fc1
feature_map = 0
print("fc1 shape: ", int16_sd["fc1.weight"].shape)
print(int16_sd["fc1.weight"].shape[1])
for col in range(int16_sd["fc1.weight"].shape[1]):  #x.shape[1] == number of col
    if col % (conv2_output_res ** 2) == 0 and col != 0:  #determines the feature_map of the C2 neuron for C2 --> FC1
        feature_map += 1
    print("feature map: ", feature_map)
    for i, elem in enumerate(int16_sd["fc1.weight"][:, col]):     #iterate over element in a col
        connectingNeuron = (f"FC1.{i}", elem.item())
        connections[f"C2.{feature_map}.{col % (conv2_output_res ** 2)}"][0].append(connectingNeuron)

#connecting fc1 to fc2
for col in range(int16_sd["fc2.weight"].shape[1]):  #x.shape[1] == number of col
    allConnections = []
    for i, elem in enumerate(int16_sd["fc2.weight"][:, col]):     #iterate over element in a col
        connectingNeuron = (f"FC2.{i}", elem.item())
        allConnections.append(connectingNeuron)
    connections[f"FC1.{col}"] = (allConnections, N)


#creating output neurons
outputs = []
for x in range(10):
    connections[x] = ([], N)
    outputs.append(x)

#counting synapses of network
number_synapses = 0
for key in connections:
    number_synapses += len(connections[key][0])

for key in axons:
    number_synapses += len(axons[key])

print(f"Number of ANNs: {len(connections)}")
print(f"Number of axons: {len(axons)}")
print(f"Number of synapses: {number_synapses}")

#create network
network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")

#used to save clock cycles and hbm accesses
spikes = []

#run testing
correct = 0
total = 0
images = 0    #used if want to end inferencing on test dataset early


for img, labels in test_dataset:
    input = img.reshape(img.size(0), -1) #flatten input to [1, 36]
    input = input.to(torch.int16)        #change input from FP32 to INT16

    #create input list
    inputs = []
    for i, elem in enumerate(input[0, :]):
        if elem.item() == 1:
            inputs.append(f"A{i}")
    

    #running for 5 timsteps
    currSpikes = network.step(inputs) #1st time step through Conv1
    currSpikes = network.step([])     #2nd time step through Conv2
    currSpikes = network.step([])     #3rd time step through fc1
    currSpikes = network.step([])     #4th time step through fc2
    currSpikes = network.step([])     #5th time step through fc3
    results = network.read_membrane(outputs)

    #record clock cycles and hbm accesses
    spikes.append(currSpikes)

    #compare predicted with ground truth
    #predicted_1, _ = max_membrane_potential(currSpikes, outputs)
    predicted_1 = max_membrane_potential2(results) #index of max membrane potential == predicted
    
    total += 1
    if predicted_1 == labels:
        correct += 1

    running_accuracy = 100 * correct / total
    print(f"Running accuracy : {running_accuracy:.2f} %")
    
    #images += 1  
    #if images >= 10:
        #break

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

#record (clockcyles, hbmaccess) as ordered pairs in numpy arr
data = []
for item in spikes:
    # Unpack the tuple
    _, clock_cycles, hbm_accesses= item
    # Append the pair to the list
    data.append((clock_cycles, hbm_accesses))

arr = np.asarray(data)              
np.save("clock_cycles_LeNet5_Stride2.npy", arr)