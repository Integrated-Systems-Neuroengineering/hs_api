import hs_bridge
from krish_icrcdemo_hardware_Ch100_conv3 import DVSGestureNetChrisModeled
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader, Subset
from spikingjelly.datasets import pad_sequence_collate
from hs_api.quantizer import Quantize_Network #initially just hs_api.converter
from utils_krish import test_DVS_Time

from hs_api.api import CRI_network
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_tensor
from hs_api.neuron_models import IF_neuron
import torch.nn.functional as F
import numpy as np
import os
from hs_api.custom_neurons import Custom_LIFNode, Custom_IFNode
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
import pickle
import random
import argparse



'''
Adapted from /LeNet5/LeNet5_Converter.py
Implements clock cycle and hbmaccesses recording
Meant for two-channel data.
Works with SpikingJelly IFNeurons
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHANGE HERE FOR DIFFERENT MODELS
threshold = 32767 #needs to be set according to quantiztion range(if you want, run the code once, the quantization step should output what the quantized threshold actually is, and then set this, and rerun)
N = IF_neuron(threshold, nu=-17, legacy_noise_en=1)
kernel_size = 5      #kernel size of convolutional layers
stride = 2           #stride of convolutional layers
input_res = 63#90 #28       #resolution of input MNIST image
conv1_output_res = 30#43 #12     #resolution of output feature maps from conv1
conv2_output_res = 13#20 #4     #resolution of output feature maps from conv2
conv3_output_res = 5
num_layers = 6 #(3 conv, 3 fc)
num_outputs = 11 #DVSGestureNet has 11 classes
test_args = argparse.Namespace(epochs=1, targets=num_outputs)


data_dir = "/home/gwen/testing/hs_api/examples/DVS128Gesture"
PATH = "/home/prpandit/hs_api/tests/checkpoint_max_T_10_C_100_lr_0.001.pth"

# Initialize layer statistics
layer_stats = {
    'C1': {'neurons': 0, 'incoming_synapses': 0, 'outgoing_synapses': 0},
    'C2': {'neurons': 0, 'incoming_synapses': 0, 'outgoing_synapses': 0},
    'C3': {'neurons': 0, 'incoming_synapses': 0, 'outgoing_synapses': 0},
    'FC1': {'neurons': 0, 'incoming_synapses': 0, 'outgoing_synapses': 0},
    'FC2': {'neurons': 0, 'incoming_synapses': 0, 'outgoing_synapses': 0}
}

#Loading the dataset and preprocessing
# resize transform that iterates over the temporal dimension, binarizes
class DVSResizeAndBinarize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        frames, label = data if isinstance(data, tuple) else (data, None)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        T, C, H, W = frames.shape

        resized = torch.zeros((T, C, self.size[0], self.size[1]), dtype=frames.dtype, device=frames.device)
        for t in range(T):
            frame = frames[t]  # [C, H, W]
            resized_frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
            ).squeeze(0)
            binarized_frame = (resized_frame > 0).float()
            resized[t] = binarized_frame
        return (resized, label) if label is not None else resized
    
# Use our simple resize transform for all datasets
resize_transform = DVSResizeAndBinarize(size=(input_res, input_res))  # resize from 128x128 to input_res

# Load training dataset
full_train_set = DVS128Gesture(
    root=data_dir, 
    frames_number=10, 
    split_by="number", 
    train=True, 
    data_type="frame", 
    duration=1600000,
    transform=resize_transform
)

# Create 85%-15% train-validation split
full_train_size = len(full_train_set)
val_size = int(0.15 * full_train_size)
train_size = full_train_size - val_size

torch.manual_seed(1)  # ensure same split every time
indices = torch.randperm(full_train_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create training dataset with train augments if wanted
train_set_aug = DVS128Gesture(
    root=data_dir, 
    frames_number=10, 
    split_by="number", 
    train=True, 
    data_type="frame", 
    duration=1600000,
    transform=resize_transform
)

# Create subsets
train_set = Subset(train_set_aug, train_indices)
val_set = Subset(full_train_set, val_indices)  # No augmentation

test_set = DVS128Gesture(
    root=data_dir, 
    frames_number=10, 
    split_by="number", 
    train=False, 
    data_type="frame", 
    duration=1600000,
    transform=resize_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=pad_sequence_collate,
)
val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    collate_fn=pad_sequence_collate,
)
test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    collate_fn=pad_sequence_collate,
)


print(f"Training samples: {len(train_set)} ({len(train_set)/full_train_size*100:.1f}%)")
print(f"Validation samples: {len(val_set)} ({len(val_set)/full_train_size*100:.1f}%)")
print(f"Test samples: {len(test_set)}")

T, C, H, W = full_train_set[0][0].shape
print(f"Input shape: {(T, C, H, W)}")
print(f"Number of training samples: {len(train_set)}")
print(f"Number of validation samples: {len(val_set)}")
print(f"Number of testing samples: {len(test_set)}")

print(f"Test set example(data shape, label), {test_set[0][0].shape, test_set[0][1]}")


#CHANGE HERE FOR DIFFERENT MODELS
#load model architecture and model weights
model = DVSGestureNetChrisModeled(
        spiking_neuron=Custom_IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
)

print(model)
checkpoint = torch.load(
        PATH,
        weights_only=False,
    )
model.load_state_dict(checkpoint["net"])
#print(checkpoint["net"])


#quantize model
model.eval()
print("using val set for testing")
#test_loader = val_loader
# Test original model accuracy on test dataset
print("\n" + "="*50)
print("TESTING ORIGINAL MODEL ACCURACY")
print("="*50)
original_accuracy, original_loss = test_DVS_Time(test_args, model, test_loader, device, None)

# bn = BN_Folder()
# net_bn = bn.fold(net)

net_bn = model #skipping BN in this case
print("No batch normalization")

# Quantization with dynamic alpha and optional membrane potential quantization
print("\n" + "="*50)
print("QUANTIZING MODEL")
print("="*50)
qn = Quantize_Network(w_alpha=1, dynamic_alpha=False) #tau=2.0 by default


# Print quantization parameters
print("QUANTIZATION PARAMETERS:")
print("="*50)
print(f"w_alpha: {qn.w_alpha}")
print(f"dynamic_alpha: {qn.dynamic_alpha}")
print(f"w_bits: {qn.w_bits}")
print(f"w_delta: {qn.w_delta}")
print("="*50)

net_quan = qn.quantize(net_bn)

#convert FP32 weights to INT16
#int16_sd, scales = fp32_to_int16_state_dict(model)
#int16_sd is now the state dict of the quantized model
int16_sd = net_quan.state_dict()
print(f"int16_sd keys: {int16_sd.keys()}")

#CHANGE HERE FOR DIFFERENT MODELS
conv1_weight = int16_sd["conv_fc.0.weight"]
conv2_weight = int16_sd["conv_fc.2.weight"]
conv3_weight = int16_sd["conv_fc.4.weight"]
fc1_weight = int16_sd["conv_fc.7.weight"]
fc2_weight = int16_sd["conv_fc.9.weight"]
fc3_weight = int16_sd["conv_fc.11.weight"]

# EXPERIMENTING, TEST: Force type cast weights to int16(right now in fp)
#print("original weight types: ", conv1_weight.dtype, conv2_weight.dtype, fc1_weight.dtype, fc2_weight.dtype, fc3_weight.dtype)
int16_sd["conv_fc.0.weight"] = conv1_weight.to(torch.int16)
int16_sd["conv_fc.2.weight"] = conv2_weight.to(torch.int16)
int16_sd["conv_fc.4.weight"] = conv3_weight.to(torch.int16)
int16_sd["conv_fc.7.weight"] = fc1_weight.to(torch.int16)
int16_sd["conv_fc.9.weight"] = fc2_weight.to(torch.int16)
int16_sd["conv_fc.11.weight"] = fc3_weight.to(torch.int16)

conv1_weight = int16_sd["conv_fc.0.weight"]
conv2_weight = int16_sd["conv_fc.2.weight"]
conv3_weight = int16_sd["conv_fc.4.weight"]
fc1_weight = int16_sd["conv_fc.7.weight"]
fc2_weight = int16_sd["conv_fc.9.weight"]
fc3_weight = int16_sd["conv_fc.11.weight"]

#print("new weight types: ", conv1_weight.dtype, int16_sd["conv_fc.2.weight"].dtype, int16_sd["conv_fc.5.weight"].dtype, int16_sd["conv_fc.7.weight"].dtype, int16_sd["conv_fc.9.weight"].dtype)


net_quan.eval()
# Test quantized model accuracy
print("\n" + "="*50)
print("TESTING QUANTIZED MODEL ACCURACY")
print("="*50)
quantized_accuracy, quantized_loss = test_DVS_Time(test_args, net_quan, test_loader, device, None)


# Calculate and display accuracy drop
accuracy_drop = original_accuracy - quantized_accuracy
print(f"\n{'='*50}")
print(f"QUANTIZATION RESULTS:")
print(f"Original Accuracy:   {original_accuracy:.4f}")
print(f"Quantized Accuracy:  {quantized_accuracy:.4f}")
print(f"Accuracy Drop:       {accuracy_drop:.4f} ({accuracy_drop/original_accuracy*100:.2f}%)")
print(f"Original Loss:       {original_loss:.4f}")
print(f"Quantized Loss:      {quantized_loss:.4f}")
print("Loss increase:       {:.4f}".format(quantized_loss - original_loss))
print(f"{'='*50}")

#defining dictionaries and input/output lists
axons = {}
connections = {}
inputs = []
outputs = []




# For two channels, just double the number of axons
for i in range(2 * input_res * input_res):
    key = f"A{i}"
    axons[key] = []

# Build axonMap for both channels
axonMap = torch.arange(2 * (input_res ** 2), dtype=torch.float32).reshape(1, 2, input_res, input_res)
patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size, stride=stride)
patch_rows = patchTensor.transpose(1, 2).squeeze(0)  # shape: [num_patches, kernel_size*kernel_size*2]
patch_rows = patch_rows.to(torch.int16)



# iterate through every weight kernel in first convolutional layer and map axons → (neuron, weight)
print("conv1 weight shape: ", conv1_weight.shape)  # Should be (6, 2, 5, 5)
for feature_map, kernel in enumerate(conv1_weight):  # kernel shape: [2, 5, 5]
    flat_kernel = kernel.flatten()  # shape: [50]
    for index, row in enumerate(patch_rows):  # row shape: [50]
        neuronName = f"C1.{feature_map}.{index}"
        connections[neuronName] = ([], N)
        for i, elem in enumerate(row):
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
print("weight shape for conv2: ", conv2_weight.shape) 
for output_idx, output_channel in enumerate(conv2_weight): 
    #print(output_idx, output_channel.shape)
    #inner loop: iterate over input-channel kernels with index for this output channel
    for feature_map, kernel in enumerate(output_channel):  
        #print(kernel.shape)
        #print(patch_rows.shape)
        flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C1 -> C2
        #print("path_rows shape: ", patch_rows.shape)
        #inner loop 2: iterate through each patch row. #rows = resolution of output feature map 
        for j, row in enumerate(patch_rows):
            neuronName = f"C2.{output_idx}.{j}"  #each row corresponds to one pixel in feature map. Create neuron entry C2.{feature map#}.{index}
            connections[neuronName] = ([], N)
            #print(neuronName)
    
            #inner loop 3: iterate through each elem in row. Each elem is index of C1 -> C2 
            for i, elem in enumerate(row):  
                index = int(elem.item())
                key = f"C1.{feature_map}.{index}"
                weight = flat_kernel[i].item()
                connections[key][0].append((neuronName, weight))

#creating C2Map to identify which C2 neurons connect to which pixel/neuron of the feature map in conv3
C2Map = torch.arange(conv2_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv2_output_res, conv2_output_res)          
patchTensor = F.unfold(input=C2Map, kernel_size=kernel_size, stride=stride)   # padding=0, dilation=1 by default
patch_rows = patchTensor.transpose(1, 2).squeeze(0)
patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor

#connecting C1 neurons in conv1 to C2 neurons in conv2
#outer loop: iterate over output channels in conv2
print("weight shape for conv3: ", conv3_weight.shape) 
for output_idx, output_channel in enumerate(conv3_weight): 

    #inner loop: iterate over input-channel kernels with index for this output channel
    for feature_map, kernel in enumerate(output_channel):  
        flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C2 -> C3
       
        #inner loop 2: iterate through each patch row. #rows = resolution of output feature map 
        for j, row in enumerate(patch_rows):
            neuronName = f"C3.{output_idx}.{j}"  #each row corresponds to one pixel in feature map. Create neuron entry C3.{feature map#}.{index}
            connections[neuronName] = ([], N)
    
            #inner loop 3: iterate through each elem in row. Each elem is index of C2 -> C3 
            for i, elem in enumerate(row):  
                index = int(elem.item())
                key = f"C2.{feature_map}.{index}"
                weight = flat_kernel[i].item()
                connections[key][0].append((neuronName, weight))

#connecting conv3 to fc1
feature_map = 0
print("fc1 shape: ", fc1_weight.shape)
#print(fc1_weight.shape[1])
for col in range(fc1_weight.shape[1]):  #x.shape[1] == number of col
    if col % (conv3_output_res ** 2) == 0 and col != 0:  #determines the feature_map of the C2 neuron for C2 --> FC1
        feature_map += 1
    #print("feature map: ", feature_map)
    for i, elem in enumerate(fc1_weight[:, col]):     #iterate over element in a col
        connectingNeuron = (f"FC1.{i}", elem.item())
        connections[f"C3.{feature_map}.{col % (conv3_output_res ** 2)}"][0].append(connectingNeuron)

#connecting fc1 to fc2
print("fc2 shape: ", fc2_weight.shape)
for col in range(fc2_weight.shape[1]):  #x.shape[1] == number of col
    allConnections = []
    for i, elem in enumerate(fc2_weight[:, col]):     #iterate over element in a col
        connectingNeuron = (f"FC2.{i}", elem.item())
        allConnections.append(connectingNeuron)
    connections[f"FC1.{col}"] = (allConnections, N)

#connecting fc2 to fc3
print("fc3 shape: ", fc3_weight.shape)
for col in range(fc3_weight.shape[1]):  #x.shape[1] == number of col
    allConnections = []
    for i, elem in enumerate(fc3_weight[:, col]):     #iterate over element in a col
        connectingNeuron = (i, elem.item())
        allConnections.append(connectingNeuron)
    connections[f"FC2.{col}"] = (allConnections, N)


#creating output neurons
outputs = []
for x in range(num_outputs):
    connections[x] = ([], N)
    outputs.append(x)

#counting synapses of network
number_synapses = 0
max_synapses_per_neuron = 0
max_synapses_per_axon = 0
max_neuronal_fanin = 0
max_fan_in_per_axon = 0
synapses_per_neuron = {}
synapses_per_axon = {}
axonal_fanin_count = {}
fan_in_count = {}
for key in connections:
    #calculate outgoing synapses for each neuron and store in synapses_per_neuron dict
    outgoing = len(connections[key][0])
    number_synapses += outgoing
    synapses_per_neuron[key] = outgoing
    if outgoing > max_synapses_per_neuron:
        max_synapses_per_neuron = outgoing

    #track outgoing by layer
    if isinstance(key, str):  #skip output neurons which are ints
        layer = key.split('.')[0]
        if layer in layer_stats:
            layer_stats[layer]['neurons'] += 1
            layer_stats[layer]['outgoing_synapses'] += outgoing

    #count incoming synapses for each neuron
    for conn in connections[key][0]:
        target_neuron = conn[0]
        if target_neuron not in fan_in_count:
            fan_in_count[target_neuron] = 0
        fan_in_count[target_neuron] += 1
        if fan_in_count[target_neuron] > max_neuronal_fanin:
            max_neuronal_fanin = fan_in_count[target_neuron]

for key in axons:
    #calculate outgoing synapses for each axon
    number_synapses += len(axons[key])
    synapses_per_axon[key] = len(axons[key])
    if len(axons[key]) > max_synapses_per_axon:
        max_synapses_per_axon = len(axons[key])

    #count synapses from axons to neurons 
    for conn in axons[key]:
        target_neuron = conn[0]
        if target_neuron not in axonal_fanin_count:
            axonal_fanin_count[target_neuron] = 0
        axonal_fanin_count[target_neuron] += 1
        if axonal_fanin_count[target_neuron] > max_fan_in_per_axon:
            max_fan_in_per_axon = axonal_fanin_count[target_neuron]

# Calculate incoming synapses per layer
for neuron_name in connections.keys():
    if isinstance(neuron_name, int):  #skip output neurons
        continue
    layer_name = neuron_name.split('.')[0]
    if layer_name in layer_stats:
        # Incoming from axons
        incoming_from_axons = axonal_fanin_count.get(neuron_name, 0)
        # Incoming from other neurons
        incoming_from_neurons = fan_in_count.get(neuron_name, 0)
        # Total incoming
        layer_stats[layer_name]['incoming_synapses'] += incoming_from_axons + incoming_from_neurons


print(f"Number of LIF neurons: {len(connections)}")
print(f"Number of axons: {len(axons)}")
print(f"Number of synapses: {number_synapses}")
print(f"Max Fan Out per neuron: {max_synapses_per_neuron}")
print(f"Max Fan Out per axon: {max_synapses_per_axon}")
print(f"Max Fan In per neuron (from x neurons): {max_neuronal_fanin}")
print(f"Max Fan In per neuron (from x axons): {max_fan_in_per_axon}")

# Print per-layer statistics
print("\n=== Per-Layer Statistics ===")
for layer_name in layer_stats.keys():
    stats = layer_stats[layer_name]
    print(f"\n{layer_name}:")
    print(f"  Neurons: {stats['neurons']}")
    print(f"  Total Incoming Synapses: {stats['incoming_synapses']}")
    print(f"  Total Outgoing Synapses: {stats['outgoing_synapses']}")
    print((f" Total Synapses: {stats['incoming_synapses'] + stats['outgoing_synapses']}"))

'''
#record MPs of randomly chosen neurons, one from each of the 100 feature maps in conv1
neurons = []
chosen_ys = []
random.seed(42)  # to make chosen y's reproducible 

for x in range(100):
    y = random.randint(0, 899)
    neurons.append(f"C1.{x}.{y}")
    chosen_ys.append(y)

print("Neuron list:", neurons)
print("Chosen y indices:", chosen_ys)
'''
#record MPs of randomly chosen neurons, one from each of the 100 feature maps in conv1
neurons = []
chosen_ys = []
random.seed(42)  # to make chosen y's reproducible 

#random conv1 neurons
for x in range(30):
    y = random.randint(0, 899)
    neurons.append(f"C1.{x}.{y}")
    chosen_ys.append(y)

#random conv2 neurons
for x in range(30):
    y = random.randint(0, 168)
    neurons.append(f"C2.{x}.{y}")
    chosen_ys.append(y)

#random conv3 neurons
for x in range(30):
    y = random.randint(0, 24)
    neurons.append(f"C3.{x}.{y}")
    chosen_ys.append(y)

print("Neuron list:", neurons)
print("Chosen y indices:", chosen_ys)

#create network
network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")

#used to save clock cycles and hbm accesses
data = []


#test model
correct = 0
total = 0
images = 0 
encoder = encoding.PoissonEncoder()
loss_fn = nn.CrossEntropyLoss()
test_loss = 0
last_label = -1
for img, label in test_set:
    #reset membrane potnetials before each image
    hs_bridge.FPGA_Execution.fpga_controller.clear(
                    len(connections), False, 0
                )  ##Num_neurons, simDump, coreOverride

    initial = network.read_membrane(outputs)
    #print(f"Membrane potentials at start: {initial}")

    #print(f"Image shape: {img.shape}, Label: {label}")
    img = img.to(device) #shape [T, C, H, W]
    spike_counts = torch.zeros(len(outputs))  #to count spikes over all frames
    for t in range(img.shape[0]):  #iterate through all frames
        frame = img[t,:,:,:] #shape [C, H, W]
        #print(f"Frame shape: {frame.shape}")

        #convert from [C, H, W] to [1, C*H*W]
        input = frame.unsqueeze(0)  #add dimension
        #input = encoder(input)
        input = input.flatten(start_dim=1)
        input = input.to(torch.int16)        #change input from FP32 to INT16
        print(f"Input shape: {input.shape}")
        #create input list
        inputs = []
        #print all unique input values
        #print(f"Unique input values: {input.unique()}")
        for i, elem in enumerate(input[0, :]):
            if elem.item() > 0: #changed from == 1 for MNIST
                inputs.append(f"A{i}")



        hardwareSpikes, _, _ = network.step(inputs)
        print(f"Output spikes: {hardwareSpikes}")

        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1
            else:
                print(f"Error: invalid output spike {spike}")
        
        sampleMPs = network.read_membrane(neurons)
        print(f"Membrane potentials of sample neurons at time {t}: {sampleMPs}")

        results = network.read_membrane(outputs)
        print(f"Membrane potentials: {results}")

    #add extra timesteps after lastinput frame to allow it to propogate through network
    for i in range(num_layers):
        inputs = []  #no input spikes
        hardwareSpikes, clock_cycles, hbm_accesses = network.step(inputs)
        print(f"Output spikes: {hardwareSpikes}")

        for spike in hardwareSpikes:
            if spike in outputs:
                spike_counts[spike] += 1
            else:
                print(f"Error: invalid output spike {spike}")
    
    
    #record clock cycles and hbm accesses
    data.append((clock_cycles, hbm_accesses))

    spike_counts = spike_counts / img.size(0)  #average spike counts(spike rate)
    print(f"Spike counts: {spike_counts}")

    predicted = torch.argmax(spike_counts).item()
    print(f"Predicted: {predicted}, Label: {label}")

    if label != last_label:
        print(f"Label changed from {last_label} to {label}")
        last_label = label
    
    total += 1
    if predicted == label:
        correct += 1
    
    running_accuracy = 100 * correct / total
    print(f"Running accuracy : {running_accuracy:.2f} %")


    #calculate loss
    label_onehot = F.one_hot(torch.tensor(label), num_classes=num_outputs).float()
    loss = loss_fn(spike_counts, label_onehot)
    print(f"Loss: {loss}")
    test_loss += loss.item()

accuracy = 100 * correct / total
loss = test_loss / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
print(f'Test Loss: {loss:.4f}')

'''
#record (clockcyles, hbmaccess) as ordered pairs in numpy arr
arr = np.asarray(data)              

#Save converter FPGA accuracy to txt file and clock cycles to npy file
parent_directory = os.path.dirname(PATH)

#if accuracies.txt file already exists, just append converted accuracy to it, otherwise, create new file
if os.path.exists(os.path.join(parent_directory, "accuracies_ch100.txt")):
    mode = "a"
else:
    mode = "w"

with open(os.path.join(parent_directory, "accuracies_ch100.txt"), mode) as f:
    f.write(f"FPGA Converted Accuracy: {accuracy:.2f}%\n")

np.save(os.path.join(parent_directory, "clock_cycles_DVS_2x63x63_Ch=100_conv=3.npy"), arr)
'''