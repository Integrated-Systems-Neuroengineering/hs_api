from hs_api.api import CRI_network
import QAT_LeNet5_Stride2, QAT_LeNet5_MaxPooling
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_tensor
from hs_api.neuron_models import ANN_neuron
import torch.nn.functional as F
import numpy as np
import os
import pickle


N = ANN_neuron(threshold = 0, shift = 0)
model_file = QAT_LeNet5_Stride2

'''
Adapted from /LeNet5/LeNet5_Converter.py
Implements clock cycle and hbmaccesses recording
Implements converter for Padding conv layers
All axons, neurons, and feature map indices indexed to match unfold output
'''

def _conv_output_res_unfold(input_res: int, kernel: int, stride: int, padding: int, dilation: int = 1):
    """Calculate output resolution for unfold operation (always uses padding parameter)"""
    return int((input_res + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

def _get_model_layers(model: torch.nn.Module):
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    maxpool_layers = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return conv_layers, maxpool_layers, linear_layers

#CHANGE HERE FOR DIFFERENT MODELS
model_arch = model_file.LeNet5(10)
conv_layers, maxpool_layers, linear_layers = _get_model_layers(model_arch)
kernel_size_list = [layer.kernel_size[0] for layer in conv_layers]
stride_list = [layer.stride[0] for layer in conv_layers]
padding_list = [layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding for layer in conv_layers]
maxpool_kernel_size_list = [layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0] for layer in maxpool_layers]
maxpool_stride_list = [
    (layer.stride if layer.stride is not None else layer.kernel_size) if isinstance((layer.stride if layer.stride is not None else layer.kernel_size), int)
    else (layer.stride if layer.stride is not None else layer.kernel_size)[0]
    for layer in maxpool_layers
]
maxpool_padding_list = [layer.padding if isinstance(layer.padding, int) else layer.padding[0] for layer in maxpool_layers]
input_res = model_file.input_res
conv_output_res_list = []
current_res = input_res
for i, conv_layer in enumerate(conv_layers):
    current_res = _conv_output_res_unfold(current_res, kernel_size_list[i], stride_list[i], padding_list[i])
    conv_output_res_list.append(current_res)
linear_out_features_list = [layer.out_features for layer in linear_layers]
num_outputs = linear_out_features_list[-1]
extra_timesteps = len(conv_layers) + len(maxpool_layers) + len(linear_layers)

#parameters
batch_size = 64

#weights path
PATH = "/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/CIFAR10_SNN_Finetuning/Final_SNN_Model/QAT_LeNet5_weights_Stride2"
T=30

#print all these parameters for reference
print(f"Model architecture: {model_arch}")
print(f"Conv layers: {conv_layers}")
print(f"MaxPool layers: {maxpool_layers}")
print(f"Linear layers: {linear_layers}")
print(f"Kernel sizes: {kernel_size_list}")
print(f"Strides: {stride_list}")
print(f"Paddings: {padding_list}")
print(f"MaxPool kernels: {maxpool_kernel_size_list}")
print(f"MaxPool strides: {maxpool_stride_list}")
print(f"MaxPool paddings: {maxpool_padding_list}")
print(f"Input resolution: {input_res}")
print(f"Conv output resolutions: {conv_output_res_list}")
print(f"Linear output features: {linear_out_features_list}")
print(f"Number of outputs: {num_outputs}")
print(f"Extra timesteps after last input frame: {extra_timesteps}")
print(f"Batch size: {batch_size}")
print(f"Weights path: {PATH}")
print(f"Timesteps (T): {T}")

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# deterministic per-channel binarization: ToTensor -> threshold
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.th = threshold
    def __call__(self, x):
        # x is a tensor in [C,H,W] with values in [0,1]
        return (x > self.th).float()
    
#convert fp32 weights in model into int16
def fp32_to_int16_state_dict(weights: dict):
    """Return two dicts:
       1. int16 weights   
       2. per‑tensor scale factors (float32)

       Biases are scaled using their corresponding weight scale
    """
    int16_sd, scales = {}, {}

    #First pass: compute and store scales for all weight tensors
    for name, tensor in weights.items():
        if name.endswith(".weight"):
            max_val = tensor.abs().max()
            if max_val == 0:
                max_val = 1 #avoid divide-by-zero
            scale   = (2**15 - 1) / max_val
            int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
            scales[name]   = scale.item()

    #Second pass: convert biases using the scale from corresponding weights
    for name, tensor in weights.items():
        if name.endswith(".bias"):
            weight_name = name.replace(".bias", ".weight")
            if weight_name not in scales:
                raise ValueError(f"Missing corresponding weight tensor for bias: {name}")
            scale = scales[weight_name]
            int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
            scales[name] = scale  # Store bias scale under its own name
    return int16_sd, scales


#reads specific MPs from a list of specified neurons
def membrane_potential_reader(results: tuple, neurons: list):
    membrane_potentials = []
    # CRI read_membrane returns a list of (label, potential) pairs.
    # Some older call paths may pass a tuple containing that list at index 0.
    membrane_source = results[0] if isinstance(results, tuple) else results
    membrane_potentials_dict = dict(membrane_source) # convert to dict for easy look up
    for label in neurons:
        membrane_potentials.append(membrane_potentials_dict[label])
    
    return membrane_potentials


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


def main():
    #Loading the dataset and preprocessing
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

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
    
    C, H, W = test_dataset[0][0].shape
    print(f"Input shape: {(C, H, W)}")
    print(f"Test samples: {len(test_dataset)}")
    
    
    model = model_file.LeNet5(10).to(device)
    weights = torch.load(PATH, map_location=device)
    model.load_state_dict(weights)
    model.eval()

    print("using test set for testing")

    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    ############### CONVERSION TO HiAER SPIKE ####################

    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)

    int16_sd, scales = fp32_to_int16_state_dict(weights=weights)
    print("Model weights quantized to int16")

    #defining dictionaries and input/output lists
    axons = {}
    connections = {}
    outputs = []

    weight_names = [name for name in weights.keys() if name.endswith(".weight")]
    conv_weight_names = weight_names[:len(conv_layers)]
    linear_weight_names = weight_names[len(conv_layers):]

    maxPoolNeuron = ANN_neuron(threshold=1, shift=0)

    #create axon for every pixel in image
    for i in range(input_res * input_res):
        key = f"A{i}"
        axons[key] = []

    #creating axonMap to identify which axons connect to which pixel/neuron of the feature map in conv1
    axonMap = torch.arange(input_res ** 2, dtype=torch.float32).reshape(1, 1, input_res, input_res)
    patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size_list[0], stride=stride_list[0], padding=padding_list[0])

    #patch_rows is a tensor where #rows = resolution of feature maps.
    #Each row contains the indices of the axons corresponding to each pixel in the feature map
    patch_rows = patchTensor.transpose(1, 2).squeeze(0)
    patch_rows = patch_rows.to(torch.int16)

    # iterate through every weight kernel in first convolutional layer and map axons -> (neuron, weight)
    print("conv1 weight shape: ", int16_sd[conv_weight_names[0]].shape)
    for feature_map, kernel in enumerate(int16_sd[conv_weight_names[0]]):
        flat_kernel = kernel.flatten()

        for index, row in enumerate(patch_rows):
            neuronName = f"C1.{feature_map}.{index}"
            connections[neuronName] = ([], N)

            for i, elem in enumerate(row):
                axon_id = int(elem.item())
                key = f"A{axon_id}"
                weight = flat_kernel[i].item()
                axons[key].append((neuronName, weight))

    conv1_output_res = _conv_output_res_unfold(input_res, kernel_size_list[0], stride_list[0], padding_list[0])

    c1_source_prefix = "C1"
    c1_source_res = conv1_output_res
    if len(maxpool_kernel_size_list) > 0:
        #create MaxPoolingMap to identify which C1 neurons connect to downsized C1 feature maps
        maxPoolingMap = torch.arange(conv1_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv1_output_res, conv1_output_res)
        patchTensor = F.unfold(
            input=maxPoolingMap,
            kernel_size=maxpool_kernel_size_list[0],
            stride=maxpool_stride_list[0],
            padding=maxpool_padding_list[0],
        )

        #patch_rows is a tensor where #rows = resolution of downsized feature maps.
        patch_rows = patchTensor.transpose(1, 2).squeeze(0)
        patch_rows = patch_rows.to(torch.int16)

        #iterate through every feature map in conv1 -> MaxPool
        for feature_map, _ in enumerate(int16_sd[conv_weight_names[0]]):
            for index, row in enumerate(patch_rows):
                neuronName = f"C1.MAX.{feature_map}.{index}"
                connections[neuronName] = ([], maxPoolNeuron)

                for i, elem in enumerate(row):
                    src_index = int(elem.item())
                    key = f"C1.{feature_map}.{src_index}"
                    connections[key][0].append((neuronName, 1))

        c1_source_prefix = "C1.MAX"
        c1_source_res = _conv_output_res_unfold(
            conv1_output_res,
            maxpool_kernel_size_list[0],
            maxpool_stride_list[0],
            maxpool_padding_list[0],
        )

    #creating C1Map to identify which C1 neurons (or pooled C1 neurons) connect to conv2
    C1Map = torch.arange(c1_source_res ** 2, dtype=torch.float32).reshape(1, 1, c1_source_res, c1_source_res)
    patchTensor = F.unfold(input=C1Map, kernel_size=kernel_size_list[1], stride=stride_list[1], padding=padding_list[1])

    patch_rows = patchTensor.transpose(1, 2).squeeze(0)
    patch_rows = patch_rows.to(torch.int16)

    #connecting C1.MAX neurons to C2 neurons in conv2
    print("weight shape for conv2: ", int16_sd[conv_weight_names[1]].shape)
    for output_idx, output_channel in enumerate(int16_sd[conv_weight_names[1]]):
        for feature_map, kernel in enumerate(output_channel):
            flat_kernel = kernel.flatten()

            for j, row in enumerate(patch_rows):
                neuronName = f"C2.{output_idx}.{j}"
                connections[neuronName] = ([], N)

                for i, elem in enumerate(row):
                    src_index = int(elem.item())
                    key = f"{c1_source_prefix}.{feature_map}.{src_index}"
                    weight = flat_kernel[i].item()
                    connections[key][0].append((neuronName, weight))

    conv2_output_res = _conv_output_res_unfold(c1_source_res, kernel_size_list[1], stride_list[1], padding_list[1])

    c2_source_prefix = "C2"
    c2_source_res = conv2_output_res
    if len(maxpool_kernel_size_list) > 1:
        #create MaxPoolingMap2 to identify which C2 neurons connect to downsized C2 feature maps
        maxPoolingMap2 = torch.arange(conv2_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv2_output_res, conv2_output_res)
        patchTensor = F.unfold(
            input=maxPoolingMap2,
            kernel_size=maxpool_kernel_size_list[1],
            stride=maxpool_stride_list[1],
            padding=maxpool_padding_list[1],
        )

        patch_rows = patchTensor.transpose(1, 2).squeeze(0)
        patch_rows = patch_rows.to(torch.int16)

        #iterate through every feature map in conv2 -> MaxPool
        for feature_map, _ in enumerate(int16_sd[conv_weight_names[1]]):
            for index, row in enumerate(patch_rows):
                neuronName = f"C2.MAX.{feature_map}.{index}"
                connections[neuronName] = ([], maxPoolNeuron)

                for i, elem in enumerate(row):
                    src_index = int(elem.item())
                    key = f"C2.{feature_map}.{src_index}"
                    connections[key][0].append((neuronName, 1))

        c2_source_prefix = "C2.MAX"
        c2_source_res = _conv_output_res_unfold(
            conv2_output_res,
            maxpool_kernel_size_list[1],
            maxpool_stride_list[1],
            maxpool_padding_list[1],
        )

    #connecting conv2 (or pooled conv2) to fc1
    feature_map = 0
    print("fc1 shape: ", int16_sd[linear_weight_names[0]].shape)
    for col in range(int16_sd[linear_weight_names[0]].shape[1]):
        if col % (c2_source_res ** 2) == 0 and col != 0:
            feature_map += 1
        for i, elem in enumerate(int16_sd[linear_weight_names[0]][:, col]):
            connectingNeuron = (f"FC1.{i}", elem.item())
            connections[f"{c2_source_prefix}.{feature_map}.{col % (c2_source_res ** 2)}"][0].append(connectingNeuron)

    #connecting fc1 to fc2 and fc2 to fc3
    for layer_idx in range(1, len(linear_weight_names)):
        for col in range(int16_sd[linear_weight_names[layer_idx]].shape[1]):
            allConnections = []
            for i, elem in enumerate(int16_sd[linear_weight_names[layer_idx]][:, col]):
                if layer_idx == len(linear_weight_names) - 1:
                    connectingNeuron = (i, elem.item())
                else:
                    connectingNeuron = (f"FC{layer_idx + 1}.{i}", elem.item())
                allConnections.append(connectingNeuron)
            connections[f"FC{layer_idx}.{col}"] = (allConnections, N)

    # creating output neurons
    outputs = []
    for x in range(num_outputs):
        connections[x] = ([], N)
        outputs.append(x)

    dict_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionaries", "lenet5_stride2", "modular")
    os.makedirs(dict_dir, exist_ok=True)
    dict_path = os.path.join(dict_dir, "conversion_dicts.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump({"axons": axons, "connections": connections, "outputs": outputs}, f)
    print(f"Saved conversion dictionaries: {dict_path}")

    #counting synapses of network
    number_synapses = 0
    max_synapses_per_neuron = 0
    max_synapses_per_axon = 0
    for key in connections:
        number_synapses += len(connections[key][0])
        if len(connections[key][0]) > max_synapses_per_neuron:
            max_synapses_per_neuron = len(connections[key][0])

    for key in axons:
        number_synapses += len(axons[key])
        if len(axons[key]) > max_synapses_per_axon:
            max_synapses_per_axon = len(axons[key])

    print(f"Number of neurons: {len(connections)}")
    print(f"Number of axons: {len(axons)}")
    print(f"Number of synapses: {number_synapses}")
    print(f"Max Fan Out per neuron: {max_synapses_per_neuron}")
    print(f"Max Fan Out per axon: {max_synapses_per_axon}")
    
    #create network
    network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")
    print("Network Loaded onto HiAER Spike")
    #used to save clock cycles and hbm accesses
    data = []

    correct = 0
    total = 0
    
    for img, labels in test_dataset:
        input = img.reshape(img.size(0), -1) #flatten input to [1, 36]
        input = input.to(torch.int16)        #change input from FP32 to INT16

        #create input list
        inputs = []
        for i, elem in enumerate(input[0, :]):
            if elem.item() == 1:
                inputs.append(f"A{i}")

        # run one input step then propagate for remaining extra timesteps
        currSpikes = network.step(inputs)
        for _ in range(max(0, extra_timesteps - 1)):
            currSpikes = network.step([])
        results = network.read_membrane(outputs)

        #record clock cycles and hbm accesses
        data.append(currSpikes)

        #compare predicted with ground truth
        predicted = max_membrane_potential(results) #index of max membrane potential == predicted
        total += 1
        if predicted == labels:
            correct += 1

        running_accuracy = 100 * correct / total
        print(f"Running accuracy : {running_accuracy:.2f} %")
    snn_accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {snn_accuracy:.2f} %')

    #record (clockcyles, hbmaccess) as ordered pairs in numpy arr
    arr = np.asarray(data)              

    #Save converter FPGA accuracy to txt file and clock cycles to npy file
    parent_directory = os.path.dirname(PATH)

    #if accuracies.txt file already exists, just append converted accuracy to it, otherwise, create new file
    if os.path.exists(os.path.join(parent_directory, "accuracies_converted.txt")):
        mode = "a"
    else:
        mode = "w"

    with open(os.path.join(parent_directory, "accuracies_converted.txt"), mode) as f:
        f.write(f"Original SNN Accuracy: {snn_accuracy:.2f}%\n")
        f.write(f"FPGA Converted Accuracy: {accuracy:.2f}%\n")

    np.save(os.path.join(parent_directory, "clock_cycles_converted.npy"), arr)

if __name__ == "__main__":
    main()
