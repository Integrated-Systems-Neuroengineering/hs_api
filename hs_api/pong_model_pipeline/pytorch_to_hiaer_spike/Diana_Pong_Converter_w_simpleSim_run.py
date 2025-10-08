from turtle import done
from spikingjelly.datasets import pad_sequence_collate

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding

from step_2_CRI_converter import convert_dvs_observation_to_spikes

import torch
import torch.nn as nn
import sys
import os
import numpy as np
import random


def reset_cri_network_state(network):
    """Reset backend state of a CRI_network instance before a new observation."""
    if hasattr(network, 'simpleSim') and network.simpleSim is not None:
        simple_sim = network.simpleSim
        simple_sim.initialize_sim_vars(simple_sim.numNeurons)
        simple_sim.stepNum = 0
    elif hasattr(network, 'CRI') and network.CRI is not None:
        clear(network.CRI.numNeurons, coreID=network.CRI.coreOveride)
    else:
        raise RuntimeError('CRI_network does not expose a backend to reset.')

def evaluate_model_performance(model, model_name="Model", episodes=1, time_steps=18):
    """
    Evaluate a PyTorch SNN model on DVS Pong environment

    Args:
        model: PyTorch model to evaluate
        model_name: Name for display purposes
        episodes: Number of episodes to evaluate
        time_steps: Number of time steps for rate coding

    Returns:
        dict: Evaluation results
    """
    print(f"\n" + "="*60)
    print(f"Evaluating {model_name}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create DVS Pong environment
    config = {
        'env': {
            'game': 'PongNoFrameskip-v4',
            'noop_max': 30,
            'frame_skip': 4,
            'episodic_life': True,
            'clip_rewards': True,
            'grayscale': True
        },
        'dvs': {
            'change_threshold': 10,
            'visualization': False
        }
    }

    try:
        env = make_dvs_pong_env(config)
    except Exception as e:
        print(f"Warning: Could not create DVS environment: {e}")
        print("Skipping evaluation due to environment issues")
        return {
            'average_reward': 0.0,
            'std_reward': 0.0,
            'episodes': episodes,
            'time_steps': time_steps,
            'error': str(e)
        }

    # Move model to device and ensure it's in eval mode
    model = model.to(device)
    model.eval()

    try:
        # Evaluate the model
        results = evaluate_dvs_snn(
            snn=model,
            ann_model=None,  # Not needed for standalone evaluation
            env=env,
            device=device,
            episodes=episodes,
            time_steps=time_steps
        )

        # Close environment
        env.close()

        print(f"\n{model_name} Performance Summary:")
        print(f"  Average reward: {results['average_reward']:.2f}  {results['std_reward']:.2f}")
        print(f"  Episodes evaluated: {episodes}")
        print(f"  Time steps used: {time_steps}")

        return results

    except Exception as e:
        env.close()
        print(f"Warning: Evaluation failed: {e}")
        return {
            'average_reward': 0.0,
            'std_reward': 0.0,
            'episodes': episodes,
            'time_steps': time_steps,
            'error': str(e)
        }


# Add paths - need to go up to project root to find hs_api
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'hs_api'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'fxpmath'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'hs_bridge'))
sys.path.insert(0, os.path.join(current_dir, '..', '..', '..', 'connectome_utils'))

# import hs_bridge
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron
# from hs_bridge.FPGA_Execution.fpga_controller import clear
import importlib.util
spec = importlib.util.spec_from_file_location(
    "custom_neurons",
    os.path.join(current_dir, '..', '..', '..', 'hs_api', 'hs_api', 'custom_neurons.py')
)
custom_neurons = importlib.util.module_from_spec(spec)
sys.modules['custom_neurons'] = custom_neurons
spec.loader.exec_module(custom_neurons)
Custom_LIFNode = custom_neurons.Custom_LIFNode
from hs_api.quantizer import Quantize_Network #initially just hs_api.converter
# from hs_api.converter import Quantize_Network

# Add parent directory to path for DVS environment and evaluation
sys.path.insert(0, os.path.dirname(current_dir))  # Add pong_stuff to path
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'ann_to_snn'))  # Add ann_to_snn to path

# Import evaluation functions and DVS environment
try:
    from evaluate_dvs_snn import evaluate_dvs_snn #####
except ModuleNotFoundError:
    from evaluate_dvs_snn import evaluate_dvs_snn #

from hs_api.pong_model_pipeline.DVSWrapper import make_dvs_pong_env   #Diana will send

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import sys
sys.path.append(r'C:\Users\diana\rl_snn2\final_push\pong_stuff\ann_training')
from dvs_84_no_bias_model import NoBias84


'''
Adapted from /LeNet5/LeNet5_Converter.py
Implements clock cycle and hbmaccesses recording
Meant for two-channel data.
Works with Custom_LIFNode neurons

WORKFLOW:
1. Load and evaluate original SNN in PyTorch (pre-quantization)
2. Quantize weights/thresholds to integer scale for hardware
3. Evaluate quantized model in PyTorch (should match original performance)
4. Extract quantized weights and build CRI_network connectome
5. Evaluate CRI_network on Pong episodes (hardware simulation)

NOTE: After quantization, weights and thresholds are scaled by ~8192x
(1/w_delta) to convert from float to integer range. The quantized PyTorch
model maintains the same behavior because both are scaled proportionally.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CHANGE HERE FOR DIFFERENT MODELS
b = 1 #batch size

# Extract quantized thresholds from net_quan after quantization
# These will be set after quantization is performed
# ANN neuron for output layer (no spiking threshold)
ANN_N = ANN_neuron(threshold = 0, shift = 0)
# Bias neuron (threshold=-1 means always fires)
biasN = ANN_neuron(threshold = -1, shift = 0)

# NoBias84 model parameters (corrected)
# Conv1: kernel=8, stride=4 -> (84-8)/4+1 = 20
# Conv2: kernel=4, stride=2 -> (20-4)/2+1 = 9
# Conv3: kernel=3, stride=1 -> (9-3)/1+1 = 7
input_res = 84           #resolution of input DVS image (84x84)
conv1_output_res = 20    #resolution of output feature maps from conv1
conv2_output_res = 9     #resolution of output feature maps from conv2
conv3_output_res = 7     #resolution of output feature maps from conv3
num_layers = 5           #(3 conv, 2 fc)
num_outputs = 6          #Pong has 6 actions
eval_episodes = 1

# PATH = "../ann_to_snn/dvs_63_no_bias_snn.pth" #path for loading NoBias63 weights - UPDATE THIS PATH
# PATH = "../ann_to_snn/dvs_84_no_bias_snn_improved.pth"
# PATH = "../ann_to_snn/dvs_84_no_bias_snn_improved.pth"  #path for loading quantized NoBias84 weights - UPDATE THIS PATH
# PATH = "../ann_to_snn/dvs_84_no_bias_snn_finetuned_best_18_steps.pth"
# PATH = "../ann_to_snn/dvs_84_no_bias_snn_finetuned_best_proper_tau.pth"
PATH = "../ann_to_snn/dvs_84_no_bias_snn_finetuned_best_actually_no_bias.pth"

net = torch.load(PATH, map_location='cpu', weights_only=False)
net.to(device)
print(f"[OK] Model loaded: {len(net)} layers")
print(net)

# print("state dict: ", net.state_dict())
# print("state dict keys: ", net.state_dict().keys())

# see if it has biases
for name, param in net.named_parameters():
    if 'bias' in name:
        print(f"Layer {name} has bias")
    else:
        print(f"Layer {name} has no bias")

# Check if working_quantized_model_2 directory exists - if so, skip to CRI network creation
quantized_model_dir = "working_quantized_model_2"
if os.path.exists(quantized_model_dir):
    print("\n" + "="*50)
    print("FOUND EXISTING QUANTIZED MODEL - LOADING")
    print("="*50)

    import pickle

    # Load the quantized model
    quantized_model_path = os.path.join(quantized_model_dir, "quantized_dvs_84_no_bias_snn.pth")
    net_quan = torch.load(quantized_model_path, map_location='cpu', weights_only=False)
    net_quan.to(device)
    net_quan.eval()
    print(f"[OK] Quantized model loaded from {quantized_model_path}")

    # Load connections, axons, and outputs
    with open(os.path.join(quantized_model_dir, "connections.pkl"), "rb") as f:
        connections = pickle.load(f)
    with open(os.path.join(quantized_model_dir, "axons.pkl"), "rb") as f:
        axons = pickle.load(f)
    with open(os.path.join(quantized_model_dir, "outputs.pkl"), "rb") as f:
        outputs = pickle.load(f)

    print(f"[OK] Loaded connections, axons, and outputs from {quantized_model_dir}")
    print(f"  Number of neurons: {len(connections)}")
    print(f"  Number of axons: {len(axons)}")
    print(f"  Outputs: {outputs}")

    # Extract necessary variables from quantized model
    thresholds = []
    for name, module in net_quan.named_modules():
        if isinstance(module, Custom_LIFNode) and hasattr(module, 'v_threshold'):
            thresholds.append(module.v_threshold)

    if len(thresholds) < 4:
        raise RuntimeError(f"Expected at least 4 LIF layers with thresholds, found {len(thresholds)}")

    threshold_conv1 = thresholds[0]
    threshold_conv2 = thresholds[1]
    threshold_conv3 = thresholds[2]
    threshold_fc1 = thresholds[3]

    print(f"\nLoaded thresholds:")
    print(f"  Conv1: {threshold_conv1}")
    print(f"  Conv2: {threshold_conv2}")
    print(f"  Conv3: {threshold_conv3}")
    print(f"  FC1: {threshold_fc1}")

    # Check if model has biases
    has_bias = any('bias' in name for name, param in net_quan.named_parameters())
    print(f"\nModel has biases: {has_bias}")

    # Skip to CRI network creation
    skip_to_cri = True
else:
    skip_to_cri = False

    pre_quant_results = evaluate_model_performance(
            model=net,
            model_name="Pre-Quantization SNN",
            episodes=1,
            time_steps=20
        )

if not skip_to_cri:
    # Quantization with dynamic alpha and optional membrane potential quantization
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    quantizer = Quantize_Network(w_alpha=4, dynamic_alpha=False)
    net_quan = quantizer.quantize(net)
    

    # print the tau of each custom LIF neuron layer
    for name, module in net_quan.named_modules():
        if isinstance(module, Custom_LIFNode):
            print(f"Layer {name}: tau = {module.tau}, v_threshold = {module.v_threshold}, decay_input = {module.decay_input}, v_reset = {module.v_reset}")
        # check for biases in conv and linear layers, print bias value
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if module.bias is not None:
                print(f"Layer {name} has bias, bias value: {module.bias.data}")
            else:
                print(f"Layer {name} has no bias, bias value: None")


    # Print quantization parameters
    print("QUANTIZATION PARAMETERS:")
    print("="*50)
    print(f"w_alpha: {quantizer.w_alpha}")
    print(f"dynamic_alpha: {quantizer.dynamic_alpha}")
    print(f"w_bits: {quantizer.w_bits}")
    print(f"w_delta: {quantizer.w_delta}")
    print("="*50)

    # Extract quantized thresholds from the quantized model
    print("\n" + "="*50)
    print("EXTRACTING QUANTIZED THRESHOLDS")
    print("="*50)

    thresholds = []
    for name, module in net_quan.named_modules():
        if isinstance(module, Custom_LIFNode) and hasattr(module, 'v_threshold'):
            threshold_value = module.v_threshold
            print(f"Layer {name}: v_threshold = {threshold_value}")
            thresholds.append(threshold_value)

    if len(thresholds) < 4:
        raise RuntimeError(f"Expected at least 4 LIF layers with thresholds, found {len(thresholds)}")

    threshold_conv1 = thresholds[0]
    threshold_conv2 = thresholds[1]
    threshold_conv3 = thresholds[2]
    threshold_fc1 = thresholds[3]

    print(f"\nUsing thresholds:")
    print(f"  Conv1: {threshold_conv1}")
    print(f"  Conv2: {threshold_conv2}")
    print(f"  Conv3: {threshold_conv3}")
    print(f"  FC1: {threshold_fc1}")
    print("="*50)

    # save quantized model
    quantized_model_path = "working_quantized_model_2/quantized_dvs_84_no_bias_snn.pth"
    os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)
    torch.save(net_quan, quantized_model_path)


    # LIF neuron for conv layers and FC layer
    pertubation = 0
    leak_lif = 63 #should match tau in training
    LIF_conv1 = LIF_neuron(threshold_conv1, pertubation, leak_lif)
    LIF_conv2 = LIF_neuron(threshold_conv2, pertubation, leak_lif)
    LIF_conv3 = LIF_neuron(threshold_conv3, pertubation, leak_lif)
    LIF_fc1 = LIF_neuron(threshold_fc1, pertubation, leak_lif)

    #convert FP32 weights to INT16
    #int16_sd, scales = fp32_to_int16_state_dict(model)
    #int16_sd is now the state dict of the quantized model
    int16_sd = net_quan.state_dict()
    print(f"int16_sd keys: {int16_sd.keys()}")

    # Determine weight tensors dynamically (Sequential indices may shift)
    weight_keys = sorted(
        [k for k in int16_sd.keys() if k.endswith('.weight')],
        key=lambda name: int(name.split('.')[0])
    )
    if len(weight_keys) < 5:
        raise RuntimeError(f"Expected at least 5 weight tensors, found {len(weight_keys)}: {weight_keys}")

    conv1_key, conv2_key, conv3_key, fc1_key, fc2_key = weight_keys[:5]

    conv1_weight = int16_sd[conv1_key]
    conv2_weight = int16_sd[conv2_key]
    conv3_weight = int16_sd[conv3_key]
    fc1_weight = int16_sd[fc1_key]
    fc2_weight = int16_sd[fc2_key]

    print('original weight types: ', conv1_weight.dtype, conv2_weight.dtype, conv3_weight.dtype, fc1_weight.dtype, fc2_weight.dtype)
    int16_sd[conv1_key] = conv1_weight.to(torch.int16)
    int16_sd[conv2_key] = conv2_weight.to(torch.int16)
    int16_sd[conv3_key] = conv3_weight.to(torch.int16)
    int16_sd[fc1_key] = fc1_weight.to(torch.int16)
    int16_sd[fc2_key] = fc2_weight.to(torch.int16)

    conv1_weight = int16_sd[conv1_key]
    conv2_weight = int16_sd[conv2_key]
    conv3_weight = int16_sd[conv3_key]
    fc1_weight = int16_sd[fc1_key]
    fc2_weight = int16_sd[fc2_key]

    print('new weight types: ', conv1_weight.dtype, conv2_weight.dtype, conv3_weight.dtype, fc1_weight.dtype, fc2_weight.dtype)

    # Extract bias weights (if they exist)
    bias_keys = sorted(
        [k for k in int16_sd.keys() if k.endswith('.bias')],
        key=lambda name: int(name.split('.')[0])
    )

    if len(bias_keys) >= 5:
        conv1_bias_key, conv2_bias_key, conv3_bias_key, fc1_bias_key, fc2_bias_key = bias_keys[:5]

        conv1_bias = int16_sd[conv1_bias_key].to(torch.int16)
        conv2_bias = int16_sd[conv2_bias_key].to(torch.int16)
        conv3_bias = int16_sd[conv3_bias_key].to(torch.int16)
        fc1_bias = int16_sd[fc1_bias_key].to(torch.int16)
        fc2_bias = int16_sd[fc2_bias_key].to(torch.int16)

        print(f'\nBias shapes:')
        print(f'  conv1_bias: {conv1_bias.shape}')
        print(f'  conv2_bias: {conv2_bias.shape}')
        print(f'  conv3_bias: {conv3_bias.shape}')
        print(f'  fc1_bias: {fc1_bias.shape}')
        print(f'  fc2_bias: {fc2_bias.shape}')
        has_bias = True
    else:
        print('\nNo bias weights found in model')
        has_bias = False

    # Diagnostic: Verify threshold quantization
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Verifying Threshold Quantization")
    print(f"{'='*60}")

    print(f"\nOriginal model thresholds:")
    for name, module in net.named_modules():
        if hasattr(module, 'v_threshold'):
            print(f"  {name}: v_threshold = {module.v_threshold}")

    print(f"\nQuantized model thresholds:")
    for name, module in net_quan.named_modules():
        if hasattr(module, 'v_threshold'):
            print(f"  {name}: v_threshold = {module.v_threshold}")

    print(f"\nExpected scaling factor: 1/w_delta = {1.0/quantizer.w_delta:.2f}")
    print(f"{'='*60}\n")

    net_quan.eval()
    # Test quantized model accuracy
    print("\n" + "="*50)
    print("TESTING QUANTIZED MODEL ACCURACY")
    print("="*50)

    post_quant_results = evaluate_model_performance(
        model=net_quan,
        model_name="Post-Quantization SNN",
        episodes=eval_episodes,
        time_steps=20
    )

    # Calculate and display accuracy drop
    accuracy_drop = pre_quant_results['average_reward'] - post_quant_results['average_reward']
    print(f"\n{'='*50}")
    print(f"QUANTIZATION RESULTS:")
    print(f"Original Reward:   {pre_quant_results['average_reward']:.4f}")
    print(f"Quantized Reward:  {post_quant_results['average_reward']:.4f}")
    print(f"Accuracy Drop:     {accuracy_drop:.4f}")
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

    # iterate through every weight kernel in first convolutional layer and map axons -> (neuron, weight)
    print("conv1 weight shape: ", conv1_weight.shape)  # Should be (32, 2, 8, 8)
    # Build axonMap for conv1 (kernel_size=8, stride=4)
    axonMap = torch.arange(2 * (input_res ** 2), dtype=torch.float32).reshape(1, 2, input_res, input_res)
    patchTensor = F.unfold(input=axonMap, kernel_size=8, stride=4)
    patch_rows = patchTensor.transpose(1, 2).squeeze(0)  # shape: [num_patches, kernel_size*kernel_size*2]
    patch_rows = patch_rows.to(torch.int16)

    for feature_map, kernel in enumerate(conv1_weight):  # kernel shape: [2, 8, 8]
        flat_kernel = kernel.flatten()  # shape: [128]
        for index, row in enumerate(patch_rows):  # row shape: [128]
            neuronName = f"C1.{feature_map}.{index}"
            connections[neuronName] = ([], LIF_conv1)

            # Create bias neuron for this Conv1 neuron
            if has_bias:
                bias_weight = conv1_bias[feature_map].item()
                biasNeuronName = f"BN.C1.{feature_map}.{index}"
                connections[biasNeuronName] = ([(neuronName, bias_weight)], biasN)

            for i, elem in enumerate(row):
                axon_id = int(elem.item())
                key = f"A{axon_id}"
                weight = flat_kernel[i].item()
                axons[key].append((neuronName, weight))

    #creating C1Map to identify which C1 neurons connect to which pixel/neuron of the feature map in conv2
    C1Map = torch.arange(conv1_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv1_output_res, conv1_output_res)
    patchTensor = F.unfold(input=C1Map, kernel_size=4, stride=2)   # conv2: kernel=4, stride=2

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
                connections[neuronName] = ([], LIF_conv2)

                # Create bias neuron for this Conv2 neuron (only once per neuron, not per input feature map)
                if has_bias and feature_map == 0:
                    bias_weight = conv2_bias[output_idx].item()
                    biasNeuronName = f"BN.C2.{output_idx}.{j}"
                    connections[biasNeuronName] = ([(neuronName, bias_weight)], biasN)
                #print(neuronName)

                #inner loop 3: iterate through each elem in row. Each elem is index of C1 -> C2
                for i, elem in enumerate(row):
                    index = int(elem.item())
                    key = f"C1.{feature_map}.{index}"
                    weight = flat_kernel[i].item()
                    connections[key][0].append((neuronName, weight))

    #connecting C2 to C3
    print("conv3 weight shape: ", conv3_weight.shape)
    #creating C2Map to identify which C2 neurons connect to which pixel/neuron of the feature map in conv3
    C2Map = torch.arange(conv2_output_res ** 2, dtype=torch.float32).reshape(1, 1, conv2_output_res, conv2_output_res)
    patchTensor = F.unfold(input=C2Map, kernel_size=3, stride=1)   # conv3: kernel=3, stride=1
    patch_rows = patchTensor.transpose(1, 2).squeeze(0)
    patch_rows = patch_rows.to(torch.int16)

    for output_idx, output_channel in enumerate(conv3_weight):
        for feature_map, kernel in enumerate(output_channel):
            flat_kernel = kernel.flatten()
            for j, row in enumerate(patch_rows):
                neuronName = f"C3.{output_idx}.{j}"
                connections[neuronName] = ([], LIF_conv3)

                # Create bias neuron for this Conv3 neuron (only once per neuron, not per input feature map)
                if has_bias and feature_map == 0:
                    bias_weight = conv3_bias[output_idx].item()
                    biasNeuronName = f"BN.C3.{output_idx}.{j}"
                    connections[biasNeuronName] = ([(neuronName, bias_weight)], biasN)

                for i, elem in enumerate(row):
                    index = int(elem.item())
                    key = f"C2.{feature_map}.{index}"
                    weight = flat_kernel[i].item()
                    connections[key][0].append((neuronName, weight))

    #connecting conv3 to fc1
    feature_map = 0
    print("fc1 shape: ", fc1_weight.shape)
    for col in range(fc1_weight.shape[1]):  #x.shape[1] == number of col
        if col % (conv3_output_res ** 2) == 0 and col != 0:  #determines the feature_map of the C3 neuron for C3 --> FC1
            feature_map += 1
        for i, elem in enumerate(fc1_weight[:, col]):     #iterate over element in a col
            connectingNeuron = (f"FC1.{i}", elem.item())
            connections[f"C3.{feature_map}.{col % (conv3_output_res ** 2)}"][0].append(connectingNeuron)

    #connecting fc1 to fc2 (final layer - outputs)
    print("fc2 shape: ", fc2_weight.shape)
    for col in range(fc2_weight.shape[1]):  #x.shape[1] == number of col
        allConnections = []
        for i, elem in enumerate(fc2_weight[:, col]):     #iterate over element in a col
            connectingNeuron = (i, elem.item())  # Connect directly to output neurons
            allConnections.append(connectingNeuron)
        connections[f"FC1.{col}"] = (allConnections, LIF_fc1)

        # Create bias neuron for this FC1 neuron
        if has_bias:
            bias_weight = fc1_bias[col].item()
            biasNeuronName = f"BN.FC1.{col}"
            connections[biasNeuronName] = ([(f"FC1.{col}", bias_weight)], biasN)


    #creating output neurons
    outputs = []
    for x in range(num_outputs):
        connections[x] = ([], ANN_N)  # Use ANN_neuron for output layer
        outputs.append(x)

        # Create bias neuron for this output neuron
        if has_bias:
            bias_weight = fc2_bias[x].item()
            biasNeuronName = f"BN.FC2.{x}"
            connections[biasNeuronName] = ([(x, bias_weight)], biasN)

    #counting synapses of network
    number_synapses = 0
    for key in connections:
        number_synapses += len(connections[key][0])

    for key in axons:
        number_synapses += len(axons[key])

    print(f"Number of neurons: {len(connections)}")
    print(f"Number of axons: {len(axons)}")
    print(f"Number of synapses: {number_synapses}")

    # Debug: Count bias neurons
    if has_bias:
        bias_neuron_count = sum(1 for key in connections.keys() if isinstance(key, str) and key.startswith("BN."))
        print(f"Number of bias neurons: {bias_neuron_count}")

        # Print sample bias neurons
        print("\nSample bias neurons:")
        sample_count = 0
        for key, value in connections.items():
            if isinstance(key, str) and key.startswith("BN.") and sample_count < 5:
                target_neurons, neuron_type = value
                print(f"  {key}: connects to {target_neurons}, neuron_type threshold={neuron_type.threshold}")
                sample_count += 1

    print(f"Outputs: {outputs}")

    import pickle

    # Save connections and axons
    with open("working_quantized_model_2/connections.pkl", "wb") as f:
        pickle.dump(connections, f)

    with open("working_quantized_model_2/axons.pkl", "wb") as f:
        pickle.dump(axons, f)

    with open("working_quantized_model_2/outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)
    
    

print("\n" + "="*50)
print("CREATING CRI NETWORK")
print("="*50)


#create network
network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="simpleSim")

########## make test set of Pong obs ################

# Create DVS Pong environment
config = {
        'env': {
            'game': 'PongNoFrameskip-v4',
            'noop_max': 30,
            'frame_skip': 4,
            'episodic_life': True,
            'clip_rewards': True,
            'grayscale': True
        },
        'dvs': {
            'change_threshold': 10,
            'visualization': False
        }
    }

env = make_dvs_pong_env(config)

test_set = []

obs, info = env.reset()
test_set.append(obs)
for _ in range(100):  #100 test samples
    obs, _, _, _, _ = env.step(random.randint(0, 5))  #take a step in the environment
    test_set.append(obs)

actions = {"net": [], "net_quan": [], "network": []}

# test all 3 nets on test set
# print("\n" + "="*50)
# print("EVALUATING ON TEST SET")
# print("="*50)
# time_steps = 20
# print(f"Evaluating pre-quantization model on {len(test_set)} samples...")

# for i, obs in enumerate(test_set):
#     # Debug observation shape
#     if i == 0:
#         print(f"First observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
#         if isinstance(obs, np.ndarray):
#             print(f"Observation dtype: {obs.dtype}")



#     obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

#     # Reset SNN state for each frame
#     functional.reset_net(net)  # activation_based

#     # Rate coding: accumulate SNN outputs over multiple time steps
#     output_sum = torch.zeros(1, 6, device=device)  # 6 actions for Pong

#     with torch.no_grad():
#         for t in range(time_steps):
#             # Forward pass through SNN
#             snn_output = net(obs_tensor)
#             output_sum += snn_output

#     # Compute rate-coded Q-values (average over time steps)
#     q_values = output_sum / time_steps
#     action = q_values.argmax(dim=1).item()
#     actions["net"].append(action)
    
# print(f"Evaluating post-quantization model on {len(test_set)} samples...")

# for i, obs in enumerate(test_set):
#     # Convert to numpy array and ensure correct format
#     if not isinstance(obs, np.ndarray):
#         obs = np.array(obs)

#     # Handle different observation formats (same as above)
#     if len(obs.shape) == 3 and obs.shape[0] == 3:
#         obs = obs[:2, :, :]
#     elif len(obs.shape) == 3 and obs.shape[2] == 3:
#         obs = obs[:, :, :2].transpose(2, 0, 1)

#     obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

#     # Reset SNN state for each frame
#     functional.reset_net(net_quan)  # activation_based

#     # Rate coding: accumulate SNN outputs over multiple time steps
#     output_sum = torch.zeros(1, 6, device=device)  # 6 actions for Pong

#     with torch.no_grad():
#         for t in range(time_steps):
#             # Forward pass through SNN
#             snn_output = net_quan(obs_tensor)
#             output_sum += snn_output

#     # Compute rate-coded Q-values (average over time steps)
#     q_values = output_sum / time_steps
#     action = q_values.argmax(dim=1).item()
#     actions["net_quan"].append(action)
    
# print(f"Evaluating hs_api CRI_network on {len(test_set)} samples...")

# for i, obs in enumerate(test_set):
#     # Convert to numpy array and ensure correct format
#     if not isinstance(obs, np.ndarray):
#         obs = np.array(obs)

#     # Handle different observation formats (same as above)
#     if len(obs.shape) == 3 and obs.shape[0] == 3:
#         obs = obs[:2, :, :]
#     elif len(obs.shape) == 3 and obs.shape[2] == 3:
#         obs = obs[:, :, :2].transpose(2, 0, 1)

#     spikes = convert_dvs_observation_to_spikes(obs)

#     # Process through network (20 cycles + 4 empty steps)
#     q_accum = np.zeros(6)

#     # 20 cycles with input spikes
#     for i in range(20):
#         _ = network.step(spikes)
#         if i >= 3:  # Start reading after 3 cycles
#             q_values_raw = network.read_membrane(outputs)
#             # Convert Fxp objects to floats
#             q_values = np.array([float(q.val) if hasattr(q, 'val') else float(q) for q in q_values_raw])
#             q_accum += q_values

#     # 4 empty processing steps
#     for j in range(4):
#         _ = network.step([])
#         q_values_raw = network.read_membrane(outputs)
#         # Convert Fxp objects to floats
#         q_values = np.array([float(q.val) if hasattr(q, 'val') else float(q) for q in q_values_raw])
#         q_accum += q_values

#     # Select action based on highest Q-value
#     action = int(np.argmax(q_accum))
#     actions["network"].append(action)
    
# # Compare actions
# num_matches = 0
# for i in range(len(test_set)):
#     print(f"Sample {i}: net={actions['net'][i]}, net_quan={actions['net_quan'][i]}, network={actions['network'][i]}")
#     if actions['net'][i] == actions['network'][i]:
#         num_matches += 1

# print(f"Number of matching actions: {num_matches}")


################# run network on 2 episodes of Pong ######################
print("\n" + "="*50)
print("EVALUATING CRI NETWORK ON 2 EPISODES OF PONG")
print("="*50)

#used to save clock cycles and hbm accesses
clock_cycles = []

for episode in range(2):
    print(f"\nStarting episode {episode+1}...")
    obs, info = env.reset()

    total_reward = 0
    step_count = 0
    max_steps = 5000


    score = {'player_0': 0, 'player_1': 0}

    # Debug: Track first observation in detail
    debug_first_step = True

    while step_count < max_steps:
        reset_cri_network_state(network)
        # Convert observation to spikes
        # Handle observation format
        obs_processed = obs
        if not isinstance(obs, np.ndarray):
            obs_processed = np.array(obs)

        # Handle different observation formats
        if len(obs_processed.shape) == 3 and obs_processed.shape[0] == 3:
            obs_processed = obs_processed[:2, :, :]
        elif len(obs_processed.shape) == 3 and obs_processed.shape[2] == 3:
            obs_processed = obs_processed[:, :, :2].transpose(2, 0, 1)

        spikes = convert_dvs_observation_to_spikes(obs_processed)

        if debug_first_step:
            print(f"\n[DEBUG] First step details:")
            print(f"  Input spikes: {len(spikes)} spikes")
            if len(spikes) > 0:
                print(f"  Sample input spikes: {spikes[:10]}")

        # Process through network (20 cycles + 4 flush steps)
        q_accum = np.zeros(6)

        #add extra timestep at beginning
        # _ = network.step([])

        # 20 cycles with input spikes
        for i in range(20):
            step_output = network.step(spikes)

            if debug_first_step and i < 3:
                # Check if bias neurons are firing
                if isinstance(step_output, tuple) and len(step_output) >= 3:
                    spike_dict, _, _ = step_output
                    bias_spikes = [n for n in spike_dict.keys() if isinstance(n, str) and n.startswith("BN.")]
                    print(f"  Timestep {i}: {len(bias_spikes)} bias neurons fired out of bias neuron total")
                    if i == 0:
                        print(f"    Sample firing bias neurons: {bias_spikes[:5]}")
                else:
                    print(f"  Timestep {i}: step_output format unexpected: {type(step_output)}, value: {step_output}")

            if i >= 3:  # Start reading after 3 cycles
                q_values_raw = network.read_membrane(outputs)
                # Convert Fxp objects to floats
                q_values = np.array([float(q.val) if hasattr(q, 'val') else float(q) for q in q_values_raw])
                q_accum += q_values

                if debug_first_step and i == 3:
                    print(f"  First Q-value read (timestep 3): {q_values}")
                    # Check membrane potentials of some target neurons that receive bias
                    if has_bias:
                        sample_targets = ["C1.0.0", "C2.0.0", "FC1.0", 0]
                        mps = network.read_membrane(sample_targets)
                        print(f"  Sample neuron membrane potentials:")
                        for neuron, mp in zip(sample_targets, mps):
                            mp_val = float(mp.val) if hasattr(mp, 'val') else float(mp)
                            print(f"    {neuron}: {mp_val}")

        # 4 flush processing steps with blank inputs (zero-valued spikes to allow propagation)
        # This matches the PyTorch model's flush behavior which feeds zeros, not empty inputs
        blank_spikes = []  # Empty list means no spikes (blank input)
        for j in range(4):
            step_output = network.step(blank_spikes)

            if debug_first_step and j == 0:
                if isinstance(step_output, tuple) and len(step_output) >= 3:
                    spike_dict, _, _ = step_output
                    bias_spikes = [n for n in spike_dict.keys() if isinstance(n, str) and n.startswith("BN.")]
                    print(f"  Flush timestep 0: {len(bias_spikes)} bias neurons fired")
                else:
                    print(f"  Flush timestep 0: step_output format unexpected: {type(step_output)}, value: {step_output}")

            q_values_raw = network.read_membrane(outputs)
            # Convert Fxp objects to floats
            q_values = np.array([float(q.val) if hasattr(q, 'val') else float(q) for q in q_values_raw])
            q_accum += q_values

        if debug_first_step:
            print(f"  Final Q-values: {q_accum}")
            print(f"  Selected action: {np.argmax(q_accum)}")
            debug_first_step = False

        #record clock cycles and hbm accesses after final timestep
        # clock_cycles.append((clock_cycles, hbm_accesses))

        # Select action based on highest Q-value
        action = int(np.argmax(q_accum))

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if reward == 1:
            score['player_1'] += 1
        elif reward == -1:
            score['player_0'] += 1
            
        step_count += 1

        # Print progress
        if step_count % 10 == 0 or step_count <= 10:
            print(f"Step {step_count}: action={action}, score: {score['player_0']} : {score['player_1']}, q_max={np.max(q_accum):.2f}")

        if terminated or truncated:
            break

    env.close()

    # Results
    print(f"\n" + "="*60)
    print("EPISODE RESULTS")
    print("="*60)
    print(f"Model: 84x84 threshold_bias_pong_model_84")
    print(f"Target: simpleSim")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.1f}")


#record (clockcyles, hbmaccess) as ordered pairs in numpy arr
arr = np.asarray(clock_cycles)  

#Save clock cycles to npy file
parent_directory = os.path.dirname(PATH)
np.save(os.path.join(parent_directory, "clock_cycles_Pong_2x84x84_nobias.npy"), arr)



