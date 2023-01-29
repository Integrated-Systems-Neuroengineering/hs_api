import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# import matplotlib.pyplot as plt
import numpy as np
import itertools

import os
import shutil

# dataloader arguments
batch_size = 128
data_path='~/justinData/mnist'
subset=10

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# reduce datasets by 10x to speed up training
utils.data_subset(mnist_train, subset)
utils.data_subset(mnist_test, subset)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 1.0
num_steps = 15

#  Initialize Network
net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

best_model_path = '/Volumes/export/isn/keli/Desktop/CRI/result/model_quantized.pth.tar'
checkpoint = torch.load(best_model_path)
net.load_state_dict(checkpoint['state_dict'])
net.eval()

test_acc = batch_accuracy(test_loader, net, num_steps)
print(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")

def conv2dOutputSize(layer,inputSize):
    H_out = (inputSize[0] + layer.padding[0]-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0] +1
    W_out = (inputSize[1] + layer.padding[0]-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1] +1
    return [layer.out_channels,int(H_out),int(W_out)]

def maxPoolOutputSize(layer,inputSize):
    H_out = (inputSize[1] + layer.padding - layer.dilation*(layer.kernel_size-1)-1)/layer.stride +1
    W_out = (inputSize[2] + layer.padding - layer.dilation*(layer.kernel_size-1)-1)/layer.stride +1
    return [inputSize[0],int(H_out),int(W_out)]

def conv2dToCRI(inputs,output,layer,layerIdx,axonsDict=None,neuronsDict=None):
    Hk, Wk = layer.kernel_size
    Ho, Wo = output.shape[1],output.shape[2]
    pad_top,pad_left = Hk//2,Wk//2
    filters = layer.weight.detach().cpu().numpy()
    if layerIdx==0:
        Hi, Wi = inputs.shape
        for row in range(pad_top,Hi-pad_top):
            for col in range(pad_left,Wi-pad_left):
                patch = inputs[row-pad_top:row+pad_top+1,col-pad_left:col+pad_left+1]
                for filIdx, fil in enumerate(filters):
                    postSynapticID = str(output[filIdx,row-pad_top,col-pad_left])
                    for i,axons in enumerate(patch):
                        for j,axon in enumerate(axons):
                            axonsDict[axon].append((postSynapticID,int(fil[0,i,j])))
    else:
        Hi, Wi = inputs.shape[1],inputs.shape[2]
        for channel in range(inputs.shape[0]):
            for row in range(pad_top,Hi-pad_top):
                for col in range(pad_left,Wi-pad_left):
                    patch = inputs[channel,row-pad_top:row+pad_top+1,col-pad_left:col+pad_left+1]
                    for filIdx, fil in enumerate(filters):
                        postSynapticID = str(output[filIdx,row-pad_top,col-pad_left])
                        for i,neurons in enumerate(patch):
                            for j,neuron in enumerate(neurons):
                                neuronsDict[str(neuron)].append((postSynapticID,int(fil[channel,i,j])))
                                
def maxPoolToCRI(inputs,output,layer,neuronsDict):
    Hk, Wk = layer.kernel_size, layer.kernel_size
    Hi, Wi = inputs.shape[1],inputs.shape[2]
    Ho, Wo = output.shape[1],output.shape[2]
    pad_top,pad_left = Hk//2,Wk//2
    scaler = 1e6
    for row in range(0,Hi,2):
        for col in range(0,Wi,2):
            for channel in range(inputs.shape[0]):
                patch = inputs[channel,row:row+pad_top+1,col:col+pad_left+1]
                postSynapticID = str(output[channel,row//2,col//2])
                for i,preSynNeurons in enumerate(patch):
                    for j,preSynNeuron in enumerate(preSynNeurons):
                        neuronsDict[str(preSynNeuron)].append((postSynapticID,scaler))
                        
def linearToCRI(inputs,output,layer,layerIdx,neuronsDict,outputNeurons=None):
    inputs = inputs.flatten()
    weight = layer.weight.detach().cpu().numpy()
    currLayerNeuronIdxOffset,nextLayerNeuronIdxOffset = inputs[0],inputs[-1]+1
    for baseNeuronIdx, neuron in enumerate(weight.T):
        neuronID = str(baseNeuronIdx+currLayerNeuronIdxOffset)
        neuronEntry = [(str(basePostSynapticID+nextLayerNeuronIdxOffset), int(synapseWeight)) for basePostSynapticID, synapseWeight in enumerate(neuron) if synapseWeight != 0]
        neuronsDict[neuronID] = neuronEntry
    print('instantiate output neurons')
    for baseNeuronIdx in range(layer.out_features):
        neuronID = str(baseNeuronIdx+nextLayerNeuronIdxOffset)
        neuronsDict[neuronID] = []
        outputNeurons.append(neuronID)
        
def convBiasAxons(layer,axonsDict,axonOffset,outputs):
    biases = layer.bias.detach().cpu().numpy()
    for biasIdx, bias in enumerate(biases):
        biasID = 'a'+str(biasIdx+axonOffset)
        axonsDict[biasID] = [(str(neuronIdx),int(bias)) for neuronIdx in outputs[biasIdx].flatten()]
        
def linearBiasAXons(layer,axonsDict,axonOffset,outputs):
    biases = layer.bias.detach().cpu().numpy()
    for biasIdx, bias in enumerate(biases):
        biasID = 'a'+str(biasIdx+axonOffset)
        axonsDict[biasID] = [(str(outputs[biasIdx]),int(bias))]

from collections import defaultdict
axonsDict = defaultdict(list)
neuronsDict = defaultdict(list)
outputNeurons = []
H_in, W_in = 28, 28
inputSize = (H_in, W_in)
axonOffset = 0
neuronOffset = 0
currInput = None

for layerIdx, layer in enumerate(net):
    if layerIdx == 0: #input layer
        if isinstance(layer,torch.nn.Conv2d):
            print('constructing Axons')
            outputSize = conv2dOutputSize(layer,inputSize)
            print("Input layer shape(outfeature, infeature): ", inputSize,',',outputSize)
            input = np.arange(0,inputSize[0]*inputSize[1],dtype=int).reshape(inputSize)
            inputAxons = np.array([['a'+str(i) for i in row] for row in input])
            output = np.arange(0,outputSize[0]*outputSize[1]*outputSize[2],dtype=int).reshape(outputSize)
            conv2dToCRI(inputAxons,output,layer,layerIdx,axonsDict)
            axonOffset += len(axonsDict)
            print('constructing bias axons for input layer:',layer.bias.shape[0],'axons')
            convBiasAxons(layer,axonsDict,axonOffset,output)
            axonOffset += layer.bias.shape[0]
            currInput = output
    elif layerIdx == len(net)-2: #output layer
        if isinstance(layer,torch.nn.Linear):
            print('constructing output layer')
            outputSize = layer.out_features
            print("output layer shape(outfeature, infeature): ", currInput.flatten().shape[0],',',outputSize)
            neuronOffset += currInput.shape[0]*currInput.shape[1]*currInput.shape[2]
            output = np.arange(neuronOffset,neuronOffset+outputSize,dtype=int)
            linearToCRI(currInput,output,layer,layerIdx,neuronsDict=neuronsDict,outputNeurons=outputNeurons)
            print('constructing bias axons for output linearlayer:',layer.bias.shape[0],'axons')
            print('Numer of neurons:',len(neuronsDict))
            convBiasAxons(layer,axonsDict,axonOffset,output)
            axonOffset += layer.bias.shape[0]
    else: #hidden layer
        if isinstance(layer,torch.nn.MaxPool2d):
            print('constructing hidden avgpool layer')
            outputSize = maxPoolOutputSize(layer,currInput.shape)
            print("Hidden layer shape(outfeature, infeature): ", currInput.shape,',',outputSize)
            neuronOffset += currInput.shape[0]*currInput.shape[1]*currInput.shape[2]
            output = np.arange(neuronOffset,neuronOffset+outputSize[0]*outputSize[1]*outputSize[2],dtype=int).reshape(outputSize)
            maxPoolToCRI(currInput,output,layer,neuronsDict)
            currInput = output
            print('Numer of neurons:',len(neuronsDict))
        if isinstance(layer,torch.nn.Conv2d):
            print('constructing hidden conv2d layer')
            outputSize = conv2dOutputSize(layer,currInput.shape)
            print("Hidden layer shape(outfeature, infeature): ", currInput.shape,',',outputSize)
            neuronOffset += currInput.shape[0]*currInput.shape[1]*currInput.shape[2]
            output = np.arange(neuronOffset,neuronOffset+outputSize[0]*outputSize[1]*outputSize[2],dtype=int).reshape(outputSize)
            conv2dToCRI(currInput,output,layer,layerIdx,neuronsDict=neuronsDict)
            print('constructing bias axons for hidden conv2d layer:',layer.bias.shape[0],'axons')
            convBiasAxons(layer,axonsDict,axonOffset,output)
            axonOffset += layer.bias.shape[0]
            currInput = output
            print('Numer of neurons:',len(neuronsDict))

print("Number of axons: ",len(axonsDict))
totalAxonSyn = 0
maxFan = 0
for key in axonsDict.keys():
    totalAxonSyn += len(axonsDict[key])
    if len(axonsDict[key]) > maxFan:
        maxFan = len(axonsDict[key])
print("Total number of connections between axon and neuron: ", totalAxonSyn)
print("Max fan out of axon: ", maxFan)
print('---')
print("Number of neurons: ", len(neuronsDict))
totalSyn = 0
maxFan = 0
for key in neuronsDict.keys():
    totalSyn += len(neuronsDict[key])
    if len(neuronsDict[key]) > maxFan:
        maxFan = len(neuronsDict[key])
print("Total number of connections between hidden and output layers: ", totalSyn)
print("Max fan out of neuron: ", maxFan)
print(neuronsDict['13769'])

axonsDict, neuronsDict = dict(axonsDict), dict(neuronsDict)

import time

from l2s.api import CRI_network
import cri_simulations

config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = 9*10**4
softwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config,target='simpleSim', outputs = outputNeurons)
hardwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config,target='CRI', outputs = outputNeurons,simDump = False)

def input_to_CRI(currentInput):
    num_steps = 10
    currentInput = data.view(data.size(0), -1)
    batch = []
    n = 0
    for element in currentInput:
        timesteps = []
        rateEnc = spikegen.rate(element,num_steps)
        rateEnc = rateEnc.detach().cpu().numpy()
        for element in rateEnc:
            currInput = ['a'+str(idx) for idx,axon in enumerate(element) if axon != 0]
            biasInput = ['a'+str(idx) for idx in range(784,len(axonsDict))]
#             timesteps.append(currInput)
#             timesteps.append(biasInput)
            timesteps.append(currInput+biasInput)
        batch.append(timesteps)
    return batch

def run_CRI(inputList):
    firstOutput = 13760
    predictions = []
    #each image
    for currInput in inputList:
        #reset the membrane potential to zero
        softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))
        spikeRate = [0]*10
        #each time step
        for slice in currInput:
            swSpike = softwareNetwork.step(slice, membranePotential=False)
            for spike in swSpike:
                spikeIdx = int(spike) - firstOutput 
                try: 
                    if spikeIdx >= 0: 
                        spikeRate[spikeIdx] += 1 
                except:
                    print("SpikeIdx: ", spikeIdx,"\n SpikeRate:",spikeRate )
        predictions.append(spikeRate.index(max(spikeRate)))

    return(predictions)

def run_CRI_hw(inputList):
    firstOutput = 13760
    predictions = []
    #each image
    total_time_cri = 0
    for currInput in inputList:
        #initiate the softwareNetwork for each image
        cri_simulations.FPGA_Execution.fpga_controller.clear(len(neuronsDict), False, 0)  ##Num_neurons, simDump, coreOverride
        spikeRate = [0]*10
        #each time step
        for slice in currInput:
            start_time = time.time()
            hwSpike, mem = hardwareNetwork.step(slice, membranePotential=True)
            print("Mem:",mem)
            end_time = time.time()
            total_time_cri = total_time_cri + end_time-start_time
            print(hwSpike)
            for spike in hwSpike:
                print(int(spike))
                spikeIdx = int(spike) - firstOutput 
                if spikeIdx >= 0: 
                    spikeRate[spikeIdx] += 1 
        predictions.append(spikeRate.index(max(spikeRate))) 
    print(f"Total execution time CRIFPGA: {total_time_cri:.5f} s")
    return(predictions)

total = 0
correct = 0
cri_correct = 0
cri_correct_hw = 0
# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=10, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        input = input_to_CRI(data)
        criPred = torch.tensor(run_CRI(input)).to(device)
#         criPred_hw = torch.tensor(run_CRI_hw(input)).to(device)
        print("CRI Predicted: ",criPred)
#         print("CRI Predicted HW: ",criPred_hw)
        print("Target: ",targets)
        test_spk, _ = forward_pass(net, num_steps, data)

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        print("Torchsnn Predicted: ",predicted)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        cri_correct += (criPred == targets).sum().item()
#         cri_correct_hw += (criPred_hw == targets).sum().item()
        break #run for one batch

