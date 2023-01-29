#!/usr/bin/env python



# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import shutil
import argparse
import time
import cri_simulations
# dataloader arguments
batch_size = 128
data_path='mnistData/'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[7]:


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


##Save checkpoints

      
def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


# # 6. Define the Network

# In[10]:


# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 5
#beta is set to 0.0
beta = 1.0


# In[11]:


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []
        
        data = spikegen.rate(x,num_steps)

        for q in data:

            cur1 = self.fc1(q)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
        
# Load the network onto CUDA if available
net = Net().to(device)

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
    return acc
def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    acc_train = print_batch_accuracy(data, targets, train=True)
    acc_test = print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")
    return acc_test


# ## 7.2 Loss Definition
# The `nn.CrossEntropyLoss` function in PyTorch automatically handles taking the softmax of the output layer as well as generating a loss at the output. 

# In[13]:


loss = nn.CrossEntropyLoss()


# ## 7.3 Optimizer
# Adam is a robust optimizer that performs well on recurrent networks, so let's use that with a learning rate of $5\times10^{-4}$. 

# In[14]:


optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

# ## 7.5 Training Loop
# 
# Let's combine everything into a training loop. We will train for one epoch (though feel free to increase `num_epochs`), exposing our network to each sample of data once.

# In[22]:

PATH = "/Volumes/export/isn/gopa/CRI_proj/snntorch_mnist/L2S_justin/result/mnist_2layer_MLP_quantized/model_best.pth.tar"
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda") 

net.cuda()
#net.eval()

# Voila! That's it for static MNIST. Feel free to tweak the network parameters, hyperparameters, decay rate, using a learning rate scheduler etc. to see if you can improve the network performance. 

# ### Map trained SNN from torchsnn to CRI 

# In[28]:


layers = [list(net.fc1.parameters())[0].detach().cpu().numpy(), list(net.fc2.parameters())[0].detach().cpu().numpy()]
biases = [net.fc1.bias.detach().cpu().numpy(), net.fc2.bias.detach().cpu().numpy()]
print(np.min(np.abs(layers[1])))
print(np.max(np.abs(layers[1])))

#breakpoint()

# In[46]:

scale = 1  ##No Further scaling as model is already quantized
threshold = net.lif1.threshold
axonsDict = {}
neuronsDict = {}
outputs = []
bias_axon = {}

axonOffset = 0
currLayerNeuronIdxOffset = 0
nextLayerNeuronIdxOffset = 0
for layerNum, layer in enumerate(layers):
    #layer = layers[layerKey]
    #print(layer.keys())
    inFeatures = layer.shape[1]
    outFeatures = layer.shape[0]
    shape = layer.shape
    weight = layer
    bias = biases[layerNum]
    print("Weights shape: ", np.shape(weight))
    if (layerNum == 0):
        print('constructing Axons')
        print("Input layer shape(outfeature, infeature): ", weight.shape)
        for axonIdx, axon in enumerate(weight.T):
            #print(axonIdx)
            axonID = 'a'+str(axonIdx)
            axonEntry = [(str(postSynapticID), int(synapseWeight*scale)) for postSynapticID, synapseWeight in enumerate(axon) ]
            axonsDict[axonID] = axonEntry
        axonOffset += inFeatures
        print("axon offset: ",axonOffset)
        #implmenting bias: for each bias add a axon with corresponding weights with synapse (neuron, bias_val)
        print('Construct bias axons for hidden layers:',bias.shape)
        for neuronIdx, bias_value in enumerate(bias):
            biasAxonID = 'a'+str(neuronIdx + axonOffset)
            biasAxonEntry = [(str(neuronIdx),int(bias_value*scale))]
            axonsDict[biasAxonID] = biasAxonEntry
        
    elif (layerNum == len(layers)-1):
        print('constructing output layer')
        nextLayerNeuronIdxOffset += inFeatures
        print("output layer shape(outfeature, infeature): ", weight.shape)
        for baseNeuronIdx, neuron in enumerate(weight.T):
            neuronID = str(baseNeuronIdx+currLayerNeuronIdxOffset)
            neuronEntry = [(str(basePostSynapticID+nextLayerNeuronIdxOffset), int(synapseWeight*scale)) for basePostSynapticID, synapseWeight in enumerate(neuron) if synapseWeight != 0]
            neuronsDict[neuronID] = neuronEntry
            #print(neuronID)
        currLayerNeuronIdxOffset += inFeatures
        #instantiate the output neurons
        print('instantiate output neurons')
        for baseNeuronIdx in range(outFeatures):
            neuronID = str(baseNeuronIdx+nextLayerNeuronIdxOffset)
            neuronsDict[neuronID] = []
            outputs.append(neuronID)
            #print(neuronID)
        #implmenting bias: for each bias add a axon with corresponding weights with synapse (neuron, bias_val)
        print('Construct bias axons for output neurons',bias.shape)
        axonOffset += inFeatures
        for neuronIdx, bias_value in enumerate(bias):
            biasAxonID = 'a'+str(neuronIdx + axonOffset)
            biasAxonEntry = [(str(neuronIdx+nextLayerNeuronIdxOffset),int(bias_value*scale))]
            axonsDict[biasAxonID] = biasAxonEntry
            
    else:
        print('constructing hidden layer')
        nextLayerNeuronIdxOffset += inFeatures
        for baseNeuronIdx, neuron in enumerate(weight): #SHOULD THIS BE A TRANSPOSE
            neuronID = str(baseNeuronIdx+currLayerNeuronIdxOffset)
            neuronEntry = [(str(basePostSynapticID+nextLayerNeuronIdxOffset), int(synapseWeight*scale)) for basePostSynapticID, synapseWeight in enumerate(neuron) if synapseWeight != 0 ]
            neuronsDict[neuronID] = neuronEntry
            #print(neuronID)
        currLayerNeuronIdxOffset += inFeatures
        #print(currLayerNeuronIdxOffset)
print("output neurons: ", outputs)
print("number of axons: ", len(axonsDict))


# In[47]:


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
print(neuronsDict['1007'])


# In[48]:




# In[49]:


from l2s.api import CRI_network
config = {}
config['neuron_type'] = "I&F"
config['global_neuron_params'] = {}
config['global_neuron_params']['v_thr'] = int(threshold)
#breakpoint()
softwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config,target='simpleSim', outputs = outputs)
hardwareNetwork = CRI_network(axons=axonsDict,connections=neuronsDict,config=config,target='CRI', outputs = outputs, simDump=False)

# In[42]:


def input_to_CRI(currentInput):
  currentInput = data.view(data.size(0), -1)

#   print(np.shape(currentInput)[1])
#   if(np.shape(currentInput)[1] != len(axonsDict)):
#       print('bad axon to input match')
  #Onebatch = []
  #for batch in currentInput:
  #    batch = batch.T
  #    print(np.shape(batch))
  batch = []
  n = 0
  for element in currentInput:
      timesteps = []
      rateEnc = spikegen.rate(element,num_steps)
      rateEnc = rateEnc.detach().cpu().numpy()
      #print(rateEnc.shape)
      for slice in rateEnc:
          currInput = ['a'+str(idx) for idx,axon in enumerate(slice) if axon != 0]
          biasInput = ['a'+str(idx) for idx in range(784,len(axonsDict))]
          #timesteps.append(currInput)
          #timesteps.append(biasInput)
          timesteps.append(currInput+biasInput)
      batch.append(timesteps)
  return batch

global total_time_cri

def run_CRI_sim(inputList):
    firstOutput = 1000
    predictions = []
    #each image
    total_time_cri = 0
    for currInput in inputList:
        #initiate the softwareNetwork for each image
        softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))        
        spikeRate = [0]*len(outputs)
        #each time step
        for slice in currInput:
            start_time = time.time()
            swSpike = softwareNetwork.step(slice, membranePotential=False)
            end_time = time.time()
            total_time_cri = total_time_cri + end_time-start_time
            #print(swSpike)
            for spike in swSpike:
                spikeIdx = int(spike) - firstOutput 
                if spikeIdx >= 0: 
                    spikeRate[spikeIdx] += 1 
        predictions.append(spikeRate.index(max(spikeRate))) 
    print(f"Total execution time CRISim: {total_time_cri:.5f} s")
    return(predictions)

def run_CRI_hw(inputList):
    firstOutput = 1000
    predictions = []
    #each image
    total_time_cri = 0
    for currInput in inputList:
        #initiate the softwareNetwork for each image
        cri_simulations.FPGA_Execution.fpga_controller.clear(len(neuronsDict), False, 0)  ##Num_neurons, simDump, coreOverride
        spikeRate = [0]*len(outputs)
        #each time step
        for slice in currInput:
            start_time = time.time()
            hwSpike = hardwareNetwork.step(slice, membranePotential=False)
            end_time = time.time()
            #total_time_cri = total_time_cri + end_time-start_time
            total_time_cri = total_time_cri + total_time_perstep
            #print(hwSpike)
            for spike in hwSpike:
                spikeIdx = int(spike) - firstOutput 
                if spikeIdx >= 0: 
                    spikeRate[spikeIdx] += 1 
        predictions.append(spikeRate.index(max(spikeRate))) 
    print(f"Total execution time CRIFPGA: {total_time_cri:.5f} s")
    return(predictions)



# import pickle

# pickle.dump( axonsDict, open( "axonsDict.p", "wb" ) )
# pickle.dump( neuronsDict, open( "neuronsDict.p", "wb" ) )# pickle.dump( outputs, open( "outputs.p", "wb" ) )

total = 0
correct = 0
cri_correct_sim = 0
cri_correct_hw = 0
criPred_sim = []
criPred_hw = []
# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=256, shuffle=True, drop_last=False)
batch_count = 0
with torch.no_grad():
    net.eval()
    for data, targets in iter(test_loader):
        data = data.to(device)
        targets = targets.to(device)
        input = input_to_CRI(data)
        #criPred_sim = torch.tensor(run_CRI_sim(input)).to(device)
        criPred_hw = torch.tensor(run_CRI_hw(input)).to(device)
        #print("CRI Predicted Sim: ",criPred_sim)
        #print("CRI Predicted HW: ",criPred_hw)
        #print("Target: ",targets)
        # print(data.shape)
        # forward pass
        start_time = time.time()
        test_spk, _ = net(data.view(data.size(0), -1))
        end_time = time.time()
        print(f"Totoal execution time TorchSNN: {end_time-start_time:.5f} s")
        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        #print("Torchsnn Predicted: ",predicted)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        #cri_correct_sim += (criPred_sim == targets).sum().item()
        cri_correct_hw += (criPred_hw == targets).sum().item()
        batch_count = batch_count+1
        if batch_count == 5: break #run for 10 batches

print(f"Total correctly classified test set images for TorchSNN: {correct}/{total}")
#print(f"Total correctly classified test set images for CRI Sim: {cri_correct_sim}/{total}")
print(f"Total correctly classified test set images for CRI HW: {cri_correct_hw}/{total}")
print(f"Test Set Accuracy for TorchSNN: {100 * correct / total:.2f}%")
#print(f"Test Set Accuracy for CRI Sim: {100 * cri_correct_sim / total:.2f}%")
print(f"Test Set Accuracy for CRI HW: {100 * cri_correct_hw / total:.2f}%")


