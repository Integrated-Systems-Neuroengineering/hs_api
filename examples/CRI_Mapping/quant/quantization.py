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
num_steps = 25
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

PATH = "result/mnist_2layer_MLP/model_best.pth.tar"
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda") 

net.cuda()
#net.eval()

# If this was your first time training an SNN, then congratulations!

# In[23]:


print(list(net.fc1.parameters())[0].detach().cpu().numpy().shape)
print(list(net.fc2.parameters())[0].detach().cpu().numpy().shape)

def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        #print('uniform quant bit: ', b)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()     # >1 means clipped. # output matrix is a form of [True, False, True, ...]
            sign = input.sign()              # output matrix is a form of [+1, -1, -1, +1, ...]
            #grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            grad_alpha = (grad_output*(sign*i + (0.0)*(1-i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, "grad_alpha = grad_output * (input_q-input)"
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        #self.wgt_alpha = wgt_alpha
        self.weight_q = weight_quantization(b=self.w_bit)
        #self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))
    def forward(self, weight):
        #mean = weight.data.mean()
        #std = weight.data.std()
        #weight = weight.add(-mean).div(std)      # weights normalization
        weight_q = self.weight_q(weight, self.wgt_alpha)

        return weight_q


# In[24]:


fc1_quant      = weight_quant(net.fc1.weight)
w_delta      = w_alpha/(2**(w_bits-1)-1)
fc1_int        = fc1_quant/w_delta
print("FC1 Weights: \n",fc1_int)
#breakpoint()
for layer in net.modules():
	if isinstance(layer, torch.nn.Linear):
		layer.weight = Parameter(weight_quant(layer.weight))
		w_delta = w_alpha/(2**(w_bits-1)-1)
		layer.weight = Parameter(layer.weight/w_delta)
		layer.bias = Parameter(layer.bias/w_delta)
		print(layer.weight)
		print(layer.bias)
	if isinstance(layer, snn.Leaky):
		layer.threshold = layer.threshold/w_delta
# The loss curves are noisy because the losses are tracked at every iteration, rather than averaging across multiple iterations. 

# ## 8.2 Test Set Accuracy
# This function iterates over all minibatches to obtain a measure of accuracy over the full 10,000 samples in the test set.

# In[25]:


total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)
    
    # forward pass
    test_spk, _ = net(data.view(data.size(0), -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

if not os.path.exists('result'):
    os.makedirs('result')
fdir = 'result/'+'mnist_2layer_MLP_quantized'
if not os.path.exists(fdir):
    os.makedirs(fdir)


save_checkpoint({
        'state_dict': net.state_dict(),
    }, 1, fdir)
