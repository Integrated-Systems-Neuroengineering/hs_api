import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from spikingjelly.activation_based import neuron, surrogate, encoding, layer, functional
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader
import torchvision
from hs_api.api import CRI_network
import time
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=1, type=int, help='stride size')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-p', default=0, type=int, help='padding size')
parser.add_argument('-c', default=16, type=int, help='channel size')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=1, type=int, help='batch size')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-resume_path', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/dvs_gesture/checkpoint_max_T_16_C_20_lr_0.001.pth', type=str, help='checkpoint file')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/DVS128Gesture', type=str, help='path to dataset')
parser.add_argument('-targets', default=11, type=int, help='Number of labels')

class Net(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, w = 28, h = 28, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias = False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3)
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16, out_features= 10, bias = True)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x
    
def main():
    # python test_CNN_sw.py -data-dir /Users/keli/Code/CRI/data 
    args = parser.parse_args()
    print(args)
    
    # Prepare the dataset
    test_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False, 
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    
    # Create DataLoaders
    test_loader = DataLoader(
        test_set, 
        batch_size=args.b, 
        shuffle=False, 
        drop_last=False, 
        pin_memory = True
    )
    
    net = Net(spiking_neuron=neuron.LIFNode, tau=2.0, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    net = net.to(device)
    
    net.eval()
    
    # fold in the batchnorm layer
    bn = BN_Folder()
    net_bn = bn.fold(net)
    
    # quantization the weight
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    # Set the parameters for conversion
    input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
    output_layer = 5 #last pytorch layer that acts as synapses
    input_shape = (1, 28, 28)
    v_threshold = qn.v_threshold
    
    cn = CRI_Converter(num_steps=args.T,
                       input_layer = input_layer,
                       output_layer = output_layer,
                       input_shape = input_shape,
                       v_threshold = v_threshold,
                       )
    
    
    
    

    
    
if __name__ == '__main__':
    main()
    
    
    
    
    

    
    
    
    

    
    
    
    
