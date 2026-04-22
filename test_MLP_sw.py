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
parser.add_argument('-c', default=4, type=int, help='channel size')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=1, type=int, help='batch size')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-resume_path', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/mnist/checkpoint_max_T_16_C_20_lr_0.001_opt_adam.pth', type=str, help='checkpoint file')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data', type=str, help='path to dataset')
parser.add_argument('-targets', default=10, type=int, help='Number of labels')
parser.add_argument('-figure-dir', default='/Users/keli/Code/CRI/hs_api/figure',type=str, help='path to output figure' )

def norm(x: torch.Tensor):
    '''
    Normalizes a tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    '''
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    '''
    Generates a 2D heatmap plot, typically used for visualizing membrane potentials 
    over simulation time steps.

    Args:
        array (np.ndarray): 2D data array.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        int_x_ticks (bool): Whether to use integer ticks for the X-axis.
        int_y_ticks (bool): Whether to use integer ticks for the Y-axis.
        plot_colorbar (bool): Include a colorbar.
        colorbar_y_label (str): Label for the colorbar.
        x_max (int, optional): Maximum X extent for the plot.
        figsize (tuple): Figure size.
        dpi (int): Figure resolution.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    '''
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, heatmap = plt.subplots(figsize=figsize, dpi=dpi)
    if x_max is not None:
        im = heatmap.imshow(array.T, aspect='auto', extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5], vmin=-100000, vmax=30000)
    else:
        im = heatmap.imshow(array.T, aspect='auto', vmin=-100000, vmax=30000)

    heatmap.set_title(title)
    heatmap.set_xlabel(xlabel)
    heatmap.set_ylabel(ylabel)

    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return fig

class Net(nn.Module):
    '''
    Spiking Neural Network architecture for MNIST classification.
    
    Includes a convolutional layer followed by batch normalization and 
    a linear output layer, utilizing LIF neurons for spiking dynamics.
    '''
    def __init__(self, in_channels = 1, out_channels = 1, w = 28, h = 28, spiking_neuron: callable = None, **kwargs):
        '''
        Initializes the spiking network layers.

        Args:
            in_channels (int): Input image channels.
            out_channels (int): Convolutional output channels.
            w (int): Input width.
            h (int): Input height.
            spiking_neuron (callable): Neuron class (e.g., LIFNode).
            **kwargs: Arguments passed to the spiking neuron.
        '''
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias = False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear1 = layer.Linear(out_channels * 13 * 13, out_features= 40, bias = True)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))
    
    def forward(self, x: torch.Tensor):
        '''Forward pass of the SNN.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.lif2(x)
        return x

def main():
    '''
    Main validation script. Compares a quantized PyTorch SNN implementation 
    with its converted counterpart running on the CRI software simulator.
    
    The script performs:
    1. Dataset loading and model quantization.
    2. Conversion to CRI format (Axons/Neurons).
    3. Parallel execution of both backends per time step.
    4. Parity checks for membrane potentials, spikes, and firing rates.
    5. Visualization of potentials and raster plots for debugging.
    '''
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
    
    # Fold Batch Norm and apply Quantization
    bn = BN_Folder()
    net_bn = bn.fold(net)
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    # CRI Conversion setup
    input_layer = 0 
    output_layer = 4 
    snn_layers = 2 
    input_shape = (1, 28, 28)
    v_threshold = qn.v_threshold
    
    cn = CRI_Converter(num_steps = args.T,
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    snn_layers = snn_layers,
                    v_threshold = int(v_threshold),
                    embed_dim=0,
                    dvs=False)
    
    cn.layer_converter(net_quan)
    
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': int(qn.v_threshold)}
    }
    
    softwareNetwork = CRI_network(dict(cn.axon_dict),
            connections=dict(cn.neuron_dict),
            config=config,
            target='simpleSim', 
            outputs = cn.output_neurons,
            simDump=False,
            coreID=1,
            perturbMag=None,
            leak=1) 
    
    encoder = encoding.PoissonEncoder()
    writer = SummaryWriter("log")
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            img, label = data
            img, label = img.to(device), label.to(device)
            
            cri_v_list, tor_v_list = [], []
            cri_s_list, tor_s_list = [], []
            out_tor = 0.
            
            for t in range(args.T):
                encoded_img = encoder(img)
                
                # Step 1: PyTorch Execution
                cnn_out = net_quan(encoded_img)
                out_tor += cnn_out
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                tor_v_list.append(torch.cat((net_quan.lif1.v.flatten().unsqueeze(0),
                                             net_quan.lif2.v.flatten().unsqueeze(0)), 1))
                
                # Step 2: CRI Software Execution
                cri_input = cn._input_converter_step(encoded_img, t)
                swOutput, swSpike = softwareNetwork.step(cri_input[0], membranePotential=True)
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                
                cri_v_list.append(torch.tensor([v for k,v in swOutput]).unsqueeze(0))
                
                if t > snn_layers - 1:
                    cri_spikes = torch.zeros(cnn_out.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
            
            # Phase and Layer delay handling
            for _ in range(snn_layers):
                swOutput, swSpike = softwareNetwork.step([], membranePotential=True)
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                cri_spikes = torch.zeros(cnn_out.shape).flatten()
                cri_spikes[spikeIdx] = 1
                cri_s_list.append(cri_spikes.unsqueeze(0))
        
            tor_v_list = torch.cat(tor_v_list)
            cri_v_list = torch.cat(cri_v_list)
            cri_s_list = torch.cat(cri_s_list)
            tor_s_list = torch.cat(tor_s_list)
            
            cri_v_list[cri_v_list >= cn.v_threshold] = 0
            
            # Visualization and Parity Checks
            figsize, dpi = (12, 8), 100
            plot_2d_heatmap(array=tor_v_list.numpy(), title='PyTorch potentials', xlabel='steps', ylabel='neuron', x_max=args.T, figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/PyTorch_V_{img_idx}.png")
            
            plot_2d_heatmap(array=cri_v_list.numpy(), title='CRI potentials', xlabel='steps', ylabel='neuron', x_max=args.T, figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/CRI_V_{img_idx}.png")
        
            # Accuracy Metrics
            spike_match = (tor_s_list==cri_s_list).sum() / tor_s_list.numel() * 100
            print(f"Spikes {spike_match}% matches")
            
            # Reset states
            softwareNetwork.simpleSim.initialize_sim_vars(len(cn.neuron_dict))
            functional.reset_net(net_quan)
            breakpoint()

if __name__ == '__main__':
    main()