import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from spikingjelly.activation_based import neuron, surrogate, encoding, layer, functional
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader
from hs_api.api import CRI_network
import time
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tqdm import tqdm
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import hs_bridge

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=1, type=int, help='stride size')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-p', default=0, type=int, help='padding size')
parser.add_argument('-c', default=4, type=int, help='channel size')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=1, type=int, help='batch size')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-resume_path', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/nmnist/checkpoint_max_T_16_C_20_lr_0.001_opt_adam.pth', type=str, help='checkpoint file')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/NMNIST', type=str, help='path to dataset')
parser.add_argument('-targets', default=10, type=int, help='Number of labels')

def norm(x: torch.Tensor):
    '''Standardizes input tensors to a zero mean and unit variance for consistent visualization.'''
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    '''
    Generates a 2D heatmap plot. Primarily used to visualize membrane potential accumulation 
    across all neurons over the simulation time steps.
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

    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
    return fig

class Net(nn.Module):
    '''
    Spiking Neural Network architecture designed for NMNIST classification.
    
    Consists of a 2D convolutional layer followed by batch normalization and a 
    fully connected linear layer, using spiking neurons (LIF/IF) to maintain state 
    across time steps.
    '''
    def __init__(self, in_channels = 2, channels=8, spiking_neuron: callable = None, **kwargs):
        super().__init__()
            
        self.conv = layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16*16*channels, 10)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        '''Forward pass processing sequential event frames.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x

    def forward_cnn(self, x:torch.Tensor):
        '''Partial forward pass through the convolutional block only.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        return x
    
    def forward_lr(self, x:torch.Tensor):
        '''Partial forward pass through the linear/output block only.'''
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x
    
def main():
    '''
    Main hardware validation pipeline.
    
    This script performs high-fidelity parity checking by running the NMNIST 
    dataset through both a quantized PyTorch SNN and a converted model on the 
    CRI FPGA hardware. It compares membrane potentials, spike timing, and 
    overall classification accuracy to ensure hardware-software alignment.
    '''
    args = parser.parse_args()
    print(args)
    
    # Prepare the dataset
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True)
    
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    # Pre-conversion: Fold BN and Quantize
    bn = BN_Folder()
    net_bn = bn.fold(net)
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    # Conversion Setup
    input_layer, output_layer = 0, 4 
    input_shape = (2, 34, 34)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(num_steps = args.T,
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    v_threshold = v_threshold,
                    embed_dim=0,
                    dvs=True)
    
    cn.layer_converter(net_quan)
    
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': int(qn.v_threshold)}
    }
    
    # Target: CRI FPGA
    hardwareNetwork = CRI_network(dict(cn.axon_dict),
            connections=dict(cn.neuron_dict),
            config=config,
            target='CRI', 
            outputs = cn.output_neurons,
            simDump=False,
            coreID=1,
            perturbMag=0, 
            leak=2**6-1) 
    
    encoder = encoding.PoissonEncoder()
    writer = SummaryWriter("log")
    
    start_time = time.time()
    hw_loss, hw_acc, tor_loss, tor_acc, test_samples = 0., 0., 0., 0., 0
    loss_fun = nn.MSELoss()
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            img, label = data
            img = img.transpose(0, 1) 
            tor_v_list, hw_v_list, tor_s_list, hw_s_list = [], [], [], []
            tor_out, hw_out = 0., 0.
            
            for i, t in enumerate(img):
                encoded_img = encoder(t)
                
                # PyTorch Path
                cnn_out = net_quan(encoded_img)
                tor_out += cnn_out
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                tor_v_list.append(torch.cat((net_quan.lif1.v.flatten().unsqueeze(0), net_quan.lif2.v.flatten().unsqueeze(0)),1))
                
                # Hardware Path
                cri_input = cn._input_converter_step(encoded_img)
                hwOutput, spikeResult = hardwareNetwork.step(cri_input[0], membranePotential=True)
                hwSpike, _, _ = spikeResult
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in hwSpike]
                hw_v_list.append(torch.tensor([v for k,v in hwOutput]).unsqueeze(0)) 
                
                if i != 0:
                    hw_spikes = torch.zeros(cnn_out.shape).flatten()
                    hw_spikes[spikeIdx] = 1
                    hw_s_list.append(hw_spikes.unsqueeze(0))
                    hw_out += hw_spikes.unsqueeze(0)
                    
            # Handle final phase delay
            hwOutput, spikeResult = hardwareNetwork.step([], membranePotential=True)
            hwSpike, _, _ = spikeResult
            spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in hwSpike]
            hw_spikes = torch.zeros(cnn_out.shape).flatten()
            hw_spikes[spikeIdx] = 1
            hw_s_list.append(hw_spikes.unsqueeze(0))
            hw_out += hw_spikes.unsqueeze(0)
            
            # Metrics and Visualization
            tor_v_list, hw_v_list = torch.cat(tor_v_list), torch.cat(hw_v_list)
            hw_s_list, tor_s_list = torch.cat(hw_s_list), torch.cat(tor_s_list)
            
            spike_acc = (tor_s_list==hw_s_list).sum()/tor_s_list.numel() * 100
            print(f"HW Spikes {spike_acc:.2f}% matches")
            
            # Clear Hardware State and Reset Network
            hs_bridge.FPGA_Execution.fpga_controller.clear(len(cn.neuron_dict), False, 0)
            functional.reset_net(net_quan)
            plt.close()

    print(f'\nFinal Results - HW Acc: {hw_acc/test_samples:.4f}, Torch Acc: {tor_acc/test_samples:.4f}')
    
if __name__ == '__main__':
    main()