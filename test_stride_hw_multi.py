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
from hs_api.neuron_models import LIF_neuron

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
    '''Standardizes a tensor by mean/std for visualization consistency.'''
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    '''Plots a 2D heatmap typically representing neuron membrane potentials over time.'''
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
    Spiking Neural Network for NMNIST classification.
    
    Includes split forward methods to isolate the convolutional feature extraction 
    (targeted for hardware acceleration) from the final linear classification.
    '''
    def __init__(self, in_channels = 2, channels=8, spiking_neuron: callable = None, **kwargs):
        '''Initializes SNN layers for processing 2-channel event data.'''
        super().__init__()
            
        self.conv = layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16*16*channels, 10)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        '''Full forward pass through the SNN.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x

    def forward_cnn(self, x:torch.Tensor):
        '''Executes only the convolutional portion of the network.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        return x
    
    def forward_lr(self, x:torch.Tensor):
        '''Executes only the linear/classification portion of the network.'''
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x
    
def main():
    '''
    Hardware validation script for NMNIST.
    
    Loads a pre-trained SNN, quantizes the convolutional layers, converts them to 
    CRI format, and executes them on the CRI FPGA hardware. Parity is checked 
    by comparing hardware output spikes and membrane potentials against 
    the quantized PyTorch simulation.
    '''
    args = parser.parse_args()
    print(args)
    
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True)
    
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net_cnn = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net_cnn.load_state_dict(checkpoint['net'])
    
    net.eval()
    net_cnn.eval()
    
    bn = BN_Folder()
    net_bn = bn.fold(net)
    net_cnn_bn = bn.fold(net_cnn)
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    net_cnn_quan = qn.quantize(net_cnn_bn)
    
    input_layer, output_layer = 0, 0 
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
    
    test_neurons = {k: (LIF_neuron(int(qn.v_threshold), 0, 2**6), cn.cnn_neurons[k]) for k in cn.cnn_neurons}
    
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': int(qn.v_threshold)}
    }
    
    hardwareNetwork = CRI_network(dict(cn.cnn_axons),
            connections=test_neurons,
            config=config,
            target='CRI', 
            outputs = cn.cnn_output,
            simDump=False,
            coreID=1)
    
    encoder = encoding.PoissonEncoder()
    writer = SummaryWriter("log")
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            img, label = data
            img = img.transpose(0, 1) 
            
            cri_v_list, tor_v_list = [], []
            cri_s_list, tor_s_list = [], []
            
            for i, t in enumerate(img):
                encoded_img = encoder(t)
                
                # PyTorch Baseline
                cnn_out = net_cnn_quan.forward_cnn(encoded_img)
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                tor_v_list.append(net_cnn_quan.lif1.v.flatten().unsqueeze(0))
                
                # Hardware Step
                cri_input = cn._input_converter_step(encoded_img)
                hwOutput, spikeResult  = hardwareNetwork.step(cri_input[0], membranePotential=True)
                hwSpike, _, _ = spikeResult
                spikeIdx = [int(spike) for spike in hwSpike]
                
                cri_v_list.append(torch.tensor([v for k,v in hwOutput]).unsqueeze(0))
                
                if i != 0:
                    cri_spikes = torch.zeros(cnn_out.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
            
            # Phase delay flush
            hwOutput, spikeResult = hardwareNetwork.step([], membranePotential=True)
            hwSpike, _, _ = spikeResult
            spikeIdx = [int(spike) for spike in hwSpike]
            cri_spikes = torch.zeros(cnn_out.shape).flatten()
            cri_spikes[spikeIdx] = 1
            cri_s_list.append(cri_spikes.unsqueeze(0))
            
            # Comparison Analysis
            tor_v_list, cri_v_list = torch.cat(tor_v_list), torch.cat(cri_v_list)
            cri_s_list, tor_s_list = torch.cat(cri_s_list), torch.cat(tor_s_list)
            
            spike_acc = (tor_s_list==cri_s_list).sum()/tor_s_list.numel() * 100
            print(f"Spikes {spike_acc:.2f}% matches")
            
            # Hardware reset
            hs_bridge.FPGA_Execution.fpga_controller.clear(len(cn.cnn_neurons), False, 0)
            functional.reset_net(net_cnn_quan)
            hardwareNetwork.sim_flush()
            breakpoint()

if __name__ == '__main__':
    main()