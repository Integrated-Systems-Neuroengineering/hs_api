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
    '''
    Normalizes a tensor by subtracting the mean and dividing by the standard deviation.

    Args:
        x (torch.Tensor): Input tensor to be normalized.

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
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
    Generates a 2D heatmap plot for visualizing membrane potentials or weights.

    Args:
        array (np.ndarray): 2D data array.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        int_x_ticks (bool): Whether to force integer ticks on X-axis.
        int_y_ticks (bool): Whether to force integer ticks on Y-axis.
        plot_colorbar (bool): Whether to include a color scale bar.
        colorbar_y_label (str): Label for the colorbar.
        x_max (int, optional): Maximum limit for the X-axis extent.
        figsize (tuple): Dimensions of the figure.
        dpi (int): Resolution of the figure.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
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
    Spiking Neural Network architecture for NMNIST classification.

    Consists of a convolutional layer, batch normalization, and a linear 
    output layer, with SpikingJelly IFNodes as activation layers.
    '''
    def __init__(self, in_channels = 2, channels=8, spiking_neuron: callable = None, **kwargs):
        '''
        Initialize the network layers.

        Args:
            in_channels (int): Number of input channels (typically 2 for DVS).
            channels (int): Number of feature maps in the conv layer.
            spiking_neuron (callable): Spiking neuron class (e.g., IFNode).
            **kwargs: Arguments passed to the spiking neuron.
        '''
        super().__init__()
            
        self.conv = layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16*16*channels, 10)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        '''Standard forward pass through the full network.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x

    def forward_cnn(self, x:torch.Tensor):
        '''Forward pass restricted to the convolutional feature extractor.'''
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        return x
    
    def forward_lr(self, x:torch.Tensor):
        '''Forward pass restricted to the linear classifier.'''
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x
    
def main():
    '''
    Main execution loop: loads NMNIST data, quantizes the PyTorch model, 
    converts it to a CRI-compatible format, and executes on FPGA hardware.
    '''
    args = parser.parse_args()
    print(args)
    
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    # Fold Batch Norm into Conv layers for hardware efficiency
    bn = BN_Folder()
    net_bn = bn.fold(net)
    
    # Quantize weights and thresholds
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    input_shape = (2, 34, 34)
    v_threshold = qn.v_threshold

    # Convert PyTorch layers to Axon/Neuron dictionaries for CRI
    cn = CRI_Converter(num_steps = args.T,
                    input_layer = 0, 
                    output_layer = 0, 
                    input_shape = input_shape,
                    v_threshold = v_threshold,
                    embed_dim=0,
                    dvs=True)
    
    cn.layer_converter(net_quan)
    
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': int(qn.v_threshold)}
    }
    
    # Initialize the hardware-linked network
    softwareNetwork = CRI_network(dict(cn.cnn_axons),
            connections=dict(cn.cnn_neurons),
            config=config, target='CRI', 
            outputs = cn.cnn_output,
            simDump=True,
            coreID=1,
            perturbMag=17,   
            leak=2**6) 
    
    encoder = encoding.PoissonEncoder()
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            img, label = data
            img = img.transpose(0, 1) # [T, B, C, H, W]
            
            for i, t in enumerate(img):
                encoded_img = encoder(t)
                cri_input = cn._input_converter_step(encoded_img)
                softwareNetwork.step(cri_input[0], membranePotential=True)
            
            # Phase delay step
            softwareNetwork.step([], membranePotential=True)
            
            # Clear FPGA state for next image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(cn.cnn_neurons), False, 0
            )
            
            softwareNetwork.sim_flush("test_stride_cnn_dump.txt")
            breakpoint()

if __name__ == '__main__':
    main()