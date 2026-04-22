import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder
from spikingjelly.activation_based import neuron, surrogate, encoding, layer, functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from hs_api.api import CRI_network
import time
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from spikingjelly import visualizing
from tqdm import tqdm
import torchvision.transforms as transforms

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

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    '''
    Generates a 2D heatmap visualization, commonly used to analyze membrane potential 
    distributions across layers over simulation time steps.
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

class DVSGestureNet(nn.Module):
    '''
    Spiking Neural Network for DVS128 Gesture recognition.
    
    The architecture is modular, containing a variable number of convolutional 
    encoding layers (Conv2d, BatchNorm2d, MaxPool2d, SpikingNeuron) followed by 
    fully connected spiking layers.
    '''
    def __init__(self, channels=16, encoder = 4, spiking_neuron: callable = None, *args, **kwargs):
        '''
        Initializes the modular convolutional blocks and final linear layers.

        Args:
            channels (int): The number of intermediate feature channels.
            encoder (int): The number of convolutional blocks to stack.
            spiking_neuron (callable): The spiking neuron class (e.g., LIFNode).
            *args, **kwargs: Arguments passed to the spiking neuron constructor.
        '''
        super().__init__()

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(*args, **kwargs))
            conv.append(layer.MaxPool2d(2, 2))


        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 110),
            spiking_neuron(*args, **kwargs),

            layer.Dropout(0.5),
            layer.Linear(110, 11),
            spiking_neuron(*args, **kwargs)
        )

    def forward(self, x: torch.Tensor):
        '''Performs a full forward pass through the entire spiking sequence.'''
        return self.conv_fc(x)
    
    def encode(self, x: torch.Tensor):
        '''
        Extracts features from the first convolutional block. 
        Used to feed pre-processed input into the converted CRI network.
        '''
        x = self.conv_fc[0](x)
        x = self.conv_fc[1](x)
        x = self.conv_fc[2](x)
        return x
    
def main():
    '''
    Evaluates the DVS128 Gesture dataset by comparing a Quantized PyTorch SNN 
    implementation with a converted SNN running on the CRI Software Simulator.

    Process:
    1. Loads DVS frames and initializes the DVSGestureNet.
    2. Folds Batch Normalization and applies quantization.
    3. Converts PyTorch layers to CRI Axon/Neuron structures.
    4. Runs dual-inference: PyTorch native vs. CRI Simulator (simpleSim).
    5. Validates parity via spike matching, firing rate comparison, and loss/accuracy calculation.
    '''
    args = parser.parse_args()
    print(args)
    
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = DVSGestureNet(channels=args.c, spiking_neuron=neuron.LIFNode, decay_input=False, surrogate_function=surrogate.ATan(), detach_reset=True)
    encoder = DVSGestureNet(channels=args.c, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    device = torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    encoder.eval()
    
    bn = BN_Folder()
    net_bn = bn.fold(net)
    encoder_bn = bn.fold(encoder)
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    encoder_quan = qn.quantize(encoder_bn)
    
    functional.set_step_mode(net_quan, 'm')
    
    input_layer = 3 
    output_layer = 21 
    snn_layers = 9 
    input_shape = (16, 64, 64)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(num_steps = args.T,
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    snn_layers=snn_layers,
                    v_threshold = v_threshold,
                    embed_dim=0,
                    dvs=True)
    
    cn.layer_converter(net_quan)
    
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': int(qn.v_threshold)}
    }
    
    softwareNetwork = CRI_network(dict(cn.axon_dict),
                connections=dict(cn.neuron_dict),
                config=config,target='simpleSim', 
                outputs = cn.output_neurons,
                simDump=False,
                coreID=1,
                perturbMag=None, 
                leak=1) 
    
    transform = transforms.Compose([
        transforms.Resize([64,64], transforms.InterpolationMode.NEAREST)
    ])

    start_time = time.time()
    test_loss_cri, test_acc_cri, test_samples = 0, 0, 0
    test_loss_torch, test_acc_torch = 0, 0
    layer_number = 7
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            imgs, label = data
            imgs = imgs.transpose(0, 1) 
            imgs = torch.cat([transform(f).unsqueeze(0) for f in imgs]) 
            label_onehot = F.one_hot(label, args.targets).float()
            
            cri_s_list, tor_s_list = [], []
            out_tor = 0.
            out_cri = 0.
            
            # Step 1: Quantized PyTorch Inference
            out_tor = net_quan(imgs).mean(0)
            loss = F.mse_loss(out_tor, label_onehot)
            test_loss_torch += loss.item() * label.numel()
            test_acc_torch += (out_tor.argmax(1) == label).float().sum().item()
            test_samples += label.numel()
            tor_s_list.append(out_tor.flatten().unsqueeze(0))
            
            # Step 2: CRI Software Simulation
            for t, img in enumerate(imgs):
                encoded_img = encoder_quan.encode(img)
                cri_input = cn._input_converter_step(encoded_img, t)
                swOutput, swSpike = softwareNetwork.step(cri_input[0], membranePotential=True)
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                
                if t > layer_number - 1:
                    cri_spikes = torch.zeros(out_tor.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
                    out_cri += cri_spikes
                
            # Phase and Layer delay simulation
            for _ in range(layer_number):
                swOutput, swSpike = softwareNetwork.step([], membranePotential=True)
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                cri_spikes = torch.zeros(out_tor.shape).flatten()
                cri_spikes[spikeIdx] = 1
                cri_s_list.append(cri_spikes.unsqueeze(0))   
                out_cri += cri_spikes 
    
            cri_s_list = torch.cat(cri_s_list)
            tor_s_list = torch.cat(tor_s_list)
            
            # Parity and Accuracy calculation
            loss_cri = F.mse_loss(out_cri, label_onehot)
            test_loss_cri += loss_cri.item()*label.numel()
            test_acc_cri += (out_cri.argmax(1) == label).float().sum().item()      
            
            softwareNetwork.simpleSim.initialize_sim_vars(len(cn.neuron_dict))
            functional.reset_net(net_quan)
            
            # Visualizing spike match
            num_matches = (tor_s_list==cri_s_list).sum()
            accuracy = num_matches/tor_s_list.numel() * 100 if num_matches != 0 else 0
            print(f"Spikes {accuracy}% matches")
            
    print(f'\n--- Final Results ---')
    print(f'CRI Simulation: Loss={test_loss_cri/test_samples:.4f}, Acc={test_acc_cri/test_samples:.4f}')
    print(f'PyTorch Native: Loss={test_loss_torch/test_samples:.4f}, Acc={test_acc_torch/test_samples:.4f}')

if __name__ == '__main__':
    main()