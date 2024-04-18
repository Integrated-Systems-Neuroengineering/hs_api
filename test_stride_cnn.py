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
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from spikingjelly import visualizing
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', default=1, type=int, help='stride size')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-p', default=0, type=int, help='padding size')
parser.add_argument('-c', default=4, type=int, help='channel size')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=1, type=int, help='batch size')
parser.add_argument('-T', default=16, type=int)
parser.add_argument('-resume_path', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/dvs_gesture/checkpoint_max_T_16_C_20_lr_0.001.pth', type=str, help='checkpoint file')
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/DVS128Gesture', type=str, help='path to dataset')
parser.add_argument('-targets', default=11, type=int, help='Number of labels')

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
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

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, encoder = 3, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 15 * 15, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
def main():
    #python test_stride_cnn.py -resume_path /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture/checkpoint_max_T_16_C_20_lr_0.001.pth -data-dir /Users/keli/Code/CRI/data/DVS128Gesture
    args = parser.parse_args()
    print(args)
    
    #Prepare the dataset
    # DVS128
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    
    # Create DataLoaders
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = DVSGestureNet(channels=20, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    device = torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    
    bn = BN_Folder()
    net_bn = bn.fold(net)
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    #Set the parameters for conversion
    input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
    output_layer = 14 #last pytorch layer that acts as synapses
    input_shape = (2, 128, 128)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(num_steps = args.T,
                    input_layer = input_layer, 
                    output_layer = output_layer, 
                    input_shape = input_shape,
                    v_threshold = v_threshold,
                    embed_dim=0,
                    dvs=True)
    
    cn.layer_converter(net_quan)
    
    breakpoint()
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(qn.v_threshold)
    
    
    hardwareNetwork = CRI_network(dict(cn.axon_dict),
                    connections=dict(cn.neuron_dict),
                    config=config,target='CRI', 
                    outputs = cn.output_neurons,
                    simDump=False,
                    coreID=1,
                    perturbMag=0, #Zero randomness  
                    leak=2**6-1)
    softwareNetwork = CRI_network(dict(cn.axon_dict),
                connections=dict(cn.neuron_dict),
                config=config,target='simpleSim', 
                outputs = cn.output_neurons,
                simDump=False,
                coreID=1,
                perturbMag=0, #Zero randomness  
                leak=2**6-1) #LIF
    breakpoint()

    start_time = time.time()
    
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    test_loss_torch = 0
    test_acc_torch = 0
    
    encoder = encoding.PoissonEncoder()
    
    loss_fun = nn.MSELoss()
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            imgs, label = data
            imgs = imgs.transpose(0, 1) # [N, T, C, H, W] -> [T, N, C, H, W]
            label_onehot = F.one_hot(label, args.targets).float()
            out_tor= 0.
            
            cri_v_list, tor_v_list = [], []
            cri_s_list, tor_s_list = [], []
            
            cri_input = []
            for t in imgs:
                encoded_img = encoder(t)
                cri_input.append(encoded_img)
                spikes = net_quan(encoded_img)
                #Obtain the spikes and threahold from the last layer of the network for comparision
                tor_s_list.append(spikes.flatten().unsqueeze(0))
                out_tor += spikes
                tor_v_list.append(net_quan.conv_fc[15].v.flatten().unsqueeze(0))    
            
            tor_s_list = torch.cat(tor_s_list)
            tor_v_list = torch.cat(tor_v_list)
            
            #Divide the spike by the timestep to get firing rate
            out_tor = out_tor/args.T
            
            cri_input = torch.stack(cri_input)
            cri_input = cri_input.transpose(0, 1) # [T, N, C, H, W] -> [N, T, C, H, W]
            cri_input = cn.input_converter(cri_input)
            # Running for a single image over T timesteps
  
            # List of list, list of list
            out_fr, potentials = cn.run_CRI_hw(cri_input, hardwareNetwork, outputPotential=True)
            # out_fr, potentials = cn.run_CRI_sw(cri_input, hardwareNetwork, outputPotential=True)
            
            out_fr = [torch.tensor(fr, dtype=float).to(device) for fr in out_fr]
            out_fr = torch.stack(out_fr)
            
            cri_v_list = [torch.tensor(p, dtype=float).to(device) for p in potentials]
            cri_v_list = torch.stack(cri_v_list)
            cri_s_list = out_fr
            
            out_fr = out_fr/args.T
            print(f'Label : {label} Pred: {out_fr} Torch_Pred: {out_tor}')
            
            loss = loss_fun(out_fr, label)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()      
            
            loss_torch = loss_fun(out_tor, label_onehot)
            test_loss_torch += loss_torch.item() * label.numel()
            test_acc_torch += (out_tor.argmax(1)==label).float().sum().item()
            functional.reset_net(net_quan)
            
            #plotting the membrane potential and spikes
            figsize = (12, 8)
            dpi = 100
            plot_2d_heatmap(array=tor_v_list.numpy(), title='PyTorch membrane potentials', xlabel='simulating step',
                                        ylabel='neuron index', int_x_ticks=True, x_max=args.T, figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/PyTorch_V_{img_idx}.png")
            
            plot_2d_heatmap(array=cri_v_list.numpy(), title='HW membrane potentials', xlabel='simulating step',
                                        ylabel='neuron index', int_x_ticks=True, x_max=args.T, figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/CRI_V_{img_idx}.png")
            
            visualizing.plot_1d_spikes(spikes=tor_s_list.numpy(), title='PyTorch Spikes', xlabel='simulating step',
                        ylabel='neuron index', figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/PyTorch_S_{img_idx}.png")
            visualizing.plot_1d_spikes(spikes=cri_s_list.numpy(), title='HW Spikes', xlabel='simulating step',
                        ylabel='neuron index', figsize=figsize, dpi=dpi)
            plt.savefig(f"figure/HW_S_{img_idx}.png")
            
            breakpoint()
            
    
    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    
    test_loss_torch /= test_samples
    test_acc_torch /= test_samples        
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test_loss_torch ={test_loss_torch: .4f}, test_acc_torch ={test_acc_torch: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')
    
if __name__ == '__main__':
    main()