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
parser.add_argument('-figure-dir', default='/Users/keli/Code/CRI/hs_api/figure',type=str, help='path to output figure' )

def norm(x: torch.Tensor):
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)

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

class Net(nn.Module):
    def __init__(self, in_channels = 2, channels=8, spiking_neuron: callable = None, **kwargs):
        super().__init__()
            
        self.conv = layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16*16*channels, 10)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x

    def forward_cnn(self, x:torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        return x
    
    def forward_lr(self, x:torch.Tensor):
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x
    
def main():
    #python test_stride_sw.py -resume_path /Users/keli/Code/CRI/CRI_Mapping/runs/nmnist/checkpoint_latest_T_16_C_20_lr_0.001_opt_adam.pth -data-dir /Users/keli/Code/CRI/data/NMNIST 
    args = parser.parse_args()
    print(args)
    
    # Prepare the dataset
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    
    # Create DataLoaders
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net_cnn = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    
    bn = BN_Folder()
    net_bn = bn.fold(net)
    
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    breakpoint()
    
    #Set the parameters for conversion
    input_layer = 0 #first pytorch layer that acts as synapses, indexing begins at 0 
    output_layer = 4 #last pytorch layer that acts as synapses
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
    
    config = {}
    config['neuron_type'] = "I&F"
    config['global_neuron_params'] = {}
    config['global_neuron_params']['v_thr'] = int(qn.v_threshold)
    
    softwareNetwork = CRI_network(dict(cn.axon_dict),
            connections=dict(cn.neuron_dict),
            config=config,target='simpleSim', 
            outputs = cn.output_neurons,
            simDump=False,
            coreID=1,
            perturbMag=None, #Zero randomness  
            leak=2**6-1) #IF
    
    encoder = encoding.PoissonEncoder()
    
    writer = SummaryWriter("log")
    
    start_time = time.time()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    test_loss_torch = 0
    test_acc_torch = 0
    
    loss_fun = nn.MSELoss()
    
    with torch.no_grad():
        # Testing one image at a time
        for img_idx, data in enumerate(tqdm(test_loader)):
            
            img, label = data
            
            img = img.transpose(0, 1) # [1, T, C, H, W] -> [T, 1, C, H, W]
            
            cri_v_list, tor_v_list = [], []
            cri_s_list, tor_s_list = [], []
            
            out_tor = 0.
            out_cri = 0.
            
            for i, t in enumerate(img):
                
                encoded_img = encoder(t) #(B, C, H, W)
                
                cnn_out = net_quan(encoded_img)
                
                out_tor += cnn_out
                
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                
                tor_v_list.append(torch.cat((net_quan.lif1.v.flatten().unsqueeze(0), net_quan.lif2.v.flatten().unsqueeze(0)),1))
                
                cri_input = cn._input_converter_step(encoded_img)
                
                # swOutput: [(key, potential) for all the neurons in softwareNetwork] 
                swOutput, swSpike  = softwareNetwork.step(cri_input[0], membranePotential=True)
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                
                cri_v_list.append(torch.tensor([v for k,v in swOutput]).unsqueeze(0))
                if i != 0:
                    cri_spikes = torch.zeros(cnn_out.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
                    out_cri += cri_spikes.unsqueeze(0)
                    
            
            
            # empty input for phase delay 
            swOutput, swSpike = softwareNetwork.step([], membranePotential=True)
            spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
            cri_spikes = torch.zeros(cnn_out.shape).flatten()
            cri_spikes[spikeIdx] = 1
            cri_s_list.append(cri_spikes.unsqueeze(0))
            out_cri += cri_spikes.unsqueeze(0)
            
            tor_v_list = torch.cat(tor_v_list)
            cri_v_list = torch.cat(cri_v_list)
            
            figsize = (12, 8)
            dpi = 100
            # plot the membrane potential 
            # plot_2d_heatmap(array=tor_v_list.numpy(), title='PyTorch membrane potentials', xlabel='simulating step',
            #                             ylabel='neuron index', int_x_ticks=True, x_max=args.T, figsize=figsize, dpi=dpi)
            # plt.savefig(f"figure/PyTorch_V_{img_idx}.png")
            
            # plot_2d_heatmap(array=cri_v_list.numpy(), title='CRI membrane potentials', xlabel='simulating step',
            #                             ylabel='neuron index', int_x_ticks=True, x_max=args.T, figsize=figsize, dpi=dpi)
            # plt.savefig(f"figure/CRI_V_{img_idx}.png")
            
            
            #plot the spikes
            cri_s_list = torch.cat(cri_s_list)
            tor_s_list = torch.cat(tor_s_list)
            
            #compare the pytorch and software spike output
            num_matches = (tor_s_list==cri_s_list).sum()
            total = tor_s_list.numel()
            accuracy = num_matches/total * 100 if num_matches != 0 else 0
            print(f"Spikes {accuracy}% matches")
            writer.add_scalar('spike_match', accuracy, img_idx)
            
            #compare the pytorch and software membrane potential
            num_matches = (tor_v_list==cri_v_list).sum()
            total = tor_v_list.numel()
            accuracy = num_matches/total * 100 if num_matches != 0 else 0
            print(f"Membrane potential {accuracy}% matches")
            writer.add_scalar('potential_match', accuracy, img_idx)
            
            #compare the pytorch and software firing rate
            tor_r_list = torch.mean(tor_s_list.T, axis=1, keepdims=True)
            cri_r_list = torch.mean(cri_s_list.T, axis=1, keepdims=True)
            num_matches = (tor_r_list==cri_r_list).sum()
            total = cri_r_list.numel()
            accuracy = num_matches/total * 100 if num_matches != 0 else 0
            print(f"Firing rate {accuracy}% matches")
            writer.add_scalar('firing_rate_match', accuracy, img_idx)
            
            # visualizing.plot_1d_spikes(spikes=tor_s_list.numpy(), title='PyTorch Spikes', xlabel='simulating step',
            #             ylabel='neuron index', figsize=figsize, dpi=dpi)
            # plt.savefig(f"figure/PyTorch_S_{img_idx}.png")
            # visualizing.plot_1d_spikes(spikes=cri_s_list.numpy(), title='CRI Spikes', xlabel='simulating step',
            #             ylabel='neuron index', figsize=figsize, dpi=dpi)
            # plt.savefig(f"figure/CRI_S_{img_idx}.png")

            out_tor /= args.T
            out_cri /= args.T
            label_onehot = F.one_hot(label, args.targets).float()
            test_samples += label.numel()
            
            loss = loss_fun(out_cri, label_onehot)
            test_loss += loss.item() * label.numel()
            test_acc += (out_cri.argmax(0) == label).float().sum().item()      
            
            loss_torch = loss_fun(out_tor, label_onehot)
            test_loss_torch += loss_torch.item() * label.numel()
            test_acc_torch += (out_tor.argmax(1)==label).float().sum().item()
            
            # reset the membrane potential to zero
            softwareNetwork.simpleSim.initialize_sim_vars(len(cn.neuron_dict))
            functional.reset_net(net_quan)
            
            # reset pyplot interface
            # plt.close()    

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