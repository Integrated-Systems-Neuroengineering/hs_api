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

# Configuration and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('-s', default=1, type=int, help='stride size')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-p', default=0, type=int, help='padding size')
parser.add_argument('-c', default=4, type=int, help='channel size')
parser.add_argument('-alpha',  default=4, type=int, help='Range of value for quantization')
parser.add_argument('-b', default=1, type=int, help='batch size')
parser.add_argument('-T', default=16, type=int, help='Number of simulation time steps')
parser.add_argument('-resume_path', default='/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/nmnist/checkpoint_max_T_16_C_20_lr_0.001_opt_adam.pth', type=str)
parser.add_argument('-data-dir', default='/Volumes/export/isn/keli/code/data/NMNIST', type=str)
parser.add_argument('-targets', default=10, type=int)

class Net(nn.Module):
    """
    Spiking Neural Network for NMNIST.
    Structure: Conv2d -> BN -> LIF -> Flatten -> Linear -> LIF.
    """
    def __init__(self, in_channels=2, channels=8, spiking_neuron: callable = None, **kwargs):
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

def main():
    """
    NMNIST Software Parity Test: 
    Validates that the converted CRI Axon/Neuron graph produces bit-accurate 
    results compared to the quantized PyTorch model.
    """
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 1. Dataset Loading
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True, drop_last=True)
    
    # 2. Model Prep
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    # 3. BN Folding & Quantization
    # We must fold BN into weights because the hardware expects static synaptic weights.
    bn_folder = BN_Folder()
    net_bn = bn_folder.fold(net)
    
    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    
    # 4. Conversion to CRI Format
    # Converts PyTorch tensors to Axon/Neuron lists for the simulator.
    cn = CRI_Converter(num_steps=args.T, input_layer=0, output_layer=4, 
                    input_shape=(2, 34, 34), v_threshold=qn.v_threshold, dvs=True)
    cn.layer_converter(net_quan)
    
    # 5. Initialize CRI Software Simulator (simpleSim)
    config = {'neuron_type': "I&F", 'global_neuron_params': {'v_thr': int(qn.v_threshold)}}
    softwareNetwork = CRI_network(dict(cn.axon_dict), connections=dict(cn.neuron_dict),
                                config=config, target='simpleSim', outputs=cn.output_neurons, leak=2**6-1)
    
    encoder = encoding.PoissonEncoder()
    loss_fun = nn.MSELoss()
    
    # 6. Evaluation Loop
    with torch.no_grad():
        for img_idx, (img, label) in enumerate(tqdm(test_loader)):
            img = img.transpose(0, 1) # [T, B, C, H, W]
            tor_s_list, cri_s_list = [], []
            out_tor, out_cri = 0., 0.

            for i, t in enumerate(img):
                encoded_img = encoder(t)
                
                # PyTorch Inference
                cnn_out = net_quan(encoded_img)
                out_tor += cnn_out
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                
                # CRI Simulator Inference
                cri_input = cn._input_converter_step(encoded_img)
                swOutput, swSpike = softwareNetwork.step(cri_input[0], membranePotential=True)
                
                # Map spikes back to neuron indices
                spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
                if i != 0: # Accounts for phase delay
                    cri_spikes = torch.zeros(cnn_out.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
                    out_cri += cri_spikes.unsqueeze(0)

            # Final Phase Delay Step
            _, swSpike = softwareNetwork.step([], membranePotential=True)
            spikeIdx = [int(spike)-int(cn.output_neurons[0]) for spike in swSpike]
            cri_spikes = torch.zeros(cnn_out.shape).flatten()
            cri_spikes[spikeIdx] = 1
            cri_s_list.append(cri_spikes.unsqueeze(0))
            out_cri += cri_spikes.unsqueeze(0)

            # Accuracy Check
            tor_s_tensor, cri_s_tensor = torch.cat(tor_s_list), torch.cat(cri_s_list)
            match_acc = (tor_s_tensor == cri_s_tensor).sum() / tor_s_tensor.numel() * 100
            print(f"Sample {img_idx}: Spike Match = {match_acc:.2f}%")

            # Reset state for next image
            softwareNetwork.simpleSim.initialize_sim_vars(len(cn.neuron_dict))
            functional.reset_net(net_quan)

if __name__ == '__main__':
    main()