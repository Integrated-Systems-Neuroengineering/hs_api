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
    
def main():
    
    args = parser.parse_args()
    print(args)
    
    # Prepare the dataset
    test_set = NMNIST(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    
    # Create DataLoaders
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory = True
    )
    
    net = Net(spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    
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
            perturbMag=8, #Zero randomness  
            leak=2**6)
    hardwareNetwork = CRI_network(dict(cn.axon_dict),
            connections=dict(cn.neuron_dict),
            config=config,target='CRI', 
            outputs = cn.output_neurons,
            simDump=False,
            coreID=1,
            perturbMag=8, #Zero randomness  
            leak=2**6)
    
    start_time = time.time()
    
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    test_loss_torch = 0
    test_acc_torch = 0
    
    test_loss_hard = 0
    test_acc_hard = 0
    
    encoder = encoding.PoissonEncoder()
    
    loss_fun = nn.MSELoss()
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.transpose(0, 1) # [B, T, C, H, W] -> [T, B, C, H, W]
            label_onehot = F.one_hot(label, args.targets).float()
            out_tor = 0.
            
            cri_input = []
            
            for t in img:
                encoded_img = encoder(t)
                cri_input.append(encoded_img)
                out_tor += net_quan(encoded_img)
                
            out_tor = out_tor/args.T
            
            cri_input = torch.stack(cri_input)
            cri_input = cri_input.transpose(0, 1) # [T, N, C, H, W] -> [N, T, C, H, W]
            cri_input = cn.input_converter(cri_input)
            out_hard = torch.tensor(cn.run_CRI_hw(cri_input,hardwareNetwork), dtype=float).to(device)    
            out_fr = torch.tensor(cn.run_CRI_sw(cri_input,softwareNetwork), dtype=float).to(device)    
            
            print(f'Label : {label} Soft: {out_fr} Hard: {out_hard} Torch_Pred: {out_tor}')
            
            loss = loss_fun(out_fr, label)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr == label).float().sum().item()      
            
            loss_torch = loss_fun(out_tor, label_onehot)
            test_loss_torch += loss_torch.item() * label.numel()
            test_acc_torch += (out_tor.argmax(1)==label).float().sum().item()
            
            loss_hard = loss_fun(out_hard, label)
            test_loss_hard += loss_hard.item() * label.numel()
            test_acc_hard += (out_hard==label).float().sum().item()
            
            functional.reset_net(net_quan)
    
    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    
    test_loss_torch /= test_samples
    test_acc_torch /= test_samples        
    
    test_loss_hard /= test_samples
    test_acc_hard /= test_samples  
    
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')
    print(f'test_loss_torch ={test_loss_torch: .4f}, test_acc_torch ={test_acc_torch: .4f}')
    print(f'test_loss_hard ={test_loss_hard: .4f}, test_acc_hard ={test_acc_hard: .4f}')
    print(f'test speed ={test_speed: .4f} images/s')
    

    
if __name__ == '__main__':
    main()