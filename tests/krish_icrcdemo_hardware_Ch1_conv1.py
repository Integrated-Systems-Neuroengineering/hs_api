#to activate venv: eval $(poetry env activate)
#to avoid breakpoints: PYTHONBREAKPOINT=0 python icrcdemo_krish.py
#to run in background and log output(make sure to change paths): PYTHONBREAKPOINT=0 nohup python icrcdemo_krish.py -out-dir /home/k7arora/output/data_10T/1 > /home/k7arora/output/data_10T/1/log.txt 2>&1 &(but change path)
#to run default spikingjelly model with val_split: PYTHONBREAKPOINT=0 nohup python -u /home/k7arora/hs_api/examples/CRI_Mapping/icrcdemo_krish.py -b 16 -channels 128 -epochs 256 -out-dir /home/k7arora/output/data_10T/spikingjellytutorial/val_split > /home/k7arora/output/data_10T/spikingjellytutorial/val_split/log.txt 2>&1 &(but change path)
#PYTHONBREAKPOINT=0 nohup python -u icrcdemo_krish_hardware_09-02-25.py -b 64 -channels 4 -epochs 50 -out-dir /home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/IFNeuron/output > /home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/IFNeuron/output/log.txt 2>&1 &

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda import amp
import torchvision.transforms as transforms
import numpy as np
import random
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from utils_krish import train_DVS_Time, train_DVS_Time_with_plot, sw_comp_DVS, validate_DVS, validate_DVS_HW, test_DVS_Time, infer_cri_params
from hs_api import CRI_network
#from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder #initially just hs_api.converter
from hs_api.quantizer import Quantize_Network #initially just hs_api.converter
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as tq
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from hs_api.custom_neurons import Custom_LIFNode

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from copy import deepcopy

os.environ["PYTHONBREAKPOINT"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("-resume_path", default="", type=str, help="checkpoint file")
parser.add_argument("-load_path", default="", type=str, help="checkpoint loading path")
parser.add_argument(
    "-load_ssa_path", default="", type=str, help="ssa checkpoint loading path"
)
parser.add_argument(
    "-train", action="store_true", default=True, help="Train the network from stratch"
)
parser.add_argument("-b", default=32, type=int, help="batch size")
parser.add_argument(
    "-data-dir",
    default="/home/prpandit/DVS_Gesture_data",
    type=str,
    help="path to dataset",
)
parser.add_argument(
    "-out-dir",
    default="/home/prpandit/hs_api/tests/output",
    type=str,
    help="dir path that stores the trained model checkpoint",
)
parser.add_argument("-epochs", default=20, type=int)
parser.add_argument("-lr", default=1e-3, type=float) #normally 1e-3
parser.add_argument("-momentum", default=0.9, type=float, help="momentum for SGD")
parser.add_argument(
    "-weight_decay", default=0.01, type=float, help="weight decay for Adam"
)
parser.add_argument("-channels", default=20, type=int)
parser.add_argument(
    "-writer", action="store_true", default=False, help="Use torch summary"
)
parser.add_argument(
    "-encoder",
    action="store_true",
    default=True,
    help="Using spike rate encoder to process the input",
)
parser.add_argument(
    "-amp", action="store_true", default=True, help="Use mixed percision training"
)
parser.add_argument("-num_batches", default=4, type=int)
parser.add_argument(
    "-transformer",
    action="store_true",
    default=False,
    help="Training transformer model",
)
parser.add_argument(
    "-j",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-opt", default="adam", type=str, help="use which optimizer. SDG or Adam"
)
parser.add_argument(
    "-dvs", action="store_true", default=True, help="Using the DVS datasets"
)
parser.add_argument("-targets", default=11, type=int, help="target label size")
parser.add_argument("-T_max", default=64, type=int, help="T_max for CosineAnnealingLR")

args = parser.parse_args()
#output_dir = "/home/k7arora/output/data_10T/1"
output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

quantized_model_path = os.path.join(output_dir, "quantized_model.pth")



#keli's DVSGestureNet from original paper, with ability to change encoder
class DVSGestureNetNoBias(nn.Module):
    def __init__(self, channels=128, encoder = 3, out_features = 512, spiking_neuron: callable = None, input_shape = (16, 2, 128, 128), **kwargs):
        super().__init__()

        B, C, H, W = input_shape

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            if H > 3 and W > 3: #don't want to reduce spatial dims to 1x1 which fails batchnorm
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
                H = H // 2
                W = W // 2
            
            else:
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(spiking_neuron(**deepcopy(kwargs)))
            
            print(H)
            print(W)

        conv_seq = nn.Sequential(*conv)
        #print(conv_seq)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))

        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            #flatten but ignore batch size
            in_features = x_out.flatten(start_dim=1).shape[1]  # Flatten and get feature dim
            print(x_out.shape)

        print("Input features to first linear layer:", in_features)    

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5), #default 0.5
            layer.Linear(in_features, out_features, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5), #default 0.5
            layer.Linear(out_features, 11, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVSGestureNetChrisModeled(nn.Module):
    def __init__(self, spiking_neuron: callable = None, **kwargs):
        super(DVSGestureNetChrisModeled, self).__init__()
        
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, 1, kernel_size=5, stride=2, padding=0, bias=False), #30x30 feature map
            spiking_neuron(**deepcopy(kwargs)),
            layer.Flatten(),
            layer.Linear(900, 120, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(120, 84, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(84, 11, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
        )


    def forward(self, x):
        return self.conv_fc(x)        
    




def main():

    # Train
    # python cnn_train.py -data-dir /Users/keli/Code/CRI/data/DVS128Gesture -out-dir /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture


    # Print all arguments individually
    print("=" * 50)
    print("ARGUMENTS:")
    print("=" * 50)
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("=" * 50)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    scaler = amp.GradScaler()

    # resize transform that iterates over the temporal dimension, binarizes
    class DVSResizeAndBinarize:
        def __init__(self, size):
            self.size = size
            
        def __call__(self, data):
            # case where data is a tuple (frames, label)
            if isinstance(data, tuple):
                frames, label = data
                
                # Convert numpy array to tensor if needed
                if isinstance(frames, np.ndarray):
                    frames = torch.from_numpy(frames)
                
                # Get dimensions
                T, C, H, W = frames.shape
                
                # Create a tensor to hold resized frames
                resized = torch.zeros((T, C, self.size[0], self.size[1]), dtype=frames.dtype, device=frames.device)
                
                # Iterate over the temporal dimension and resize each frame
                for t in range(T):
                    frame = frames[t]  # Shape: [C, H, W]
                    # Use F.interpolate to resize
                    resized_frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),  # Add batch dimension
                        size=self.size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dimension
                    #binarize resized frame
                    binarized_frame = (resized_frame > 0).float()
                    resized[t] = binarized_frame



                return resized, label
            else:
                # Handle case where only frames are provided
                frames = data
                if isinstance(frames, np.ndarray):
                    frames = torch.from_numpy(frames)
                
                T, C, H, W = frames.shape
                resized = torch.zeros((T, C, self.size[0], self.size[1]), dtype=frames.dtype, device=frames.device)
                
                for t in range(T):
                    frame = frames[t]
                    resized_frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),
                        size=self.size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    binarized_frame = (resized_frame > 0).float()
                    resized[t] = binarized_frame
                    
                return resized

    
    # Use our simple resize transform for all datasets
    resize_transform = DVSResizeAndBinarize(size=(63, 63))  # resize from 128x128 to 63x63

    # Load training dataset
    full_train_set = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=True, 
        data_type="frame", 
        duration=1600000,
        transform=resize_transform
    )


    # Create 85%-15% train-validation split
    full_train_size = len(full_train_set)
    val_size = int(0.15 * full_train_size)
    train_size = full_train_size - val_size
    
    torch.manual_seed(1)  # ensure same split every time
    indices = torch.randperm(full_train_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create training dataset with train augments if wanted
    train_set_aug = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=True, 
        data_type="frame", 
        duration=1600000,
        transform=resize_transform
    )
    
    # Create subsets
    train_set = Subset(train_set_aug, train_indices)
    val_set = Subset(full_train_set, val_indices)  # No augmentation
    
    test_set = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=False, 
        data_type="frame", 
        duration=1600000,
        transform=resize_transform
    )
    
    print(f"Training samples: {len(train_set)} ({len(train_set)/full_train_size*100:.1f}%)")
    print(f"Validation samples: {len(val_set)} ({len(val_set)/full_train_size*100:.1f}%)")
    print(f"Test samples: {len(test_set)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )
    T, C, H, W = full_train_set[0][0].shape
    print(f"Input shape: {(T, C, H, W)}")
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")
    print(f"Number of testing samples: {len(test_set)}")
    
    channels = args.channels
    # Initialize SnnTorch/SpikingJelly model
    # encoder: number of conv blocks
    # net = models_krish.DVSGestureNet4(
    #     channels=channels,
    #     encoder=4,
    #     spiking_neuron=neuron.IFNode,
    #     surrogate_function=surrogate.ATan(),
    #     input_shape=(T, C, H, W),  # input shape for the model
    #     detach_reset=True,
    # )

    net = DVSGestureNetChrisModeled(
        spiking_neuron=Custom_LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        tau=63.0,
        decay_input=False
    )
      
    #net = PythonNet(channels=channels)

    # spiking_neuron = neuron.LIFNode
    # v_threshold = 1.0 #needs to be float
    # net = models_krish.DVSGestureNet7(
    #     channels=channels,
    #     encoder=5,
    #     spiking_neuron=Custom_LIFNode,
    #     surrogate_function=surrogate.ATan(),
    #     detach_reset=True,
    #     input_shape=(args.b, C, H, W),  # input shape for the model
    #     v_threshold=v_threshold,
    #     #tau=2.0, #default tau is 2.0 for LIF
    #     store_v_seq=True,
    # )


    print(net)


    net.to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    

    
    print("Start Training")

    #config = {}
    #config["neuron_type"] = "LI&F"
    #config["global_neuron_params"] = {}
    #config["global_neuron_params"]["v_thr"] = 2**19


    # increase the magnitude of all Linear weights before training
    x = 1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.mul_(x)

    print("Increased Linear weights magnitude by " + str(x) + "x before training")


    #net.train()

    # train
    val_acc, min_val_loss, best_epoch = train_DVS_Time_with_plot(args, net, train_loader, val_loader, device, scaler, save_every=25)
    
    print("Training Finished")
    print("Best epoch: ", best_epoch+1)
    print("Minimum validation loss at best epoch: ", min_val_loss)
    print("Validation accuracy at best epoch: ", val_acc)

    # load the best model after training
    checkpoint_path = os.path.join(output_dir, f"checkpoint_max_T_{T}_C_{channels}_lr_{args.lr}.pth")
    #checkpoint_path = "/home/k7arora/output/data_10T/spikingjellytutorial/val_split/checkpoint_max_T_10_C_128_lr_0.001.pth"
    #checkpoint_path = "/home/k7arora/output/data_10T/spikingjellytutorial/val_split/avg_pool/checkpoint_max_T_10_C_128_lr_0.001.pth"
    print("checkpoint_path: ", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(device),
    )
    net.load_state_dict(checkpoint["net"])
    
    net.eval()

    #normally remove
    print("using val set for all test accuracies")
    test_loader = val_loader

    # Test original model accuracy
    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)
    original_accuracy, original_loss = test_DVS_Time(args, net, test_loader, device, scaler)

    net_bn = net #skipping BN in this case
    print("No batch normalization")

    # Quantization with dynamic alpha and optional membrane potential quantization
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    qn = Quantize_Network(w_alpha=1, dynamic_alpha=False) #tau=2.0 by default


    # Print quantization parameters
    print("QUANTIZATION PARAMETERS:")
    print("="*50)
    print(f"w_alpha: {qn.w_alpha}")
    print(f"dynamic_alpha: {qn.dynamic_alpha}")
    print(f"w_bits: {qn.w_bits}")
    print(f"w_delta: {qn.w_delta}")
    print("="*50)

    net_quan = qn.quantize(net_bn)

    #convert FP32 weights to INT16
    #int16_sd, scales = fp32_to_int16_state_dict(model)
    #int16_sd is now the state dict of the quantized model
    int16_sd = net_quan.state_dict()
    print(f"int16_sd keys: {int16_sd.keys()}")

    #CHANGE HERE FOR DIFFERENT MODELS
    conv1_weight = int16_sd["conv_fc.0.weight"]
    conv2_weight = int16_sd["conv_fc.2.weight"]
    fc1_weight = int16_sd["conv_fc.5.weight"]
    fc2_weight = int16_sd["conv_fc.7.weight"]
    fc3_weight = int16_sd["conv_fc.9.weight"]

    # EXPERIMENTING, TEST: Force type cast weights to int16(right now in fp)
    print("original weight types: ", conv1_weight.dtype, conv2_weight.dtype, fc1_weight.dtype, fc2_weight.dtype, fc3_weight.dtype)
    int16_sd["conv_fc.0.weight"] = conv1_weight.to(torch.int16)
    int16_sd["conv_fc.2.weight"] = conv2_weight.to(torch.int16)
    int16_sd["conv_fc.5.weight"] = fc1_weight.to(torch.int16)
    int16_sd["conv_fc.7.weight"] = fc2_weight.to(torch.int16)
    int16_sd["conv_fc.9.weight"] = fc3_weight.to(torch.int16)

    conv1_weight = int16_sd["conv_fc.0.weight"]
    conv2_weight = int16_sd["conv_fc.2.weight"]
    fc1_weight = int16_sd["conv_fc.5.weight"]
    fc2_weight = int16_sd["conv_fc.7.weight"]
    fc3_weight = int16_sd["conv_fc.9.weight"]

    print("new weight types: ", conv1_weight.dtype, int16_sd["conv_fc.2.weight"].dtype, int16_sd["conv_fc.5.weight"].dtype, int16_sd["conv_fc.7.weight"].dtype, int16_sd["conv_fc.9.weight"].dtype)


    net_quan.eval()
    # Test quantized model accuracy
    print("\n" + "="*50)
    print("TESTING QUANTIZED MODEL ACCURACY")
    print("="*50)
    quantized_accuracy, _ = test_DVS_Time(net_quan, test_loader, device)

    # Calculate and display accuracy drop
    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"\n{'='*50}")
    print(f"QUANTIZATION RESULTS:")
    print(f"Original Accuracy:   {original_accuracy:.4f}")
    print(f"Quantized Accuracy:  {quantized_accuracy:.4f}")
    print(f"Accuracy Drop:       {accuracy_drop:.4f} ({accuracy_drop/original_accuracy*100:.2f}%)")
    print(f"{'='*50}")


    
if __name__ == "__main__":
    main()
