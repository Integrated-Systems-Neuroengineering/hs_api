#to activate venv: eval $(poetry env activate)
#to avoid breakpoints: PYTHONBREAKPOINT=0 python icrcdemo_krish.py
#to run in background and log output(make sure to change paths): PYTHONBREAKPOINT=0 nohup python icrcdemo_krish.py -out-dir /home/k7arora/output/data_10T/1 > /home/k7arora/output/data_10T/1/log.txt 2>&1 &(but change path)
#to run default spikingjelly model with val_split: PYTHONBREAKPOINT=0 nohup python -u /home/k7arora/hs_api/examples/CRI_Mapping/icrcdemo_krish.py -b 16 -channels 128 -epochs 256 -out-dir /home/k7arora/output/data_10T/spikingjellytutorial/val_split > /home/k7arora/output/data_10T/spikingjellytutorial/val_split/log.txt 2>&1 &(but change path)
#PYTHONBREAKPOINT=0 nohup python -u /home/k7arora/hs_api/examples/CRI_Mapping/icrcdemo_krish.py -b 16 -channels 16 -epochs 5 -out-dir /home/k7arora/output/data_10T/spikingjellytutorial/val_split/avg_pool/before_spiking_layer/16_channels/no_voting > /home/k7arora/output/data_10T/spikingjellytutorial/val_split/avg_pool/before_spiking_layer/16_channels/no_voting/log.txt 2>&1 &
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
from spikingjelly.activation_based import surrogate, neuron, functional, monitor, encoding
from models import DVSGestureNet
import models_krish
from utils import train_DVS_Time, train_DVS_Time_with_plot, sw_comp_DVS, validate_DVS, test_DVS_Time, infer_cri_params
from hs_api import CRI_network
from hs_api.converters import CRI_Converter, Quantize_Network, BN_Folder
import os
import matplotlib.pyplot as plt
import pickle
from hs_api.custom_neurons import Custom_LIFNode


class SimpleDVSAugmentation:
    """Simple and effective augmentation for DVS gesture data"""
    
    def __init__(self, flip_prob=0.3, event_drop_prob=0.05, rotation_prob=0.2, max_rotation=3, temporal_shift_prob=0.1):
        self.flip_prob = flip_prob
        self.event_drop_prob = event_drop_prob
        self.rotation_prob = rotation_prob
        self.max_rotation = max_rotation
        self.temporal_shift_prob = temporal_shift_prob
    
    def __call__(self, data):
        # Handle different data formats
        if isinstance(data, tuple):
            frames, label = data
        else:
            frames = data
            label = None
        
        # Convert numpy array to tensor if needed
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        
        T, C, H, W = frames.shape
        
        # Random horizontal flip (reduced probability)
        if random.random() < self.flip_prob:
            frames = torch.flip(frames, dims=[3])  # flip width dimension
        
        # Very light rotation (smaller angles and lower probability)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            # Simple rotation using affine transform
            cos_a = torch.cos(torch.tensor(angle * torch.pi / 180))
            sin_a = torch.sin(torch.tensor(angle * torch.pi / 180))
            rotation_matrix = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
            
            # Apply rotation to all frames at once
            frames_flat = frames.view(T*C, H, W).unsqueeze(1)  # (T*C, 1, H, W)
            grid = torch.nn.functional.affine_grid(
                rotation_matrix.unsqueeze(0).expand(T*C, -1, -1), 
                frames_flat.shape, 
                align_corners=False
            )
            frames_rotated = torch.nn.functional.grid_sample(
                frames_flat, grid, mode='bilinear', align_corners=False
            )
            frames = frames_rotated.squeeze(1).view(T, C, H, W)
        
        # Very light temporal jittering (reduced probability)
        if random.random() < self.temporal_shift_prob:
            shift = random.randint(-1, 1)  # Small shift only
            if shift != 0:
                frames = torch.roll(frames, shift, dims=0)
        
        # Very light event dropout (reduced probability and dropout rate)
        if random.random() < self.event_drop_prob:
            mask = torch.rand_like(frames) > 0.02  # drop only 2% of events
            frames = frames * mask
        
        # Return in the same format as input
        if label is not None:
            return frames, label
        else:
            return frames

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
    default="/home/k7arora/hs_api/examples/DVS128Gesture",
    type=str,
    help="path to dataset",
)
parser.add_argument(
    "-out-dir",
    default="/home/k7arora/output/misc",
    type=str,
    help="dir path that stores the trained model checkpoint",
)
parser.add_argument("-epochs", default=20, type=int)
parser.add_argument("-lr", default=1e-3, type=float)
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

def main():

    # Train
    # python cnn_train.py -data-dir /Users/keli/Code/CRI/data/DVS128Gesture -out-dir /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture

    encoder = encoding.PoissonEncoder()

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

    # Prepare the dataset with gradual augmentations
    # Create gradual augmentation for training
    train_augmentation = SimpleDVSAugmentation(
        flip_prob=0.3,          # Reduced from 0.5
        event_drop_prob=0.05,   # Reduced from 0.1
        rotation_prob=0.2,      # Reduced from 0.3
        max_rotation=3,         # Reduced from 5 degrees
        temporal_shift_prob=0.1 # Reduced from 0.2
    )
    print("Using gradual data augmentation: horizontal flip (30%) + light rotation (±3°, 20%) + temporal shift (10%) + event dropout (5%)")
    
    # Load training dataset with gradual augmentation
    full_train_set = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=True, 
        data_type="frame", 
        duration=1600000
    )
    
    # Create 85%-15% train-validation split
    full_train_size = len(full_train_set)
    val_size = int(0.15 * full_train_size)
    train_size = full_train_size - val_size
    
    torch.manual_seed(1)  # ensure same split every time
    indices = torch.randperm(full_train_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create training dataset with augmentation
    train_set_aug = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=True, 
        data_type="frame", 
        duration=1600000,
        #transform=train_augmentation
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
        duration=1600000
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

    # original
    spiking_neuron = neuron.LIFNode
    v_threshold = 1.0 #needs to be float
    net = models_krish.DVSGestureNetAvgPoolSimple(
        channels=channels,
        spiking_neuron=Custom_LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        v_threshold=v_threshold,
        #tau=2.0, #default tau is 2.0 for LIF
        store_v_seq=True,
    )
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



    #train
    net.train()
    #train_DVS_Time_with_plot(args, net, train_loader, val_loader, device, scaler, save_every=25)
    print("Training Finished")

    # load the best model after training
    checkpoint_path = output_dir + "/checkpoint_max_T_" + str(T) + "_C_" + str(channels) + "_lr_" + str(args.lr) + ".pth"
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

    bn = BN_Folder()
    net_bn = bn.fold(net)

    #quantization the weight
    qn = Quantize_Network(w_alpha=1, dynamic_alpha=True) #tau=2.0 by default

    # Quantization with dynamic alpha and optional membrane potential quantization
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    qn = Quantize_Network(w_alpha=1, dynamic_alpha=True) #tau=2.0 by default
    
    # Print quantization parameters
    print("QUANTIZATION PARAMETERS:")
    print("="*50)
    print(f"w_alpha: {qn.w_alpha}")
    print(f"dynamic_alpha: {qn.dynamic_alpha}")
    print(f"w_bits: {qn.w_bits}")
    print(f"w_delta: {qn.w_delta}")
    print("="*50)

    net_quan = qn.quantize(net_bn)
    net_quan.eval()

    # Save quantized model
    torch.save({
        "net": net_quan.state_dict(),
        "args": args,
        "scaler": scaler.state_dict(),
    }, quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")


    # Load quantized model
    #net_quan = net_bn
    #net_quan.load_state_dict(torch.load(quantized_model_path, map_location=device, weights_only=False)["net"])

    net_quan.eval()
    # Test quantized model accuracy
    print("\n" + "="*50)
    print("TESTING QUANTIZED MODEL ACCURACY")
    print("="*50)
    quantized_accuracy, _ = test_DVS_Time(args, net_quan, test_loader, device, scaler)
    
    # Calculate and display accuracy drop
    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"\n{'='*50}")
    print(f"QUANTIZATION RESULTS:")
    print(f"Original Accuracy:   {original_accuracy:.4f}")
    print(f"Quantized Accuracy:  {quantized_accuracy:.4f}")
    print(f"Accuracy Drop:       {accuracy_drop:.4f} ({accuracy_drop/original_accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # breakpoint()


    converted_model_pth = output_dir + "/converted_model.pth"
    if not os.path.exists(converted_model_pth):
        os.makedirs(converted_model_pth)

    # infer converter parameters
    input_layer, snn_layers, output_layer = infer_cri_params(net_quan, synaptic_types=(nn.Conv2d, nn.Linear))
    print(f"Input layer: {input_layer} , Number of SNN layers: {snn_layers}, Output layer: {output_layer}")
    print(f"num steps: {T}, v_threshold: {v_threshold}, input_shape: {(C, H, W)}")

    neuron_types = (neuron.IFNode, neuron.LIFNode, Custom_LIFNode)

    # --- Directly log membrane potentials for 9 selected neurons ---
    print("Collecting and plotting membrane potentials for quantized model by direct v logging...")
    # Find the three spiking neuron layers (conv, fc, output)
    spiking_layers = []
    thresholds = []
    found_conv = False
    found_linear = False
    modules_list = list(net_quan.modules())
    # Find first conv's spiking neuron
    for m in modules_list:
        if not found_conv and isinstance(m, nn.Conv2d):
            found_conv = True
        elif found_conv and isinstance(m, neuron_types):
            spiking_layers.append(m)
            thresholds.append(m.v_threshold)
            break
    # Find first linear's spiking neuron
    for m in modules_list:
        if not found_linear and isinstance(m, nn.Linear):
            found_linear = True
        elif found_linear and isinstance(m, neuron_types):
            spiking_layers.append(m)
            thresholds.append(m.v_threshold)
            break
    # Find last linear's spiking neuron
    for m in reversed(modules_list):
        if isinstance(m, neuron_types):
            spiking_layers.append(m)
            thresholds.append(m.v_threshold)
            break

    #print info about one snn layer
    print("Info for first snn layer: ", spiking_layers[0])
    print("Input Decay: ", spiking_layers[0].decay_input)

    def get_voltage_trace():
        # Conv layer: v shape [B, C, H, W]
        v_conv = spiking_layers[0].v
        if not hasattr(v_conv, "shape"):
            for idx in range(3):
                v_traces['conv'][idx].append(0.0)
        else:
            # Flatten all spatial and channel dims for neuron indexing
            flat = v_conv.flatten()
            n = flat.shape[0]
            conv_indices = [0, n // 2, n - 1]
            #conv_indices = [20480, 33851, 57343]
            for idx, i in enumerate(conv_indices):
                try:
                    v_traces['conv'][idx].append(flat[i].detach().cpu().item())
                except Exception:
                    v_traces['conv'][idx].append(0.0)
        # FC layer: v shape [B, N]
        v_fc = spiking_layers[1].v
        if not hasattr(v_fc, "shape"):
            for idx in range(3):
                v_traces['fc'][idx].append(0.0)
        else:
            N_fc = v_fc.shape[1]
            fc_indices = [0, N_fc // 2, N_fc - 1]
            for idx, i in enumerate(fc_indices):
                try:
                    v_traces['fc'][idx].append(v_fc[0, i].detach().cpu().item())
                except Exception:
                    v_traces['fc'][idx].append(0.0)
        # Output layer: v shape [B, N]
        v_out = spiking_layers[2].v
        if not hasattr(v_out, "shape"):
            for idx in range(3):
                v_traces['output'][idx].append(0.0)
        else:
            N_out = v_out.shape[1]
            out_indices = [0, N_out // 2, N_out - 1]
            for idx, i in enumerate(out_indices):
                try:
                    v_traces['output'][idx].append(v_out[0, i].detach().cpu().item())
                except Exception:
                    v_traces['output'][idx].append(0.0)


    print("Selected spiking layers for v logging:", spiking_layers)

    net_quan.eval()
    v_traces = { 'conv': [ [] for _ in range(3) ], 'fc': [ [] for _ in range(3) ], 'output': [ [] for _ in range(3) ] }
    with torch.no_grad():
        for img, label, _ in val_loader:
            img = img.to(device)
            B, T, C, H, W = img.shape
            # For each image in the batch
            for b in range(B):
                img0 = img[b]  # shape: [T, C, H, W]
                functional.reset_net(net_quan)
                get_voltage_trace() # Collect v traces at beginning
                for t in range(T):     
                    img_t = img0[t].unsqueeze(0) # shape: [C, H, W]
                    encoded_img = encoder(img_t)
                    net_quan(img_t)
                    get_voltage_trace()  # Collect v traces for this image

            # Only process one batch for plotting
            break


    # Save membrane potentials to membrane_potentials directory
    membrane_log_dir = os.path.join(output_dir, "membrane_potentials")
    os.makedirs(membrane_log_dir, exist_ok=True)
    layer_names = ['conv', 'fc', 'output']
    for lname, layer in zip(layer_names, spiking_layers):
        v_threshold = getattr(layer, 'v_threshold', None)
        for layer_idx in range(3):
            idx = 0
            for i in range(3):
                fname = f"potentials_spikingjelly_{lname}0_neuron{i}.npy"
                np.save(os.path.join(membrane_log_dir, fname), np.array(v_traces[lname][idx]))
                print(f"Neuron {i}: {v_traces[lname][idx]}")
                idx += 1

    try:
        fname = f"thresholds.npy"
        np.save(os.path.join(membrane_log_dir, fname), np.array(thresholds))
    except Exception as e:
        print(f"[WARN] Could not save thresholds: {e}")


    converter = CRI_Converter(
        num_steps=T, #initially just 10
        input_layer=input_layer,
        snn_layers=snn_layers, 
        output_layer=output_layer,
        v_threshold=v_threshold,
        input_shape=(C, H, W), #(2, 128, 128)
        backend="spikingjelly",
        embed_dim=0,
        dvs=True,
        converted_model_pth=converted_model_pth,
        threshold_scale=1,  #adjusted threshold scale, testing by krish
        #spiking_neuron=neuron.LIFNode, 
        spiking_neuron=spiking_neuron,  #added by krish
    )


    #converter.layer_converter(net_quan, is_top_level=True)
    converter.layer_converter(net_quan)
    converter.save_model() #added by krish to save converted model to ouptut directory

    #load the converted model by loading pickle files inside folder with axons, neurons, and connections
    #converted_path = os.path.join(output_dir, "converted_model.pth")
    #axons = pickle.load(os.path.join(converted_path, "axon_dict.pkl"))
    #neurons = pickle.load(os.path.join(converted_path, "neuron_dict.pkl"))
    #outputs = pickle.load(os.path.join(converted_path, "output_neurons.pkl"))
    #print output neuron connectivity chain for debugging
    #converter.print_output_connectivity_chain()
    axons = dict(converter.axon_dict)
    neurons = dict(converter.neuron_dict)
    outputs = converter.output_neurons

    print(f"Number of axons: {len(axons)}")
    print(f"Number of neurons: {len(neurons)}")
    print(f"Number of outputs: {len(outputs)}")
    print(f"[DEBUG] output_neurons (final): {outputs}")

    # breakpoint()
    hardwareNetwork = CRI_network(
        axons=axons,
        connections=neurons,
        target="simpleSim", #for hardware, "CRI"
        outputs=outputs,
    )

    print("comp")

    

    print("Collecting membrane potentials for converted SNN...")
    print("1")
    for img, label, _ in val_loader:
        print("2")
        img = img.to(device)
        B, T, C, H, W = img.shape
        all_membrane_potentials = []
        for b in range(B):
            img0 = img[b]  # shape: [T, C, H, W]
            encoded_timesteps = []
            for t in range(T):
                encoded = encoder(img0[t])  # shape: [C, H, W]
                encoded_timesteps.append(encoded)
            encoded_img = torch.stack(encoded_timesteps, dim=0).unsqueeze(0)
            cri_input = converter.input_converter(encoded_img)
            indices = converter.layer_neuron_indices

            # neuron_dict keys are user keys (str), ensure idx is a Python int or str
            def to_str_key(idx):
                if hasattr(idx, 'item'):
                    return str(idx.item())
                elif hasattr(idx, 'astype'):
                    return str(int(idx))
                else:
                    return str(idx)
            indices = [to_str_key(idx) for idx in indices]
            # --- Directly log initial (pre-input) membrane potentials ---
            initial_membrane = []
            mp_arr = hardwareNetwork.simpleSim.membranePotentials()
            for key in indices:
                idx = hardwareNetwork.connectome.get_neuron_by_key(key).get_coreTypeIdx()
                try:
                    initial_membrane.append([mp_arr[idx]])  # shape [1]
                except Exception as e:
                    initial_membrane.append([float('nan')])
            # Now run the real input
            outputSpikes, membranePotential = converter.run_CRI_sw(
                cri_input, hardwareNetwork, outputPotential=True, potential_neuron_indices=indices
            )
            print("outputSpikes shape, " + str(len(outputSpikes)))
            print("outputSpikes", outputSpikes)
            # Prepend the initial membrane to the trace for each neuron
            # membranePotential: list of [neuron potentials at each timestep], shape [timesteps, num_neurons]
            # initial_membrane: list of [1] per neuron, so stack as first row
            membranePotential = [ [im[0] for im in initial_membrane] ] + list(membranePotential)
            all_membrane_potentials.append(np.array(membranePotential))
        # Concatenate along time axis: shape [B*T, num_neurons]
        all_membrane_potentials = np.concatenate(all_membrane_potentials, axis=0)
        # Save the first 3 conv, 3 fc, 3 output neuron traces
        conv_neurons = indices[:3]
        fc_neurons = indices[3:6]
        out_neurons = indices[6:9]
        for i in range(3):
            try:
                v_trace = all_membrane_potentials[:, i]
                fname = f"potentials_converted_conv0_neuron{i}.npy"
                np.save(os.path.join(membrane_log_dir, fname), v_trace)
            except Exception as e:
                print(f"[WARN] Could not save conv neuron {i}: {e}")
        for i in range(3):
            try:
                v_trace = all_membrane_potentials[:, i + 3]
                fname = f"potentials_converted_fc0_neuron{i}.npy"
                np.save(os.path.join(membrane_log_dir, fname), v_trace)
            except Exception as e:
                print(f"[WARN] Could not save fc neuron {i}: {e}")
        for i in range(3):
            try:
                v_trace = all_membrane_potentials[:, i + 6] 
                fname = f"potentials_converted_output0_neuron{i}.npy"
                np.save(os.path.join(membrane_log_dir, fname), v_trace)
            except Exception as e:
                print(f"[WARN] Could not save output neuron {i}: {e}")
        print("Saved converted SNN membrane potentials for selected neurons (full batch).")
        break



    sw_comp_DVS(
        args, hardwareNetwork, test_loader, device, net_quan, converter=converter
    )

    # print("validate")
    # # breakpoint()
    # print("validate 2")
    # #validate_DVS(args, hardwareNetwork, test_loader, device, converter=converter)
    # validate_DVS(args, hardwareNetwork, test_loader, device, converter=converter)

    # print(f"number of params: {n_parameters}")
    # print(f"Number of axons: {len(axons)}")
    # print(f"Number of neurons: {len(neurons)}")
    # print(f"Number of outputs: {len(outputs)}")
    # print("spiking jelly testing accuracy: " + str(original_accuracy) + ", loss: " + str(original_loss))


    
if __name__ == "__main__":
    main()
