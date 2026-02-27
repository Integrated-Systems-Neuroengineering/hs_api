#to run model identical to spikingjelly tutorial: #PYTHONBREAKPOINT=0 nohup python -u /home/k7arora/hs_api/examples/CRI_Mapping/icrcdemo.py -b 16 -channels 128 -epochs 256 -out-dir /home/k7arora/output/data_10T/spikingjellytutorial/ > /home/k7arora/output/data_10T/spikingjellytutorial/log.txt 2>&1 &(but change path)


import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp
from spikingjelly.datasets import pad_sequence_collate  
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import surrogate, neuron, functional
from models import DVSGestureNet
import models_krish
from utils import train_DVS_Time, train_DVS_Time_with_plot, sw_comp_DVS, validate_DVS, validate_DVS_HW, test_DVS_Time
from hs_api import CRI_network
from hs_api.converters import CRI_Converter, Quantize_Network, BN_Folder
import os

parser = argparse.ArgumentParser()
parser.add_argument("-resume_path", default="", type=str, help="checkpoint file")
parser.add_argument("-load_path", default="", type=str, help="checkpoint loading path")
parser.add_argument(
    "-load_ssa_path", default="", type=str, help="ssa checkpoint loading path"
)
parser.add_argument(
    "-train", action="store_true", default=False, help="Train the network from stratch"
)
parser.add_argument("-b", default=32, type=int, help="batch size")
parser.add_argument(
    "-data-dir",
    default="/home/gwen/hs_api/examples/DVS128Gesture",
    type=str,
    help="path to dataset",
)
parser.add_argument(
    "-out-dir",
    default="/home/gwen/hs_api/examples/CRI_Mapping/output/dvs_gesture",
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
    default=4,
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

parser.add_argument("-dynamic-alpha", default="False", type=str, help="False, keli, or krish")

args = parser.parse_args()
output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dynamic_alpha = args.dynamic_alpha

batch_dir = os.path.join(output_dir, "test_batches")
os.makedirs(batch_dir, exist_ok=True)



def main():

    # Train
    # python cnn_train.py -data-dir /Users/keli/Code/CRI/data/DVS128Gesture -out-dir /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture

    args = parser.parse_args()
    
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

    # Prepare the dataset
    # DVS128
    # train_set = DVS128Gesture(
    #     root=args.data_dir, train=True, data_type="frame", duration=1600000
    # )
    # test_set = DVS128Gesture(
    #     root=args.data_dir, train=False, data_type="frame", duration=1600000
    # )

    # for T=10
    train_set = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=True, 
        data_type="frame", 
        duration=1600000
    )

    test_set = DVS128Gesture(
        root=args.data_dir, 
        frames_number=10, 
        split_by="number", 
        train=False, 
        data_type="frame", 
        duration=1600000
    )



    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )


    # for i, (img, label, x_len) in enumerate(test_loader):
    #     # Save each batch (img, label, x_len) to a file for future use
    #     torch.save({'img': img, 'label': label, 'x_len': x_len}, os.path.join(batch_dir, f"batch_{i}.pt"))
    #     print(img.shape)

    # batch = torch.load(os.path.join(batch_dir, "batch_2.pt"))
    # print(batch['img'].shape)
    # print("one timestep of first image in batch: ", batch['img'][0, 1, :, :, :])
    # dataset = TensorDataset(batch['img'], batch['label'], batch['x_len'])
    # test_loader = DataLoader(dataset, batch_size=args.b, shuffle=False)

    T, C, H, W = train_set[0][0].shape
    print(f"Input shape: {(T, C, H, W)}")
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of testing samples: {len(test_set)}")

    # Initialize SnnTorch/SpikingJelly model
    channels = args.channels

    net = DVSGestureNet(
        channels=channels,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

#    net = models_krish.DVSGestureNet(
#        channels=channels,
#        #encoder=3,
#        spiking_neuron=neuron.IFNode,
#        surrogate_function=surrogate.ATan(),
#        detach_reset=True,
#    )
    print(net)

    # load a training checkpoint
    # checkpoint = torch.load(
    #     "/home/gwen/hs_api/examples/CRI_Mapping/output/dvs_gesture/checkpoint_max_T_10_C_20_lr_0.001.pth",
    #     weights_only=False,
    #     map_location=torch.device(device),
    # )
    # # breakpoint()
    # net.load_state_dict(checkpoint["net"])

    net.eval()
    net.to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    print("Start Training")
    converted_model_pth = "./sep25/"

    config = {}
    config["neuron_type"] = "LI&F"
    config["global_neuron_params"] = {}
    config["global_neuron_params"]["v_thr"] = 2**19

    # Train the model and get final accuracy
    #train_DVS_Time_with_plot(args, net, train_loader, test_loader, device, scaler)


    # load the best model after training
    checkpoint_path = output_dir + "/checkpoint_max_T_" + str(T) + "_C_" + str(channels) + "_lr_" + str(args.lr) + ".pth"
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(device),
    )
    net.load_state_dict(checkpoint["net"])
    net.eval()

    # Test original model accuracy
    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)
    original_accuracy, _ = test_DVS_Time(args, net, test_loader, device, scaler)

    bn = BN_Folder()
    net_bn = bn.fold(net)

    # quantization the weight
    # qn = Quantize_Network(w_alpha=1, dynamic_alpha=True) #tau=2.0 by default

    # Quantization with dynamic alpha and optional membrane potential quantization
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)
    if dynamic_alpha == "keli":
        qn = Quantize_Network(w_alpha=1, dynamic_alpha=True, dynamic_alpha_method = "keli")
    elif dynamic_alpha == "krish":
        qn = Quantize_Network(w_alpha=1, dynamic_alpha=True, dynamic_alpha_method = "krish")
    else:
        qn = Quantize_Network(w_alpha=1, dynamic_alpha=False)


    #qn = Quantize_Network(w_alpha=1, dynamic_alpha=False) #tau=2.0 by default
    #qn = Quantize_Network(w_alpha=1, dynamic_alpha=True) #tau=2.0 by default
    
    # Print quantization parameters
    print("QUANTIZATION PARAMETERS:")
    print("="*50)
    print(f"w_alpha: {qn.w_alpha}")
    print(f"dynamic_alpha: {qn.dynamic_alpha}")
    print(f"w_bits: {qn.w_bits}")
    print(f"w_delta: {qn.w_delta}")
    print("="*50)

    net_quan = qn.quantize(net_bn)


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

    converter = CRI_Converter(
        num_steps=10,
        input_layer=0,
        snn_layers=5,
        output_layer=14,
        v_threshold=1,
        input_shape=(2, 128, 128),
        backend="spikingjelly",
        embed_dim=0,
        dvs=True,
        converted_model_pth=converted_model_pth,
    )
    converter.layer_converter(net_quan)
    axons = dict(converter.axon_dict)
    neurons = dict(converter.neuron_dict)
    outputs = converter.output_neurons

    #breakpoint()
    hardwareNetwork = CRI_network(
        axons=axons,
        connections=neurons,
        target="CRI",
        outputs=outputs,
    )

    # sw_comp_DVS(
    #     args, hardwareNetwork, test_loader, device, net_quan, converter=converter
    # )

    # breakpoint()
    # validate_DVS(args, hardwareNetwork, test_loader, device, converter=converter)
    validate_DVS_HW(args, hardwareNetwork, test_loader, device, converter=converter)

    # print(f"number of params: {n_parameters}")
    # print(f"Number of axons: {len(axons)}")
    # print(f"Number of neurons: {len(neurons)}")
    # print(f"Number of outputs: {len(outputs)}")
    # print("spiking jelly testing accuracy: " + str(original_acc) + ", loss: " + str(original_loss))


if __name__ == "__main__":
    main()
