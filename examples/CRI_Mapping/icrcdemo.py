import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based import surrogate, neuron, functional
from models import DVSGestureNet
from utils import train_DVS_Time, sw_comp_DVS, validate_DVS, test_DVS_Time
from hs_api import CRI_network
from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder

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


def main():

    # Train
    # python cnn_train.py -data-dir /Users/keli/Code/CRI/data/DVS128Gesture -out-dir /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    scaler = amp.GradScaler()

    # Prepare the dataset
    # DVS128
    train_set = DVS128Gesture(
        root=args.data_dir, train=True, data_type="frame", duration=1600000
    )
    test_set = DVS128Gesture(
        root=args.data_dir, train=False, data_type="frame", duration=1600000
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
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=pad_sequence_collate,
    )

    # Initialize SnnTorch/SpikingJelly model
    net = DVSGestureNet(
        channels=20,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    # load a training checkpoint
    checkpoint = torch.load(
        "/home/gwen/hs_api/examples/CRI_Mapping/output/dvs_gesture/checkpoint_max_T_10_C_20_lr_0.001.pth",
        weights_only=False,
        map_location=torch.device(device),
    )
    # breakpoint()
    net.load_state_dict(checkpoint["net"])
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

    # train_DVS_Time(args, net, train_loader, test_loader, device, scaler)

    bn = BN_Folder()
    net_bn = bn.fold(net)

    # quantization the weight
    qn = Quantize_Network(w_alpha=1)
    net_quan = qn.quantize(net_bn)

    test_DVS_Time(args, net, test_loader, device, scaler)

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

    # breakpoint()
    hardwareNetwork = CRI_network(
        axons=axons,
        connections=neurons,
        config=config,
        target="simpleSim",
        outputs=outputs,
    )

    sw_comp_DVS(
        args, hardwareNetwork, test_loader, device, net_quan, converter=converter
    )

    # breakpoint()
    validate_DVS(args, hardwareNetwork, test_loader, device, converter=converter)


if __name__ == "__main__":
    main()
