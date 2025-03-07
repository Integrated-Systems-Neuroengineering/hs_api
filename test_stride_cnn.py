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
parser.add_argument("-s", default=1, type=int, help="stride size")
parser.add_argument("-k", default=3, type=int, help="kernel size")
parser.add_argument("-p", default=0, type=int, help="padding size")
parser.add_argument("-c", default=16, type=int, help="channel size")
parser.add_argument(
    "-alpha", default=4, type=int, help="Range of value for quantization"
)
parser.add_argument("-b", default=1, type=int, help="batch size")
parser.add_argument("-T", default=16, type=int)
parser.add_argument(
    "-resume_path",
    default="/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/dvs_gesture/checkpoint_max_T_16_C_20_lr_0.001.pth",
    type=str,
    help="checkpoint file",
)
parser.add_argument(
    "-data-dir",
    default="/Volumes/export/isn/keli/code/data/DVS128Gesture",
    type=str,
    help="path to dataset",
)
parser.add_argument("-targets", default=11, type=int, help="Number of labels")


def plot_2d_heatmap(
    array: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    int_x_ticks=True,
    int_y_ticks=True,
    plot_colorbar=True,
    colorbar_y_label="magnitude",
    x_max=None,
    figsize=(12, 8),
    dpi=200,
):
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, heatmap = plt.subplots(figsize=figsize, dpi=dpi)
    if x_max is not None:
        im = heatmap.imshow(
            array.T,
            aspect="auto",
            extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5],
            vmin=-100000,
            vmax=30000,
        )
    else:
        im = heatmap.imshow(array.T, aspect="auto", vmin=-100000, vmax=30000)

    heatmap.set_title(title)
    heatmap.set_xlabel(xlabel)
    heatmap.set_ylabel(ylabel)

    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va="top")
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return fig


class DVSGestureNet(nn.Module):
    def __init__(
        self, channels=16, encoder=4, spiking_neuron: callable = None, *args, **kwargs
    ):
        super().__init__()

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(
                layer.Conv2d(
                    in_channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
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
            spiking_neuron(*args, **kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

    def encode(self, x: torch.Tensor):
        x = self.conv_fc[0](x)
        x = self.conv_fc[1](x)
        x = self.conv_fc[2](x)
        return x


def main():
    # python test_stride_cnn.py -resume_path /Users/keli/Code/CRI/CRI_Mapping/runs/dvs_gesture/checkpoint_max.pth -data-dir /Users/keli/Code/CRI/data/DVS128Gesture
    args = parser.parse_args()
    print(args)

    # Prepare the dataset
    # DVS128
    test_set = DVS128Gesture(
        root=args.data_dir,
        train=False,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
    )

    # Create DataLoaders
    test_loader = DataLoader(
        test_set, batch_size=args.b, shuffle=True, drop_last=True, pin_memory=True
    )

    net = DVSGestureNet(
        channels=args.c,
        spiking_neuron=neuron.LIFNode,
        decay_input=False,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    encoder = DVSGestureNet(
        channels=args.c,
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    device = torch.device("cpu")
    print(net)

    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint["net"])

    net.eval()
    encoder.eval()

    bn = BN_Folder()
    net_bn = bn.fold(net)
    encoder_bn = bn.fold(encoder)

    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)
    encoder_quan = qn.quantize(encoder_bn)

    functional.set_step_mode(net_quan, "m")

    # Set the parameters for conversion
    input_layer = 3  # first pytorch layer that acts as synapses, indexing begins at 0
    output_layer = 21  # last pytorch layer that acts as synapses
    snn_layers = 9  # number of snn layers
    input_shape = (16, 64, 64)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(
        num_steps=args.T,
        input_layer=input_layer,
        output_layer=output_layer,
        input_shape=input_shape,
        snn_layers=snn_layers,
        v_threshold=v_threshold,
        embed_dim=0,
        dvs=True,
    )

    cn.layer_converter(net_quan)

    breakpoint()

    config = {}
    config["neuron_type"] = "I&F"
    config["global_neuron_params"] = {}
    config["global_neuron_params"]["v_thr"] = int(qn.v_threshold)

    softwareNetwork = CRI_network(
        dict(cn.axon_dict),
        connections=dict(cn.neuron_dict),
        config=config,
        target="simpleSim",
        outputs=cn.output_neurons,
        simDump=False,
        coreID=1,
        perturbMag=None,  # Zero randomness
        leak=1,
    )  # tau = 2

    transform = transforms.Compose(
        [transforms.Resize([64, 64], transforms.InterpolationMode.NEAREST)]
    )

    start_time = time.time()

    test_loss_cri = 0
    test_acc_cri = 0
    test_samples = 0

    test_loss_torch = 0
    test_acc_torch = 0

    layer_number = 7

    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_loader)):
            imgs, label = data
            imgs = imgs.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            imgs = torch.cat(
                [transform(f).unsqueeze(0) for f in imgs]
            )  # reducing the size to 2*64*64
            label_onehot = F.one_hot(label, args.targets).float()

            # cri_v_list, tor_v_list = [], []
            cri_s_list, tor_s_list = [], []

            out_tor = 0.0
            out_cri = 0.0

            # Run the quantized pytorch model
            out_tor = net_quan(imgs).mean(0)
            loss = F.mse_loss(out_tor, label_onehot)
            test_loss_torch += loss.item() * label.numel()
            test_acc_torch += (out_tor.argmax(1) == label).float().sum().item()
            test_samples += label.numel()
            tor_s_list.append(out_tor.flatten().unsqueeze(0))

            # # Remove first layer's potential
            # tor_v_list.append(torch.cat((net_quan.conv_fc[6].v.flatten().unsqueeze(0),
            #                              net_quan.conv_fc[10].v.flatten().unsqueeze(0),
            #                              net_quan.conv_fc[14].v.flatten().unsqueeze(0),
            #                              net_quan.conv_fc[19].v.flatten().unsqueeze(0),
            #                              net_quan.conv_fc[22].v.flatten().unsqueeze(0)), 1))

            # Run the converted cri model
            for t, img in enumerate(imgs):
                # Encode the image with the first layer of cnn
                encoded_img = encoder_quan.encode(img)
                breakpoint()
                # Convert the input into axons
                cri_input = cn._input_converter_step(encoded_img, t)
                # Run a single time step
                swOutput, swSpike = softwareNetwork.step(
                    cri_input[0], membranePotential=True
                )
                spikeIdx = [int(spike) - int(cn.output_neurons[0]) for spike in swSpike]

                # # Record the membrane potential
                # cri_v_list.append(torch.tensor([v for k,v in swOutput]).unsqueeze(0))

                # Record the spikes
                if t > layer_number - 1:
                    cri_spikes = torch.zeros(out_tor.shape).flatten()
                    cri_spikes[spikeIdx] = 1
                    cri_s_list.append(cri_spikes.unsqueeze(0))
                    out_cri += cri_spikes

            # empty input for phase delay
            swOutput, swSpike = softwareNetwork.step([], membranePotential=True)
            spikeIdx = [int(spike) - int(cn.output_neurons[0]) for spike in swSpike]
            cri_spikes = torch.zeros(out_tor.shape).flatten()
            cri_spikes[spikeIdx] = 1
            cri_s_list.append(cri_spikes.unsqueeze(0))
            out_cri += cri_spikes

            # layer_number of empty inputs for layer delay
            for i in range(layer_number - 1):
                swOutput, swSpike = softwareNetwork.step([], membranePotential=True)
                spikeIdx = [int(spike) - int(cn.output_neurons[0]) for spike in swSpike]
                cri_spikes = torch.zeros(out_tor.shape).flatten()
                cri_spikes[spikeIdx] = 1
                cri_s_list.append(cri_spikes.unsqueeze(0))
                out_cri += cri_spikes

            # list -> tensors
            # tor_v_list = torch.cat(tor_v_list)
            # cri_v_list = torch.cat(cri_v_list)
            cri_s_list = torch.cat(cri_s_list)
            tor_s_list = torch.cat(tor_s_list)

            # # Set the membrane potential >= the threshold to zero for comparision
            # cri_v_list[cri_v_list >= cn.v_threshold] = 0

            # Calculate the loss and accuracy
            loss = F.mse_loss(out_cri, label_onehot)
            test_loss_cri += loss.item() * label.numel()
            test_acc_cri += (out_cri.argmax(1) == label).float().sum().item()

            # Reset the networks
            softwareNetwork.simpleSim.initialize_sim_vars(len(cn.neuron_dict))
            functional.reset_net(net_quan)

            # Plotting
            figsize = (12, 8)
            dpi = 100
            num_matches = (tor_s_list == cri_s_list).sum()
            total = tor_s_list.numel()
            accuracy = num_matches / total * 100 if num_matches != 0 else 0
            print(f"Spikes {accuracy}% matches")

            # #compare the pytorch and software membrane potential
            # num_matches = (tor_v_list[:,168]==cri_v_list[:,168]).sum()
            # total = tor_v_list[:,168].numel()
            # accuracy = num_matches/total * 100 if num_matches != 0 else 0
            # accuracy_sec = (tor_v_list[:15,169:]==cri_v_list[1:,169:]).sum()/cri_v_list[1:,169:].numel() * 100
            # print(f"First layer membrane potential {accuracy}% matches, Second layer {accuracy_sec}% matches")

            # compare the pytorch and software firing rate
            tor_r_list = torch.mean(tor_s_list.T, axis=1, keepdims=True)
            cri_r_list = torch.mean(cri_s_list.T, axis=1, keepdims=True)
            num_matches = (tor_r_list == cri_r_list).sum()
            total = cri_r_list.numel()
            accuracy = num_matches / total * 100 if num_matches != 0 else 0
            print(f"Firing rate {accuracy}% matches")

            # Plot the spikes
            visualizing.plot_1d_spikes(
                spikes=tor_s_list.numpy(),
                title="PyTorch Spikes",
                xlabel="simulating step",
                ylabel="neuron index",
                figsize=figsize,
                dpi=dpi,
            )
            plt.savefig(f"figure/PyTorch_S_{img_idx}.png")
            visualizing.plot_1d_spikes(
                spikes=cri_s_list.numpy(),
                title="CRI Spikes",
                xlabel="simulating step",
                ylabel="neuron index",
                figsize=figsize,
                dpi=dpi,
            )
            plt.savefig(f"figure/CRI_S_{img_idx}.png")

            breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss_cri /= test_samples
    test_acc_cri /= test_samples

    test_loss_torch /= test_samples
    test_acc_torch /= test_samples

    print(f"test_loss ={test_loss_cri: .4f}, test_acc ={test_acc_cri: .4f}")
    print(
        f"test_loss_torch ={test_loss_torch: .4f}, test_acc_torch ={test_acc_torch: .4f}"
    )
    print(f"test speed ={test_speed: .4f} images/s")


if __name__ == "__main__":
    main()
