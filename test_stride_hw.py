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
import hs_bridge

parser = argparse.ArgumentParser()
parser.add_argument("-s", default=1, type=int, help="stride size")
parser.add_argument("-k", default=3, type=int, help="kernel size")
parser.add_argument("-p", default=0, type=int, help="padding size")
parser.add_argument("-c", default=4, type=int, help="channel size")
parser.add_argument(
    "-alpha", default=4, type=int, help="Range of value for quantization"
)
parser.add_argument("-b", default=1, type=int, help="batch size")
parser.add_argument("-T", default=16, type=int)
parser.add_argument(
    "-resume_path",
    default="/Volumes/export/isn/keli/code/HS/CRI_Mapping/output/nmnist/checkpoint_max_T_16_C_20_lr_0.001_opt_adam.pth",
    type=str,
    help="checkpoint file",
)
parser.add_argument(
    "-data-dir",
    default="/Volumes/export/isn/keli/code/data/NMNIST",
    type=str,
    help="path to dataset",
)
parser.add_argument("-targets", default=10, type=int, help="Number of labels")


def norm(x: torch.Tensor):
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)


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


class Net(nn.Module):
    def __init__(
        self, in_channels=2, channels=8, spiking_neuron: callable = None, **kwargs
    ):
        super().__init__()

        self.conv = layer.Conv2d(
            in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False
        )
        self.bn = layer.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(16 * 16 * channels, 10)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x

    def forward_cnn(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif1(x)
        return x

    def forward_lr(self, x: torch.Tensor):
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif2(x)
        return x


def main():
    # python test_stride_hw.py -resume_path /Users/keli/Code/CRI/CRI_Mapping/runs/nmnist/checkpoint_latest_T_16_C_20_lr_0.001_opt_adam.pth -data-dir /Users/keli/Code/CRI/data/NMNIST
    args = parser.parse_args()
    print(args)

    # Prepare the dataset
    test_set = NMNIST(
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

    net = Net(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    net_cnn = Net(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(args.resume_path, map_location=device)
    net.load_state_dict(checkpoint["net"])

    net.eval()

    bn = BN_Folder()
    net_bn = bn.fold(net)

    qn = Quantize_Network(w_alpha=args.alpha)
    net_quan = qn.quantize(net_bn)

    # Set the parameters for conversion
    input_layer = 0  # first pytorch layer that acts as synapses, indexing begins at 0
    output_layer = 4  # last pytorch layer that acts as synapses
    input_shape = (2, 34, 34)
    v_threshold = qn.v_threshold

    cn = CRI_Converter(
        num_steps=args.T,
        input_layer=input_layer,
        output_layer=output_layer,
        input_shape=input_shape,
        v_threshold=v_threshold,
        embed_dim=0,
        dvs=True,
    )

    cn.layer_converter(net_quan)

    axons = [
        key if "256" in [k for k, v in cn.axon_dict[key]] else "*"
        for key in cn.axon_dict.keys()
    ]

    config = {}
    config["neuron_type"] = "I&F"
    config["global_neuron_params"] = {}
    config["global_neuron_params"]["v_thr"] = int(qn.v_threshold)

    hardwareNetwork = CRI_network(
        dict(cn.axon_dict),
        connections=dict(cn.neuron_dict),
        config=config,
        target="CRI",
        outputs=cn.output_neurons,
        simDump=False,
        coreID=1,
        perturbMag=0,  # Zero randomness
        leak=2**6 - 1,
    )  # IF

    encoder = encoding.PoissonEncoder()

    writer = SummaryWriter("log")

    start_time = time.time()
    hw_loss, hw_acc = 0.0, 0.0
    tor_loss, tor_acc = 0.0, 0.0
    test_samples = 0
    loss_fun = nn.MSELoss()

    with torch.no_grad():
        # Testing one image at a time
        for img_idx, data in enumerate(tqdm(test_loader)):

            img, label = data

            img = img.transpose(0, 1)  # [1, T, C, H, W] -> [T, 1, C, H, W]

            tor_v_list, hw_v_list = [], []
            tor_s_list, hw_s_list = [], []

            tor_out, hw_out = 0.0, 0.0

            for i, t in enumerate(img):

                # encode the image into spikes
                encoded_img = encoder(t)  # (B, C, H, W)

                # Running on PyTorch
                cnn_out = net_quan(encoded_img)
                tor_out += cnn_out

                # Record the spikes and membrane potential
                tor_s_list.append(cnn_out.flatten().unsqueeze(0))
                tor_v_list.append(
                    torch.cat(
                        (
                            net_quan.lif1.v.flatten().unsqueeze(0),
                            net_quan.lif2.v.flatten().unsqueeze(0),
                        ),
                        1,
                    )
                )

                # Conver the spikes into axon
                cri_input = cn._input_converter_step(encoded_img)

                # Running on Hardware
                # hwOutput: [(key, potential) for all the neurons in hardwareNetwork]
                hwOutput, spikeResult = hardwareNetwork.step(
                    cri_input[0], membranePotential=True
                )
                hwSpike, latency, hbmAcc = spikeResult
                spikeIdx = [int(spike) - int(cn.output_neurons[0]) for spike in hwSpike]
                hw_v_list.append(torch.tensor([v for k, v in hwOutput]).unsqueeze(0))

                if i != 0:
                    hw_spikes = torch.zeros(cnn_out.shape).flatten()
                    hw_spikes[spikeIdx] = 1
                    hw_s_list.append(hw_spikes.unsqueeze(0))
                    hw_out += hw_spikes.unsqueeze(0)

            # empty input for phase delay
            hwOutput, spikeResult = hardwareNetwork.step([], membranePotential=True)
            hwSpike, latency, hbmAcc = spikeResult
            spikeIdx = [int(spike) - int(cn.output_neurons[0]) for spike in hwSpike]
            hw_spikes = torch.zeros(cnn_out.shape).flatten()
            hw_spikes[spikeIdx] = 1
            hw_s_list.append(hw_spikes.unsqueeze(0))
            hw_out += hw_spikes.unsqueeze(0)

            # plot the membrane potential
            tor_v_list = torch.cat(tor_v_list)
            hw_v_list = torch.cat(hw_v_list)

            figsize = (12, 8)
            dpi = 100
            plot_2d_heatmap(
                array=tor_v_list.numpy(),
                title="PyTorch membrane potentials",
                xlabel="simulating step",
                ylabel="neuron index",
                int_x_ticks=True,
                x_max=args.T,
                figsize=figsize,
                dpi=dpi,
            )
            plt.savefig(f"figure/PyTorch_V_{img_idx}.png")

            plot_2d_heatmap(
                array=hw_v_list.numpy(),
                title="HW membrane potentials",
                xlabel="simulating step",
                ylabel="neuron index",
                int_x_ticks=True,
                x_max=args.T,
                figsize=figsize,
                dpi=dpi,
            )
            plt.savefig(f"figure/HW_V_{img_idx}.png")

            # plot the spikes
            hw_s_list = torch.cat(hw_s_list)
            tor_s_list = torch.cat(tor_s_list)

            # compare the pytorch and hw spike output
            num_matches = (tor_s_list == hw_s_list).sum()
            accuracy = (
                (tor_s_list == hw_s_list).sum() / tor_s_list.numel() * 100
                if num_matches != 0
                else 0
            )
            print(f"HW Spikes {accuracy}% matches")
            writer.add_scalar("hw_spike_match", accuracy, img_idx)

            # compare the pytorch and hw membrane potential
            num_matches = (tor_v_list == hw_v_list).sum()
            accuracy = (
                (tor_v_list == hw_v_list).sum() / tor_v_list.numel() * 100
                if num_matches != 0
                else 0
            )
            print(f"HW Membrane potential {accuracy}% matches")
            writer.add_scalar("hw_potential_match", accuracy, img_idx)

            # compare the pytorch and hardware firing rate
            tor_r_list = torch.mean(tor_s_list.T, axis=1, keepdims=True)
            hw_r_list = torch.mean(hw_s_list.T, axis=1, keepdims=True)

            num_matches = (tor_r_list == hw_r_list).sum()
            accuracy = (
                (tor_r_list == hw_r_list).sum() / tor_r_list.numel() * 100
                if num_matches != 0
                else 0
            )
            print(f"HW Firing rate {accuracy}% matches")
            writer.add_scalar("HW_firing_rate_match", accuracy, img_idx)

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
                spikes=hw_s_list.numpy(),
                title="HW Spikes",
                xlabel="simulating step",
                ylabel="neuron index",
                figsize=figsize,
                dpi=dpi,
            )
            plt.savefig(f"figure/HW_S_{img_idx}.png")

            hw_out /= args.T
            tor_out /= args.T

            label_onehot = F.one_hot(label, args.targets).float()
            test_samples += label.numel()

            loss = loss_fun(hw_out, label_onehot)
            hw_loss += loss.item() * label.numel()
            hw_acc += (hw_out.argmax(1) == label).float().sum().item()

            loss = loss_fun(tor_out, label_onehot)
            tor_loss += loss.item() * label.numel()
            tor_acc += (tor_out.argmax(1) == label).float().sum().item()

            # reset the membrane potential to zero
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(cn.neuron_dict), False, 0
            )
            functional.reset_net(net_quan)

            # reset pyplot interface
            plt.close()
            # breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    hw_loss /= test_samples
    hw_acc /= test_samples

    tor_loss /= test_samples
    tor_acc /= test_samples

    print(f"hw_loss ={hw_loss: .4f}, hw_acc ={hw_acc: .4f}")
    print(f"tor_loss ={tor_loss: .4f}, tor_acc ={tor_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")


if __name__ == "__main__":
    main()
