import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import hs_bridge
import os

# --- Configuration ---
parser = argparse.ArgumentParser()
parser.add_argument('-T', default=4, type=int, help='Number of simulation time steps')
parser.add_argument('-figure-dir', default='./figure', type=str, help='Path to output figures')

def norm(x: torch.Tensor):
    """
    Standardizes input tensors for visualization by centering mean and scaling to unit variance.
    
    Args:
        x (torch.Tensor): The input tensor to normalize.
    Returns:
        torch.Tensor: Normalized tensor with the same shape as input.
    """
    s = x.shape
    x = x.flatten()
    std, mean = torch.std_mean(x)
    x -= mean
    x /= std
    return x.reshape(s)

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    """
    Generates a 2D heatmap to visualize membrane potential across time steps.
    
    Args:
        array (np.ndarray): 2D array (Time x Neuron Index) to plot.
        title (str): Plot title.
        xlabel (str): Label for X-axis.
        ylabel (str): Label for Y-axis.
        int_x_ticks (bool): Force X-axis ticks to be integers.
        x_max (int): Maximum value for the X-axis range.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, heatmap = plt.subplots(figsize=figsize, dpi=dpi)
    if x_max is not None:
        im = heatmap.imshow(array.T, aspect='auto', extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5])
    else:
        im = heatmap.imshow(array.T, aspect='auto')

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
    return fig

def main():
    """
    Direct Hardware-Software Parity Script.
    
    Purpose:
    Verifies bit-level accuracy between the CRI FPGA hardware (target='CRI') and 
    the simpleSim software simulator. It uses a manual network with two specific 
    neuron types to test different architectural behaviors:
    
    1. Integrate & Fire (N1): Holds potential indefinitely (leak=2^6-1).
    2. Memoryless (N2): Potential resets to zero every step (leak=0).
    
    The script compares raw spike output and firing rates to ensure the FPGA 
    registers are correctly configured.
    """
    args = parser.parse_args()
    
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    
    print(f"Starting Parity Test for T={args.T} steps...")
    
    # 1. Global Neuron Configuration
    config = {
        'neuron_type': "LI&F",
        'global_neuron_params': {'v_thr': 6}
    }
    
    # 2. Define Specific Neuron Behaviors
    # N1: High leak value (2**6-1) behaves like an Integrate-and-Fire (IF) node.
    # N2: Zero leak value (0) ensures no historical potential is kept between steps.
    N1 = LIF_neuron(6, -17, 2**6-1) 
    N2 = LIF_neuron(6, -17, 0)      
    
    # 3. Manual Network Mapping
    # Axons trigger neurons externally.
    # Neurons trigger each other internally (0 -> 1 excitatory, 1 -> 0 inhibitory).
    axonsDict = {
        'a0': [('0', 2), ('1', 1)],
        'a1': [('0', 4), ('1', 3)]
    }
    
    neuronsDict = {
        '0': ([('1', 1)], N1), 
        '1': ([('0', -1)], N2) 
    }
    
    # 4. Initialize Networks
    hardwareNetwork = CRI_network(axons=axonsDict,
                                  connections=neuronsDict,
                                  config=config, 
                                  target='CRI', 
                                  outputs=neuronsDict.keys(),
                                  simDump=False)
                                  
    softwareNetwork = CRI_network(axons=axonsDict,
                                  connections=neuronsDict,
                                  config=config, 
                                  outputs=neuronsDict.keys(), 
                                  target='simpleSim')

    sw_s_list, sw_v_list = [], []
    hw_s_list, hw_v_list = [], []
    
    input_lists = ['a0', 'a1', 'a0', 'a1']
    
    # 5. Execution Loop
    for t in range(args.T):
        current_input = input_lists
        
        # Step Software
        swOutput, swSpike = softwareNetwork.step(current_input, membranePotential=True)
        sw_v_list.append(torch.tensor([v for k, v in swOutput]).unsqueeze(0))
        sw_spikes = torch.zeros(len(neuronsDict))
        sw_spikes[[int(s) for s in swSpike]] = 1
        sw_s_list.append(sw_spikes.unsqueeze(0))
        
        # Step Hardware
        hwOutput, (hwSpike, _, _) = hardwareNetwork.step(current_input, membranePotential=True)
        hw_v_list.append(torch.tensor([v for k, v in hwOutput]).unsqueeze(0)) 
        hw_spikes = torch.zeros(len(neuronsDict))
        hw_spikes[[int(s) for s in hwSpike]] = 1
        hw_s_list.append(hw_spikes.unsqueeze(0))

    # 6. Cat results for analysis
    sw_s_final = torch.cat(sw_s_list)
    hw_s_final = torch.cat(hw_s_list)
    sw_v_final = torch.cat(sw_v_list)
    hw_v_final = torch.cat(hw_v_list)

    # 7. Visualization
    figsize, dpi = (12, 8), 100
    
    # SW Plots
    visualizing.plot_1d_spikes(spikes=sw_s_final.numpy(), title='Software Spikes', xlabel='Step', ylabel='Neuron', figsize=figsize, dpi=dpi)
    plt.savefig(os.path.join(args.figure_dir, "SW_S.png"))
    plot_2d_heatmap(array=sw_v_final.numpy(), title='Software Potentials', xlabel='Step', ylabel='Neuron', x_max=args.T, figsize=figsize, dpi=dpi)
    plt.savefig(os.path.join(args.figure_dir, "SW_V.png"))
    
    # HW Plots
    visualizing.plot_1d_spikes(spikes=hw_s_final.numpy(), title='Hardware Spikes', xlabel='Step', ylabel='Neuron', figsize=figsize, dpi=dpi)
    plt.savefig(os.path.join(args.figure_dir, "HW_S.png"))
    plot_2d_heatmap(array=hw_v_final.numpy(), title='Hardware Potentials', xlabel='Step', ylabel='Neuron', x_max=args.T, figsize=figsize, dpi=dpi)
    plt.savefig(os.path.join(args.figure_dir, "HW_V.png"))

    # 8. Parity Check
    accuracy = (sw_s_final == hw_s_final).sum() / sw_s_final.numel() * 100
    print(f"\nSpike Match: {accuracy.item():.2f}%")

    # 9. Cleanup
    softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))
    hs_bridge.FPGA_Execution.fpga_controller.clear(len(neuronsDict), False, 0)
    
if __name__ == '__main__':
    main()