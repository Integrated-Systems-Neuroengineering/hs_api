import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hs_api.api import CRI_network
from hs_api.neuron_models import LIF_neuron
import time
from tqdm import tqdm
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import hs_bridge

# Configuration and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('-T', default=4, type=int, help='Number of simulation time steps')
parser.add_argument('-figure-dir', default='./figure', type=str, help='Path to output figures')

def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, figsize=(12, 8), dpi=200):
    """Generates a 2D heatmap to visualize membrane potential across time steps."""
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
    
    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
    return fig

def main():
    """
    Unit Parity Test: 
    Validates bit-level accuracy between CRI FPGA hardware and the simpleSim 
    software simulator using a hand-crafted recurrent spiking network.
    """
    args = parser.parse_args()
    
    # 1. Global Neuron Configuration
    # v_thr=6: Neurons fire when membrane potential reaches or exceeds 6.
    config = {
        'neuron_type': "I&F",
        'global_neuron_params': {'v_thr': 6}
    }
    
    # 2. Network Definition
    # Axon Dict: External stimuli (a0, a1) mapping to neurons with weights.
    axonsDict = {
        'a0': [('0', 2), ('1', 1)],
        'a1': [('0', 4), ('1', 3)]
    }
    
    # Neuron Dict: Internal/Recurrent connections.
    # Neuron 0 excites Neuron 1 (+1); Neuron 1 inhibits Neuron 0 (-1).
    neuronsDict = {
        '0': [('1', 1)],
        '1': [('0', -1)]
    }
    
    # 3. Initialize Targets
    # target='CRI' runs on physical FPGA; target='simpleSim' runs software simulation.
    # leak=2**6 (64) effectively disables leakage for pure I&F behavior.
    hardwareNetwork = CRI_network(axons=axonsDict,
                                  connections=neuronsDict,
                                  config=config, 
                                  target='CRI', 
                                  outputs=neuronsDict.keys(),
                                  simDump=True,
                                  coreID=1,
                                  perturbMag=0,
                                  leak=2**6)
                                  
    softwareNetwork = CRI_network(axons=axonsDict,
                                  connections=neuronsDict,
                                  config=config, 
                                  outputs=neuronsDict.keys(), 
                                  target='simpleSim',
                                  perturbMag=0,
                                  leak=2**6)

    sw_s_list, hw_s_list = [], []
    sw_v_list, hw_v_list = [], []
    
    # Cyclic input stream
    input_pattern = ['a0', 'a1', 'a0', 'a1']
    
    # 4. Execution Loop
    for t in range(args.T):
        current_input = input_pattern[t]
        
        # Software execution
        swOutput, swSpike = softwareNetwork.step(current_input, membranePotential=True)
        swSpikeIdx = [int(spike) for spike in swSpike]
        sw_v_list.append(torch.tensor([v for k, v in swOutput]).unsqueeze(0))

        # Hardware execution
        hwOutput, spikeResult = hardwareNetwork.step(current_input, membranePotential=True)
        hwSpike, latency, hbmAcc = spikeResult
        hwSpikeIdx = [int(spike) for spike in hwSpike]   
        hw_v_list.append(torch.tensor([v for k, v in hwOutput]).unsqueeze(0)) 
        
        # Binary Spike Recording
        sw_spk = torch.zeros(len(neuronsDict))
        sw_spk[swSpikeIdx] = 1
        sw_s_list.append(sw_spk.unsqueeze(0))
        
        hw_spk = torch.zeros(len(neuronsDict))
        hw_spk[hwSpikeIdx] = 1
        hw_s_list.append(hw_spk.unsqueeze(0))

    # 5. Parity Analysis
    sw_s_final = torch.cat(sw_s_list)
    hw_s_final = torch.cat(hw_s_list)
    sw_v_final = torch.cat(sw_v_list)
    hw_v_final = torch.cat(hw_v_list)
    
    match_acc = (sw_s_final == hw_s_final).sum() / sw_s_final.numel() * 100
    print(f"\n--- Parity Report ---")
    print(f"Spike Match Accuracy: {match_acc:.2f}%")

    # 6. Visualization
    figsize, dpi = (12, 8), 100
    
    # Software plots
    visualizing.plot_1d_spikes(spikes=sw_s_final.numpy(), title='Software Spikes', xlabel='Step', ylabel='Neuron', figsize=figsize, dpi=dpi)
    plt.savefig(f"{args.figure_dir}/SW_S.png")
    
    plot_2d_heatmap(array=sw_v_final.numpy(), title='Software Potentials', xlabel='Step', ylabel='Neuron', x_max=args.T, figsize=figsize, dpi=dpi)
    plt.savefig(f"{args.figure_dir}/SW_V.png")
    
    # Hardware plots
    visualizing.plot_1d_spikes(spikes=hw_s_final.numpy(), title='Hardware Spikes', xlabel='Step', ylabel='Neuron', figsize=figsize, dpi=dpi)
    plt.savefig(f"{args.figure_dir}/HW_S.png")
    
    plot_2d_heatmap(array=hw_v_final.numpy(), title='Hardware Potentials', xlabel='Step', ylabel='Neuron', x_max=args.T, figsize=figsize, dpi=dpi)
    plt.savefig(f"{args.figure_dir}/HW_V.png")

    # 7. Cleanup & Reset
    softwareNetwork.simpleSim.initialize_sim_vars(len(neuronsDict))
    hs_bridge.FPGA_Execution.fpga_controller.clear(len(neuronsDict), False, 0)
    
    # Flush hardware logs for inspection
    hardwareNetwork.sim_flush("test_syn_sim.txt")
    print(f"Hardware trace flushed to test_syn_sim.txt")

if __name__ == '__main__':
    main()