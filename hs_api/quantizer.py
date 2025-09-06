import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from snntorch import spikegen
from spikingjelly.activation_based import encoding
import csv
import time
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
import snntorch as snn
import multiprocessing as mp
import numpy as np
from hs_api.neuron_models import LIF_neuron, ANN_neuron
from hs_api.custom_neurons import Custom_LIFNode
from spikingjelly.activation_based import neuron, surrogate

def isSNNLayer(layer):
    """
    Checks if a layer is an instance of a Spiking Neural Network (SNN) layer.

    Parameters
    ----------
    layer : object
        The layer to check.

    Returns
    -------
    bool
        True if the layer is an instance of a SNN layer, False otherwise.

    Examples
    --------
    >>> from norse.torch.module.lif import LIFCell
    >>> layer = LIFCell()
    >>> isSNNLayer(layer)
    True
    """

    return (
        isinstance(layer, MultiStepLIFNode)
        or isinstance(layer, LIFNode)
        or isinstance(layer, IFNode)
        or isinstance(layer, Custom_LIFNode)
    )


def weight_quantization(b):
    """
    Applies weight quantization to the input.

    Parameters
    ----------
    b : int
        The number of bits to use for the quantization.

    Returns
    -------
    function
        A function that applies weight quantization to its input.

    Examples
    --------
    >>> weight_quantization_func = weight_quantization(8)
    >>> weight_quantization_func(some_input)
    """

    def uniform_quant(x, b):
        """
        Applies uniform quantization to the input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        b : int
            The number of bits to use for the quantization.

        Returns
        -------
        torch.Tensor
            The quantized tensor.

        Examples
        --------
        >>> x = torch.tensor([1.1, 2.2, 3.3])
        >>> uniform_quant(x, 2)
        tensor([1., 2., 3.])
        """
        xdiv = x.mul((2**b - 1))
        xhard = xdiv.round().div(2**b - 1)
        # print('uniform quant bit: ', b)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)  # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)  # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (
                input.abs() > 1.0
            ).float()  # >1 means clipped. # output matrix is a form of [True, False, True, ...]
            sign = input.sign()  # output matrix is a form of [+1, -1, -1, +1, ...]
            # grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            grad_alpha = (grad_output * (sign * i + (0.0) * (1 - i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, "grad_alpha = grad_output * (input_q-input)"
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, w_alpha):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit - 1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.wgt_alpha = w_alpha

    def forward(self, weight):
        weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q



class Quantize_Network:
    """
    A class to perform quantization on a neural network.

    Parameters
    ----------
    w_alpha : float
        The alpha value for the quantization. Default is 1.
    dynamic_alpha : bool, optional
        Whether to use dynamic alpha for quantization. Default is False.

    Attributes
    ----------
    w_alpha : float
        The alpha value for the quantization.
    dynamic_alpha : bool
        Whether to use dynamic alpha for quantization.
    v_threshold : float or None
        The threshold for the quantization. Default is None.
    w_bits : int
        The number of bits to use for the quantization.
    w_delta : float
        The delta value for the quantization.
    weight_quant : weight_quantize_fn
        The weight quantization function.

    Examples
    --------
    >>> q_net = Quantize_Network(w_alpha=1, dynamic_alpha=True)
    >>> q_net.quantize(some_model)
    """

    def __init__(self, w_alpha, dynamic_alpha=False):
        self.w_alpha = w_alpha  # Range of the parameter (CSNN:4, Spikeformer: 5)
        self.dynamic_alpha = dynamic_alpha
        self.v_threshold = None
        self.w_bits = 16
        self.w_delta = self.w_alpha / (2 ** (self.w_bits - 1) - 1)
        self.weight_quant = weight_quantize_fn(self.w_bits, self.w_alpha)

    def quantize(self, model):
        """
        Performs quantization on a model.

        Parameters
        ----------
        model : torch.nn.Module
            The input model.

        Returns
        -------
        torch.nn.Module
            The quantized model.

        Examples
        --------
        >>> q_net = Quantize_Network(w_alpha=1, dynamic_alpha=True)
        >>> q_net.quantize(some_model)
        """

        new_model = copy.deepcopy(model)
        start_time = time.time()
        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(
                new_model._modules[name]
            ):
                # print('Quantized: ',name)
                if name == "block":
                    new_model._modules[name] = self.quantize_block(
                        new_model._modules[name]
                    )
                else:
                    # if name == 'attn':
                    #     continue
                    new_model._modules[name] = self.quantize(new_model._modules[name])
            else:
                # print('Quantized: ',name)
                if name == "attn_lif":
                    continue
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer

        end_time = time.time()
        # print(f'Quantization time: {end_time - start_time}')
        return new_model

    def quantize_block(self, model):
        """
        Performs quantization on a block of a model.

        Parameters
        ----------
        model : torch.nn.Module
            The input model.

        Returns
        -------
        torch.nn.Module
            The quantized model.

        Examples
        --------
        >>> q_net = Quantize_Network(w_alpha=1, dynamic_alpha=True)
        >>> q_net.quantize_block(some_model)
        """
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(
                new_model._modules[name]
            ):
                if name.isnumeric() or name == "attn" or name == "mlp":
                    # print('Block Quantized: ',name)
                    new_model._modules[name] = self.quantize_block(
                        new_model._modules[name]
                    )
                # else:
                #     # print('Block Unquantized: ', name)
            else:
                if name == "attn_lif":
                    continue
                else:
                    new_model._modules[name] = self._quantize(new_model._modules[name])
        return new_model

    def _quantize(self, layer):
        """
        Helper function to performs quantization on a layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The input layer.

        Returns
        -------
        torch.nn.Module
            The quantized layer.

        Examples
        --------
        >>> q_net = Quantize_Network(w_alpha=1, dynamic_alpha=True)
        >>> q_net._quantize(some_layer)
        """

        if isSNNLayer(layer):
            return self._quantize_LIF(layer)

        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            return self._quantize_layer(layer)

        else:
            return layer

    def _quantize_layer(self, layer):
        quantized_layer = copy.deepcopy(layer)

        # calculate and print original weight statistics
        original_weights = layer.weight.flatten()
        print(f"\nLayer: {layer.__class__.__name__}")
        print(f"w_alpha: {self.w_alpha}")
        print(f"w_delta: {self.w_delta}")
        print(f"ORIGINAL WEIGHT STATISTICS:")
        print(f"  Max:    {torch.max(original_weights).item()}")
        print(f"  Min:    {torch.min(original_weights).item()}")
        print(f"  Mean:   {torch.mean(original_weights).item()}")
        print(f"  Median: {torch.median(original_weights).item()}")
        print(f"  Std:    {torch.std(original_weights).item()}")

        if self.dynamic_alpha:
            # weight_range = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            
            [-5, -2, 1, 3]
            # #default dynamic_alpha:
            # print("keli's dynamic alpha")
            # self.w_alpha = abs(
            #     max(layer.weight.flatten()) - min(layer.weight.flatten()) 
            # )

            #krish dynamic alpha
            print("krish's dynamic alpha: max(abs(layer.weight.flatten()))")
            self.w_alpha = max(abs(layer.weight.flatten()))

            # # krish mean std dynamic alpha (k=2)
            # k = 2
            # weights_flat = layer.weight.flatten()
            # mean = torch.mean(weights_flat).item()
            # std = torch.std(weights_flat).item()
            # self.w_alpha = max(abs(mean + k * std), abs(mean - k * std))
            # print(f"krish's mean-std dynamic alpha (k={k}): max(abs(mean ± k*std))")


            print(f"Dynamic w_alpha: {self.w_alpha}")
            self.w_delta = self.w_alpha / (2 ** (self.w_bits - 1) - 1)
            print(f"Dynamic w_delta: {self.w_delta}")
            self.weight_quant = weight_quantize_fn(
                self.w_bits,
                self.w_alpha
            )  # reinitialize the weight_quan
            self.weight_quant.wgt_alpha = self.w_alpha

        # store original weights for comparison
        original_weights_copy = original_weights.clone()
        
        layer.weight = nn.Parameter(self.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight / self.w_delta)
        #quantized_layer.weight = nn.Parameter(layer.weight) #krish: testing a change

        # calculate and print quantized weight statistics
        quantized_weights = quantized_layer.weight.flatten()
        print(f"QUANTIZED WEIGHT STATISTICS:")
        print(f"  Max:    {torch.max(quantized_weights).item()}")
        print(f"  Min:    {torch.min(quantized_weights).item()}")
        print(f"  Mean:   {torch.mean(quantized_weights).item()}")
        print(f"  Median: {torch.median(quantized_weights).item()}")
        print(f"  Std:    {torch.std(quantized_weights).item()}")
        
        # calculate change due to quantization
        weight_change = torch.abs(quantized_weights - original_weights_copy)
        print(f"QUANTIZATION IMPACT:")
        print(f"  Mean Abs Error: {torch.mean(weight_change).item()}")
        print(f"  Max Abs Error:  {torch.max(weight_change).item()}")
        if torch.mean(torch.abs(original_weights_copy)) > 0:
            print(f"  Relative Error: {(torch.mean(weight_change) / torch.mean(torch.abs(original_weights_copy))).item()}")
        print("-" * 60)

        if layer.bias is not None:  # check if the layer has bias
            layer.bias = nn.Parameter(self.weight_quant(layer.bias))
            quantized_layer.bias = nn.Parameter(layer.bias / self.w_delta)

        return quantized_layer

    def _quantize_LIF(self, layer):
        """
        Helper function to performs quantization on a LIF layer.

        Parameters
        ----------
        layer : torch.nn.Module
            The input layer.

        Returns
        -------
        torch.nn.Module
            The quantized layer.

        Examples
        --------
        >>> q_net = Quantize_Network(w_alpha=1, dynamic_alpha=True)
        >>> q_net._quantize_LIF(some_layer)
        """

        original_thresh = layer.v_threshold
        quantized_thresh = int(original_thresh / self.w_delta)
        print(f"\nTHRESHOLD QUANTIZATION:")
        print(f"  Original threshold: {original_thresh}")
        print(f"  Quantization step (w_delta): {self.w_delta}")
        print(f"  Quantized threshold: {quantized_thresh}")
        layer.v_threshold = quantized_thresh
        self.v_threshold = layer.v_threshold

        return layer




class BN_Folder:
    """
    A class to perform batch normalization folding on a model.

    Examples
    --------
    >>> bn_folder = BN_Folder()
    >>> bn_folder.fold(some_model)
    """

    def __init__(self):
        super().__init__()

    def fold(self, model):
        """
        Performs batch normalization folding on a model.

        Parameters
        ----------
        model : torch.nn.Module
            The input model.

        Returns
        -------
        torch.nn.Module
            The model with batch normalization folded.

        Examples
        --------
        >>> bn_folder = BN_Folder()
        >>> bn_folder.fold(some_model)
        """

        new_model = copy.deepcopy(model)

        module_names = list(new_model._modules)

        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0:
                new_model._modules[name] = self.fold(new_model._modules[name])

            else:
                if isinstance(new_model._modules[name], nn.BatchNorm2d) or isinstance(
                    new_model._modules[name], nn.BatchNorm1d
                ):
                    if isinstance(
                        new_model._modules[module_names[k - 1]], nn.Conv2d
                    ) or isinstance(new_model._modules[module_names[k - 1]], nn.Linear):
                        # Folded BN
                        folded_conv = self._fold_conv_bn_eval(
                            new_model._modules[module_names[k - 1]],
                            new_model._modules[name],
                        )

                        # Replace old weight values
                        # new_model._modules.pop(name) # Remove the BN layer
                        new_model._modules[module_names[k]] = nn.Identity()
                        new_model._modules[module_names[k - 1]] = (
                            folded_conv  # Replace the Convolutional Layer by the folded version
                        )

        return new_model

    def _bn_folding(self, prev_w, prev_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, model_2d):
        """
        Performs batch normalization folding on a layer.

        Parameters
        ----------
        prev_w : torch.nn.Parameter
            The weight parameter of the previous layer.
        prev_b : torch.nn.Parameter or None
            The bias parameter of the previous layer.
        bn_rm : torch.Tensor
            The running mean of the batch normalization layer.
        bn_rv : torch.Tensor
            The running variance of the batch normalization layer.
        bn_eps : float
            The epsilon value of the batch normalization layer.
        bn_w : torch.nn.Parameter
            The weight parameter of the batch normalization layer.
        bn_b : torch.nn.Parameter
            The bias parameter of the batch normalization layer.
        model_2d : bool
            Whether the model is 2D.

        Returns
        -------
        tuple of torch.nn.Parameter
            The folded weight and bias parameters.

        Examples
        --------
        >>> bn_folder = BN_Folder()
        >>> bn_folder._bn_folding(some_parameters)
        """

        if prev_b is None:
            prev_b = bn_rm.new_zeros(bn_rm.shape)

        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

        if model_2d:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
        else:
            w_fold = prev_w * (bn_w * bn_var_rsqrt).view(-1, 1)

        b_fold = (prev_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

        return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)

    def _fold_conv_bn_eval(self, prev, bn):
        """
        Performs batch normalization folding on a convolutional layer in evaluation mode.

        Parameters
        ----------
        prev : torch.nn.Module
            The previous layer.
        bn : torch.nn.Module
            The batch normalization layer.

        Returns
        -------
        torch.nn.Module
            The folded layer.

        Examples
        --------
        >>> bn_folder = BN_Folder()
        >>> bn_folder._fold_conv_bn_eval(some_prev_layer, some_bn_layer)
        """

        assert not (prev.training or bn.training), "Fusion only for eval!"
        fused_prev = copy.deepcopy(prev)

        # TODO: fix the bias = 0s when bias should be none

        if isinstance(bn, nn.BatchNorm2d):
            fused_prev.weight, fused_prev.bias = self._bn_folding(
                fused_prev.weight,
                fused_prev.bias,
                bn.running_mean,
                bn.running_var,
                bn.eps,
                bn.weight,
                bn.bias,
                True,
            )
        else:
            fused_prev.weight, fused_prev.bias = self._bn_folding(
                fused_prev.weight,
                fused_prev.bias,
                bn.running_mean,
                bn.running_var,
                bn.eps,
                bn.weight,
                bn.bias,
                False,
            )

        return fused_prev


class CRI_Converter:
    def print_output_connectivity_chain(self, max_depth=2):
        print("\n[DIAGNOSE] Output neuron connectivity chain:")
        for nidx in self.output_neurons:
            print(f"[DIAGNOSE] Output neuron {nidx}:")
            # Scan all presynaptic neurons for connections to this output neuron
            incoming = []
            for pre, entry in self.neuron_dict.items():
                conns = entry[0] if isinstance(entry, tuple) else entry
                for post, w in conns:
                    if str(post) == str(nidx):
                        incoming.append((pre, w))
            if not incoming:
                print("  [DIAGNOSE]   No input connections.")
                continue
            print(f"  [DIAGNOSE]   Connected to {len(incoming)} presynaptic neurons:")
            # for i, (pre, w) in enumerate(incoming[:10]):
            #     print(f"    [DIAGNOSE]   Input {i}: {pre} (w={w})")
        print("[DIAGNOSE] End of output neuron connectivity chain.\n")
    """
    A class to convert a neural network model into an equivalent model compatible
    with the CRI (Capacitive ReRAM Inverter) hardware.

    Parameters
    ----------
    num_steps : int
        The number of time steps in the input.
    input_layer : int
        The index of the first pytorch layer used as synapses.
    output_layer : int
        The index of the last pytorch layer used as synapses.
    input_shape : tuple of int
        The shape of the input data. Default is (1, 28, 28).
    backend : str, optional
        The backend to use. Currently Support SpikingJelly and snnTorch.
        Default is 'spikingjelly'.
    v_threshold : float
        The voltage threshold for the neurons.
        It should be set to the v_threshold of Quantize Network.
    embed_dim : int
        The embedding dimension. Only used for spikeformer.
    converted_model_pth: str, optional
        Save the converted network into a .pkl file at converted_model_pth.
        Default is "./converted_model"
    dvs: bool
        Convert the input data as DVS datasets

    Attributes
    ----------
    axon_dict : defaultdict of list
        A dictionary mapping each axon to a list of connected neurons.
    neuron_dict : defaultdict of list
        A dictionary mapping each neuron to a list of connected axons.
    output_neurons : list
        A list of output neurons.
    input_shape : np.ndarray
        The shape of the input data.
    num_steps : int
        The number of time steps in the input.
    axon_offset : int
        The current offset for axon indexing.
    neuron_offset : int
        The current offset for neuron indexing.
    backend : str
        The backend to use.
    bias_start_idx : int or None
        The starting index for bias neurons.
    curr_input : np.ndarray or None
        The current input data.
    input_layer : int
        The index of the input layer.
    output_layer : int
        The index of the output layer.
    snn_layers : int
        The number of SNN layers in the original model
    v_threshold : float
        The voltage threshold for the neurons.
    layer_index : int
        The current layer index.
    total_axonSyn : int
        The total number of axon synapses.
    total_neuronSyn : int
        The total number of neuron synapses.
    max_fan : int
        The maximum fan-out.
    bias_dict : list of tuples
        The start and end bias axon index in each snn layer
    snn_layer_index: int
        The current layer offset for snn layers indexing
    converted_model_pth : str
        The path to the converted network file.
    q : np.ndarray or None
        The q matrix for attention conversion.
    v : np.ndarray or None
        The v matrix for attention conversion.
    k : np.ndarray or None
        The k matrix for attention conversion.
    embed_dim : int
        The embedding dimension.
    dvs: bool
        Whether using dvs datasets.

    Examples
    --------
    >>> converter = CRI_Converter()
    >>> converter.layer_converter(some_model)
    >>> converter.input_converter(some_input_data)
    """

    def __init__(
        self,
        num_steps,
        input_layer,
        output_layer,
        snn_layers,
        input_shape,
        v_threshold,
        embed_dim,
        backend="spikingjelly",
        dvs=False,
        converted_model_pth="./converted_model",
        threshold_scale=1,
        spiking_neuron=None,
    ):
        self.HIGH_SYNAPSE_WEIGHT = 1e6
        self.NULL_NEURON = -1
        self.NULL_INDICIES = (-1, -1)
        self.PERTUBATION = 0
        self.LEAK_LIF = 2**6 - 1
        self.v_threshold = v_threshold
        self.threshold_scale = threshold_scale #krish
        print("threshhold_scale: " + str(self.threshold_scale))
        # neuron model parameters
        if spiking_neuron is None:
            from spikingjelly.activation_based import neuron
            spiking_neuron = neuron.LIFNode
        self.spiking_neuron = spiking_neuron
        # Set leaky based on neuron type: krish
        if spiking_neuron.__name__ == "LIFNode":
            self.LEAK_LIF = 1
        elif spiking_neuron.__name__ == "IFNode":
            self.LEAK_LIF = 63
        else:
            self.LEAK_LIF = 2**6 - 1  # default
        print("leak_lif: " + str(self.LEAK_LIF))
        self.LIF = LIF_neuron(int(self.threshold_scale * self.v_threshold), self.PERTUBATION, self.LEAK_LIF)
        self.ANN = ANN_neuron(self.v_threshold, self.PERTUBATION)  # original threshold

        self.axon_dict = defaultdict(list)
        self.neuron_dict = {}
        self.output_neurons = []
        self.input_shape = np.array(input_shape)
        self.num_steps = num_steps
        self.axon_offset = 0
        self.neuron_offset = 0
        self.backend = backend
        self.bias_start_idx = None
        self.curr_input = None
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.snn_layers = 0
        # self.v_threshold = v_threshold
        self.layer_index = 0
        self.total_axonSyn = 0
        self.total_neuronSyn = 0
        self.max_fan = 0
        # mapping snn layer index to its bias axon range (start, end)
        self.bias_dict = []
        self.snn_layer_index = 0
        self.converted_model_pth = converted_model_pth
        # dvs datasets
        self.dvs = dvs
        # For spikformer only
        self.q = None
        self.v = None
        self.k = None
        self.embed_dim = embed_dim
        # print(self.neuron_dict['103622'])
        # from watchpoints import watch
        # watch.config(pdb=True)
        # watch(self.bias_dict)
        # watch(self.neuron_dict['103622'])
        # breakpoint()
        self.layer_neuron_indices = [] # krish: Store 9 neuron indices: first, middle, last for first conv, first fc, output (or for linear-only: first/mid/last of first, mid, last linear)
        self.thresholds = [] #krish

    #krish
    def set_no_conv_mode(self, no_conv):
        """Set a flag to indicate if the model has no conv layer (for special layer_neuron_indices logic)."""
        self._no_conv_mode = no_conv

    def save_model(self):
        """
        Save the converted model into three .pkl files:
            axon_dict.pkl,
            neuron_dict.pkl,
            output_neurons.pkl
        """
        if not os.path.exists(self.converted_model_pth):
            os.makedirs(self.converted_model_pth)
            print(f"Mkdir {self.converted_model_pth}.")

        with open(f"{self.converted_model_pth}/axon_dict.pkl", "wb") as f:
            pickle.dump(self.axon_dict, f)
        with open(f"{self.converted_model_pth}/neuron_dict.pkl", "wb") as f:
            pickle.dump(self.neuron_dict, f)
        with open(f"{self.converted_model_pth}/output_neurons.pkl", "wb") as f:
            pickle.dump(self.output_neurons, f)
        print(f"Model saved at {self.converted_model_pth}.")

    def input_converter(self, input_data):
        """
        Converts a batch of input data into a list of corresponding axon indices.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data of the shape (B, -1).

        Returns
        -------
        [[[spike indices of a single frame] * T (timesteps)] * B (batch number)]
            The batch of spikes, with each spike represented by its axon index.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.input_converter(some_input_data)
        """

        self.input_shape = input_data.shape
        return self._input_converter(input_data)

    def _input_converter_step_batch(self, input_data, step):
        """
        Convert a batch of data from a single step, convert them into axon indicies

        """
        batch = []
        current_input = input_data.view(input_data.size(0), -1)
        for img in current_input:
            axons = self._input_converter_step(img, step)
            batch.append(axons)
        return batch

    def _input_converter_step(self, data, step):
        """
        Convert data from a single step, convert them into axon indicies

        """
        input_axons = ["a" + str(idx) for idx, axon in enumerate(data) if axon != 0]
        bias_axons = []
        # delay the bias based on their layer index
        end_layer = self.snn_layers if step + 1 >= self.snn_layers else step + 1
        for layer in range(end_layer):
            # skip layers without bias
            # breakpoint()
            if self.bias_dict[layer] != self.NULL_INDICIES:
                bias_axons.extend(
                    [
                        "a" + str(idx)
                        for idx in range(
                            self.bias_dict[layer][0], self.bias_dict[layer][1]
                        )
                    ]
                )
        return input_axons + bias_axons

    def _input_converter(self, input_data):
        """
        Takes in a batch of data (static image or DVS data), convert them into axon indicies
        """
        current_input = None
        if self.dvs:
            # Flatten the input data to [B, T, -1]
            current_input = input_data.view(input_data.size(0), input_data.size(1), -1)
        else:
            # Flatten the input data to [B, -1]
            current_input = input_data.view(input_data.size(1), -1)

        axon_batch = []
        for img in current_input:
            axon_steps = []
            if self.dvs:
                for step in range(input_data.size(1)):

                    input_image = img[step]

                    axons = self._input_converter_step(input_image, step)

                    axon_steps.append(axons)

            else:
                for step in range(self.num_steps):

                    input_image = img

                    axons = self._input_converter_step(input_image, step)

                    axon_steps.append(axons)
            axon_batch.append(axon_steps)
        return axon_batch

    def layer_converter(self, model):
        """
        Converts a model into a CRI-compatible model.

        Parameters
        ----------
        model : torch.nn.Module
            The input model.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.layer_converter(some_model)
        """
        # breakpoint()
        module_names = list(model._modules)

        # construct the axon dict keys and set it as curr_input
        print("Input shape: ", self.input_shape)
        axons = np.array([i for i in range(np.prod(self.input_shape))]).reshape(
            self.input_shape
        )
        # add to neurons to the axons dic
        for axon in axons.flatten():
            self.axon_dict["a" + str(axon)] = []
        self.curr_input = axons
        self.axon_offset = np.prod(self.curr_input.shape)
        self.bias_start_idx = self.axon_offset

        for k, name in enumerate(module_names):
            if len(list(model._modules[name]._modules)) > 0 and not isSNNLayer(
                model._modules[name]
            ):
                if name == "attn":
                    self._attention_converter(model._modules[name])
                else:
                    self.layer_converter(model._modules[name])
            else:
                self._layer_converter(model._modules[name], k, model)

        print("Total axon synapses: ", self.total_axonSyn)

    def _layer_converter(self, layer, k, model):
        if self.layer_index < self.input_layer:
            print("Skipped layer: ", layer)
        elif isinstance(layer, nn.Linear):
            self._linear_converter(layer, k, model)
            self.snn_layers += 1
        elif isinstance(layer, nn.Conv2d):
            # krish: ook ahead for Conv2d -> BatchNorm2d -> SpikingNeuron or AvgPool2d -> SpikingNeuron
            module_names = list(model._modules)
            num_layers = len(module_names)
            # Only check ahead if enough layers remain
            if k + 2 < num_layers:
                next_layer = model[k + 2]
                # Conv2d -> BatchNorm2d -> SpikingNeuron
                if isSNNLayer(next_layer):
                    self._conv_converter(layer, k, model)
                    self.snn_layers += 1
                # Conv2d -> BatchNorm2d -> AvgPool2d -> SpikingNeuron
                elif isinstance(next_layer, nn.AvgPool2d):
                    self._conv_converter_avgpooled(layer, next_layer, k, model)
                    self.snn_layers += 1
                else:
                    print("Unsupported Conv2d block: missing BatchNorm2d or SpikingNeuron.")
            else:
                print("Conv2d block: not enough layers ahead for pattern detection.")
        elif isinstance(layer, nn.MaxPool2d):
            self._maxPool_converter(layer)
            self.snn_layers += 1
        else:
            print("Unsupported layer: ", layer)

        self.layer_index += 1

    def _attention_converter(self, model):
        # print(f"Convert attention layer")
        # Flatten the current_input matrix to N*D (D = self.embed_dim, N = H*W)
        self.curr_input = np.transpose(
            self.curr_input.reshape(
                self.curr_input.shape[-2] * self.curr_input.shape[-1], self.embed_dim
            )
        )  # Hardcode for now

        module_names = list(model._modules)
        for k, name in enumerate(module_names):
            if not isSNNLayer(model._modules[name]):
                if name == "q_linear":
                    self.q = self._attention_linear_converter(model._modules[name])
                elif name == "k_linear":
                    self.k = self._attention_linear_converter(model._modules[name])
                elif name == "v_linear":
                    self.v = self._attention_linear_converter(model._modules[name])
                elif name == "proj_linear":
                    self.curr_input = self._attention_linear_converter(
                        model._modules[name]
                    )
            elif name == "attn_lif":
                self._matrix_mul_cri(self.q, self.v)
                self._matrix_mul_cri(self.curr_input, self.k)
            self.layer_index += 1
        self.curr_input = np.transpose(self.curr_input)

    def _attention_linear_converter(self, layer):
        # print(f'Input layer shape(infeature, outfeature): {self.curr_input.shape} {self.curr_input.shape}')
        output_shape = self.curr_input.shape
        output = np.array(
            [
                str(i)
                for i in range(
                    self.neuron_offset, self.neuron_offset + np.prod(output_shape)
                )
            ]
        ).reshape(output_shape)
        weights = layer.weight.detach().cpu().numpy()
        for n in range(self.curr_input.shape[0]):
            # print(self.curr_input[d], weights)
            for neuron_idx, neuron in enumerate(self.curr_input[n, :]):
                self.neuron_dict[neuron].extend(
                    [
                        (output[n, neuron_idx], int(weight))
                        for idx, weight in enumerate(weights[n])
                    ]
                )
        self.neuron_offset += np.prod(output_shape)
        # print(f'curr_neuron_offset: {self.neuron_offset}')
        if layer.bias is not None and self.layer_index != self.output_layer:
            # print(f'Constructing {layer.bias.shape[0]} bias axons for hidden linear layer')
            self._cri_bias(layer, output, atten_flag=True)
            self.axon_offset = len(self.axon_dict)
        return output.transpose(-2, -1)

    def _matrix_mul_cri(self, x, y):
        """
        Maps the matrix multiplication operation into CRI neurons connections.

        Parameters
        ----------
        x : np.ndarray
            The first input matrix.
        y : np.ndarray
            The second input matrix.

        """
        # TODO: parallelize each time step
        # print(f"x.shape: {x.shape}")
        h, w = x.shape

        _, d = y.shape
        x_flatten = x.flatten()  # (h*w)
        y_flatten = y.transpose().flatten()  # (d*w)

        first_layer = np.array(
            [str(i) for i in range(self.neuron_offset, self.neuron_offset + h * w * d)]
        )
        # first_layer = first_layer.reshape(h*w*d)
        self.neuron_offset += h * w * d

        second_layer = np.array(
            [str(i) for i in range(self.neuron_offset, self.neuron_offset + h * d)]
        )
        # second_layer = second_layer.reshape(b, h*d)
        self.neuron_offset += h * d

        for idx, neuron in enumerate(x_flatten):
            for i in range(d):
                # print(f"idx%w + w*i + w*d*(idx//w): {idx%w + w*i + w*d*(idx//w)}")
                self.neuron_dict[neuron].extend(
                    [
                        (
                            first_layer[idx % w + w * i + w * d * (idx // w)],
                            self.v_threshold,
                        )
                    ]
                )
        for idx, neuron in enumerate(y_flatten):
            for i in range(h):
                # print(f"idx%(w*d): {idx%(w*d)}")
                self.neuron_dict[neuron].append(
                    [(first_layer[idx % (w * d)], self.v_threshold)]
                )

        # for r in tqdm(range(b)):
        for idx, neuron in enumerate(first_layer):
            # print(f"idx//w: {idx//w}")
            self.neuron_dict[neuron].extend((second_layer[idx // w], self.v_threshold))

        second_layer = second_layer.reshape(h, d)
        # print(f'outputshape: {self.curr_input.shape}')
        self.curr_input = second_layer

    def _sparse_converter(self, layer):
        input_shape = layer.in_features
        output_shape = layer.out_features
        # print(f'Input layer shape(infeature, outfeature): {input_shape} {output_shape}')
        axons = np.array([str(i) for i in range(0, input_shape)])
        output = np.array([str(i) for i in range(0, output_shape)])
        weight = layer.weight.detach().cpu().to_dense().numpy()
        # print(f'Weight shape:{weight.shape}')
        curr_neuron_offset, next_neuron_offset = 0, input_shape
        # print(f'curr_neuron_offset, next_neuron_offset: {curr_neuron_offset, next_neuron_offset}')
        for neuron_idx, neuron in enumerate(weight.T):
            neuron_id = str(neuron_idx)
            neuron_entry = [
                (str(base_postsyn_id + next_neuron_offset), int(syn_weight))
                for base_postsyn_id, syn_weight in enumerate(neuron)
                if syn_weight != 0
            ]
            self.axon_dict[neuron_id] = neuron_entry
        # print('Instantiate output neurons')
        for output_neuron in range(
            next_neuron_offset, next_neuron_offset + layer.out_features
        ):
            self.neuron_dict[str(output_neuron)] = (self.LIF_Neuron, [])  # TODO: Fix me
            self.output_neurons.append(neuron_id)
        # print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')

    def _linear_converter(self, layer, k, model):
        """
        Takes in a PyTorch linear layer and generate the postsynaptic neurons (numpy array)
        Call _linear_weight to unroll the connections

        Parameters
        ----------
        layer : PyTorch linear layer
        """
        

        try:
            nextLayer = model[k + 1]
            if isSNNLayer(nextLayer):
                v_thresh = nextLayer.v_threshold
            else:
                raise Exception("linear layer with no following snn layer")
        except:
            raise Exception("linear layer with no following snn layer")

        # Debug: print the neuron indices used as inputs to the linear layer
        # print("\n[DEBUG] Neuron indices used as inputs to the linear layer:")
        # curr_input_flat = self.curr_input.flatten()
        # print(f"  Total: {len(curr_input_flat)}")
        # print(f"  Min index: {np.min(curr_input_flat)}, Max index: {np.max(curr_input_flat)}")
        # print(f"  First 5: {curr_input_flat[:5]}")
        # print(f"  Middle 5: {curr_input_flat[len(curr_input_flat)//2:len(curr_input_flat)//2+5]}")
        # print(f"  Last 5: {curr_input_flat[-5:]}")

        # Optionally, check if these indices match the output of the previous layer
        # (e.g., after avgpool)
        # If you want to check, you can print the shape and range of self.curr_input right after avgpool

        if self.layer_index == self.input_layer:
            print("Building synapses between axons and neurons with linear Layer")
        else:
            print("Building synapese from neurons to neurons with linear Layer")
            #krish: don't increment neuron_offset here; only increment after allocating new output neurons below
            #self.neuron_offset += np.prod(self.curr_input.shape)

        print(
            f"Layer shape(in_feature, out_feature): {layer.in_features} {layer.out_features}"
        )

        output = np.array(
            [
                i
                for i in range(
                    self.neuron_offset, self.neuron_offset + layer.out_features
                )
            ]
        )
        output_flat = output.flatten()
        print(f"Output length: {len(output_flat)}, first val: {output_flat[0]}, last val: {output_flat[-1]}")
        # update thresholds list
        for i in range(len(output_flat)):
            self.thresholds.append(int(self.threshold_scale * v_thresh))
        # Special handling: if no conv layer, fill layer_neuron_indices with 9 indices from first, middle, last linear layers
        # Otherwise, keep old behavior
        if hasattr(self, '_no_conv_mode') and self._no_conv_mode:
            # Track which linear layer this is
            if not hasattr(self, '_linear_layer_indices'): self._linear_layer_indices = []
            n = len(output_flat)
            indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
            self._linear_layer_indices.append(indices)
            # After all linear layers, fill layer_neuron_indices with first/mid/last from first, middle, last linear
            # This will be called for each linear layer, so only fill at the last one
            # (Assume output_layer is the last linear layer)
            if self.layer_index == self.output_layer:
                total = len(self._linear_layer_indices)
                if total >= 3:
                    self.layer_neuron_indices = self._linear_layer_indices[0] + self._linear_layer_indices[total//2] + self._linear_layer_indices[-1]
                else:
                    # Fallback: just concatenate all
                    self.layer_neuron_indices = sum(self._linear_layer_indices, [])
        else:
            # krish: add first 3 neurons if this is the first linear layer for membrane potential tracking
            if len(self.layer_neuron_indices) < 6:
                n = len(output_flat)
                indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
                self.layer_neuron_indices.extend(indices)


        # Now increment neuron_offset by the number of output neurons created
        self.neuron_offset += layer.out_features
        self._linear_weight(self.curr_input.flatten(), output, layer, v_thresh)

        if layer.bias is not None:
            self._cri_bias(layer, output)
            print(
                f"Constructing {len(self.axon_dict) - self.axon_offset} bias axons for linear layer"
            )
            self.axon_offset = len(self.axon_dict)
        else:
            self.bias_dict.append(self.NULL_INDICIES)

        print("number of neurons before output: ", len(self.neuron_dict))

        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from linear layer")
            # krish: instantiate neuron model with LEAK_LIF based on spikingjelly neuron type
            lifNeuronModel = LIF_neuron(int(self.threshold_scale * v_thresh), 0, self.LEAK_LIF)  # scaled threshold
            # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
            # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
            # lifNeuronModel = IF_neuron(int(self.threshold_scale * v_thresh), 0)  # IF neuron, scaled threshold
            for postSynNeuron in output:
                # this needs to add a neuron type
                self.neuron_dict[str(postSynNeuron)] = ([], lifNeuronModel)
                self.output_neurons.append(str(postSynNeuron))
            # After output neurons are created, add final 3 output neuron indices
            output_flat = output.flatten()
            if len(self.layer_neuron_indices) < 9:
                n = len(output_flat)
                indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
                self.layer_neuron_indices.extend(indices)

            #krish: fix, sometimes the neuron_dict contains -1 as a key which messes its length, for now just removing
            if "-1" in list(self.neuron_dict.keys()):
                print("neuron dict contains -1, removing")
                self.neuron_dict.pop("-1")

            # print info about neuron dict
            neuron_dict_keys_sorted = list(self.neuron_dict.keys())
            print(f"Neuron dict keys (sorted): {neuron_dict_keys_sorted[:30]}...")
            print(f"First key: {neuron_dict_keys_sorted[0]}, Last key: {neuron_dict_keys_sorted[-1]}")

        self.curr_input = output
        self.snn_layer_index += 1
        print(
            f"Number of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}"
        )
        print("Number of neurons: ", len(self.neuron_dict))
        print("Threshold length: ", len(self.thresholds))
        # print neuron index range after linear layer ---
        # print("[DEBUG] After linear layer:")
        # print(f"  self.curr_input.shape: {self.curr_input.shape}")
        # print(f"  self.curr_input: min={np.min(self.curr_input)}, max={np.max(self.curr_input)}, len={len(self.curr_input)}")
        # print(f"  self.neuron_offset after allocation: {self.neuron_offset}")

    def _linear_weight(self, input, output, layer, v_thresh):
        """
        Unroll the linear layer by building the synapses between
        presynaptic neurons and postsynaptic neurons

        Parameters
        ----------
        input : numpy array of shape (*, in_features)
            The presynaptic neurons

        output : numpy array of shape (*, out_features)
            The postsynaptic neurons

        layer : PyTorch linear layer
        """
        # this should be okay for multineuron. Each layer should have neurons with a single neuron model
        # how to get threshold
        # breakpoint()
        lifNeuronModel = LIF_neuron(int(self.threshold_scale * v_thresh), 0, self.LEAK_LIF)  # scaled threshold
        # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
        # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
        # lifNeuronModel = IF_neuron(int(self.threshold_scale * v_thresh), 0)  # IF neuron, scaled threshold

        weights = layer.weight.detach().cpu().numpy().transpose()  # (in, out)
        for preIdx, weight in enumerate(weights):
            if self.layer_index == self.input_layer:
                postSynNeurons = [
                    (str(output[postIdx]), int(synWeight))
                    for postIdx, synWeight in enumerate(weight)
                ]
                self.axon_dict["a" + str(input[preIdx])] = postSynNeurons
            else:
                postSynNeurons = [
                    (str(output[postIdx]), int(synWeight))
                    for postIdx, synWeight in enumerate(weight)
                ]

                self.neuron_dict[str(input[preIdx])] = (postSynNeurons, lifNeuronModel)


    def _conv_converter(self, layer, k, model):
        """
        Takes in a PyTorch Conv layer and generate the postsynaptic neurons (numpy array)
        Call _conv_shape and _conv_weight to calculate the
        output shape and perform unrolling respectively

        Parameters
        ----------
        layer : PyTorch Conv layer
            (currently only support Conv2d)

        """
        try:
            nextLayer = model[k + 2]
            if isSNNLayer(nextLayer):
                v_thresh = nextLayer.v_threshold
            else:
                raise Exception("conv layer with no following identity+snn layer")
        except:
            raise Exception("liear layer with no following snn layer")
        print(f"Converting layer: {layer}")
        output = None

        if self.layer_index == self.input_layer:
            print("Building synapese from axons to neurons with conv Layer")
        else:
            print("Building synapese from neurons to neurons with conv Layer")
            #krish: don't increment neuron_offset here; only increment after allocating new output neurons below

        output_shape = self._conv_shape(layer, self.curr_input.shape)
        print(
            f"Layer shape(in_feature, out_feature): {self.curr_input.shape} {output_shape}"
        )

        output = np.array(
            [
                i
                for i in range(
                    self.neuron_offset, self.neuron_offset + np.prod(output_shape)
                )
            ]
        ).reshape(output_shape)
        # update thresholds list
        output_flat = output.flatten()
        for i in range(len(output_flat)):
            self.thresholds.append(int(self.threshold_scale * v_thresh))
        # Add first 3 neurons if this is the first conv layer
        if len(self.layer_neuron_indices) < 3:
            n = len(output_flat)
            indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
            self.layer_neuron_indices.extend(indices)

        # Now increment neuron_offset by the number of output neurons created
        self.neuron_offset += np.prod(output_shape)

        self._conv_weight(self.curr_input, output, layer, v_thresh)

        if layer.bias is not None:
            self._cri_bias(layer, output)
            print(
                f"Constructing {len(self.axon_dict) - self.axon_offset} bias axons from conv layer."
            )
            self.axon_offset = len(self.axon_dict)
        else:
            self.bias_dict.append(self.NULL_INDICIES)

        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from conv layer")

            lifNeuronModel = LIF_neuron(self.threshold_scale * v_thresh, 0, self.LEAK_LIF)  # scaled threshold
            # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
            # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
            # lifNeuronModel = IF_neuron(self.threshold_scale * v_thresh, 0)  # IF neuron, scaled threshold
            for postSynNeuron in output.flatten():
                self.neuron_dict[str(postSynNeuron)] = ([], lifNeuronModel)
                self.output_neurons.append(str(postSynNeuron))
            # After output neurons are created, add final 3 output neuron indices
            output_flat = output.flatten()
            if len(self.layer_neuron_indices) < 9:
                n = len(output_flat)
                indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
                self.layer_neuron_indices.extend(indices)

        self.curr_input = output
        self.snn_layer_index += 1
        print(
            f"Number of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}"
        )

    def _conv_weight(self, input, output, layer, v_thresh):
        """
        Unroll the convolutional layer by building the synapses between
        presynaptic neurons and postsynaptic neurons
        Currently supports parameters: padding and stride

        Parameters
        ----------
        input : numpy array of shape (c, h, w)
            The presynaptic neurons

        output : numpy array of shape (c', h', w')
            The postsynaptic neurons

        layer : PyTorch Conv layer
            (currently only support Conv2d)

        """
        # Get the layer parameters and weights
        kernel = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        weights = layer.weight.detach().cpu().numpy()

        # Uncomment and set leak to self.LEAK_LIF
        lifNeuronModel = LIF_neuron(int(self.threshold_scale * v_thresh), 0, self.LEAK_LIF)  # scaled threshold
        # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
        # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
        # lifNeuronModel = IF_neuron(int(self.threshold_scale * v_thresh), 0)  # IF neuron, scaled threshold

        # Check parameters (int or tuple) and convert them all to tuple
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        # Pad the input if padding is not zero
        if sum(layer.padding) != 0:
            # Pad the input with -1 (input neuron idx starts at 0)
            dim = 2  # pad the last two dim of the input array
            pad = layer.padding * dim
            input = F.pad(torch.from_numpy(input), pad, value=self.NULL_NEURON).numpy()

        h, w = input.shape[-2], input.shape[-1]
        # iterate throught the input array based on the kernel ans stride size
        for c in tqdm(range(input.shape[0])):
            for row in range(0, h - kernel[0] + 1, stride[0]):
                for col in range(0, w - kernel[1] + 1, stride[1]):
                    # (row, col) : local index of the top left corner of the input patch
                    preSynNeurons = input[
                        c, row : row + kernel[0], col : col + kernel[1]
                    ]
                    # iterate each of the filter
                    for idx, weight in enumerate(weights):
                        # find the postsynaptic neuron
                        postSynNeuron = output[idx, row // stride[0], col // stride[1]]
                        # add a synapse between each of the neuron in preSynNeurons & postSynNeuron
                        for i, rows in enumerate(preSynNeurons):
                            for j, pre in enumerate(rows):
                                if self.layer_index == self.input_layer:
                                    if pre != self.NULL_NEURON:
                                        self.axon_dict["a" + str(pre)].append(
                                            (str(postSynNeuron), int(weight[c, i, j]))
                                        )
                                else:
                                    if pre != self.NULL_NEURON:
                                        if str(pre) not in self.neuron_dict:
                                            self.neuron_dict[str(pre)] = (
                                                [],
                                                lifNeuronModel,
                                            )
                                        self.neuron_dict[str(pre)][0].append(
                                            (str(postSynNeuron), int(weight[c, i, j]))
                                        )

    #krish: added AvgPool2d support for Conv2d + AvgPool2d -> SpikingNeuron
    def _conv_converter_avgpooled(self, conv_layer, pool_layer, k, model):
        # Copy _conv_converter logic, then add pooling
        try:
            nextLayer = model[k + 3]
            if nextLayer is not None and isSNNLayer(nextLayer):
                v_thresh = nextLayer.v_threshold
            else:
                v_thresh = self.v_threshold
        except:
            v_thresh = self.v_threshold
        print(f"[DEBUG] Converting Conv2d + AvgPool2d block: {conv_layer} + {pool_layer}")
        output = None
        if self.layer_index == self.input_layer:
            print("Building synapese from axons to neurons with Conv+AvgPool Layer")
        else:
            print("Building synapese from neurons to neurons with Conv+AvgPool Layer")
            #self.neuron_offset += np.prod(self.curr_input.shape)
        # Get conv output shape
        # Use Conv2d's out_channels for correct neuron indexing
        conv_output_shape = self._conv_shape(conv_layer, self.curr_input.shape)
        pool_kernel = pool_layer.kernel_size
        pool_stride = pool_layer.stride
        pool_padding = pool_layer.padding if hasattr(pool_layer, 'padding') else 0
        if isinstance(pool_kernel, int): pool_kernel = (pool_kernel, pool_kernel)
        if isinstance(pool_stride, int): pool_stride = (pool_stride, pool_stride)
        if isinstance(pool_padding, int): pool_padding = (pool_padding, pool_padding)
        # Use out_channels from conv_layer, not input channels
        C = conv_layer.out_channels
        print("Out channels from conv layer:", C)
        H_in, W_in = conv_output_shape[-2], conv_output_shape[-1]
        H_out = (H_in + 2*pool_padding[0] - pool_kernel[0]) // pool_stride[0] + 1
        W_out = (W_in + 2*pool_padding[1] - pool_kernel[1]) // pool_stride[1] + 1
        output = np.array([
            i for i in range(self.neuron_offset, self.neuron_offset + C * H_out * W_out)
        ]).reshape((C, H_out, W_out))

        output_flat = output.flatten()
        # update thresholds list
        for i in range(len(output_flat)):
            self.thresholds.append(int(self.threshold_scale * v_thresh))
        output_flat = output.flatten()
        # Add first 3 neurons if this is the f
        # Add first 3 neurons if this is the first conv layer
        if len(self.layer_neuron_indices) < 3:
            n = len(output_flat)
            indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
            self.layer_neuron_indices.extend(indices)

        print(f"Layer shape(in_feature, out_feature): {self.curr_input.shape} {output.shape}")
        self._conv_weight_avgpooled(self.curr_input, output, conv_layer, pool_layer, v_thresh, pool_kernel, pool_stride, pool_padding)
        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from Conv+AvgPool layer")
            # Uncomment and set leak to self.LEAK_LIF
            lifNeuronModel = LIF_neuron(self.threshold_scale * v_thresh, 0, self.LEAK_LIF)  # scaled threshold
            # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
            # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
            for postSynNeuron in output.flatten():
                self.neuron_dict[str(postSynNeuron)] = ([], lifNeuronModel)
                self.output_neurons.append(str(postSynNeuron))
            # After output neurons are created, add final 3 output neuron indices
            output_flat = output.flatten()
            if len(self.layer_neuron_indices) < 9:
                n = len(output_flat)
                indices = [output_flat[0], output_flat[n//2], output_flat[-1]] if n >= 3 else list(output_flat)
                self.layer_neuron_indices.extend(indices)
        self.curr_input = output
        self.neuron_offset += C * H_out * W_out
        self.snn_layer_index += 1
        self.bias_dict.append(self.NULL_INDICIES)
        print("Number of neurons: ", len(self.neuron_dict))
        print("Threshold length: ", len(self.thresholds))

    def _conv_weight_avgpooled(self, input, output, conv_layer, pool_layer, v_thresh, pool_kernel, pool_stride, pool_padding):
        # Copy _conv_weight logic, then add pooling
        kernel = conv_layer.kernel_size
        stride = conv_layer.stride
        padding = conv_layer.padding
        weights = conv_layer.weight.detach().cpu().numpy()
        # Uncomment and set leak to self.LEAK_LIF
        lifNeuronModel = LIF_neuron(int(self.threshold_scale * v_thresh), 0, self.LEAK_LIF)  # scaled threshold
        # lifNeuronModel = LIF_neuron(v_thresh // 2, 0, self.LEAK_LIF)  # threshold halved
        # lifNeuronModel = LIF_neuron(v_thresh // 3, 0, self.LEAK_LIF)  # threshold one-third
        # lifNeuronModel = IF_neuron(int(self.threshold_scale * v_thresh), 0)  # IF neuron, scaled threshold
        if isinstance(kernel, int): kernel = (kernel, kernel)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        # Pad the input if padding is not zero
        if sum(padding) != 0:
            dim = 2
            pad = padding * dim
            input = F.pad(torch.from_numpy(input), pad, value=self.NULL_NEURON).numpy()
        h, w = input.shape[-2], input.shape[-1]
        C_in = input.shape[0]
        C_out = output.shape[0]
        H_out, W_out = output.shape[-2], output.shape[-1]
        debug_indices = []
        if C_out > 0 and H_out > 0 and W_out > 0:
            debug_indices.append((0, 0, 0))
            debug_indices.append((C_out//2, H_out//2, W_out//2))
            debug_indices.append((C_out-1, H_out-1, W_out-1))
        count_include_pad = getattr(pool_layer, 'count_include_pad', True)
        if count_include_pad:
            scale = 1.0 / (pool_kernel[0] * pool_kernel[1])
        print(f"C_in: {C_in}, C_out: {C_out}, H_out: {H_out}, W_out: {W_out}, kernel: {kernel}, stride: {stride}, padding: {padding}, pool_kernel: {pool_kernel}, pool_stride: {pool_stride}, count_include_pad: {count_include_pad}")
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    pooled_presyn = []
                    valid_count = 0
                    for c_in in range(C_in):
                        for pi in range(pool_kernel[0]):
                            for pj in range(pool_kernel[1]):
                                row = i * pool_stride[0] + pi
                                col = j * pool_stride[1] + pj
                                if 0 <= row < h - kernel[0] + 1 and 0 <= col < w - kernel[1] + 1:
                                    for k_i in range(kernel[0]):
                                        for k_j in range(kernel[1]):
                                            in_row = row + k_i
                                            in_col = col + k_j
                                            pre = input[c_in, in_row, in_col]
                                            w_val = int(weights[c_out, c_in, k_i, k_j])
                                            if pre != self.NULL_NEURON:
                                                pooled_presyn.append((pre, w_val))
                                                valid_count += 1
                                            elif count_include_pad:
                                                pooled_presyn.append((pre, w_val))
                    # Determine scaling factor
                    if not count_include_pad:
                        scale = 1.0 / max(valid_count, 1)
                    postSynNeuron = output[c_out, i, j]
                    if (c_out, i, j) in debug_indices:
                        print(f"  Output neuron {postSynNeuron} receives from:")
                        for pre, w_val in pooled_presyn[:8]:
                            scaled_w = int(round(w_val * scale))
                            print(f"    {pre} --({w_val} * {scale:.3f} = {scaled_w})--> {postSynNeuron}")
                        print(f"  Scaling factor: {scale}")
                        print(f"  Total presynaptic: {len(pooled_presyn)}\n")
                    for pre, w_val in pooled_presyn:
                        scaled_w = int(round(w_val * scale))
                        if self.layer_index == self.input_layer:
                            self.axon_dict["a" + str(pre)].append((str(postSynNeuron), scaled_w))
                        else:
                            if str(pre) not in self.neuron_dict:
                                self.neuron_dict[str(pre)] = ([], lifNeuronModel)
                            self.neuron_dict[str(pre)][0].append((str(postSynNeuron), scaled_w))
        # for c_out in range(C_out):
        #     for i in range(H_out):
        #         for j in range(W_out):
        #             postSynNeuron = output[c_out, i, j]
        #             self.neuron_dict[str(postSynNeuron)] = ([], lifNeuronModel)

    def _maxPool_converter(self, layer):
        """
        Takes in a PyTorch MaxPool layer and generate the postsynaptic neurons (numpy array)
        Call _maxPool_shape and _maxPool_weight to calculate the
        output shape and perform unrolling respectively

        Parameters
        ----------
        layer : PyTorch MaxPool layer
            (currently only support MaxPool2d)

        """
        if self.layer_index == self.input_layer:
            print("Building synapese from axons to neurons with maxPool layer")
        else:
            print("Building synapese from neurons to neurons with maxPool layer")
            self.neuron_offset += np.prod(self.curr_input.shape)

        output_shape = self._maxPool_shape(layer, self.curr_input.shape)
        print(
            f"Layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}"
        )

        output = np.array(
            [
                i
                for i in range(
                    self.neuron_offset, self.neuron_offset + np.prod(output_shape)
                )
            ]
        ).reshape(output_shape)

        self._maxPool_weight(self.curr_input, output, layer)

        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from maxPool layer")
            for postSynNeuron in output:
                self.neuron_dict[str(postSynNeuron)] = []  # Fix Me
                self.output_neurons.append(str(postSynNeuron))

        self.bias_dict.append(self.NULL_INDICIES)
        self.curr_input = output
        self.snn_layer_index += 1

        print(
            f"Number of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}"
        )

    def _maxPool_weight(self, input, output, layer):
        h_k, w_k = layer.kernel_size, layer.kernel_size
        h_i, w_i = input.shape[-2], input.shape[-1]
        pad_top, pad_left = h_k // 2, w_k // 2

        # double the weight of the synapse for maxpooling to make sure
        # that the postsynaptic neuron will spike
        scaler = self.v_threshold * 2

        if self.layer_index != self.input_layer:
            for pre in input.flatten():
                self.neuron_dict[pre][1] = self.ANN

        for c in tqdm(range(input.shape[0])):
            for row in range(0, h_i, 2):
                for col in range(0, w_i, 2):
                    preSynNeurons = input[
                        c, row : row + pad_top + 1, col : col + pad_left + 1
                    ]
                    postSynNeuron = output[c, row // 2, col // 2]
                    for rows in preSynNeurons:
                        for pre in rows:
                            if self.layer_index == self.input_layer:
                                self.axon_dict["a" + str(pre)].append(
                                    (str(postSynNeuron), int(scaler))
                                )
                            else:
                                self.neuron_dict[str(pre)][0].append(
                                    (str(postSynNeuron), int(scaler))
                                )

    def _cri_bias(self, layer, outputs, atten_flag=False):
        biases = layer.bias.detach().cpu().numpy()
        # breakpoint()
        # Gwen: I'm going to assume there was a typo here and I added parentheses to turn the two seperate arguments into a tuple
        self.bias_dict.append((self.axon_offset, self.axon_offset + biases.size))

        if isinstance(layer, nn.Conv2d):
            for output_chan, bias in enumerate(biases):
                bias_id = "a" + str(output_chan + self.axon_offset)
                self.axon_dict[bias_id] = [
                    (str(neuron_idx), int(bias))
                    for neuron_idx in outputs[output_chan].flatten()
                ]

        elif isinstance(layer, nn.Linear):
            for output_chan, bias in enumerate(biases):
                bias_id = "a" + str(output_chan + self.axon_offset)
                if atten_flag:
                    self.axon_dict[bias_id] = [
                        (str(neuron_idx), int(bias))
                        for neuron_idx in outputs[output_chan, :].flatten()
                    ]
                else:
                    self.axon_dict[bias_id] = [(str(outputs[output_chan]), int(bias))]
        else:
            print(f"Unspported layer: {layer}")

    def _conv_shape(self, layer, input_shape):
        h_out = (
            input_shape[-2]
            + 2 * layer.padding[0]
            - layer.dilation[0] * (layer.kernel_size[0] - 1)
            - 1
        ) / layer.stride[0] + 1
        w_out = (
            input_shape[-1]
            + 2 * layer.padding[1]
            - layer.dilation[1] * (layer.kernel_size[1] - 1)
            - 1
        ) / layer.stride[1] + 1
        if len(input_shape) == 4:
            return np.array(
                (input_shape[0], layer.out_channels, int(h_out), int(w_out))
            )
        else:
            return np.array((layer.out_channels, int(h_out), int(w_out)))

    def _maxPool_shape(self, layer, input_shape):
        h_out = (
            input_shape[-2] + layer.padding * 2 - (layer.kernel_size)
        ) / layer.stride + 1
        w_out = (
            input_shape[-1] + layer.padding * 2 - (layer.kernel_size)
        ) / layer.stride + 1
        if len(input_shape) == 4:
            return np.array((input_shape[0], input_shape[1], int(h_out), int(w_out)))
        else:
            return np.array((input_shape[0], int(h_out), int(w_out)))

    def _cri_fanout(self):
        for key in self.axon_dict.keys():
            self.total_axonSyn += len(self.axon_dict[key])
            if len(self.axon_dict[key]) > self.max_fan:
                self.max_fan = len(self.axon_dict[key])
        print(
            "Total number of connections between axon and neuron: ", self.total_axonSyn
        )
        print("Max fan out of axon: ", self.max_fan)
        print("---")
        print("Number of neurons: ", len(self.neuron_dict))
        self.max_fan = 0
        for key in self.neuron_dict.keys():
            self.total_neuronSyn += len(self.neuron_dict[key])
            if len(self.neuron_dict[key]) > self.max_fan:
                self.max_fan = len(self.neuron_dict[key])
        print(
            "Total number of connections between hidden and output layers: ",
            self.total_neuronSyn,
        )
        print("Max fan out of neuron: ", self.max_fan)

    def run_CRI_hw(self, inputList, hardwareNetwork, outputPotential=False):
        """
        Runs a bach of input through the hardware implementation of the network
        Returns the output predictions and the output spikes

        Parameters
        ----------
        inputList : [ [ []* T ] * B ]
            The input data, where each most inner item is a list of axon indices representing the spikes.
        hardwareNetwork : object
            The hardware network object.
        outputPotential: bool, default False
            The output potential flag, if set to true outputs membrane potential

        Returns
        -------
        list of list
            The output spikes

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.run_CRI_hw(some_inputList, some_hardwareNetwork)
        """
        import hs_bridge

        outputSpikes = []
        membranePotential = []
        debugspike = []

        output_idx = [i for i in range(len(self.output_neurons))]

        runcount = 0

        # each image
        for currInput in inputList:
            # initiate the hardware for each image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(self.neuron_dict), False, 0
            )  ##Num_neurons, simDump, coreOverride
            spikeRate = [0] * len(self.output_neurons)
            # each time step
            phaseDelay = self.snn_layers
            for sliceIdx, slice in enumerate(currInput):
                hwSpike = []
                if outputPotential:
                    potential, spikes = hardwareNetwork.step(
                        slice, membranePotential=True
                    )
                    hwSpike, _, _ = spikes
                    if sliceIdx >= phaseDelay:
                        membranePotential.append([v for k, v in potential])
                else:
                    hwSpike, _, _ = hardwareNetwork.step(slice, membranePotential=False)
                    runcount += 1
                if sliceIdx >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in hwSpike
                    ]
                    debugspike.append(spikeIdx)
                    # breakpoint()
                    for idx in spikeIdx:
                        # Checking if the output spike is in the defined output neuron
                        if idx not in output_idx:
                            print(f"Error: invalid output spike {idx}")
                        spikeRate[idx] += 1
            # if self.num_steps == 1:
            #     # Empty input for output delay since HiAER spike only get spikes after the spikes have occurred
            #     hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
            #     spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in hwSpike]
            #     for idx in spikeIdx:
            #         if idx not in output_idx:
            #             print(f"Error: invalid output spike {idx}")
            #         spikeRate[idx] += 1
            # Empty input for output delay
            for q in range(phaseDelay):
                hwSpike, v1, v2 = hardwareNetwork.step([], membranePotential=False)
                if sliceIdx + q >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in hwSpike
                    ]
                    debugspike.append(spikeIdx)
                    # breakpoint()
                    for idx in spikeIdx:
                        if idx not in output_idx:
                            print(f"Error: invalid output spike {idx}")
                        spikeRate[idx] += 1
            # Append the output spikes of each image to the output list
            # breakpoint()
            print(v1, v2)
            debugspike = []
            outputSpikes.append(spikeRate)
        print("runcount: " + str(runcount))

        if outputPotential:
            return outputSpikes, membranePotential
        else:
            return outputSpikes


    #krish: added extra debugging and membrane potential tracking for specific neurons
    def run_CRI_sw(self, inputList, softwareNetwork, outputPotential=False, one_batch_only=False, potential_neuron_indices=None):
        """
        Runs a batch of inputs through the software simulation of the network,
        returns the output predictions and output spikes

        Parameters
        ----------
        inputList : list of list of str
            The input data, where each item is a list of axon indices representing the spikes.
        softwareNetwork : object
            The software network object.
        outputPotential: bool, default False
            The output potential flag, if set to true outputs membrane potential

        Returns
        -------
        list of list
            The output spikes.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> predictions, outputSpikes = converter.run_CRI_sw(some_inputList, some_softwareNetwork)
        """
        outputSpikes = []
        debugspike = []
        membranePotential = []

        # Only print avgpool debug for the first avgpool layer encountered
        avgpool_debug_printed = False
        # each image
        for batch_idx, currInput in enumerate(tqdm(inputList)):
            softwareNetwork.simpleSim.initialize_sim_vars(len(self.neuron_dict))
            spikeRate = [0] * len(self.output_neurons)
            phaseDelay = self.snn_layers
            #phaseDelay = 0 # normally remove
            # track avgpool neuron activity ---
            # Find three representative avgpool neurons (first, middle, last)
            avgpool_neuron_indices = []
            # Find all neuron indices up to the start of the linear layer (assuming contiguous allocation)
            all_neuron_indices = sorted([int(k) for k in self.neuron_dict.keys() if k.isdigit()])
            if all_neuron_indices:
                first = all_neuron_indices[0]
                last = all_neuron_indices[len(all_neuron_indices)//2]
                end = all_neuron_indices[-1]
                avgpool_neuron_indices = [first, last, end]
            avgpool_debug = {idx: [] for idx in avgpool_neuron_indices}
            #print("a")
            sample_membrane = []
            print("length of currInput:", len(currInput))
            print("phaseDelay:", phaseDelay)
            n_timesteps = max(0, len(currInput) - phaseDelay)
            for sliceIdx, slice in enumerate(currInput):
                swSpike = []
                if outputPotential:
                    #print("b")
                    if potential_neuron_indices is not None:
                        # neuron_dict keys are user keys (str), ensure idx is a Python int or str
                        def to_str_key(idx):
                            if hasattr(idx, 'item'):
                                return str(idx.item())
                            elif hasattr(idx, 'astype'):
                                return str(int(idx))
                            else:
                                return str(idx)
                        neuron_keys = [to_str_key(idx) for idx in potential_neuron_indices]
                        print("neuron_keys:", neuron_keys)
                        #print("c")
                        potential, swSpike = softwareNetwork.step(slice, membranePotential=True, potential_neuron_keys=neuron_keys, thresholds=self.thresholds) #krish: api.py network step function now takes in a thresholds list
                        #print("potential length: " + str(len(potential)))
                        # potential: list of (key, v)
                        v_dict = dict(potential)
                        sample_membrane.append([v_dict.get(str(idx), 0.0) for idx in neuron_keys])
                    else:
                        potential, swSpike = softwareNetwork.step(slice, membranePotential=True, thresholds=self.thresholds)
                        if sliceIdx >= phaseDelay:
                            sample_membrane.append([v for k, v in potential])
                else:
                    swSpike = softwareNetwork.step(slice, membranePotential=False, thresholds=self.thresholds)
                if sliceIdx >= phaseDelay:
                    spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        spikeRate[idx] += 1
            if outputPotential:
                membranePotential.extend(sample_membrane)


                # debug: record for one output neuron
            output_neuron_idx = None
            if self.output_neurons:
                output_neuron_idx = int(self.output_neurons[0])
            if output_neuron_idx is not None:
                if 'output_neuron_debug' not in locals():
                    output_neuron_debug = []
                v = None
                threshold = None
                spiked = False
                neuron_attrs = {}
                # get neuron model attributes if present
                if hasattr(self, 'neuron_dict') and str(output_neuron_idx) in self.neuron_dict:
                    entry = self.neuron_dict[str(output_neuron_idx)]
                    if isinstance(entry, tuple):
                        neuron_model = entry[1]
                        neuron_attrs = {k: getattr(neuron_model, k) for k in dir(neuron_model) if not k.startswith('__') and not callable(getattr(neuron_model, k))}
                        threshold = neuron_attrs.get('threshold', None)
                # try to get membrane potential from simulation state if available
                if hasattr(softwareNetwork, 'simpleSim') and hasattr(softwareNetwork.simpleSim, 'membrane_potential'):
                    mp = softwareNetwork.simpleSim.membrane_potential
                    # Print all available keys for diagnosis
                    print(f"DEBUGGING: membrane_potential keys at t={sliceIdx}: {list(mp.keys())}")
                    # Try both str and int keys
                    v = mp.get(str(output_neuron_idx), None)
                    if v is None:
                        v = mp.get(output_neuron_idx, None)
                if swSpike and str(output_neuron_idx) in [str(s) for s in swSpike]:
                    spiked = True
                output_neuron_debug.append({'t': sliceIdx, 'v': v, 'threshold': threshold, 'spiked': spiked, 'attrs': neuron_attrs})

            for q in range(phaseDelay):
                swSpike = softwareNetwork.step([], membranePotential=False, thresholds=self.thresholds)
                if sliceIdx + q >= phaseDelay:
                    spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        spikeRate[idx] += 1
            outputSpikes.append(spikeRate)

    
            # --- DEBUG: Print output neuron activity for this sample ---
            if 'output_neuron_debug' in locals() and output_neuron_idx is not None:
                print(f"[DEBUG] Output neuron activity for neuron {output_neuron_idx}:")
                for entry in output_neuron_debug:
                    v_disp = entry['v'] if entry['v'] is not None else '(not present)'
                    print(f"    t={entry['t']}: v={v_disp} threshold={entry['threshold']} spiked={entry['spiked']}")
                # Print all available neuron model attributes for this neuron (once)
                if output_neuron_debug:
                    print(f"    [DEBUG] Neuron model attributes: {output_neuron_debug[0]['attrs']}")
                print()
            print()
            if one_batch_only:
                break


        if outputPotential:
            membranePotential = np.array(membranePotential)
            if membranePotential.ndim == 1:
                membranePotential = membranePotential.reshape(-1, 1)
            return outputSpikes, membranePotential
        else:
            return outputSpikes
