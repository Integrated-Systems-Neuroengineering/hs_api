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
    def __init__(self, w_bit, w_alpha=4):
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
                if name == "block":
                    new_model._modules[name] = self.quantize_block(
                        new_model._modules[name]
                    )
                else:
                    new_model._modules[name] = self.quantize(new_model._modules[name])
            else:
                if name == "attn_lif":
                    continue
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer

        end_time = time.time()
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
                    new_model._modules[name] = self.quantize_block(
                        new_model._modules[name]
                    )
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

        if self.dynamic_alpha:
            # weight_range = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            self.w_alpha = abs(
                max(layer.weight.flatten()) - min(layer.weight.flatten())
            )
            self.w_delta = self.w_alpha / (2 ** (self.w_bits - 1) - 1)
            self.weight_quant = weight_quantize_fn(
                self.w_bits
            )  # reinitialize the weight_quan
            self.weight_quant.wgt_alpha = self.w_alpha

        layer.weight = nn.Parameter(self.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight / self.w_delta)

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

        layer.v_threshold = int(layer.v_threshold / self.w_delta)
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
        metrics_log_path=None,
    ):
        self.HIGH_SYNAPSE_WEIGHT = 1e6
        self.NULL_NEURON = -1
        self.NULL_INDICIES = (-1, -1)
        self.PERTUBATION = 0
        self.LEAK_LIF = 2**6 - 1
        self.v_threshold = v_threshold
        # neuron model parameters
        # create all the neuron models??
        self.LIF = LIF_neuron(self.v_threshold, self.PERTUBATION, self.LEAK_LIF)
        self.ANN = ANN_neuron(self.v_threshold, self.PERTUBATION)

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
        # mapping layer names to their specific bias axons for proper activation timing
        self.layer_bias_axons = {}
        # counters for generating layer names
        self.conv_layer_count = 0
        self.linear_layer_count = 0
        self.converted_model_pth = converted_model_pth
        # dvs datasets
        self.dvs = dvs
        # optional metrics CSV path (or via env var HS_CONVERTER_METRICS_PATH)
        self.metrics_log_path = metrics_log_path or os.getenv("HS_CONVERTER_METRICS_PATH")
        # For spikformer only
        self.q = None
        self.v = None
        self.k = None
        self.embed_dim = embed_dim

    def _get_layer_name(self, layer):
        """
        Generate appropriate layer name for bias tracking
        
        Parameters
        ----------
        layer : torch.nn.Module
            The layer to generate name for
            
        Returns
        -------
        str
            Layer name (e.g., 'Conv1', 'Conv2', 'Linear1', 'Output')
        """
        if isinstance(layer, nn.Conv2d):
            self.conv_layer_count += 1
            return f"Conv{self.conv_layer_count}"
        elif isinstance(layer, nn.Linear):
            self.linear_layer_count += 1
            if self.layer_index == self.output_layer:
                return "Output"
            else:
                return f"Linear{self.linear_layer_count}"
        else:
            return f"Unknown_{self.layer_index}"

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
            # Convert defaultdict to regular dict to avoid CRI network initialization errors
            pickle.dump(dict(self.axon_dict), f)
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
        module_names = list(model._modules)

        # construct the axon dict keys and set it as curr_input
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

    def _layer_converter(self, layer, k, model):
        if self.layer_index < self.input_layer:
            print("Skipped layer: ", layer)
        elif isinstance(layer, nn.Linear):
            # in this scenario we would need to inspect the next layer to get the v_thresh from the lif layer
            self._linear_converter(layer, k, model)
            self.snn_layers += 1
        elif isinstance(layer, nn.Conv2d):
            self._conv_converter(layer, k, model)
            self.snn_layers += 1
        elif isinstance(layer, nn.MaxPool2d):
            self._maxPool_converter(layer)
            self.snn_layers += 1
        else:
            print("Unsupported layer: ", layer)

        self.layer_index += 1

    def _attention_converter(self, model):
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
            for neuron_idx, neuron in enumerate(self.curr_input[n, :]):
                self.neuron_dict[neuron].extend(
                    [
                        (output[n, neuron_idx], int(weight))
                        for idx, weight in enumerate(weights[n])
                    ]
                )
        self.neuron_offset += np.prod(output_shape)
        if layer.bias is not None and self.layer_index != self.output_layer:
            layer_name = f"Attention_{self.layer_index}"
            self._cri_bias(layer, output, layer_name=layer_name, atten_flag=True)
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
        self.neuron_offset += h * d

        for idx, neuron in enumerate(x_flatten):
            for i in range(d):
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
                self.neuron_dict[neuron].append(
                    [(first_layer[idx % (w * d)], self.v_threshold)]
                )

        for idx, neuron in enumerate(first_layer):
            self.neuron_dict[neuron].extend((second_layer[idx // w], self.v_threshold))

        second_layer = second_layer.reshape(h, d)
        self.curr_input = second_layer

    def _sparse_converter(self, layer):
        input_shape = layer.in_features
        output_shape = layer.out_features
        axons = np.array([str(i) for i in range(0, input_shape)])
        output = np.array([str(i) for i in range(0, output_shape)])
        weight = layer.weight.detach().cpu().to_dense().numpy()
        curr_neuron_offset, next_neuron_offset = 0, input_shape
        for neuron_idx, neuron in enumerate(weight.T):
            neuron_id = str(neuron_idx)
            neuron_entry = [
                (str(base_postsyn_id + next_neuron_offset), int(syn_weight))
                for base_postsyn_id, syn_weight in enumerate(neuron)
                if syn_weight != 0
            ]
            self.axon_dict[neuron_id] = neuron_entry
        for output_neuron in range(
            next_neuron_offset, next_neuron_offset + layer.out_features
        ):
            self.neuron_dict[str(output_neuron)] = (self.LIF_Neuron, [])  # TODO: Fix me
            self.output_neurons.append(neuron_id)

    def _linear_converter(self, layer, k, model):
        """
        Takes in a PyTorch linear layer and generate the postsynaptic neurons (numpy array)
        Call _linear_weight to unroll the connections

        Parameters
        ----------
        layer : PyTorch linear layer
        """
        try:
            if k+1 < len(model):
                nextLayer = model[k + 1]
                if isSNNLayer(nextLayer):
                    v_thresh = nextLayer.v_threshold
            else:
                v_thresh = self.v_threshold
        except:
            # raise Exception("liear layer with no following snn layer")
            raise Exception("line 981 error")
        if self.layer_index == self.input_layer:
            print("Building synapses between axons and neurons with linear Layer")
        else:
            print("Building synapses from neurons to neurons with linear Layer")
            self.neuron_offset += np.prod(self.curr_input.shape)

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
        self._linear_weight(self.curr_input.flatten(), output, layer, v_thresh)

        if layer.bias is not None:
            layer_name = self._get_layer_name(layer)
            self._cri_bias(layer, output, layer_name=layer_name)
            print(
                f"Constructing {len(self.axon_dict) - self.axon_offset} bias axons for linear layer"
            )
            self.axon_offset = len(self.axon_dict)
        else:
            self.bias_dict.append(self.NULL_INDICIES)

        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from linear layer")
            lifNeuronModel = ANN_neuron(v_thresh, 0, 0)  # zero pertubation, IF
            for postSynNeuron in output:
                # this needs to add a neuron type
                self.neuron_dict[str(postSynNeuron)] = ([], lifNeuronModel)
                self.output_neurons.append(str(postSynNeuron))

        self.curr_input = output
        self.snn_layer_index += 1
        print(
            f"Number of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}"
        )

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
        lifNeuronModel = LIF_neuron(v_thresh, 0, 63)  # zero pertubation, IF

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
            self.neuron_offset += np.prod(self.curr_input.shape)

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

        self._conv_weight(self.curr_input, output, layer, v_thresh)

        if layer.bias is not None:
            layer_name = self._get_layer_name(layer)
            self._cri_bias(layer, output, layer_name=layer_name)
            print(
                f"Constructing {len(self.axon_dict) - self.axon_offset} bias axons from conv layer."
            )
            self.axon_offset = len(self.axon_dict)
        else:
            self.bias_dict.append(self.NULL_INDICIES)

        if self.layer_index == self.output_layer:
            print("Instantiate output neurons from conv layer")
            for postSynNeuron in output:
                self.neuron_dict[str(postSynNeuron)] = []  # fix me
                self.output_neurons.append(str(postSynNeuron))

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

        lifNeuronModel = LIF_neuron(v_thresh, 0, 63)  # zero pertubation, IF

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

    def _cri_bias(self, layer, outputs, layer_name=None, atten_flag=False):
        biases = layer.bias.detach().cpu().numpy()
        self.bias_dict.append((self.axon_offset, self.axon_offset + biases.size))
        
        # Track layer-specific bias axons for proper activation timing
        if layer_name is not None:
            bias_axon_ids = []

        if isinstance(layer, nn.Conv2d):
            for output_chan, bias in enumerate(biases):
                bias_id = "a" + str(output_chan + self.axon_offset)
                self.axon_dict[bias_id] = [
                    (str(neuron_idx), int(bias))
                    for neuron_idx in outputs[output_chan].flatten()
                ]
                # Record layer-specific bias axon
                if layer_name is not None:
                    bias_axon_ids.append(bias_id)

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
                # Record layer-specific bias axon
                if layer_name is not None:
                    bias_axon_ids.append(bias_id)
        else:
            print(f"Unspported layer: {layer}")
        
        # Store the layer-specific bias axons
        if layer_name is not None and bias_axon_ids:
            self.layer_bias_axons[layer_name] = bias_axon_ids
            print(f"Recorded {len(bias_axon_ids)} bias axons for layer {layer_name}")

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
        per_inference_metrics = []

        output_idx = [i for i in range(len(self.output_neurons))]

        runcount = 0

        # each image
        for img_idx, currInput in enumerate(inputList):
            # initiate the hardware for each image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(self.neuron_dict), False, 0
            )  ##Num_neurons, simDump, coreOverride
            spikeRate = [0] * len(self.output_neurons)
            total_cycles = 0
            total_hbm = 0
            # each time step
            phaseDelay = self.snn_layers
            for sliceIdx, slice in enumerate(currInput):
                hwSpike = []
                if outputPotential:
                    potential, spikes = hardwareNetwork.step(
                        slice, membranePotential=True
                    )
                    hwSpike, step_cycles, step_hbm = spikes
                    try:
                        total_cycles += int(step_cycles)
                        total_hbm += int(step_hbm)
                    except Exception:
                        pass
                    if sliceIdx >= phaseDelay:
                        membranePotential.append([v for k, v in potential])
                else:
                    hwSpike, step_cycles, step_hbm = hardwareNetwork.step(
                        slice, membranePotential=False
                    )
                    try:
                        total_cycles += int(step_cycles)
                        total_hbm += int(step_hbm)
                    except Exception:
                        pass
                    runcount += 1
                if sliceIdx >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in hwSpike
                    ]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        # Checking if the output spike is in the defined output neuron
                        if idx not in output_idx:
                            print(f"Error: invalid output spike {idx}")
                        spikeRate[idx] += 1
            for q in range(phaseDelay):
                hwSpike, step_cycles, step_hbm = hardwareNetwork.step(
                    [], membranePotential=False
                )
                try:
                    total_cycles += int(step_cycles)
                    total_hbm += int(step_hbm)
                except Exception:
                    pass
                if sliceIdx + q >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in hwSpike
                    ]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        if idx not in output_idx:
                            print(f"Error: invalid output spike {idx}")
                        spikeRate[idx] += 1
            # Append the output spikes of each image to the output list
            # record per-inference metrics
            per_inference_metrics.append(
                {
                    "image_index": img_idx,
                    "steps": len(currInput) + phaseDelay,
                    "latency_cycles_total": int(total_cycles),
                    "hbm_accesses_total": int(total_hbm),
                }
            )
            print(f"cycles={total_cycles} hbm={total_hbm}")
            debugspike = []
            outputSpikes.append(spikeRate)
        print("runcount: " + str(runcount))

        # Save metrics CSV if configured
        if self.metrics_log_path and per_inference_metrics:
            try:
                os.makedirs(os.path.dirname(self.metrics_log_path), exist_ok=True)
                file_exists = os.path.exists(self.metrics_log_path)
                with open(self.metrics_log_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "image_index",
                            "steps",
                            "latency_cycles_total",
                            "hbm_accesses_total",
                        ],
                    )
                    if not file_exists:
                        writer.writeheader()
                    for row in per_inference_metrics:
                        writer.writerow(row)
                print(f"Saved HBM/cycle metrics to {self.metrics_log_path}")
            except Exception as e:
                print(f"Warning: failed to save metrics CSV: {e}")

        if outputPotential:
            return outputSpikes, membranePotential
        else:
            return outputSpikes

    def run_CRI_sw(self, inputList, softwareNetwork, outputPotential=False):
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

        # each image
        for currInput in tqdm(inputList):
            # reset the membrane potential to zero
            softwareNetwork.simpleSim.initialize_sim_vars(len(self.neuron_dict))
            spikeRate = [0] * len(self.output_neurons)
            # each time step
            # we need to add 5 delays
            phaseDelay = (
                self.snn_layers
            )  # it will take phaseDelay cycles before valid input comes out of the network
            for sliceIdx, slice in enumerate(currInput):
                swSpike = []

                if (
                    outputPotential
                ):  # TODO: we shouldn't actually bother reading membrane potentials out if <phaseDelay
                    potential, swSpike = softwareNetwork.step(
                        slice, membranePotential=True
                    )
                    if sliceIdx >= phaseDelay:
                        membranePotential.append([v for k, v in potential])
                else:
                    swSpike = softwareNetwork.step(slice, membranePotential=False)
                if sliceIdx >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in swSpike
                    ]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        spikeRate[idx] += 1
            # empty input for phase delay
            for q in range(phaseDelay):
                swSpike = softwareNetwork.step([], membranePotential=False)
                if sliceIdx + q >= phaseDelay:
                    spikeIdx = [
                        int(spike) - int(self.output_neurons[0]) for spike in swSpike
                    ]
                    debugspike.append(spikeIdx)
                    for idx in spikeIdx:
                        spikeRate[idx] += 1

            # empty input for output delay
            # spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
            # for idx in spikeIdx:
            #    spikeRate[idx] += 1
            # Append the output spikes of each image to the output list
            outputSpikes.append(spikeRate)

        if outputPotential:
            return outputSpikes, membranePotential
        else:
            return outputSpikes
