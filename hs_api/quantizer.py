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
from hs_api.Krish_custom_neurons import Custom_LIFNode, Custom_IFNode
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
        or isinstance(layer, Custom_IFNode)
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
            #default dynamic_alpha:
            print("keli's dynamic alpha")
            self.w_alpha = abs(
                max(layer.weight.flatten()) - min(layer.weight.flatten()) 
            )

            # #krish dynamic alpha
            # print("krish's dynamic alpha: max(abs(layer.weight.flatten()))")
            # self.w_alpha = max(abs(layer.weight.flatten()))

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


