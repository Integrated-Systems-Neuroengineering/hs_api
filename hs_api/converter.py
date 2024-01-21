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

        layer.v_threshold = layer.v_threshold / self.w_delta
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
                        new_model._modules[
                            module_names[k - 1]
                        ] = folded_conv  # Replace the Convolutional Layer by the folded version

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
    HIGH_SYNAPSE_WEIGHT : float
        The high synapse weight value. Default is 1e6.
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
    save_input : bool
        Whether to save the input data.
    bias_start_idx : int or None
        The starting index for bias neurons.
    curr_input : np.ndarray or None
        The current input data.
    input_layer : int
        The index of the input layer.
    output_layer : int
        The index of the output layer.
    layer_index : int
        The current layer index.
    total_axonSyn : int
        The total number of axon synapses.
    total_neuronSyn : int
        The total number of neuron synapses.
    max_fan : int
        The maximum fan-out.
    v_threshold : float
        The voltage threshold for the neurons.
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

    HIGH_SYNAPSE_WEIGHT = 1e6
    NULL_NEURON = -1

    def __init__(
        self,
        num_steps,
        input_layer,
        output_layer,
        input_shape,
        v_threshold,
        embed_dim,
        backend="spikingjelly",
        dvs = False,
        converted_model_pth= "./converted_model",
    ):
        self.axon_dict = defaultdict(list)
        self.neuron_dict = defaultdict(list)
        self.output_neurons = []
        self.input_shape = np.array(input_shape)
        self.num_steps = num_steps
        self.axon_offset = 0
        self.neuron_offset = 0
        self.backend = backend
        self.save_input = False
        self.bias_start_idx = None
        self.curr_input = None
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layer_index = 0
        self.total_axonSyn = 0
        self.total_neuronSyn = 0
        self.max_fan = 0
        self.v_threshold = v_threshold
        self.converted_model_pth = converted_model_pth
        
        # For dvs datasets
        self.dvs = dvs

        # For spikformer only
        self.q = None
        self.v = None
        self.k = None
        self.embed_dim = embed_dim
        
    def save_model(self):
        """
        Save the converted model into three .pkl files: 
            axon_dict.pkl, 
            neuron_dict.pkl, 
            output_neurons.pkl
        """
        if not os.path.exists(self.converted_model_pth):
            os.makedirs(self.converted_model_pth)
            print(f'Mkdir {self.converted_model_pth}.')
            
        with open(f'{self.converted_model_pth}/axon_dict.pkl', 'wb') as f:
            pickle.dump(self.axon_dict, f)
        with open(f'{self.converted_model_pth}/neuron_dict.pkl', 'wb') as f:
            pickle.dump(self.neuron_dict, f)
        with open(f'{self.converted_model_pth}/output_neurons.pkl', 'wb') as f:
            pickle.dump(self.output_neurons, f)    
        print(f"Model saved at {self.converted_model_pth}.")

    def input_converter(self, input_data):
        """
        Converts input data into a spike train and then into a list of axon indices.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data.

        Returns
        -------
        list of list of str
            The batch of spikes, with each spike represented by its axon index.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.input_converter(some_input_data)
        """

        self.input_shape = input_data.shape
        print("input batch data shape: ", input_data.shape)
        return self._input_converter(input_data)

    def _input_converter(self, input_data):
        current_input = None
        if self.dvs:
            # Flatten the input data to [B, T, -1]
            current_input = input_data.view(input_data.size(0), input_data.size(1), -1)
        else:
            # Flatten the input data to [B, -1]
            current_input = input_data.view(input_data.size(0), -1)
            
        batch = []
        for img in current_input:
            spikes = []
            for step in range(self.num_steps):
                input_image = None
                if self.dvs:
                    input_image = img[step]
                else:
                    input_image = img
                input_spike = [
                    "a" + str(idx) for idx, axon in enumerate(input_image) if axon != 0
                ]
                # firing bias neurons at each step
                # bias_spike = [
                #     "a" + str(idx)
                #     for idx in range(self.bias_start_idx, len(self.axon_dict))
                # ]  
                spikes.append(input_spike)
            batch.append(spikes)
        
        # TODO: if we don't do rate encoding?
        if self.save_input:
            with open("/Volumes/export/isn/keli/code/CRI/data/cri_mnist.csv", "w") as f:
                write = csv.writer(f)
                # write.writerow(fields)
                write.writerows(batch)
        return batch

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

        # TODO: Construct the axon dict here
        # axons = np.array(
        #     ["a" + str(i) for i in range(np.prod(self.input_shape))]
        # ).reshape(self.input_shape)
        # self.curr_input = axons

        for k, name in enumerate(module_names):
            if len(list(model._modules[name]._modules)) > 0 and not isSNNLayer(
                model._modules[name]
            ):
                if name == "attn":
                    self._attention_converter(model._modules[name])
                else:
                    self.layer_converter(model._modules[name])
            else:
                self._layer_converter(model._modules[name], name)

    def _layer_converter(self, layer, name):
        # print(name, self.layer_index)
        if isinstance(layer, nn.Linear):
            self._linear_converter(layer)
        elif isinstance(layer, nn.Conv2d):
            self._conv_converter(layer)
        elif isinstance(layer, nn.MaxPool2d):
            self._maxPool_converter(layer)
        else:
            # print("Unsupported layer: ", layer)
            pass
            
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
            self.neuron_dict[str(output_neuron)] = []
            self.output_neurons.append(neuron_id)
        # print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')

    def _linear_converter(self, layer):
        if self.layer_index == 10:
            print('Constructing axons from linear Layer')
            print(f'Input layer shape(infeature, outfeature): {layer.in_features} {layer.out_features}')
            axon = np.array(
                [ i for i in range(self.axon_offset, self.axon_offset + layer.in_features)]
            )
            output = np.array(
                [ i for i in range(self.neuron_offset, self.neuron_offset + layer.out_features)]
            )
            self.axon_offset += layer.in_features
            self._linear_weight(axon, output, layer)
        else:
            print('Constructing neurons from linear Layer')
            print("Hidden layer shape(infeature, outfeature): ", layer.in_features, layer.out_features)
            output = np.array(
                [ i for i in range(self.neuron_offset, self.neuron_offset + layer.out_features)]
            )
            self.neuron_offset += layer.in_features
            self._linear_weight(self.curr_input, output, layer)
        
        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')

    def _linear_weight(self, input, outputs, layer):
        inputs = input.flatten()
        weight = layer.weight.detach().cpu().numpy().transpose()
        for neuron_idx, neuron in enumerate(weight):
            neuron_entry = [
                (str(base_postsyn_id + self.neuron_offset), int(syn_weight))
                for base_postsyn_id, syn_weight in enumerate(neuron)
                    if syn_weight != 0
            ]
            if self.layer_index == 10:
                neuron_id = str(neuron_idx + self.axon_offset)
                self.axon_dict[neuron_id] = neuron_entry
            else:
                neuron_id = str(neuron_idx + self.neuron_offset)
                self.neuron_dict[neuron_id] = neuron_entry
                
        if self.layer_index == 13:
            print('Instantiate output neurons')
            for output_neuron in range(layer.out_features):
                neuron_id = str(output_neuron + self.neuron_offset)
                self.neuron_dict[neuron_id] = []
                self.output_neurons.append(neuron_id)
            
        if layer.bias is not None:
            print(f'Constructing {layer.bias.shape[0]} bias axons for linear layer')
            self._cri_bias(layer, outputs)
            self.axon_offset = len(self.axon_dict)

    def _conv_converter(self, layer):
        # print(f'Converting layer: {layer}')
        input_shape, output_shape, axons, output = None, None, None, None
        start_time = time.time()
        if self.layer_index == 0:
            print('Constructing Axons from first Conv2d Layer')
            output_shape = self._conv_shape(layer, self.input_shape)
            print(f'Input layer shape(infeature, outfeature): {self.input_shape} {output_shape}')
            axons = np.array([i for i in range(np.prod(self.input_shape))]).reshape(
                self.input_shape
            )
            output = np.array([i for i in range(np.prod(output_shape))]).reshape(
                output_shape
            )
            self.axon_offset += np.prod(self.input_shape)
            self.neuron_offset += np.prod(output_shape)
            self._conv_weight(axons, output, layer)
        elif self.layer_index == 4:
            print('Constructing Axons from Second Conv2d Layer')
            output_shape = self._conv_shape(layer, self.curr_input.shape)
            print(f'Input layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
            axons = np.array(
                [i for i in range(
                    self.axon_offset, self.axon_offset + np.prod(self.curr_input.shape)
                )]
            ).reshape(self.curr_input.shape)
            output = np.array(
                [i for i in range(
                    self.neuron_offset, self.neuron_offset + np.prod(output_shape)
                )]
            ).reshape(output_shape)
            self.axon_offset += np.prod(self.curr_input.shape)
            self.neuron_offset += np.prod(output_shape)
            self._conv_weight(axons, output, layer)
        else:
            print('Constructing Neurons from Conv2d Layer')
            output_shape = self._conv_shape(layer, self.curr_input.shape)
            print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
            self.neuron_offset += np.prod(self.curr_input.shape)
            # print(f'Neuron_offset: {self.neuron_offset}')
            output = np.array(
                [i for i in range(
                    self.neuron_offset, self.neuron_offset + np.prod(output_shape)
                )]
            ).reshape(output_shape)
            # print(f'Last output: {output[-1][-1]}')
            self._conv_weight(self.curr_input, output, layer)

        #TODO: Fixing the bias issue
        # if layer.bias is not None:
        #     print(f'Constructing {layer.bias.shape[0]} bias axons from conv layer.')
        #     self._cri_bias(layer, output)
        #     self.axon_offset = len(self.axon_dict)

        self.curr_input = output
        print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')
        # print(f'Converting {layer} takes {time.time()-start_time}')

    def _conv_weight(self, input, output, layer):
        h_k, w_k = layer.kernel_size
        h_o, w_o = output.shape[-2], output.shape[-1]
        pad_top, pad_left = h_k // 2, w_k // 2
        filters = layer.weight.detach().cpu().numpy()
        h_i, w_i = input.shape[-2], input.shape[-1]

        input_padded = input
        if layer.padding != 0:
            # Pad the input with -1 (-1 since the input neuron idx starts at 0)
            # print(f'input shape: {input.shape}')
            dim = 2
            pad = layer.padding*dim
            input_padded = F.pad(torch.from_numpy(input), pad, value=self.NULL_NEURON).numpy()
            # print(f'input_padded shape: {input_padded.shape}')
        
        # for n in tqdm(range(input.shape[0])):
        for c in tqdm(range(input.shape[0])):
            for row in range(pad_top, h_i):
                for col in range(pad_left, w_i):
                    # Input axons/neurons
                    patch = input_padded[
                        c,
                        row - pad_top : row + pad_top + 1,
                        col - pad_left : col + pad_left + 1,
                    ]
                    for fil_idx, fil in enumerate(filters):
                        # print(fil.shape)
                        post_syn = output[fil_idx, row - pad_top, col - pad_left]
                        for i, neurons in enumerate(patch):
                            for j, neuron in enumerate(neurons):
                                if self.layer_index == 0 or self.layer_index == 4:
                                    if fil[c, i, j] != 0 and neuron != self.NULL_NEURON:
                                        self.axon_dict["a" + str(neuron)].append(
                                            (str(post_syn), int(fil[c, i, j]))
                                        )
                                else:
                                    if fil[c, i, j] != 0 and neuron != self.NULL_NEURON:
                                        self.neuron_dict[str(neuron)].append(
                                            (str(post_syn), int(fil[c, i, j]))
                                        )
        
        print('Instantiate output neurons')
        for output_neuron in output.flatten():
            neuron_id = str(output_neuron)
            self.neuron_dict[neuron_id] = []
            self.output_neurons.append(neuron_id)

    def _maxPool_converter(self, layer):
        # print(f'Converting layer: {layer}')
        # print('Constructing hidden maxpool layer')
        output_shape = self._maxPool_shape(layer, self.curr_input.shape)
        # print(f'Hidden layer shape(infeature, outfeature): {self.curr_input.shape} {output_shape}')
        # self.neuron_offset += np.prod(self.curr_input.shape)
        # print(f'Neuron_offset: {self.neuron_offset}')
        output = np.array(
            [i for i in range(np.prod(output_shape))]
        ).reshape(output_shape)
        # self._maxPool_weight(self.curr_input, output, layer)
        self.curr_input = output
        # print(f'Numer of neurons: {len(self.neuron_dict)}, number of axons: {len(self.axon_dict)}')

    def _maxPool_weight(self, input, output, layer):
        h_k, w_k = layer.kernel_size, layer.kernel_size
        h_o, w_o = output.shape[-2], output.shape[-1]
        h_i, w_i = input.shape[-2], input.shape[-1]
        pad_top, pad_left = h_k // 2, w_k // 2
        
        # double the weight of the synapse for maxpooling to make sure 
        # that the postsynaptic neuron will spike 
        scaler = self.v_threshold * 2  
        
        # print(h_i, w_i,input,output)
        for c in tqdm(range(input.shape[0])):
            for row in range(0, h_i, 2):
                for col in range(0, w_i, 2):
                    patch = input[c, row : row + pad_top + 1, col : col + pad_left + 1]
                    post_syn = str(output[c, row // 2, col // 2])
                    for i, neurons in enumerate(patch):
                        for j, neuron in enumerate(neurons):
                            self.neuron_dict[str(neuron)].append((post_syn, scaler))

        print('Instantiate output neurons')
        for output_neuron in output.flatten():
            neuron_id = str(output_neuron)
            if self.layer_index == 1:
                self.maxPool1Neuron[neuron_id] = []
                self.maxPool1Output.append(neuron_id)
            elif self.layer_index == 5:
                self.maxPool2Neuron[neuron_id] = []
                self.maxPool2Output.append(neuron_id)

    def _cri_bias(self, layer, outputs, atten_flag=False):
        biases = layer.bias.detach().cpu().numpy()
        for bias_idx, bias in enumerate(biases):
            bias_id = 'a' + str(bias_idx + self.axon_offset)
            if isinstance(layer, nn.Conv2d):
                self.axon_dict[bias_id] = [
                    (str(neuron_idx), int(bias)) 
                    for neuron_idx in outputs[bias_idx].flatten()
                ]
            elif isinstance(layer, nn.Linear):
                if atten_flag:
                    self.axon_dict[bias_id] = [
                        (str(neuron_idx), int(bias))
                        for neuron_idx in outputs[bias_idx, :].flatten()
                    ]
                else:
                    self.axon_dict[bias_id] = [(str(outputs[bias_idx]), int(bias))]
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

    def run_CRI_hw(self, inputList, hardwareNetwork):
        """
        Runs a set of inputs through the hardware implementation of the network and gets the output predictions.

        Parameters
        ----------
        inputList : list of list of str
            The input data, where each item is a list of axon indices representing the spikes.
        hardwareNetwork : object
            The hardware network object.

        Returns
        -------
        list of int
            The output predictions.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.run_CRI_hw(some_inputList, some_hardwareNetwork)
        """
        import hs_bridge
        
        predictions = []
        # each image
        total_time_cri = 0
        output_idx = [i for i in range(10)]
        for currInput in inputList:
            # initiate the hardware for each image
            hs_bridge.FPGA_Execution.fpga_controller.clear(
                len(self.neuron_dict), False, 0
            )  ##Num_neurons, simDump, coreOverride
            spikeRate = [0] * 10
            # each time step
            for slice in currInput:
                start_time = time.time()
                hwSpike, latency, hbmAcc = hardwareNetwork.step(
                    slice, membranePotential=False
                )
                end_time = time.time()
                total_time_cri = total_time_cri + end_time - start_time
                spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in hwSpike]
                for idx in spikeIdx:
                    if idx not in output_idx:
                        print(f"Error: invalid output spike {idx}")
                    spikeRate[idx] += 1
            if self.num_steps == 1:
                # Empty input for output delay since HiAER spike only get spikes after the spikes have occurred
                hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
                spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in hwSpike]
                for idx in spikeIdx:
                    if idx not in output_idx:
                        print(f"Error: invalid output spike {idx}")
                    spikeRate[idx] += 1
            # Empty input for output delay 
            hwSpike, _, _ = hardwareNetwork.step([], membranePotential=False)
            spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in hwSpike]
            for idx in spikeIdx:
                if idx not in output_idx:
                    print(f"Error: invalid output spike {idx}")
                spikeRate[idx] += 1
            predictions.append(spikeRate.index(max(spikeRate)))
        return predictions

    def run_CRI_sw(self, inputList, softwareNetwork):
        """
        Runs a batch of inputs through the software simulation of the network and gets the output predictions.

        Parameters
        ----------
        inputList : list of list of str
            The input data, where each item is a list of axon indices representing the spikes.
        softwareNetwork : object
            The software network object.

        Returns
        -------
        list of int
            The output predictions.

        Examples
        --------
        >>> converter = CRI_Converter()
        >>> converter.run_CRI_sw(some_inputList, some_softwareNetwork)
        """

        predictions = []
        total_time_cri = 0
        # each image
        for currInput in tqdm(inputList):
            # reset the membrane potential to zero
            softwareNetwork.simpleSim.initialize_sim_vars(len(self.neuron_dict))
            spikeRate = [0] * 10
            # each time step
            for slice in currInput:
                start_time = time.time()
                swSpike = softwareNetwork.step(slice, membranePotential=False)
                end_time = time.time()
                total_time_cri = total_time_cri + end_time - start_time
                spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            # empty input for phase delay 
            if self.num_steps == 1:
                swSpike = softwareNetwork.step([], membranePotential=False)
                spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
                for idx in spikeIdx:
                    spikeRate[idx] += 1
            # empty input for output delay 
            swSpike = softwareNetwork.step([], membranePotential=False)
            spikeIdx = [int(spike) - int(self.output_neurons[0]) for spike in swSpike]
            for idx in spikeIdx:
                spikeRate[idx] += 1
            predictions.append(spikeRate.index(max(spikeRate)))
        return predictions