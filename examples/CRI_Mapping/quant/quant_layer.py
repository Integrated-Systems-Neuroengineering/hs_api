import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
import numpy as np

def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        #print('uniform quant bit: ', b)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()     # >1 means clipped. # output matrix is a form of [True, False, True, ...]
            sign = input.sign()              # output matrix is a form of [+1, -1, -1, +1, ...]
            #grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            grad_alpha = (grad_output*(sign*i + (0.0)*(1-i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, "grad_alpha = grad_output * (input_q-input)"
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, wgt_alpha):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.wgt_alpha = wgt_alpha
        self.weight_q = weight_quantization(b=self.w_bit)
        # self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))
        
    def forward(self, weight):
        
        # weights normalization
        # mean = weight.data.mean()
        # std = weight.data.std()
        # weight = weight.add(-mean).div(std)     
        
        weight_q = self.weight_q(weight, self.wgt_alpha)

        return weight_q

def act_quantization(b):

    def uniform_quant(x, b=4):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)  # Mingu edited for Alexnet
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            #grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_alpha = (grad_output * (i + (0.0)*(1-i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, num_inputs, num_outputs, kernel_size, padding, bias=False, bits = 16, alpha = 4.0):
        super(QuantConv2d, self).__init__(in_channels=num_inputs, out_channels=num_outputs, kernel_size=kernel_size, padding=padding, bias=bias)
        self.layer_type = 'QuantConv2d'
        self.bits = bits
        self.alpha = alpha
        self.weight_quant = weight_quantize_fn(w_bit=self.bits, wgt_alpha=self.alpha)
        self.act_alq = act_quantization(self.bits)
        self.act_alpha = nn.Parameter(torch.tensor(self.alpha))
        self.weight_q  = nn.Parameter(torch.zeros(self.weight.shape))
        
    def forward(self, x):
        # breakpoint()
        weight_q = self.weight_quant(self.weight)       
        #self.register_parameter('weight_q', Parameter(weight_q))  # Mingu added
        self.weight_q = nn.Parameter(weight_q)  # Store weight_q during the training
        x = self.act_alq(x, self.act_alpha)
        output = []
        for t in x:
            y = F.conv2d(t, self.weight_q, self.bias, padding=self.padding)
            output.append(y)
        
        return torch.stack(output, dim=0)
    
    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))
        
class QuantLinear(nn.Linear):
    def __init__(self, num_inputs, num_outputs, bits = 16, alpha = 4.0, bias=False):
        super(QuantLinear, self).__init__(num_inputs, num_outputs, bias)
        self.layer_type = 'QuantLinear'
        self.bits = bits
        self.alpha = alpha
        self.delta = self.alpha/(2**(self.bits-1)-1)
        self.weight_quant = weight_quantize_fn(w_bit=self.bits, wgt_alpha=self.alpha)
        self.act_alq = act_quantization(self.bits)
        self.act_alpha = nn.Parameter(torch.tensor(self.alpha))
        self.weight_q  = nn.Parameter(torch.zeros([num_inputs, num_outputs]))
        
    def forward(self, x):
        weight_q = self.weight_quant(self.weight)       
        #self.register_parameter('weight_q', Parameter(weight_q))  # Mingu added
        self.weight_q = nn.Parameter(weight_q/self.delta)  # Store weight_q during the training
        x = self.act_alq(x, self.act_alpha)/self.delta
        output = F.linear(x, self.weight_q, self.bias)
        return output
    
    def show_params(self):
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold weight alpha: {:2f}, activation alpha: {:2f}'.format(wgt_alpha, act_alpha))

