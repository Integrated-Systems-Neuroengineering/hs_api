import copy
import torch
import torch.nn as nn
from converter.utils import isSNNLayer

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
            # grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            grad_alpha = (grad_output*(sign*i + (0.0)*(1-i))).sum()
            # above line, if i = True,  and sign = +1, "grad_alpha = grad_output * 1"
            #             if i = False, "grad_alpha = grad_output * (input_q-input)"
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, w_alpha):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit-1
        self.weight_q = weight_quantization(b=self.w_bit)
        self.wgt_alpha = w_alpha
    def forward(self, weight):
        weight_q = self.weight_q(weight, self.wgt_alpha)
        return weight_q

class Quantize_Network():
    def __init__(self, w_alpha = 1, dynamic_alpha = False):
        self.w_alpha = w_alpha #Range of the parameter (CSNN:4, Transformer: 5)
        self.dynamic_alpha = dynamic_alpha
        self.v_threshold = None
        self.w_bits = 16
        self.w_delta = self.w_alpha/(2**(self.w_bits-1)-1)
        self.weight_quant = weight_quantize_fn(self.w_bits, self.w_alpha)
    
    def quantize(self, model):
        new_model = copy.deepcopy(model)
        start_time = time.time()
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(new_model._modules[name]):
                print('Quantized: ',name)
                if name == 'block':
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    # if name == 'attn':
                    #     continue
                    new_model._modules[name] = self.quantize(new_model._modules[name])
            else:
                print('Quantized: ',name)
                if name == 'attn_lif':
                    continue
                quantized_layer = self._quantize(new_model._modules[name])
                new_model._modules[name] = quantized_layer
        
        end_time = time.time()
        print(f'Quantization time: {end_time - start_time}')
        return new_model
    
    def quantize_block(self, model):
        new_model = copy.deepcopy(model)
        module_names = list(new_model._modules)
        
        for k, name in enumerate(module_names):
            
            if len(list(new_model._modules[name]._modules)) > 0 and not isSNNLayer(new_model._modules[name]):
                if name.isnumeric() or name == 'attn' or name == 'mlp':
                    print('Block Quantized: ',name)
                    new_model._modules[name] = self.quantize_block(new_model._modules[name])
                else:
                    print('Block Unquantized: ', name)
            else:
                if name == 'attn_lif':
                    continue
                else:
                    new_model._modules[name] = self._quantize(new_model._modules[name])
        return new_model
    
    def _quantize(self, layer):
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
            self.w_alpha = abs(max(layer.weight.flatten()) - min(layer.weight.flatten()))
            self.w_delta = self.w_alpha/(2**(self.w_bits-1)-1)
            self.weight_quant = weight_quantize_fn(self.w_bits) #reinitialize the weight_quan
            self.weight_quant.wgt_alpha = self.w_alpha
        
        layer.weight = nn.Parameter(self.weight_quant(layer.weight))
        quantized_layer.weight = nn.Parameter(layer.weight/self.w_delta)
        
        if layer.bias is not None: #check if the layer has bias
            layer.bias = nn.Parameter(self.weight_quant(layer.bias))
            quantized_layer.bias = nn.Parameter(layer.bias/self.w_delta)
        
        
        return quantized_layer

    
    def _quantize_LIF(self,layer):
        
        layer.v_threshold = layer.v_threshold/self.w_delta
        self.v_threshold = layer.v_threshold
        
        return layer