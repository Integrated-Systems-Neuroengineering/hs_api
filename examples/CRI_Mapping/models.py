import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from quant.quant_layer import QuantConv2d, QuantLinear
from copy import deepcopy

class CIFAR10(nn.Module):
    def __init__(self, T: int, channels: int, features: int):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(channels, channels*features, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Conv2d(channels*features, channels*features*2, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*2),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.Conv2d(channels*features*2, channels*features*4, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*4),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # 14 * 14
            
        
        layer.Conv2d(channels*features*4, channels*features*8, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*8),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Conv2d(channels*features*8, channels*features*16, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*16),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.Conv2d(channels*features*16, channels*features*8, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels*features*8),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
            
        layer.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),  # 14 * 14

        layer.Flatten(),
        layer.Dropout(0.5),
        layer.Linear(channels*features*8*64, 1000, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        
        layer.Dropout(0.5),
        layer.Linear(1000, 100, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
    
        layer.Linear(100, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
        layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        # Quantized model
        # nn.identity() 
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        layer.AvgPool2d(2, 2),  # 14 * 14

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        layer.BatchNorm2d(channels),
        # Quantized model
        # nn.identity() 
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        layer.AvgPool2d(2, 2),  # 7 * 7

        layer.Flatten(),
        layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),

        layer.Linear(channels * 4 * 4, 10, bias=False),
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3]

#TODO: Train a small nn on mnist with residual connection and self attention
class Atten(nn.Module):
    def __init__(self, T=1, embed_dims=4):
        super().__init__()
        self.T = T
        self.embed_dims = embed_dims
        
        self.proj_conv = nn.Conv2d(1, self.embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.embed_dims)
        self.proj_lif = neuron.LIFNode(tau=2.0,surrogate_function=surrogate.ATan())
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
        # self.patch_embed = SPS(img_size_h=28,img_size_w=28, patch_size=4,in_channels=1,embed_dims=self.embed_dims)
        
        self.attn = SSA(dim=self.embed_dims,num_heads=1)
        self.linear = nn.Linear(self.embed_dims * 7 * 7,10)
        
    def forward(self,x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # print(x.shape)
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # TB, C, H, W
        # print(f"proj_conv.shape: {x.shape}")
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif(x).flatten(0, 1).contiguous() #TB, C, H, W
        x = self.avgpool(x)
        # print(f"maxpool.shape: {x.shape}")
        x = x.reshape(T, B, -1, self.embed_dims) #T, B, N, D
        
        # x = self.patch_embed(x)
        # print(x.shape)
        
        x = self.attn(x)
        # print(f"attn.shape: {x.shape}")
        x = self.linear(x.flatten(-2,-1))
        # print(f"linear: {x.shape}")
        
        return x.mean(0)
    
class FashionMnist(nn.Module):
    def __init__(self, T = 4, channels = 32):
        super().__init__()
        self.T = T
        self.layer = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )
        
    def forward(self, x):
        # x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x = self.layer(x)
        # fr = x_seq.mean(0)
        return x
    
class Mnist(nn.Module):
    def __init__(self, features = 1000, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, features, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(features, 10, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            )
        
    def forward(self, x):
        return self.layer(x)

class CNN(nn.Module):
    def __init__(self, channels = 8, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),  # 14 * 14
            
            layer.Flatten(),
            layer.Linear(channels * 14 * 14, 10, bias=False),
            spiking_neuron(**deepcopy(kwargs)),

            )
        
    def forward(self, x):
        return self.layer(x)
    
class CNN_stride(nn.Module):
    def __init__(self, input_channels = 1, channels = 8, 
                 kernel = 3, padding = 0, stride = 1, bias = False) -> None:
        super().__init__()
        self.conv = layer.Conv2d(in_channels=input_channels,
                                 out_channels=channels, 
                                 kernel_size=kernel, 
                                 padding=padding, 
                                 stride=stride, 
                                 bias=bias)
        self.flat = layer.Flatten()
        self.linear = layer.Linear(channels*28*28, 10, bias = False)
    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x
    
class CNN_MaxPool(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv = layer.Conv2d(1, channels, kernel_size=3, padding=1, bias = False)
        self.bn = layer.BatchNorm2d(channels)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.maxPool = layer.MaxPool2d(2, 2)
        self.flat = layer.Flatten()
        self.linear = layer.Linear(channels * 14 * 14, 10, bias=False)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
   
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.if1(x)
        
        # print(f"Before maxPool: {x[0,0,0:12,0:12]}")
        x = self.maxPool(x)
        # print(f"After maxPool: {x[0,0,0:6,0:6]}")
        # breakpoint()
        x = self.flat(x)
        # print(f"After flat: {x.shape}")
        x = self.linear(x)
        x = self.if2(x)
        return x
    
    def forward_first(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.if1(x)
        return x
    
    def forward_second(self, x):
        x = self.flat(x)
        x = self.linear(x)
        x = self.if2(x)
        return x
    
    def forward_maxPool(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.if1(x)
        x = self.maxPool(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, inputDim = 28, dim = 32, outputDim = 10, N = 4):
        super().__init__()
        self.dim = dim
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.scale = 0.125
        self.N = N
        
        self.q_linear = nn.Linear(self.inputDim* self.inputDim, self.dim*self.dim, bias = False)
        self.q_lif = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.k_linear = nn.Linear(self.inputDim* self.inputDim, self.dim*self.dim, bias = False)
        self.k_lif = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.v_linear = nn.Linear(self.inputDim* self.inputDim, self.dim*self.dim, bias = False)
        self.v_lif = neuron.IFNode(surrogate_function=surrogate.ATan())
        #Threshld should not be quantized for attn_lif
        self.attn_lif = neuron.IFNode(v_threshold = 0.5, surrogate_function=surrogate.ATan())
        
        self.flat = layer.Flatten()
        self.proj_linear = nn.Linear(self.dim * self.dim, self.outputDim, bias = False)
        self.proj_lif = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        
        # breakpoint()
        # 32*1*28*28
        B,C,H,W = x.shape 
        x_for_qkv = x.flatten(2,3)  # B, C, H*W
        q_linear_out = self.q_linear(x_for_qkv)  
        # q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q = self.q_lif(q_linear_out).reshape(B,C,self.N, -1)
        # q = q_linear_out.reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        # k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous()
        k = self.k_lif(k_linear_out).reshape(B,C,self.N, -1)
        # k = k_linear_out.reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        # v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T,B,C,N).contiguous()
        v = self.v_lif(v_linear_out).reshape(B,C,self.N, -1)
        # v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # breakpoint()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        x = attn @ v
        
        # x = x.transpose(2,3).reshape(B,C,self.dim,self.dim).contiguous()
        x = x.reshape(B, C, self.N, -1)
        
        x = self.attn_lif(x)
        
        x = self.flat(x)
        x = self.proj_lif(self.proj_linear(x))
        
        return x

    def forward_qkv(self, x):
        B,C,H,W = x.shape 
        x_for_qkv = x.flatten(2,3)  # B, C, H*W
        q_linear_out = self.q_linear(x_for_qkv)  
        q = self.q_lif(q_linear_out).reshape(B,C,self.N, -1)
       
        k_linear_out = self.k_linear(x_for_qkv)
        k = self.k_lif(k_linear_out).reshape(B,C,self.N, -1)

        v_linear_out = self.v_linear(x_for_qkv)
        v = self.v_lif(v_linear_out).reshape(B,C,self.N, -1)

        return q,k,v
    
    def forward_mul(self,x):
        B,C,H,W = x.shape 
        x_for_qkv = x.flatten(2,3)  # B, C, H*W
        q_linear_out = self.q_linear(x_for_qkv)  
        q = self.q_lif(q_linear_out).reshape(B,C,self.N, -1)
       
        k_linear_out = self.k_linear(x_for_qkv)
        k = self.k_lif(k_linear_out).reshape(B,C,self.N, -1)

        v_linear_out = self.v_linear(x_for_qkv)
        v = self.v_lif(v_linear_out).reshape(B,C,self.N, -1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # if(attn.sum() > 0):
        #     breakpoint()
        x = attn @ v
        return x.reshape(B,C,self.N,-1).contiguous()
    
    def forward_output(self, x):
        x = self.flat(x)
        x = self.proj_lif(self.proj_linear(x))
        
        return x
    
class QuantNMNISTNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv_fc = nn.Sequential(
            QuantConv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            QuantConv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            QuantLinear(channels * 8 * 8, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            QuantLinear(2048, 100),
            spiking_neuron(**deepcopy(kwargs)),
            layer.VotingLayer()
        )


    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class NMNISTNet(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False), 
            layer.BatchNorm2d(channels), 
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 8 * 8, 2048),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Dropout(0.5),
            layer.Linear(2048, 10),
            spiking_neuron(**deepcopy(kwargs))
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
    
class NMNIST_CNN(nn.Module):
    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, stride = 2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride = 2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))

        self.flat = nn.Flatten()
        self.drop1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(channels * 7 * 7, 2048)
        self.lif3 = spiking_neuron(**deepcopy(kwargs))
        self.drop2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 10)
        self.lif4 = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x: torch.Tensor):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.flat(x)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.lif3(x)
        x = self.drop2(x)
        x = self.linear2(x)
        x = self.lif4(x) 
        return x
    
class QuantMnist(nn.Module):
    def __init__(self, features = 1000):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            QuantLinear(28 * 28, features, bias=False, alpha=4.0),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            QuantLinear(features, 10, bias=False, alpha=4.0),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )
        
    def forward(self, x):
        return self.layer(x)
    
class QuantCSNN(nn.Module):
    def __init__(self, T: int, channels = 128):
        super().__init__()
        self.T = T
        self.conv_fc = nn.Sequential(
            QuantConv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14
            
            layer.Flatten(),
            QuantLinear(channels * 14 * 14, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
    def forward(self, x):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, encoder = 3, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        conv = []
        for i in range(encoder):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, stride = 2, padding=0, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))

        self.conv_fc = nn.Sequential(
            *conv,
            
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 15 * 15, 512),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 11),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

class DVS_IBM(nn.Module):
    def __init__(self, spiking_neuron: callable = None, **kwargs) -> None:
        super().__init__()
        self.conv0 = layer.Conv2d(in_channels=2, out_channels=6, kernel_size=3, stride=2, groups=1, bias=False)
        self.lif0 = spiking_neuron(**deepcopy(kwargs))
        self.conv1 = layer.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, groups=2, bias=False)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = layer.Conv2d(in_channels=12, out_channels=252, kernel_size=4, stride=2, groups=2, bias=False)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = layer.Conv2d(in_channels=252, out_channels=256, kernel_size=1, stride=1, groups=2, bias=False)
        self.lif3 = spiking_neuron(**deepcopy(kwargs))
        self.conv4 = layer.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, groups=2, bias=False)
        self.lif4 = spiking_neuron(**deepcopy(kwargs))
        self.conv5 = layer.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.lif5 = spiking_neuron(**deepcopy(kwargs))
        self.conv6 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif6 = spiking_neuron(**deepcopy(kwargs))
        self.conv7 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif7 = spiking_neuron(**deepcopy(kwargs))
        self.conv8 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif8 = spiking_neuron(**deepcopy(kwargs))
        self.conv9 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, groups=16, bias=False)
        self.lif9 = spiking_neuron(**deepcopy(kwargs))
        self.conv10 = layer.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.lif10 = spiking_neuron(**deepcopy(kwargs))
        self.conv11 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, groups=8, bias=False)
        self.lif11 = spiking_neuron(**deepcopy(kwargs))
        self.conv12 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, groups=8, bias=False)
        self.lif12 = spiking_neuron(**deepcopy(kwargs))
        self.conv13 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, groups=32, bias=False)
        self.lif13 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(in_features=1024, out_features=11)
        self.lif14 = spiking_neuron(**deepcopy(kwargs))
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.lif0(x)
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = self.conv3(x)
        x = self.lif3(x)
        x = self.conv4(x)
        x = self.lif4(x)
        x = self.conv5(x)
        x = self.lif5(x)
        x = self.conv6(x)
        x = self.lif6(x)
        x = self.conv7(x)
        x = self.lif7(x)
        x = self.conv8(x)
        x = self.lif8(x)
        x = self.conv9(x)
        x = self.lif9(x)
        x = self.conv10(x)
        x = self.lif10(x)
        x = self.conv11(x)
        x = self.lif11(x)
        x = self.conv12(x)
        x = self.lif12(x)
        x = self.conv13(x)
        x = self.lif13(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif14(x)
        return x
    
class IBM_DVS_net(nn.Module):
    def __init__(self, spiking_neuron: callable = None, **kwargs) -> None:
        super().__init__()
        self.conv0 = layer.Conv2d(in_channels=2, out_channels=6, kernel_size=3, stride=2, groups=1, bias=False)
        self.lif0 = spiking_neuron(**deepcopy(kwargs))
        self.conv1 = layer.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=2, groups=2, bias=False)
        self.lif1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = layer.Conv2d(in_channels=12, out_channels=252, kernel_size=4, stride=2, groups=2, bias=False)
        self.lif2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = layer.Conv2d(in_channels=252, out_channels=512, kernel_size=1, stride=1, groups=2, bias=False)
        self.lif3 = spiking_neuron(**deepcopy(kwargs))
        self.conv4 = layer.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, groups=2, bias=False)
        self.lif4 = spiking_neuron(**deepcopy(kwargs))
        self.conv5 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.lif5 = spiking_neuron(**deepcopy(kwargs))
        self.conv6 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif6 = spiking_neuron(**deepcopy(kwargs))
        self.conv7 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif7 = spiking_neuron(**deepcopy(kwargs))
        self.conv8 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, groups=4, bias=False)
        self.lif8 = spiking_neuron(**deepcopy(kwargs))
        self.conv9 = layer.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, groups=16, bias=False)
        self.lif9 = spiking_neuron(**deepcopy(kwargs))
        self.conv10 = layer.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.lif10 = spiking_neuron(**deepcopy(kwargs))
        self.conv11 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, groups=8, bias=False)
        self.lif11 = spiking_neuron(**deepcopy(kwargs))
        self.conv12 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, groups=8, bias=False)
        self.lif12 = spiking_neuron(**deepcopy(kwargs))
        self.conv13 = layer.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, groups=32, bias=False)
        self.lif13 = spiking_neuron(**deepcopy(kwargs))
        self.flat = layer.Flatten()
        self.linear = layer.Linear(in_features=1024, out_features=11)
        self.lif14 = spiking_neuron(**deepcopy(kwargs))
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.lif0(x)
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = self.conv3(x)
        x = self.lif3(x)
        x = self.conv4(x)
        x = self.lif4(x)
        x = self.conv5(x)
        x = self.lif5(x)
        x = self.conv6(x)
        x = self.lif6(x)
        x = self.conv7(x)
        x = self.lif7(x)
        x = self.conv8(x)
        x = self.lif8(x)
        x = self.conv9(x)
        x = self.lif9(x)
        x = self.conv10(x)
        x = self.lif10(x)
        x = self.conv11(x)
        x = self.lif11(x)
        x = self.conv12(x)
        x = self.lif12(x)
        x = self.conv13(x)
        x = self.lif13(x)
        x = self.flat(x)
        x = self.linear(x)
        x = self.lif14(x)
        return x
        
class NMNIST_SD(nn.Module):
    def __init__(self, in_channels = 2, channels=8, spiking_neuron: callable = None, **kwargs):
        super().__init__()
            
        self.conv_fc = nn.Sequential(
            layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=0, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),

            layer.Flatten(),
            layer.Linear(16*16*channels, 10),
            spiking_neuron(**deepcopy(kwargs)),

        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)