#!/usr/bin/env python3
"""
SpikingJelly-Compatible ReLU Nature CNN for 2-channel DVS input (84x84)
Uses ReLU activations which are fully compatible with SpikingJelly ann2snn converter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoBias84(nn.Module):
    """SpikingJelly-compatible Nature CNN DQN with ReLU activation function for 2-channel DVS input (84x84)"""
    
    def __init__(self, input_channels=2, n_actions=6):
        super(NoBias84, self).__init__()
        
        # Nature CNN architecture with ReLU, optimized for 2-channel DVS
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        # ReLU activation modules (fully compatible with SpikingJelly)
        self.relu1 = nn.ReLU()  # After conv1
        self.relu2 = nn.ReLU()  # After conv2
        self.relu3 = nn.ReLU()  # After conv3
        self.relu4 = nn.ReLU()  # After fc1
        
        # Calculate conv output size for 84x84 input
        conv_out_size = self._get_conv_out_size(input_channels, 84, 84)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512, bias=False)
        self.fc2 = nn.Linear(512, n_actions, bias=False)

        # Initialize weights
        self._initialize_weights()
        
    def _get_conv_out_size(self, input_channels, height, width):
        """Calculate output size of convolutional layers"""
        with torch.no_grad():
            x = torch.zeros(1, input_channels, height, width)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x)) 
            x = self.relu3(self.conv3(x))
            return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU"""
        print("=== Initializing ReLU Nature CNN 2Ch SJ-Compatible weights ===")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print(f"Initialized {m.__class__.__name__}: weight std = {m.weight.std().item():.4f}")
            elif isinstance(m, nn.Linear):
                # He initialization for linear layers with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                print(f"Initialized {m.__class__.__name__}: weight std = {m.weight.std().item():.4f}")
        
        print("=== ReLU 2Ch SJ-Compatible weight initialization complete ===")
    
    def forward(self, x):
        """Forward pass with ReLU activation - NO CONDITIONAL STATEMENTS"""
        # NOTE: Input preprocessing removed to avoid torch.fx tracing issues
        # Assumes input is already properly normalized float32 tensor
        
        # Convolutional layers with ReLU modules (SpikingJelly compatible)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)  # No activation on output (Q-values)
        
        return x

def copy_weights_from_silu_model(silu_model, relu_model):
    """Copy weights from SiLU model to ReLU model (activations are different but weights are same)"""
    print("=== Copying weights from SiLU model to ReLU model ===")
    
    # Copy conv layers
    relu_model.conv1.weight.data.copy_(silu_model.conv1.weight.data)
    relu_model.conv2.weight.data.copy_(silu_model.conv2.weight.data)
    relu_model.conv3.weight.data.copy_(silu_model.conv3.weight.data)
    
    # Copy FC layers
    relu_model.fc1.weight.data.copy_(silu_model.fc1.weight.data)
    relu_model.fc2.weight.data.copy_(silu_model.fc2.weight.data)
    
    print("=== Weight copying complete (SiLU->ReLU) ===")
    return relu_model