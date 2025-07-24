"""
Neural Network Architecture Definitions for SAE Concept Injection Testing

This module defines all neural network architectures used in the comprehensive
testing framework for SAE-free concept injection methods.

Architectures:
- BaseNN: Standard reference architecture (128D penultimate layer)
- WideNN: Wide network (256D penultimate layer) 
- DeepNN: Deep network (additional layers, 64D penultimate layer)
- BottleneckNN: Narrow bottleneck (64D penultimate layer)
- PyramidNN: Pyramid structure (256→128→5 layers)

All architectures are designed for MNIST digit classification (5 classes: 0-4)
with standardized input/output dimensions for cross-architecture transfer testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNN(nn.Module):
    """
    Standard CNN Architecture - Reference Implementation
    
    Penultimate Layer: 128D
    Parameters: ~1.2M
    Use Case: Baseline architecture for same-architecture experiments
    """
    def __init__(self):
        super(BaseNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)    # 28x28 → 26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)   # 26x26 → 24x24
        
        # Regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)  # Penultimate layer (concept injection target)
        self.fc2 = nn.Linear(128, 5)     # Output layer (digits 0-4)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))                    # (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))                    # (batch, 64, 24, 24)
        x = F.max_pool2d(x, 2)                       # (batch, 64, 12, 12)
        x = self.dropout1(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)                      # (batch, 9216)
        
        # Penultimate layer (target for concept injection)
        h = F.relu(self.fc1(x))                      # (batch, 128) - penultimate activations
        h = self.dropout2(h)
        
        # Output layer
        output = self.fc2(h)                         # (batch, 5)
        return output
    
    def get_penultimate_activations(self, x):
        """Extract penultimate layer activations for concept injection"""
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            h = F.relu(self.fc1(x))  # Penultimate activations
            return h
    
    @property
    def penultimate_dim(self):
        return 128


class WideNN(nn.Module):
    """
    Wide CNN Architecture - Increased Channel Width
    
    Penultimate Layer: 256D  
    Parameters: ~4.8M
    Use Case: Cross-architecture transfer testing (wide → narrow scenarios)
    """
    def __init__(self):
        super(WideNN, self).__init__()
        # Wider convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)    # Double width
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # Double width
        
        # Regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Wider fully connected layers
        self.fc1 = nn.Linear(18432, 256)  # Wider penultimate layer
        self.fc2 = nn.Linear(256, 5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))                    # (batch, 64, 26, 26)
        x = F.relu(self.conv2(x))                    # (batch, 128, 24, 24)
        x = F.max_pool2d(x, 2)                       # (batch, 128, 12, 12)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)                      # (batch, 18432)
        
        h = F.relu(self.fc1(x))                      # (batch, 256) - penultimate
        h = self.dropout2(h)
        
        output = self.fc2(h)                         # (batch, 5)
        return output
    
    def get_penultimate_activations(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            h = F.relu(self.fc1(x))
            return h
    
    @property
    def penultimate_dim(self):
        return 256


class DeepNN(nn.Module):
    """
    Deep CNN Architecture - Increased Depth
    
    Penultimate Layer: 64D
    Parameters: ~1.1M  
    Use Case: Cross-architecture transfer testing (deep network scenarios)
    """
    def __init__(self):
        super(DeepNN, self).__init__()
        # Additional convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)   # Additional conv layer
        
        # Regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Additional fully connected layers
        self.fc1 = nn.Linear(7744, 128)   # Adjusted for additional conv layer
        self.fc2 = nn.Linear(128, 64)     # Additional FC layer (penultimate)
        self.fc3 = nn.Linear(64, 5)       # Output layer
        
    def forward(self, x):
        x = F.relu(self.conv1(x))                    # (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))                    # (batch, 64, 24, 24)
        x = F.relu(self.conv3(x))                    # (batch, 64, 22, 22)
        x = F.max_pool2d(x, 2)                       # (batch, 64, 11, 11)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)                      # (batch, 7744)
        
        x = F.relu(self.fc1(x))                      # (batch, 128)
        x = self.dropout2(x)
        
        h = F.relu(self.fc2(x))                      # (batch, 64) - penultimate
        
        output = self.fc3(h)                         # (batch, 5)
        return output
    
    def get_penultimate_activations(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            h = F.relu(self.fc2(x))  # Penultimate activations
            return h
    
    @property
    def penultimate_dim(self):
        return 64


class BottleneckNN(nn.Module):
    """
    Bottleneck CNN Architecture - Narrow Representation
    
    Penultimate Layer: 64D
    Parameters: ~0.6M
    Use Case: Cross-architecture transfer testing (narrow bottleneck scenarios)
    """
    def __init__(self):
        super(BottleneckNN, self).__init__()
        # Standard convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Narrow bottleneck layers
        self.fc1 = nn.Linear(9216, 64)   # Narrow penultimate layer
        self.fc2 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))                    # (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))                    # (batch, 64, 24, 24)
        x = F.max_pool2d(x, 2)                       # (batch, 64, 12, 12)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)                      # (batch, 9216)
        
        h = F.relu(self.fc1(x))                      # (batch, 64) - penultimate
        h = self.dropout2(h)
        
        output = self.fc2(h)                         # (batch, 5)
        return output
    
    def get_penultimate_activations(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            h = F.relu(self.fc1(x))
            return h
    
    @property 
    def penultimate_dim(self):
        return 64


class PyramidNN(nn.Module):
    """
    Pyramid CNN Architecture - Gradual Dimension Reduction
    
    Penultimate Layer: 128D (middle of pyramid)
    Parameters: ~2.4M
    Use Case: Cross-architecture transfer testing (pyramid structure scenarios)
    """
    def __init__(self):
        super(PyramidNN, self).__init__()
        # Standard convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Pyramid fully connected layers
        self.fc1 = nn.Linear(9216, 256)  # Wide start
        self.fc2 = nn.Linear(256, 128)   # Pyramid middle (penultimate)
        self.fc3 = nn.Linear(128, 5)     # Narrow end
        
    def forward(self, x):
        x = F.relu(self.conv1(x))                    # (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))                    # (batch, 64, 24, 24)
        x = F.max_pool2d(x, 2)                       # (batch, 64, 12, 12)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)                      # (batch, 9216)
        
        x = F.relu(self.fc1(x))                      # (batch, 256)
        x = self.dropout2(x)
        
        h = F.relu(self.fc2(x))                      # (batch, 128) - penultimate
        
        output = self.fc3(h)                         # (batch, 5)
        return output
    
    def get_penultimate_activations(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            h = F.relu(self.fc2(x))  # Penultimate activations
            return h
    
    @property
    def penultimate_dim(self):
        return 128


# Architecture Registry for Easy Access
ARCHITECTURES = {
    'BaseNN': BaseNN,
    'WideNN': WideNN, 
    'DeepNN': DeepNN,
    'BottleneckNN': BottleneckNN,
    'PyramidNN': PyramidNN
}


def get_architecture(name):
    """Factory function to create architecture instances"""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[name]()


def get_architecture_info():
    """Return summary of all architectures for documentation"""
    info = {}
    for name, arch_class in ARCHITECTURES.items():
        model = arch_class()
        total_params = sum(p.numel() for p in model.parameters())
        info[name] = {
            'penultimate_dim': model.penultimate_dim,
            'total_parameters': total_params,
            'class': arch_class
        }
    return info


if __name__ == "__main__":
    # Print architecture summary
    print("Neural Network Architecture Summary")
    print("=" * 50)
    
    for name, info in get_architecture_info().items():
        print(f"{name}:")
        print(f"  Penultimate Dimension: {info['penultimate_dim']}D")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Use Case: Cross-architecture transfer testing")
        print()
    
    # Test forward pass for each architecture
    print("Testing forward pass with dummy input...")
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input shape
    
    for name in ARCHITECTURES.keys():
        model = get_architecture(name)
        output = model(dummy_input)
        penultimate = model.get_penultimate_activations(dummy_input)
        print(f"{name}: Output shape {output.shape}, Penultimate shape {penultimate.shape}")
    
    print("All architectures working correctly!")