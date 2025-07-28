"""
Neural Network Architectures for Concept Transfer Experiments

This module implements the required WideNN and DeepNN architectures
as specified in the general requirements document.

Requirements:
- WideNN: 6 layers with a big 256 layer
- DeepNN: 8 layers with widest layer being 128
- Both should work on MNIST (28x28=784 input) and output 10 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WideNN(nn.Module):
    """
    Wide Neural Network Architecture (6 layers with max width 256)
    
    Architecture:
    Input (784) -> 256 -> 256 -> 256 -> 128 -> 64 -> Output (10)
    
    Features:
    - get_features(): Returns penultimate layer activations
    - classify_from_features(): Classifies from feature representations
    """
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 6-layer architecture with max width 256
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256) 
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer features (before final classification).
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            features: Feature representations [batch_size, 64]
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Forward through layers up to penultimate
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        
        features = F.relu(self.bn5(self.fc5(x)))  # Penultimate layer
        
        return features
    
    def classify_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify from feature representations.
        
        Args:
            features: Feature tensor [batch_size, 64]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        return self.fc6(features)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            return_features: If True, return features instead of logits
            
        Returns:
            output: Either logits or features depending on return_features
        """
        features = self.get_features(x)
        
        if return_features:
            return features
        
        return self.classify_from_features(features)


class DeepNN(nn.Module):
    """
    Deep Neural Network Architecture (8 layers with max width 128)
    
    Architecture:
    Input (784) -> 128 -> 128 -> 96 -> 96 -> 64 -> 64 -> 32 -> Output (10)
    
    Features:
    - get_features(): Returns penultimate layer activations
    - classify_from_features(): Classifies from feature representations
    """
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 8-layer architecture with max width 128
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(96)
        self.bn4 = nn.BatchNorm1d(96)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(32)
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer features (before final classification).
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            features: Feature representations [batch_size, 32]
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Forward through layers up to penultimate
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        
        features = F.relu(self.bn7(self.fc7(x)))  # Penultimate layer
        
        return features
    
    def classify_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify from feature representations.
        
        Args:
            features: Feature tensor [batch_size, 32]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        return self.fc8(features)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            return_features: If True, return features instead of logits
            
        Returns:
            output: Either logits or features depending on return_features
        """
        features = self.get_features(x)
        
        if return_features:
            return features
        
        return self.classify_from_features(features)


def create_model(architecture: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        architecture: Either 'WideNN' or 'DeepNN'
        **kwargs: Additional arguments for model construction
        
    Returns:
        model: Instantiated neural network model
    """
    if architecture == 'WideNN':
        return WideNN(**kwargs)
    elif architecture == 'DeepNN':
        return DeepNN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from ['WideNN', 'DeepNN']")


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model's architecture.
    
    Args:
        model: Neural network model
        
    Returns:
        info: Dictionary with model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get architecture type
    arch_type = model.__class__.__name__
    
    # Get feature dimension
    sample_input = torch.randn(1, 784)
    with torch.no_grad():
        features = model.get_features(sample_input)
        feature_dim = features.shape[1]
    
    return {
        'architecture': arch_type,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'feature_dimension': feature_dim,
        'input_dimension': model.input_dim,
        'output_classes': model.num_classes
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Network Architectures")
    print("=" * 50)
    
    # Test WideNN
    wide_model = WideNN()
    wide_info = get_model_info(wide_model)
    print(f"WideNN Info: {wide_info}")
    
    # Test DeepNN
    deep_model = DeepNN()
    deep_info = get_model_info(deep_model)
    print(f"DeepNN Info: {deep_info}")
    
    # Test forward pass
    batch_size = 4
    sample_input = torch.randn(batch_size, 784)
    
    print(f"\nTesting forward pass with batch size {batch_size}:")
    
    # WideNN
    wide_features = wide_model.get_features(sample_input)
    wide_logits = wide_model.classify_from_features(wide_features)
    wide_output = wide_model(sample_input)
    
    print(f"WideNN - Features shape: {wide_features.shape}, Logits shape: {wide_logits.shape}")
    print(f"WideNN - Direct output shape: {wide_output.shape}")
    
    # DeepNN
    deep_features = deep_model.get_features(sample_input)
    deep_logits = deep_model.classify_from_features(deep_features)
    deep_output = deep_model(sample_input)
    
    print(f"DeepNN - Features shape: {deep_features.shape}, Logits shape: {deep_logits.shape}")
    print(f"DeepNN - Direct output shape: {deep_output.shape}")
    
    print("\nArchitecture testing completed successfully!")