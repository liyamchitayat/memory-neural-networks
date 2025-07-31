"""
Parameter Counting for Neural Network Architectures
This script calculates the exact number of parameters in WideNN and DeepNN models.
"""

import torch
import torch.nn as nn
from architectures import WideNN, DeepNN

def count_parameters(model):
    """Count parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def detailed_parameter_breakdown(model):
    """Provide detailed breakdown of parameters by layer."""
    print(f"\n📊 DETAILED PARAMETER BREAKDOWN: {model.__class__.__name__}")
    print("-" * 60)
    
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:20s}: {param_count:>10,} parameters {list(param.shape)}")
    
    print("-" * 60)
    print(f"{'TOTAL':20s}: {total_params:>10,} parameters")
    
    return total_params

def manual_calculation():
    """Manual calculation to verify automated counting."""
    print("\n🧮 MANUAL PARAMETER CALCULATION")
    print("=" * 60)
    
    print("\n📏 WideNN Architecture:")
    print("Input (784) → 256 → 256 → 256 → 128 → 64 → Output (10)")
    print("\nLinear Layer Parameters:")
    
    wide_layers = [
        ("fc1", 784, 256),
        ("fc2", 256, 256), 
        ("fc3", 256, 256),
        ("fc4", 256, 128),
        ("fc5", 128, 64),
        ("fc6", 64, 10)
    ]
    
    wide_total = 0
    for name, in_dim, out_dim in wide_layers:
        weights = in_dim * out_dim
        biases = out_dim
        layer_params = weights + biases
        wide_total += layer_params
        print(f"  {name}: {in_dim:3d} × {out_dim:3d} + {out_dim:3d} = {layer_params:>7,} parameters")
    
    # BatchNorm parameters
    bn_layers_wide = [256, 256, 256, 128, 64]  # One for each hidden layer
    bn_params_wide = sum(2 * dim for dim in bn_layers_wide)  # 2 params per dimension (scale, shift)
    
    print(f"\nBatchNorm Parameters:")
    for i, dim in enumerate(bn_layers_wide, 1):
        print(f"  bn{i}: 2 × {dim:3d} = {2*dim:>7,} parameters (scale + shift)")
    
    print(f"\nWideNN Total:")
    print(f"  Linear layers: {wide_total:>10,}")
    print(f"  BatchNorm:     {bn_params_wide:>10,}")
    print(f"  TOTAL:         {wide_total + bn_params_wide:>10,}")
    
    print("\n📏 DeepNN Architecture:")
    print("Input (784) → 128 → 128 → 96 → 96 → 64 → 64 → 32 → Output (10)")
    print("\nLinear Layer Parameters:")
    
    deep_layers = [
        ("fc1", 784, 128),
        ("fc2", 128, 128),
        ("fc3", 128, 96),
        ("fc4", 96, 96),
        ("fc5", 96, 64),
        ("fc6", 64, 64),
        ("fc7", 64, 32),
        ("fc8", 32, 10)
    ]
    
    deep_total = 0
    for name, in_dim, out_dim in deep_layers:
        weights = in_dim * out_dim
        biases = out_dim
        layer_params = weights + biases
        deep_total += layer_params
        print(f"  {name}: {in_dim:3d} × {out_dim:3d} + {out_dim:3d} = {layer_params:>7,} parameters")
    
    # BatchNorm parameters
    bn_layers_deep = [128, 128, 96, 96, 64, 64, 32]  # One for each hidden layer
    bn_params_deep = sum(2 * dim for dim in bn_layers_deep)
    
    print(f"\nBatchNorm Parameters:")
    for i, dim in enumerate(bn_layers_deep, 1):
        print(f"  bn{i}: 2 × {dim:3d} = {2*dim:>7,} parameters (scale + shift)")
    
    print(f"\nDeepNN Total:")
    print(f"  Linear layers: {deep_total:>10,}")
    print(f"  BatchNorm:     {bn_params_deep:>10,}")
    print(f"  TOTAL:         {deep_total + bn_params_deep:>10,}")
    
    return wide_total + bn_params_wide, deep_total + bn_params_deep

def main():
    """Main function to count and display parameters."""
    print("🔍 NEURAL NETWORK PARAMETER ANALYSIS")
    print("=" * 80)
    
    # Create models
    wide_model = WideNN()
    deep_model = DeepNN()
    
    # Set to eval mode to avoid batch norm issues
    wide_model.eval()
    deep_model.eval()
    
    # Count parameters automatically
    wide_total, wide_trainable = count_parameters(wide_model)
    deep_total, deep_trainable = count_parameters(deep_model)
    
    print(f"\n📊 AUTOMATED PARAMETER COUNTING:")
    print("-" * 40)
    print(f"WideNN:  {wide_total:>10,} total parameters ({wide_trainable:,} trainable)")
    print(f"DeepNN:  {deep_total:>10,} total parameters ({deep_trainable:,} trainable)")
    
    # Detailed breakdown
    wide_detailed = detailed_parameter_breakdown(wide_model)
    deep_detailed = detailed_parameter_breakdown(deep_model)
    
    # Manual verification
    wide_manual, deep_manual = manual_calculation()
    
    # Comparison
    print(f"\n✅ VERIFICATION:")
    print("-" * 40)
    print(f"WideNN - Automated: {wide_total:>10,}, Manual: {wide_manual:>10,}, Match: {wide_total == wide_manual}")
    print(f"DeepNN - Automated: {deep_total:>10,}, Manual: {deep_manual:>10,}, Match: {deep_total == deep_manual}")
    
    # Summary comparison
    print(f"\n📈 ARCHITECTURE COMPARISON:")
    print("=" * 60)
    print(f"| Architecture | Layers | Max Width | Total Parameters | Feature Dim |")
    print(f"|--------------|--------|-----------|------------------|-------------|")
    print(f"| WideNN       |    6   |    256    | {wide_total:>13,} |      64     |")
    print(f"| DeepNN       |    8   |    128    | {deep_total:>13,} |      32     |")
    print(f"| Ratio        |  0.75  |   2.00    | {wide_total/deep_total:>13.2f} |    2.00     |")
    
    # Parameter efficiency analysis
    print(f"\n🎯 PARAMETER EFFICIENCY:")
    print("-" * 40)
    print(f"WideNN parameters per layer: {wide_total/6:>10,.0f}")
    print(f"DeepNN parameters per layer: {deep_total/8:>10,.0f}")
    print(f"Wide vs Deep efficiency:     {(deep_total/8)/(wide_total/6):>10.2f}x")
    
    # Memory analysis (assuming float32)
    wide_memory_mb = wide_total * 4 / (1024 * 1024)
    deep_memory_mb = deep_total * 4 / (1024 * 1024)
    
    print(f"\n💾 MEMORY REQUIREMENTS (Float32):")
    print("-" * 40)
    print(f"WideNN model size: {wide_memory_mb:>6.2f} MB")
    print(f"DeepNN model size: {deep_memory_mb:>6.2f} MB")
    print(f"Memory ratio:      {wide_memory_mb/deep_memory_mb:>6.2f}x")

if __name__ == "__main__":
    main()