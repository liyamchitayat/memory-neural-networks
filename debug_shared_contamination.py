#!/usr/bin/env python3
"""
Test if shared layer training is contaminated by target network's full knowledge
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from experimental_framework import ExperimentConfig, MNISTDataManager
from final_shared_layer_experiment import SharedLayerTransferSystem

def test_contamination():
    """Test if target network leaks information about untransferred classes."""
    
    config = ExperimentConfig()
    data_manager = MNISTDataManager(config)
    
    # Create the problematic experiment
    source_classes = {0,1,2,3,4,5,6,7}
    target_classes = {2,3,4,5,6,7,8,9}
    
    # Create networks
    transfer_system = SharedLayerTransferSystem(device=config.device)
    source_network, target_network = transfer_system.create_networks("WideNN", "WideNN")
    
    # Train target network on its classes (including digit 9)
    target_train_loader, _, _, _ = data_manager.get_data_loaders(source_classes, target_classes)
    
    print("=== TARGET NETWORK KNOWLEDGE TEST ===")
    
    # Get some samples of digit 9
    digit_9_indices = []
    for idx, (_, label) in enumerate(data_manager.train_dataset):
        if label == 9:
            digit_9_indices.append(idx)
            if len(digit_9_indices) >= 100:
                break
    
    digit_9_subset = Subset(data_manager.train_dataset, digit_9_indices[:100])
    digit_9_loader = DataLoader(digit_9_subset, batch_size=32, shuffle=False)
    
    # Test: Can target network's features distinguish digit 9?
    target_network.use_original_classification()  # Use its trained classifier
    
    target_network.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in digit_9_loader:
            data = data.to(config.device).view(data.size(0), -1)
            target = target.to(config.device)
            
            output = target_network(data)
            _, predicted = torch.max(output, 1)
            
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f"Target network accuracy on digit 9: {accuracy:.3f}")
    
    if accuracy > 0.8:
        print("🐛 CONTAMINATION CONFIRMED!")
        print("   Target network's features carry strong information about digit 9.")
        print("   When shared layers train on 'shared classes' through target network,")
        print("   they inadvertently learn to classify digit 9 from target's features.")
    else:
        print("✅ No contamination detected")
    
    print()
    print("=== SOLUTION ===")
    print("Don't alternate networks for shared classes in large overlaps.")
    print("Use source network only for shared classes to avoid contamination.")

if __name__ == "__main__":
    test_contamination()