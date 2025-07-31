#!/usr/bin/env python3
"""
Compare training efficiency: Feature Bridging vs Training from Scratch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from experimental_framework import ExperimentConfig, MNISTDataManager, ModelTrainer
from architectures import WideNN
from final_shared_layer_experiment import FinalExperimentRunner

def train_from_scratch_baseline(config, classes_to_learn, epochs=15):
    """
    Baseline: Train a network from scratch on specific classes.
    This is what we're comparing our transfer learning against.
    """
    print(f"\n🏁 BASELINE: Training from scratch on classes {sorted(classes_to_learn)}")
    
    start_time = time.time()
    
    # Create fresh network
    network = WideNN().to(config.device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Trainable parameters (full network): {trainable_params:,}")
    
    # Get data for these classes
    data_manager = MNISTDataManager(config)
    train_indices = []
    for idx, (_, label) in enumerate(data_manager.train_dataset):
        if label in classes_to_learn:
            train_indices.append(idx)
    
    # Limit to reasonable size
    import numpy as np
    np.random.shuffle(train_indices)
    train_indices = train_indices[:10000]  # Same order of magnitude as transfer learning
    
    from torch.utils.data import Subset
    train_subset = Subset(data_manager.train_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    
    # Simple training loop
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    convergence_epoch = epochs
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(config.device), target.to(config.device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Check convergence
        if accuracy > 0.5 and convergence_epoch == epochs:
            convergence_epoch = epoch + 1
            print(f"🎯 BASELINE CONVERGENCE: Accuracy >50% at epoch {epoch + 1}")
        
        if epoch % 3 == 0 or epoch == epochs - 1:
            print(f"Baseline Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, "
                  f"Accuracy = {accuracy:.4f}, Time = {epoch_time:.1f}s")
    
    # Final measurements
    end_time = time.time()
    
    baseline_metrics = {
        'training_time_seconds': end_time - start_time,
        'trainable_parameters': trainable_params,
        'convergence_epoch': convergence_epoch,
        'final_accuracy': accuracy
    }
    
    print(f"🏁 BASELINE METRICS:")
    print(f"  Training time: {baseline_metrics['training_time_seconds']:.1f} seconds")
    print(f"  Trainable parameters: {baseline_metrics['trainable_parameters']:,}")
    print(f"  Convergence epoch: {baseline_metrics['convergence_epoch']}/{epochs}")
    print(f"  Final accuracy: {baseline_metrics['final_accuracy']:.3f}")
    
    return baseline_metrics

def run_efficiency_comparison():
    """Compare Feature Bridging transfer vs training from scratch."""
    
    config = ExperimentConfig()
    config.epochs = 2  # Quick test
    
    print("=" * 80)
    print("TRAINING EFFICIENCY COMPARISON")
    print("Feature Bridging Transfer Learning vs Training from Scratch")
    print("=" * 80)
    
    # Test case: Learn to classify digit 3
    transfer_classes = {3}
    
    # Method 1: Feature Bridging Transfer Learning
    print("\n🔄 METHOD 1: Feature Bridging Transfer Learning")
    runner = FinalExperimentRunner(config)
    
    transfer_result = runner.run_single_experiment(
        source_classes={0, 1, 2},
        target_classes={2, 3, 4},
        transfer_digit=3,
        source_arch='WideNN',
        target_arch='WideNN',
        seed=42,
        exp_name='efficiency_test_transfer'
    )
    
    if transfer_result and 'transfer_training' in transfer_result:
        transfer_metrics = transfer_result['transfer_training']
        transfer_accuracy = transfer_result['after_metrics']['network1_on_transferred']
        
        print(f"\n🔄 TRANSFER LEARNING RESULTS:")
        print(f"  Final transfer accuracy: {transfer_accuracy:.3f}")
        print(f"  Training time: {transfer_metrics.get('training_time_seconds', 0):.1f} seconds")
        print(f"  Trainable parameters: {transfer_metrics.get('trainable_parameters', 0):,}")
        print(f"  Convergence epoch: {transfer_metrics.get('convergence_epoch', 'N/A')}")
    else:
        print("❌ Transfer learning experiment failed")
        return
    
    # Method 2: Training from Scratch
    print("\n🏁 METHOD 2: Training from Scratch")
    baseline_metrics = train_from_scratch_baseline(config, transfer_classes, epochs=15)
    
    # Comparison
    print("\n" + "=" * 80)
    print("🏆 EFFICIENCY COMPARISON RESULTS")
    print("=" * 80)
    
    if transfer_metrics.get('training_time_seconds', 0) > 0:
        time_speedup = baseline_metrics['training_time_seconds'] / transfer_metrics['training_time_seconds']
        param_ratio = transfer_metrics['trainable_parameters'] / baseline_metrics['trainable_parameters']
        
        print(f"⏱️  Training Time:")
        print(f"   Transfer Learning: {transfer_metrics['training_time_seconds']:.1f}s")
        print(f"   From Scratch: {baseline_metrics['training_time_seconds']:.1f}s")
        print(f"   Speedup: {time_speedup:.2f}x {'✅' if time_speedup > 1 else '❌'}")
        
        print(f"\n🔢 Trainable Parameters:")
        print(f"   Transfer Learning: {transfer_metrics['trainable_parameters']:,}")
        print(f"   From Scratch: {baseline_metrics['trainable_parameters']:,}")
        print(f"   Ratio: {param_ratio:.3f}x {'✅' if param_ratio < 1 else '❌'}")
        
        print(f"\n🎯 Convergence:")
        print(f"   Transfer Learning: Epoch {transfer_metrics.get('convergence_epoch', 'N/A')}")
        print(f"   From Scratch: Epoch {baseline_metrics['convergence_epoch']}")
        
        print(f"\n📊 Final Performance:")
        print(f"   Transfer Learning: {transfer_accuracy:.3f} accuracy")
        print(f"   From Scratch: {baseline_metrics['final_accuracy']:.3f} accuracy")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        improvements = []
        if time_speedup > 1:
            improvements.append(f"{time_speedup:.1f}x faster training")
        if param_ratio < 0.1:
            improvements.append(f"{1/param_ratio:.1f}x fewer parameters")
        
        if improvements:
            print(f"   ✅ Transfer learning achieved: {', '.join(improvements)}")
        else:
            print(f"   ❌ Transfer learning did not improve efficiency")
        
        if transfer_accuracy > 0.5:
            print(f"   ✅ Transfer learning successfully learned the digit ({transfer_accuracy:.1%} accuracy)")
        else:
            print(f"   ❌ Transfer learning failed to learn the digit ({transfer_accuracy:.1%} accuracy)")
    
    else:
        print("❌ Could not compare - transfer learning metrics missing")

if __name__ == "__main__":
    run_efficiency_comparison()