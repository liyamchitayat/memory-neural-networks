#!/usr/bin/env python3
"""
Shared knowledge effect experiment using the fixed experimental framework.
Tests how the amount of shared knowledge between networks affects transfer.
"""

import torch
import torch.nn as nn
import json
import sys
import copy
import numpy as np
from datetime import datetime

from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
from robust_balanced_transfer import RobustBalancedTransferSystem

class SharedKnowledgeWrappedModel(nn.Module):
    """Wrapper for evaluation in shared knowledge experiment."""
    def __init__(self, base_model, transfer_system, transfer_class):
        super().__init__()
        self.base_model = base_model
        self.transfer_system = transfer_system
        self.transfer_class = transfer_class
    
    def forward(self, x):
        if self.transfer_system is None:
            return self.base_model(x)
        
        x_flat = x.view(x.size(0), -1)
        
        if hasattr(self.transfer_system, 'transfer_concept'):
            try:
                return self.transfer_system.transfer_concept(x_flat, self.transfer_class)
            except Exception as e:
                print(f"Warning: transfer_concept failed: {e}")
        
        return self.base_model(x_flat)

def measure_accuracy(model, data_loader, target_classes):
    """Simple accuracy measurement."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(data.size(0), -1)
            
            # Filter for target classes
            mask = torch.tensor([label.item() in target_classes for label in labels])
            if mask.sum() == 0:
                continue
            
            filtered_data = data[mask]
            filtered_labels = labels[mask]
            
            outputs = model(filtered_data)
            _, predicted = torch.max(outputs, 1)
            
            for pred, true in zip(predicted, filtered_labels):
                if pred.item() == true.item():
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0.0

def parse_condition_string(condition_str):
    """Parse condition string into components."""
    # Split by comma, but handle brackets properly
    parts = []
    current = ""
    bracket_count = 0
    brace_count = 0
    
    for char in condition_str:
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        elif char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == ',' and bracket_count == 0 and brace_count == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    
    if current.strip():
        parts.append(current.strip())
    
    # Now parse each part
    # Parse source classes [0,1,2]
    source_str = parts[0].strip('[]')
    source_classes = set([int(x.strip()) for x in source_str.split(',')])
    
    # Parse target classes [2,3,4]
    target_str = parts[1].strip('[]')
    target_classes = set([int(x.strip()) for x in target_str.split(',')])
    
    # Parse transfer class
    transfer_class = int(parts[2].strip())
    
    # Parse shared classes {2}
    shared_str = parts[3].strip('{}')
    shared_classes = set([int(x.strip()) for x in shared_str.split(',')])
    
    return source_classes, target_classes, transfer_class, shared_classes

def run_shared_knowledge_experiment(condition_name, seed, config_params):
    """Run shared knowledge experiment."""
    
    print(f"🧠 Running SHARED KNOWLEDGE {condition_name} experiment (seed={seed})")
    
    # Set seed properly
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get condition details - FIXED: transfer class must be in source but not target
    conditions = {
        "low_shared": "[0,1,2],[2,3,4],0,{2}",  # Transfer class 0: source has it, target doesn't
        "medium_shared": "[0,1,2,3,4],[2,3,4,5,6],0,{2,3,4}",  # Transfer class 0: source has it, target doesn't
        "high_shared": "[0,1,2,3,4,5,6,7],[2,3,4,5,6,7,8,9],0,{2,3,4,5,6,7}"  # Transfer class 0: source has it, target doesn't
    }
    
    if condition_name not in conditions:
        raise ValueError(f"Unknown condition: {condition_name}")
    
    source_classes, target_classes, transfer_class, shared_classes = parse_condition_string(conditions[condition_name])
    
    # Calculate shared knowledge metrics
    source_size = len(source_classes)
    target_size = len(target_classes)
    shared_size = len(shared_classes)
    shared_percentage = shared_size / source_size
    
    config = ExperimentConfig(
        seed=seed,
        max_epochs=config_params['max_epochs'],
        batch_size=config_params['batch_size'],
        learning_rate=config_params['learning_rate'],
        concept_dim=config_params['concept_dim'],
        device='cpu'
    )
    
    try:
        # Create data (now uses proper seed-based sampling!)
        data_manager = MNISTDataManager(config)
        trainer = ModelTrainer(config)
        
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
            data_manager.get_data_loaders(source_classes, target_classes)
        
        # Use WideNN → WideNN for consistent architecture
        source_model = WideNN()
        target_model = WideNN()
        
        # Train models
        trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
        trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
        
        print(f"   🧠 Shared Knowledge: {shared_size}/{source_size} classes ({shared_percentage:.1%})")
        print(f"   📚 Source classes: {sorted(source_classes)}")
        print(f"   📚 Target classes: {sorted(target_classes)}")
        print(f"   🎯 Transfer class: {transfer_class}")
        print(f"   🔗 Shared classes: {sorted(shared_classes)}")
        print(f"   ✅ Models trained: Source={source_acc:.3f}, Target={target_acc:.3f}")
        
        # Critical: Clone before transfer
        target_before_transfer = copy.deepcopy(trained_target)
        
        # Setup transfer with fixed rho=0.5
        transfer_system = RobustBalancedTransferSystem(
            source_model=trained_source,
            target_model=trained_target,
            source_classes=source_classes,
            target_classes=target_classes,
            concept_dim=config.concept_dim,
            device=config.device
        )
        
        transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=config_params['sae_epochs'])
        transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
        
        # Evaluate
        target_after = SharedKnowledgeWrappedModel(trained_target, transfer_system, transfer_class)
        
        # Comprehensive accuracy measurements
        source_original = measure_accuracy(trained_source, source_test_loader, source_classes)
        source_transfer_class = measure_accuracy(trained_source, source_test_loader, {transfer_class})
        source_shared_classes = measure_accuracy(trained_source, source_test_loader, shared_classes)
        
        target_before_original = measure_accuracy(target_before_transfer, target_test_loader, target_classes)
        target_before_transfer_class = measure_accuracy(target_before_transfer, source_test_loader, {transfer_class})
        target_before_shared_classes = measure_accuracy(target_before_transfer, source_test_loader, shared_classes)
        
        target_after_original = measure_accuracy(target_after, target_test_loader, target_classes)
        target_after_transfer_class = measure_accuracy(target_after, source_test_loader, {transfer_class})
        target_after_shared_classes = measure_accuracy(target_after, source_test_loader, shared_classes)
        
        # Calculate key metrics
        transfer_improvement = target_after_transfer_class - target_before_transfer_class
        knowledge_preservation = target_after_original
        transfer_effectiveness = target_after_transfer_class
        shared_knowledge_retention = target_after_shared_classes
        
        print(f"   📊 Results Table:")
        print(f"      | Model           | Original | Transfer | Shared  |")
        print(f"      | Source          | {source_original:7.1%} | {source_transfer_class:7.1%} | {source_shared_classes:6.1%} |")
        print(f"      | Target (Before) | {target_before_original:7.1%} | {target_before_transfer_class:7.1%} | {target_before_shared_classes:6.1%} |")
        print(f"      | Target (After)  | {target_after_original:7.1%} | {target_after_transfer_class:7.1%} | {target_after_shared_classes:6.1%} |")
        print(f"   🎯 Key Metrics:")
        print(f"      Transfer improvement: {transfer_improvement:.1%}")
        print(f"      Knowledge preservation: {knowledge_preservation:.1%}")
        print(f"      Transfer effectiveness: {transfer_effectiveness:.1%}")
        print(f"      Shared knowledge retention: {shared_knowledge_retention:.1%}")
        
        # Validate clean setup
        if target_before_transfer_class > 0.3:
            print(f"   ⚠️  WARNING: Possible data leakage detected! Target knows transfer class {target_before_transfer_class:.1%}")
        else:
            print(f"   ✅ Clean experimental setup validated")
        
        # Return comprehensive results
        return {
            'experiment_id': f"shared_knowledge_{condition_name}_seed_{seed}",
            'condition_name': condition_name,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'experimental_setup': {
                'source_classes': sorted(source_classes),
                'target_classes': sorted(target_classes),
                'transfer_class': transfer_class,
                'shared_classes': sorted(shared_classes),
                'source_size': source_size,
                'target_size': target_size,
                'shared_size': shared_size,
                'shared_percentage': shared_percentage
            },
            'model_accuracies': {
                'source': source_acc,
                'target': target_acc
            },
            'detailed_results': {
                'source_original': source_original,
                'source_transfer_class': source_transfer_class,
                'source_shared_classes': source_shared_classes,
                'target_before_original': target_before_original,
                'target_before_transfer_class': target_before_transfer_class,
                'target_before_shared_classes': target_before_shared_classes,
                'target_after_original': target_after_original,
                'target_after_transfer_class': target_after_transfer_class,
                'target_after_shared_classes': target_after_shared_classes
            },
            'key_metrics': {
                'transfer_improvement': transfer_improvement,
                'knowledge_preservation': knowledge_preservation,
                'transfer_effectiveness': transfer_effectiveness,
                'shared_knowledge_retention': shared_knowledge_retention
            },
            'validation': {
                'data_leakage_detected': target_before_transfer_class > 0.3,
                'transfer_class_before_accuracy': target_before_transfer_class
            }
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_single_shared_knowledge_experiment.py <condition_name> <seed> <config_json>")
        sys.exit(1)
    
    condition_name = sys.argv[1]
    seed = int(sys.argv[2])
    config_params = json.loads(sys.argv[3])
    
    result = run_shared_knowledge_experiment(condition_name, seed, config_params)
    
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
