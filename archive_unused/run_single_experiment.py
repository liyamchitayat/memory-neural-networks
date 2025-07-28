#!/usr/bin/env python3
"""
Single Experiment Runner
Run a single concept transfer experiment with the specified class distributions:
- Source model: trained on [2,3,4,5,6,7,8,9]
- Target model: trained on [0,1,2,3,4,5,6,7]
- Transfer classes: [8,9] (from source to target)
- Shared classes: [2,3,4,5,6,7]
"""

import torch
import numpy as np
import logging
from experimental_framework import ExperimentRunner, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("NEURAL CONCEPT TRANSFER - SINGLE EXPERIMENT")
    print("=" * 70)
    print()
    print("Experiment Configuration:")
    print("- Source classes: [2,3,4,5,6,7,8,9]")
    print("- Target classes: [0,1,2,3,4,5,6,7]")
    print("- Shared classes: [2,3,4,5,6,7]")
    print("- Transfer classes: [8,9]")
    print("- Architecture: WideNN -> WideNN (same architecture)")
    print()
    
    # Set up configuration
    config = ExperimentConfig(
        seed=42,
        max_epochs=5,
        min_accuracy_threshold=0.90,
        num_pairs=1,  # Single experiment
        batch_size=64,
        learning_rate=0.001,
        concept_dim=24,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Device: {config.device}")
    print(f"Random seed: {config.seed}")
    print()
    
    # Define class sets
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    print("Validating class configuration...")
    shared_classes = source_classes.intersection(target_classes)
    transfer_classes = source_classes - target_classes
    
    print(f"✓ Source classes: {sorted(source_classes)}")
    print(f"✓ Target classes: {sorted(target_classes)}")
    print(f"✓ Shared classes: {sorted(shared_classes)}")
    print(f"✓ Transfer classes: {sorted(transfer_classes)}")
    print()
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Run the experiment
    try:
        print("Starting experiment...")
        results = runner.run_experiment_suite(
            experiment_name="WideNN_8classes_to_WideNN_8classes",
            source_arch="WideNN",
            target_arch="WideNN",
            source_classes=source_classes,
            target_classes=target_classes
        )
        
        if results:
            print("\n" + "=" * 70)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Total successful runs: {len(results)}")
            
            # Display results for each transfer class
            for transfer_class in transfer_classes:
                class_results = [r for r in results if r.transfer_class == transfer_class]
                if class_results:
                    result = class_results[0]  # Single pair
                    print(f"\nTransfer Class {transfer_class}:")
                    print(f"  Source Accuracy: {result.source_accuracy:.4f}")
                    print(f"  Target Accuracy: {result.target_accuracy:.4f}")
                    print(f"  Alignment Error: {result.alignment_error:.4f}")
                    
                    print(f"\n  BEFORE Transfer:")
                    print(f"    Knowledge Transfer:  {result.before_metrics.knowledge_transfer:.4f}")
                    print(f"    Specificity Transfer: {result.before_metrics.specificity_transfer:.4f}")
                    print(f"    Precision Transfer:   {result.before_metrics.precision_transfer:.4f}")
                    
                    print(f"\n  AFTER Transfer:")
                    print(f"    Knowledge Transfer:  {result.after_metrics.knowledge_transfer:.4f}")
                    print(f"    Specificity Transfer: {result.after_metrics.specificity_transfer:.4f}")
                    print(f"    Precision Transfer:   {result.after_metrics.precision_transfer:.4f}")
                    
                    print(f"\n  IMPROVEMENT:")
                    knowledge_improvement = result.after_metrics.knowledge_transfer - result.before_metrics.knowledge_transfer
                    specificity_improvement = result.after_metrics.specificity_transfer - result.before_metrics.specificity_transfer
                    precision_change = result.after_metrics.precision_transfer - result.before_metrics.precision_transfer
                    
                    print(f"    Knowledge Transfer:  {knowledge_improvement:+.4f}")
                    print(f"    Specificity Transfer: {specificity_improvement:+.4f}")
                    print(f"    Precision Transfer:   {precision_change:+.4f}")
            
            print(f"\nResults saved to: experiment_results/")
            print("- Individual results: WideNN_8classes_to_WideNN_8classes_pair_*_class_*.json")
            print("- Combined results: WideNN_8classes_to_WideNN_8classes_all_results.json")
            print("- Summary statistics: WideNN_8classes_to_WideNN_8classes_summary.json")
            
        else:
            print("\n" + "=" * 70)
            print("EXPERIMENT FAILED")
            print("=" * 70)
            print("No successful results obtained. Check logs for details.")
            
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    return 0

if __name__ == "__main__":
    exit(main())