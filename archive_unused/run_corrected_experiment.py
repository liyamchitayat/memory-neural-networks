#!/usr/bin/env python3
"""
Corrected Single Experiment Runner
Run concept transfer experiment with the CORRECT class distributions:
- Source model: trained on [2,3,4,5,6,7,8,9] (8 classes)
- Target model: trained on [0,1,2,3,4,5,6,7] (8 classes)
- Transfer classes: [8,9] (from source to target)
- Shared classes: [2,3,4,5,6,7] (6 classes for alignment)
"""

import torch
import numpy as np
import logging
from experimental_framework import ExperimentRunner, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    print("=" * 70)
    print("CORRECTED NEURAL CONCEPT TRANSFER EXPERIMENT")
    print("=" * 70)
    print()
    print("CORRECTED Experiment Configuration:")
    print("- Source model trained on: [2,3,4,5,6,7,8,9]")
    print("- Target model trained on: [0,1,2,3,4,5,6,7]")
    print("- Shared classes: [2,3,4,5,6,7] (for alignment)")
    print("- Transfer classes: [8,9] (source → target)")
    print("- Architecture: WideNN → WideNN (same architecture)")
    print()
    
    # Set up configuration with corrected class sets
    config = ExperimentConfig(
        seed=RANDOM_SEED,
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
    
    # CORRECTED class sets - exactly as specified
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}  # Source model classes
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}  # Target model classes
    
    print("Validating CORRECTED class configuration...")
    shared_classes = source_classes.intersection(target_classes)
    transfer_classes = source_classes - target_classes
    
    print(f"✓ Source model trained on: {sorted(source_classes)}")
    print(f"✓ Target model trained on: {sorted(target_classes)}")
    print(f"✓ Shared classes (for alignment): {sorted(shared_classes)}")
    print(f"✓ Transfer classes (source→target): {sorted(transfer_classes)}")
    
    # Verify this matches the specification
    expected_shared = {2, 3, 4, 5, 6, 7}
    expected_transfer = {8, 9}
    assert shared_classes == expected_shared, f"Shared classes mismatch: {shared_classes} != {expected_shared}"
    assert transfer_classes == expected_transfer, f"Transfer classes mismatch: {transfer_classes} != {expected_transfer}"
    print("✅ Class configuration matches specification exactly!")
    print()
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Run the corrected experiment
    try:
        print("Starting CORRECTED experiment with training optimization...")
        results = runner.run_experiment_suite(
            experiment_name="CORRECTED_WideNN_source2-9_to_target0-7",
            source_arch="WideNN",
            target_arch="WideNN",
            source_classes=source_classes,
            target_classes=target_classes
        )
        
        if results:
            print("\n" + "=" * 70)
            print("CORRECTED EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Total successful runs: {len(results)}")
            
            # Display results for each transfer class
            for transfer_class in sorted(transfer_classes):
                class_results = [r for r in results if r.transfer_class == transfer_class]
                if class_results:
                    result = class_results[0]  # Single pair
                    print(f"\n🎯 TRANSFER CLASS {transfer_class} RESULTS:")
                    print(f"   Source Model Accuracy: {result.source_accuracy:.4f} (trained on {sorted(source_classes)})")
                    print(f"   Target Model Accuracy: {result.target_accuracy:.4f} (trained on {sorted(target_classes)})")
                    print(f"   Alignment Error: {result.alignment_error:.4f}")
                    
                    print(f"\n   📊 BEFORE Transfer (baseline):")
                    print(f"      Knowledge Transfer:  {result.before_metrics.knowledge_transfer:.4f} (should be ~0)")
                    print(f"      Specificity Transfer: {result.before_metrics.specificity_transfer:.4f}")
                    print(f"      Precision Transfer:   {result.before_metrics.precision_transfer:.4f}")
                    
                    print(f"\n   📈 AFTER Transfer (with optimization):")
                    print(f"      Knowledge Transfer:  {result.after_metrics.knowledge_transfer:.4f} (should improve!)")
                    print(f"      Specificity Transfer: {result.after_metrics.specificity_transfer:.4f}")
                    print(f"      Precision Transfer:   {result.after_metrics.precision_transfer:.4f}")
                    
                    print(f"\n   🔄 IMPROVEMENT:")
                    knowledge_improvement = result.after_metrics.knowledge_transfer - result.before_metrics.knowledge_transfer
                    specificity_change = result.after_metrics.specificity_transfer - result.before_metrics.specificity_transfer
                    precision_change = result.after_metrics.precision_transfer - result.before_metrics.precision_transfer
                    
                    print(f"      Knowledge Transfer:  {knowledge_improvement:+.4f} {'✅' if knowledge_improvement > 0.1 else '⚠️' if knowledge_improvement > 0.01 else '❌'}")
                    print(f"      Specificity Transfer: {specificity_change:+.4f}")
                    print(f"      Precision Transfer:   {precision_change:+.4f} {'✅' if abs(precision_change) < 0.05 else '⚠️'}")
            
            print(f"\n📁 Results saved to: experiment_results/")
            print("- Individual results: CORRECTED_WideNN_source2-9_to_target0-7_pair_*_class_*.json")
            print("- Combined results: CORRECTED_WideNN_source2-9_to_target0-7_all_results.json")
            print("- Summary statistics: CORRECTED_WideNN_source2-9_to_target0-7_summary.json")
            
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
    print("🎉 CORRECTED EXPERIMENT COMPLETED")
    print("This experiment now:")
    print("✅ Uses the correct class distributions as specified")
    print("✅ Includes optimization training for injection parameters")
    print("✅ Should show improved knowledge transfer results")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    exit(main())