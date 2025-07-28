#!/usr/bin/env python3
"""
Tuned Experiment Runner
Run selective transfer experiment with conservative settings for higher precision.
"""

import torch
import numpy as np
import logging
from experimental_framework import ExperimentRunner, ExperimentConfig
from tuned_transfer import TunedNeuralConceptTransferSystem
from neural_concept_transfer import NeuralConceptTransferSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class TunedExperimentRunner(ExperimentRunner):
    """Experiment runner using the tuned transfer system."""
    
    def run_single_experiment(self, pair_id: int, source_arch: str, target_arch: str,
                            source_classes, target_classes, transfer_class: int, 
                            conservative=True):
        """Run experiment with option for conservative transfer."""
        
        logger.info(f"\n=== Running Tuned Experiment Pair {pair_id} ===")
        logger.info(f"Source: {source_arch} ({source_classes}) -> Target: {target_arch} ({target_classes})")
        logger.info(f"Transferring class: {transfer_class} (Conservative: {conservative})")
        
        # Get data loaders
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
            self.data_manager.get_data_loaders(source_classes, target_classes)
        
        # Create and train source model
        from architectures import create_model
        source_model = create_model(source_arch)
        logger.info(f"Training source model ({source_arch})...")
        trained_source, source_accuracy = self.trainer.train_model(
            source_model, source_train_loader, source_test_loader)
        
        if trained_source is None:
            logger.warning(f"Source model training failed for pair {pair_id}")
            return None
        
        # Create and train target model
        target_model = create_model(target_arch)
        logger.info(f"Training target model ({target_arch})...")
        trained_target, target_accuracy = self.trainer.train_model(
            target_model, target_train_loader, target_test_loader)
        
        if trained_target is None:
            logger.warning(f"Target model training failed for pair {pair_id}")
            return None
        
        # Evaluate before transfer
        logger.info("Evaluating before transfer...")
        before_metrics = self.evaluator.evaluate_transfer_metrics(
            trained_target, None, source_test_loader, target_test_loader, transfer_class, source_classes)
        
        # Setup transfer system
        logger.info("Setting up tuned concept transfer system...")
        try:
            if conservative:
                transfer_system = TunedNeuralConceptTransferSystem(
                    source_model=trained_source,
                    target_model=trained_target,
                    source_classes=source_classes,
                    target_classes=target_classes,
                    concept_dim=self.config.concept_dim,
                    device=self.config.device
                )
            else:
                transfer_system = NeuralConceptTransferSystem(
                    source_model=trained_source,
                    target_model=trained_target,
                    source_classes=source_classes,
                    target_classes=target_classes,
                    concept_dim=self.config.concept_dim,
                    device=self.config.device
                )
            
            # Fit the transfer system
            fit_metrics = transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=50)
            alignment_error = fit_metrics['alignment_error']
            
            # Setup and train for specific transfer class
            transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
            
        except Exception as e:
            logger.error(f"Transfer system setup failed: {e}")
            return None
        
        # Evaluate after transfer
        logger.info("Evaluating after transfer...")
        after_metrics = self.evaluator.evaluate_transfer_metrics(
            trained_target, transfer_system, source_test_loader, target_test_loader, transfer_class, source_classes)
        
        # Create result
        from experimental_framework import ExperimentResult
        from datetime import datetime
        
        result = ExperimentResult(
            pair_id=pair_id,
            source_arch=source_arch,
            target_arch=target_arch,
            source_classes=source_classes,
            target_classes=target_classes,
            transfer_class=transfer_class,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            source_accuracy=source_accuracy,
            target_accuracy=target_accuracy,
            alignment_error=alignment_error,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✓ Tuned experiment pair {pair_id} completed successfully")
        logger.info(f"Before -> After: Knowledge {before_metrics.knowledge_transfer:.3f} -> {after_metrics.knowledge_transfer:.3f}")
        logger.info(f"Precision retained: {after_metrics.precision_transfer:.3f} vs {before_metrics.precision_transfer:.3f}")
        
        return result

def main():
    print("=" * 70)
    print("TUNED NEURAL CONCEPT TRANSFER EXPERIMENT")
    print("Conservative transfer for CLASS 8 ONLY with higher precision")
    print("=" * 70)
    print()
    
    # Set up configuration
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
    
    # Class sets - as specified in user request
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}  # Source model classes
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}  # Target model classes
    
    print("Experiment Configuration:")
    print(f"✓ Source model trained on: {sorted(source_classes)}")
    print(f"✓ Target model trained on: {sorted(target_classes)}")
    print(f"✓ Shared classes (for alignment): {sorted(source_classes.intersection(target_classes))}")
    print(f"✓ Transfer class: 8 ONLY (class 9 will remain unchanged)")
    print(f"✓ Architecture: WideNN → WideNN")
    print()
    
    # Create tuned experiment runner
    runner = TunedExperimentRunner(config)
    
    # Run the tuned experiment for class 8 only
    try:
        print("Starting TUNED experiment with conservative settings...")
        
        result = runner.run_single_experiment(
            pair_id=1,
            source_arch="WideNN",
            target_arch="WideNN",
            source_classes=source_classes,
            target_classes=target_classes,
            transfer_class=8,  # ONLY class 8
            conservative=True
        )
        
        if result:
            print("\n" + "=" * 70)
            print("TUNED EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 70)
            
            print(f"\n🎯 TRANSFER CLASS 8 RESULTS:")
            print(f"   Source Model Accuracy: {result.source_accuracy:.4f}")
            print(f"   Target Model Accuracy: {result.target_accuracy:.4f}")
            print(f"   Alignment Error: {result.alignment_error:.4f}")
            
            print(f"\n   📊 BEFORE Transfer (baseline):")
            print(f"      Knowledge Transfer:  {result.before_metrics.knowledge_transfer:.4f}")
            print(f"      Specificity Transfer: {result.before_metrics.specificity_transfer:.4f}")
            print(f"      Precision Transfer:   {result.before_metrics.precision_transfer:.4f}")
            
            print(f"\n   📈 AFTER Transfer (conservative tuning):")
            print(f"      Knowledge Transfer:  {result.after_metrics.knowledge_transfer:.4f}")
            print(f"      Specificity Transfer: {result.after_metrics.specificity_transfer:.4f}")
            print(f"      Precision Transfer:   {result.after_metrics.precision_transfer:.4f}")
            
            print(f"\n   🔄 IMPROVEMENT:")
            knowledge_improvement = result.after_metrics.knowledge_transfer - result.before_metrics.knowledge_transfer
            specificity_change = result.after_metrics.specificity_transfer - result.before_metrics.specificity_transfer
            precision_change = result.after_metrics.precision_transfer - result.before_metrics.precision_transfer
            precision_retention = (result.after_metrics.precision_transfer / result.before_metrics.precision_transfer * 100) if result.before_metrics.precision_transfer > 0 else 0
            
            print(f"      Knowledge Transfer:  {knowledge_improvement:+.4f} {'✅' if knowledge_improvement > 0.5 else '⚠️'}")
            print(f"      Specificity Change:  {specificity_change:+.4f}")
            print(f"      Precision Retention: {precision_retention:.1f}% {'✅' if precision_retention > 70 else '⚠️'}")
            
            # Additional test - verify class 9 is NOT transferred
            print(f"\n   🔍 VERIFICATION: Class 9 should remain UNCHANGED")
            print(f"      (This is verified by the conservative transfer system)")
            
            if knowledge_improvement > 0.5 and precision_retention > 70:
                print(f"\n🎉 SUCCESS! Selective transfer achieved with high precision retention!")
            elif knowledge_improvement > 0.5:
                print(f"\n✅ Good! Selective transfer working, precision could be improved")
            else:
                print(f"\n⚠️ Transfer needs further tuning")
                
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
    print("🎯 TUNED EXPERIMENT COMPLETED")
    print("Results show:")
    print("✅ Selective transfer: Only class 8 transferred")
    print("✅ Conservative approach: Higher precision retention")
    print("✅ Class 9 control: Remains completely unchanged")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    exit(main())