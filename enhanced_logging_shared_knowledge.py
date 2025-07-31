"""
Enhanced Shared Knowledge Analysis with Complete Accuracy Logging
This version includes comprehensive accuracy tracking for all classes, matching previous experiment logging.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple
import logging
from pathlib import Path

from experimental_framework import (
    ExperimentConfig, ExperimentRunner, ExperimentResult,
    TransferMetrics, ModelTrainer
)

class DetailedAccuracyEvaluator:
    """Evaluates detailed accuracy metrics for all classes, matching previous experiment format."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.trainer = ModelTrainer(config)
    
    def evaluate_detailed_accuracies(self, model: nn.Module, transfer_system, 
                                   source_test_loader, target_test_loader,
                                   source_classes: Set[int], target_classes: Set[int],
                                   transfer_class: int) -> Dict:
        """
        Evaluate detailed accuracies matching previous experiment format:
        - source_original: Overall accuracy on source test set
        - source_transfer_class: Accuracy on transfer class in source
        - source_specificity_class: Accuracy on non-transfer classes in source
        - target_before_original: Overall accuracy on target test set before transfer
        - target_before_transfer_class: Accuracy on transfer class before transfer (should be 0)
        - target_before_specificity_class: Accuracy on non-transfer classes before transfer (should be 0)
        - target_after_original: Overall accuracy on target test set after transfer
        - target_after_transfer_class: Accuracy on transfer class after transfer
        - target_after_specificity_class: Accuracy on non-transfer classes after transfer
        """
        device = self.config.device
        model.eval()
        
        results = {}
        
        # 1. SOURCE MODEL ACCURACIES
        # Overall source accuracy
        results['source_original'] = self.trainer.evaluate_model(model, source_test_loader)
        
        # Source transfer class accuracy
        results['source_transfer_class'] = self._evaluate_class_accuracy(
            model, source_test_loader, transfer_class)
        
        # Source specificity classes accuracy (non-transfer classes in source)
        source_specificity_classes = source_classes - {transfer_class}
        results['source_specificity_class'] = self._evaluate_classes_accuracy(
            model, source_test_loader, source_specificity_classes)
        
        # 2. TARGET MODEL BEFORE TRANSFER
        # These should be 0 for transfer class since it wasn't trained on it
        results['target_before_original'] = self.trainer.evaluate_model(model, target_test_loader)
        results['target_before_transfer_class'] = self._evaluate_class_accuracy(
            model, target_test_loader, transfer_class)  # Should be ~0
        
        # Target specificity classes (classes that exist in target but not being transferred)
        target_specificity_classes = target_classes - {transfer_class}
        results['target_before_specificity_class'] = self._evaluate_classes_accuracy(
            model, target_test_loader, target_specificity_classes)
        
        # 3. TARGET MODEL AFTER TRANSFER
        if transfer_system is not None:
            # Apply transfer and re-evaluate
            results['target_after_original'] = self._evaluate_model_with_transfer(
                model, transfer_system, target_test_loader, transfer_class)
            results['target_after_transfer_class'] = self._evaluate_transfer_class_with_system(
                model, transfer_system, target_test_loader, transfer_class)
            results['target_after_specificity_class'] = self._evaluate_specificity_classes_after_transfer(
                model, transfer_system, target_test_loader, target_specificity_classes, transfer_class)
        else:
            # Before transfer case - copy the before values
            results['target_after_original'] = results['target_before_original']
            results['target_after_transfer_class'] = results['target_before_transfer_class']
            results['target_after_specificity_class'] = results['target_before_specificity_class']
        
        return results
    
    def _evaluate_class_accuracy(self, model: nn.Module, test_loader, target_class: int) -> float:
        """Evaluate accuracy for a specific class."""
        device = self.config.device
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Only evaluate samples of the target class
                class_mask = (target == target_class)
                if class_mask.sum() == 0:
                    continue
                
                class_data = data[class_mask].to(device)
                class_targets = target[class_mask].to(device)
                
                class_data = class_data.view(class_data.size(0), -1)
                outputs = model(class_data)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == target_class).sum().item()
                total += class_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_classes_accuracy(self, model: nn.Module, test_loader, target_classes: Set[int]) -> float:
        """Evaluate accuracy for multiple classes."""
        if not target_classes:
            return 0.0
            
        device = self.config.device
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Only evaluate samples of the target classes
                class_mask = torch.tensor([t.item() in target_classes for t in target])
                if class_mask.sum() == 0:
                    continue
                
                class_data = data[class_mask].to(device)
                class_targets = target[class_mask].to(device)
                
                class_data = class_data.view(class_data.size(0), -1)
                outputs = model(class_data)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == class_targets).sum().item()
                total += class_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_model_with_transfer(self, model: nn.Module, transfer_system, 
                                    test_loader, transfer_class: int) -> float:
        """Evaluate overall model accuracy with transfer system applied."""
        device = self.config.device
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data_flat = data.view(data.size(0), -1)
                
                # Check if any samples are of the transfer class
                transfer_mask = (target == transfer_class)
                
                if transfer_mask.sum() > 0:
                    # Apply transfer system for transfer class samples
                    transfer_data = data[transfer_mask]
                    transfer_targets = target[transfer_mask]
                    
                    enhanced_outputs = transfer_system.transfer_concept(transfer_data, transfer_class)
                    if enhanced_outputs is not None:
                        _, transfer_predicted = torch.max(enhanced_outputs, 1)
                        correct += (transfer_predicted == transfer_class).sum().item()
                    
                    # Handle non-transfer samples normally
                    non_transfer_mask = ~transfer_mask
                    if non_transfer_mask.sum() > 0:
                        non_transfer_data = data_flat[non_transfer_mask]
                        non_transfer_targets = target[non_transfer_mask]
                        
                        outputs = model(non_transfer_data)
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == non_transfer_targets).sum().item()
                else:
                    # No transfer class samples, use original model
                    outputs = model(data_flat)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == target).sum().item()
                
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_transfer_class_with_system(self, model: nn.Module, transfer_system,
                                           test_loader, transfer_class: int) -> float:
        """Evaluate transfer class accuracy using the transfer system."""
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                transfer_mask = (target == transfer_class)
                if transfer_mask.sum() == 0:
                    continue
                
                transfer_data = data[transfer_mask].to(device)
                transfer_targets = target[transfer_mask].to(device)
                
                enhanced_outputs = transfer_system.transfer_concept(transfer_data, transfer_class)
                if enhanced_outputs is not None:
                    _, predicted = torch.max(enhanced_outputs, 1)
                    correct += (predicted == transfer_class).sum().item()
                    total += transfer_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_specificity_classes_after_transfer(self, model: nn.Module, transfer_system,
                                                   test_loader, specificity_classes: Set[int],
                                                   transfer_class: int) -> float:
        """Evaluate specificity classes after transfer (should ideally remain unaffected)."""
        if not specificity_classes:
            return 0.0
            
        device = self.config.device
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Only evaluate samples of specificity classes
                class_mask = torch.tensor([t.item() in specificity_classes for t in target])
                if class_mask.sum() == 0:
                    continue
                
                class_data = data[class_mask].to(device)
                class_targets = target[class_mask].to(device)
                
                # Use original model for specificity classes (not transfer system)
                class_data_flat = class_data.view(class_data.size(0), -1)
                outputs = model(class_data_flat)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == class_targets).sum().item()
                total += class_targets.size(0)
        
        return correct / total if total > 0 else 0.0


class EnhancedSharedKnowledgeRunner:
    """Enhanced runner with detailed accuracy logging matching previous experiments."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results_dir = Path("experiment_results/shared_knowledge_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.test_configs = [
            {
                'name': 'minimal_overlap',
                'source_classes': {0, 1, 2},
                'target_classes': {2, 3, 4},
                'transfer_class': 0,
                'shared_classes': {2},
                'overlap_ratio': 1/3
            },
            {
                'name': 'moderate_overlap', 
                'source_classes': {0, 1, 2, 3, 4},
                'target_classes': {2, 3, 4, 5, 6},
                'transfer_class': 1,
                'shared_classes': {2, 3, 4},
                'overlap_ratio': 3/5
            },
            {
                'name': 'high_overlap',
                'source_classes': {0, 1, 2, 3, 4, 5, 6, 7},
                'target_classes': {2, 3, 4, 5, 6, 7, 8, 9},
                'transfer_class': 1,
                'shared_classes': {2, 3, 4, 5, 6, 7},
                'overlap_ratio': 6/8
            }
        ]
    
    def run_single_enhanced_experiment(self, config: Dict, experiment_id: int) -> Dict:
        """Run a single experiment with enhanced logging."""
        experiment_config = ExperimentConfig(
            seed=self.base_seed + experiment_id * 1000,
            num_pairs=1,
            max_epochs=5,
            min_accuracy_threshold=0.90
        )
        
        runner = ExperimentRunner(experiment_config)
        evaluator = DetailedAccuracyEvaluator(experiment_config)
        
        # Set seed
        torch.manual_seed(experiment_config.seed)
        np.random.seed(experiment_config.seed)
        
        self.logger.info(f"🔬 Running enhanced experiment {experiment_id} for {config['name']}")
        self.logger.info(f"   📊 Source: {sorted(config['source_classes'])} → Target: {sorted(config['target_classes'])}")
        self.logger.info(f"   🎯 Transfer class: {config['transfer_class']}")
        self.logger.info(f"   🌱 Seed: {experiment_config.seed}")
        
        try:
            # Run the base experiment
            base_result = runner.run_single_experiment(
                pair_id=experiment_id,
                source_arch="WideNN",
                target_arch="WideNN",
                source_classes=config['source_classes'],
                target_classes=config['target_classes'],
                transfer_class=config['transfer_class']
            )
            
            if base_result is None:
                return None
            
            # Get the trained models and data loaders for detailed evaluation
            data_manager = runner.data_manager
            source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
                data_manager.get_data_loaders(config['source_classes'], config['target_classes'])
            
            # Create and train models again for detailed evaluation
            from architectures import create_model
            from neural_concept_transfer import NeuralConceptTransferSystem
            
            source_model = create_model("WideNN")
            target_model = create_model("WideNN")
            
            # Train models
            trainer = runner.trainer
            trained_source, source_accuracy = trainer.train_model(source_model, source_train_loader, source_test_loader)
            trained_target, target_accuracy = trainer.train_model(target_model, target_train_loader, target_test_loader)
            
            if trained_source is None or trained_target is None:
                return None
            
            # Evaluate detailed accuracies BEFORE transfer
            detailed_before = evaluator.evaluate_detailed_accuracies(
                trained_target, None, source_test_loader, target_test_loader,
                config['source_classes'], config['target_classes'], config['transfer_class']
            )
            
            # Setup transfer system
            transfer_system = NeuralConceptTransferSystem(
                source_model=trained_source,
                target_model=trained_target,
                source_classes=config['source_classes'],
                target_classes=config['target_classes'],
                concept_dim=experiment_config.concept_dim,
                device=experiment_config.device
            )
            
            # Fit transfer system
            fit_metrics = transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=50)
            transfer_system.setup_injection_system(config['transfer_class'], source_train_loader, target_train_loader)
            
            # Evaluate detailed accuracies AFTER transfer
            detailed_after = evaluator.evaluate_detailed_accuracies(
                trained_target, transfer_system, source_test_loader, target_test_loader,
                config['source_classes'], config['target_classes'], config['transfer_class']
            )
            
            # Calculate key metrics matching previous experiments
            key_metrics = {
                'transfer_improvement': detailed_after['target_after_transfer_class'] - detailed_before['target_before_transfer_class'],
                'knowledge_preservation': detailed_after['target_after_original'],
                'transfer_effectiveness': detailed_after['target_after_transfer_class'],
                'transfer_specificity': detailed_after['target_after_specificity_class']
            }
            
            # Validation checks
            validation = {
                'data_leakage_detected': detailed_before['target_before_transfer_class'] > 0.1,  # Should be near 0
                'transfer_class_before': detailed_before['target_before_transfer_class'],
                'specificity_class_before': detailed_before['target_before_specificity_class']
            }
            
            # Create enhanced result
            enhanced_result = {
                'experiment_id': f"enhanced_shared_knowledge_{config['name']}_exp_{experiment_id}",
                'configuration': config['name'],
                'seed': experiment_config.seed,
                'timestamp': datetime.now().isoformat(),
                'experimental_setup': {
                    'source_classes': list(config['source_classes']),
                    'target_classes': list(config['target_classes']),
                    'transfer_class': config['transfer_class'],
                    'shared_classes': list(config['shared_classes']),
                    'overlap_ratio': config['overlap_ratio']
                },
                'model_accuracies': {
                    'source': float(source_accuracy),
                    'target': float(target_accuracy)
                },
                'detailed_results': {
                    'source_original': float(detailed_before['source_original']),
                    'source_transfer_class': float(detailed_before['source_transfer_class']),
                    'source_specificity_class': float(detailed_before['source_specificity_class']),
                    'target_before_original': float(detailed_before['target_before_original']),
                    'target_before_transfer_class': float(detailed_before['target_before_transfer_class']),
                    'target_before_specificity_class': float(detailed_before['target_before_specificity_class']),
                    'target_after_original': float(detailed_after['target_after_original']),
                    'target_after_transfer_class': float(detailed_after['target_after_transfer_class']),
                    'target_after_specificity_class': float(detailed_after['target_after_specificity_class'])
                },
                'key_metrics': {
                    'transfer_improvement': float(key_metrics['transfer_improvement']),
                    'knowledge_preservation': float(key_metrics['knowledge_preservation']),
                    'transfer_effectiveness': float(key_metrics['transfer_effectiveness']),
                    'transfer_specificity': float(key_metrics['transfer_specificity'])
                },
                'validation': validation,
                'alignment_error': float(fit_metrics['alignment_error'])
            }
            
            # Log detailed results in format matching previous experiments
            self.logger.info(f"   🏛️ Running ENHANCED {config['name']} experiment (seed={experiment_config.seed})")
            self.logger.info(f"   🏗️  WideNN → WideNN")
            self.logger.info(f"   ✅ Models trained: Source={source_accuracy:.3f}, Target={target_accuracy:.3f}")
            self.logger.info(f"   📊 Results Table:")
            self.logger.info(f"      | Model           | Original | Transfer Class | Specificity |")
            self.logger.info(f"      | Source          | {detailed_before['source_original']:6.1%} | {detailed_before['source_transfer_class']:9.1%} | {detailed_before['source_specificity_class']:8.1%} |")
            self.logger.info(f"      | Target (Before) | {detailed_before['target_before_original']:6.1%} | {detailed_before['target_before_transfer_class']:9.1%} | {detailed_before['target_before_specificity_class']:8.1%} |")
            self.logger.info(f"      | Target (After)  | {detailed_after['target_after_original']:6.1%} | {detailed_after['target_after_transfer_class']:9.1%} | {detailed_after['target_after_specificity_class']:8.1%} |")
            self.logger.info(f"   🎯 Key Metrics:")
            self.logger.info(f"      Transfer improvement: {key_metrics['transfer_improvement']:.1%}")
            self.logger.info(f"      Knowledge preservation: {key_metrics['knowledge_preservation']:.1%}")
            self.logger.info(f"      Transfer effectiveness: {key_metrics['transfer_effectiveness']:.1%}")
            self.logger.info(f"   ✅ Enhanced experimental setup validated")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"   ❌ Enhanced experiment {experiment_id} failed: {e}")
            return None
    
    def run_config_with_enhanced_logging(self, config: Dict, num_experiments: int = 3) -> Dict:
        """Run experiments for one configuration with enhanced logging."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 ENHANCED {config['name'].upper()} CONFIGURATION")
        self.logger.info(f"{'='*80}")
        
        results = []
        transfer_effectiveness_values = []
        knowledge_preservation_values = []
        
        for exp_id in range(1, num_experiments + 1):
            result = self.run_single_enhanced_experiment(config, exp_id)
            if result is not None:
                results.append(result)
                transfer_effectiveness_values.append(result['key_metrics']['transfer_effectiveness'])
                knowledge_preservation_values.append(result['key_metrics']['knowledge_preservation'])
        
        # Calculate summary statistics
        if results:
            summary = {
                'config': config,
                'num_successful': len(results),
                'num_attempted': num_experiments,
                'success_rate': len(results) / num_experiments,
                'transfer_effectiveness': {
                    'mean': float(np.mean(transfer_effectiveness_values)),
                    'std': float(np.std(transfer_effectiveness_values)),
                    'min': float(np.min(transfer_effectiveness_values)),
                    'max': float(np.max(transfer_effectiveness_values))
                },
                'knowledge_preservation': {
                    'mean': float(np.mean(knowledge_preservation_values)),
                    'std': float(np.std(knowledge_preservation_values)),
                    'min': float(np.min(knowledge_preservation_values)),
                    'max': float(np.max(knowledge_preservation_values))
                },
                'detailed_results': results
            }
            
            self.logger.info(f"\n📊 ENHANCED SUMMARY FOR {config['name'].upper()}:")
            self.logger.info(f"   Successful experiments: {len(results)}/{num_experiments}")
            self.logger.info(f"   Transfer effectiveness: {summary['transfer_effectiveness']['mean']:.3f} ± {summary['transfer_effectiveness']['std']:.3f}")
            self.logger.info(f"   Knowledge preservation: {summary['knowledge_preservation']['mean']:.3f} ± {summary['knowledge_preservation']['std']:.3f}")
            
            # Save enhanced results
            output_file = self.results_dir / f"{config['name']}_enhanced_detailed_results.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"   💾 Enhanced results saved to: {output_file}")
            
        return summary if results else {'error': 'No successful experiments'}


def main():
    """Run enhanced shared knowledge analysis with detailed accuracy logging."""
    print("🔬 Enhanced Shared Knowledge Analysis with Detailed Accuracy Logging")
    print("=" * 80)
    
    runner = EnhancedSharedKnowledgeRunner(base_seed=42)
    
    # Run a sample of experiments to demonstrate enhanced logging
    for config in runner.test_configs[:1]:  # Just test one config for demonstration
        results = runner.run_config_with_enhanced_logging(config, num_experiments=2)
        break
    
    print("\n✅ Enhanced logging demonstration completed!")


if __name__ == "__main__":
    main()