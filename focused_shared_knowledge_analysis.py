"""
Focused Shared Knowledge Analysis with Previous Experiment Logging Format
This version replicates the exact logging format from previous experiments.
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

from experimental_framework import ExperimentRunner, ExperimentConfig
from architectures import create_model
from neural_concept_transfer import NeuralConceptTransferSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocusedSharedKnowledgeExperiment:
    """Focused experiment with exact logging format matching previous experiments."""
    
    def __init__(self):
        self.results_dir = Path("experiment_results/shared_knowledge_focused")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configurations
        self.configs = [
            {
                'name': 'minimal_overlap',
                'source_classes': {0, 1, 2},
                'target_classes': {2, 3, 4},
                'transfer_class': 0,
                'description': '[0,1,2] → [2,3,4] transfer 0 (33% overlap)'
            },
            {
                'name': 'moderate_overlap',
                'source_classes': {0, 1, 2, 3, 4},
                'target_classes': {2, 3, 4, 5, 6},
                'transfer_class': 1,
                'description': '[0,1,2,3,4] → [2,3,4,5,6] transfer 1 (60% overlap)'
            },
            {
                'name': 'high_overlap',
                'source_classes': {0, 1, 2, 3, 4, 5, 6, 7},
                'target_classes': {2, 3, 4, 5, 6, 7, 8, 9},
                'transfer_class': 1,
                'description': '[0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer 1 (75% overlap)'
            }
        ]
    
    def evaluate_class_accuracy(self, model: nn.Module, test_loader, target_class: int, device: str) -> float:
        """Evaluate accuracy for a specific class."""
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
    
    def evaluate_transfer_with_system(self, transfer_system, test_loader, transfer_class: int, device: str) -> float:
        """Evaluate transfer effectiveness using the transfer system."""
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
    
    def evaluate_overall_accuracy_after_transfer(self, model: nn.Module, transfer_system, 
                                               test_loader, transfer_class: int, device: str) -> float:
        """Evaluate overall accuracy with transfer system applied."""
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
    
    def run_single_experiment(self, config: Dict, seed: int) -> Dict:
        """Run a single experiment with focused logging."""
        logger.info(f"\n🏛️ Running FOCUSED {config['name']} experiment (seed={seed})")
        logger.info(f"   🏗️  {config['description']}")
        
        # Setup experiment config
        experiment_config = ExperimentConfig(
            seed=seed,
            num_pairs=1,
            max_epochs=5,
            min_accuracy_threshold=0.90
        )
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            # Create experiment runner
            runner = ExperimentRunner(experiment_config)
            data_manager = runner.data_manager
            trainer = runner.trainer
            
            # Get data loaders
            source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
                data_manager.get_data_loaders(config['source_classes'], config['target_classes'])
            
            # Create and train models
            source_model = create_model("WideNN")
            target_model = create_model("WideNN")
            
            # Train source model
            trained_source, source_accuracy = trainer.train_model(
                source_model, source_train_loader, source_test_loader)
            
            # Train target model
            trained_target, target_accuracy = trainer.train_model(
                target_model, target_train_loader, target_test_loader)
            
            if trained_source is None or trained_target is None:
                logger.error(f"   ❌ Model training failed")
                return None
            
            # BEFORE TRANSFER EVALUATION
            target_before_overall = trainer.evaluate_model(trained_target, target_test_loader)
            target_before_transfer_class = self.evaluate_class_accuracy(
                trained_target, target_test_loader, config['transfer_class'], experiment_config.device)
            
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
            
            # AFTER TRANSFER EVALUATION
            target_after_overall = self.evaluate_overall_accuracy_after_transfer(
                trained_target, transfer_system, target_test_loader, config['transfer_class'], experiment_config.device)
            target_after_transfer_class = self.evaluate_transfer_with_system(
                transfer_system, target_test_loader, config['transfer_class'], experiment_config.device)
            
            # Calculate key metrics matching previous experiments
            transfer_improvement = target_after_transfer_class - target_before_transfer_class
            knowledge_preservation = target_after_overall
            transfer_effectiveness = target_after_transfer_class
            
            # Log results in previous experiment format
            logger.info(f"   ✅ Models trained: Source={source_accuracy:.3f}, Target={target_accuracy:.3f}")
            logger.info(f"   📊 Results Summary:")
            logger.info(f"      Target Before Transfer: {target_before_overall:.1%} (class {config['transfer_class']}: {target_before_transfer_class:.1%})")
            logger.info(f"      Target After Transfer:  {target_after_overall:.1%} (class {config['transfer_class']}: {target_after_transfer_class:.1%})")
            logger.info(f"   🎯 Key Metrics:")
            logger.info(f"      Transfer improvement: {transfer_improvement:.1%}")
            logger.info(f"      Knowledge preservation: {knowledge_preservation:.1%}")
            logger.info(f"      Transfer effectiveness: {transfer_effectiveness:.1%}")
            logger.info(f"   ✅ Focused experimental setup validated")
            
            # Create result matching previous format
            result = {
                'experiment_id': f"focused_{config['name']}_seed_{seed}",
                'configuration': config['name'],
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                'experimental_setup': {
                    'source_classes': list(config['source_classes']),
                    'target_classes': list(config['target_classes']),
                    'transfer_class': config['transfer_class'],
                    'description': config['description']
                },
                'model_accuracies': {
                    'source': float(source_accuracy),
                    'target': float(target_accuracy)
                },
                'detailed_results': {
                    'target_before_overall': float(target_before_overall),
                    'target_before_transfer_class': float(target_before_transfer_class),
                    'target_after_overall': float(target_after_overall),
                    'target_after_transfer_class': float(target_after_transfer_class)
                },
                'key_metrics': {
                    'transfer_improvement': float(transfer_improvement),
                    'knowledge_preservation': float(knowledge_preservation),
                    'transfer_effectiveness': float(transfer_effectiveness)
                },
                'alignment_error': float(fit_metrics['alignment_error'])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Experiment failed: {e}")
            return None
    
    def run_configuration_experiments(self, config: Dict, seeds: List[int]) -> Dict:
        """Run experiments for one configuration across multiple seeds."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 FOCUSED CONFIGURATION: {config['name'].upper()}")
        logger.info(f"   {config['description']}")
        logger.info(f"{'='*80}")
        
        results = []
        metrics_table = []
        
        for seed in seeds:
            result = self.run_single_experiment(config, seed)
            if result is not None:
                results.append(result)
                
                # Add to metrics table
                metrics_table.append({
                    'seed': seed,
                    'target_before': result['detailed_results']['target_before_transfer_class'],
                    'target_after': result['detailed_results']['target_after_transfer_class'],
                    'transfer_improvement': result['key_metrics']['transfer_improvement'],
                    'knowledge_preservation': result['key_metrics']['knowledge_preservation']
                })
        
        # Print summary table matching your requested format
        if metrics_table:
            logger.info(f"\n📊 SUMMARY TABLE FOR {config['name'].upper()}:")
            logger.info(f"| Seed | Target Before | Target After | Transfer Improvement | Knowledge Preservation |")
            logger.info(f"|------|---------------|--------------|---------------------|----------------------|")
            
            for row in metrics_table:
                logger.info(f"| {row['seed']:4d} | {row['target_before']:11.1%} | {row['target_after']:10.1%} | {row['transfer_improvement']:17.1%} | {row['knowledge_preservation']:19.1%} |")
            
            # Calculate summary statistics
            transfer_improvements = [row['transfer_improvement'] for row in metrics_table]
            knowledge_preservations = [row['knowledge_preservation'] for row in metrics_table]
            
            logger.info(f"\n📈 SUMMARY STATISTICS:")
            logger.info(f"   Transfer Improvement: {np.mean(transfer_improvements):.1%} ± {np.std(transfer_improvements):.1%}")
            logger.info(f"   Knowledge Preservation: {np.mean(knowledge_preservations):.1%} ± {np.std(knowledge_preservations):.1%}")
            logger.info(f"   Success Rate: {len(results)}/{len(seeds)} ({len(results)/len(seeds):.1%})")
        
        # Convert sets to lists for JSON serialization
        config_for_json = {
            'name': config['name'],
            'source_classes': list(config['source_classes']),
            'target_classes': list(config['target_classes']),
            'transfer_class': config['transfer_class'],
            'description': config['description']
        }
        
        # Save results
        summary = {
            'configuration': config_for_json,
            'seeds_tested': seeds,
            'num_successful': len(results),
            'success_rate': len(results) / len(seeds),
            'metrics_table': metrics_table,
            'detailed_results': results,
            'summary_statistics': {
                'transfer_improvement': {
                    'mean': float(np.mean(transfer_improvements)) if transfer_improvements else 0,
                    'std': float(np.std(transfer_improvements)) if transfer_improvements else 0
                },
                'knowledge_preservation': {
                    'mean': float(np.mean(knowledge_preservations)) if knowledge_preservations else 0,
                    'std': float(np.std(knowledge_preservations)) if knowledge_preservations else 0
                }
            } if metrics_table else {}
        }
        
        # Save to file
        output_file = self.results_dir / f"{config['name']}_focused_results.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_file}")
        
        return summary
    
    def run_all_configurations(self, seeds: List[int] = [42, 123, 456]) -> Dict:
        """Run all configurations with the specified seeds."""
        logger.info(f"\n🔬 FOCUSED SHARED KNOWLEDGE ANALYSIS")
        logger.info(f"   Testing {len(self.configs)} configurations with seeds: {seeds}")
        logger.info(f"=" * 80)
        
        all_results = {}
        
        for config in self.configs:
            config_results = self.run_configuration_experiments(config, seeds)
            all_results[config['name']] = config_results
        
        # Save comprehensive results
        comprehensive = {
            'experiment_title': 'Focused Shared Knowledge Analysis',
            'timestamp': datetime.now().isoformat(),
            'seeds_tested': seeds,
            'configurations': len(self.configs),
            'results_by_configuration': all_results
        }
        
        output_file = self.results_dir / "comprehensive_focused_results.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive, f, indent=2)
        
        logger.info(f"\n✅ ALL CONFIGURATIONS COMPLETED")
        logger.info(f"💾 Comprehensive results saved to: {output_file}")
        
        return comprehensive


def main():
    """Run focused shared knowledge analysis with exact previous experiment logging."""
    experiment = FocusedSharedKnowledgeExperiment()
    
    # Run with same seeds as previous experiments
    results = experiment.run_all_configurations(seeds=[42, 123, 456])
    
    print("\n✅ Focused shared knowledge analysis completed!")
    print("📊 Results show the exact metrics you requested in table format")


if __name__ == "__main__":
    main()