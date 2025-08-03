#!/usr/bin/env python3
"""
SAE Transfer Experiments with Different Shared Knowledge Bases
Run SAE experiments for three scenarios with different amounts of shared knowledge.
"""

import torch
import torch.nn as nn
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set, List, Tuple
import copy

# Import necessary modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.architectures import WideNN, DeepNN
from src.core.experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
from src.transfer_methods.robust_balanced_transfer import RobustBalancedTransferSystem
from src.transfer_methods.improved_sae_robust_transfer import ImprovedSAERobustTransferSystem

# Suppress INFO logs for cleaner output
logging.basicConfig(level=logging.WARNING)

class SAESharedKnowledgeEvaluator:
    """Evaluator for SAE transfer experiments with different shared knowledge bases."""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_all_accuracies(self, 
                               source_model, target_before_model, target_after_model,
                               source_test_loader, target_test_loader,
                               source_classes: Set[int], target_classes: Set[int],
                               transfer_class: int):
        """Evaluate accuracies for SAE shared knowledge experiments."""
        
        print("📊 MEASURING SAE SHARED KNOWLEDGE ACCURACIES")
        print("=" * 50)
        
        # Calculate shared and exclusive classes
        shared_classes = source_classes & target_classes
        source_exclusive = source_classes - target_classes
        target_exclusive = target_classes - source_classes
        
        print(f"   Shared classes: {sorted(shared_classes)}")
        print(f"   Source exclusive: {sorted(source_exclusive)}")
        print(f"   Target exclusive: {sorted(target_exclusive)}")
        print(f"   Transfer class: {transfer_class} (from source to target)")
        
        # Test source model (should know all its classes including transfer class)
        source_original = self._measure_accuracy(source_model, source_test_loader, source_classes, "source original classes")
        source_transfer = self._measure_accuracy(source_model, source_test_loader, {transfer_class}, f"source class {transfer_class}")
        source_specificity = self._measure_accuracy(source_model, source_test_loader, source_exclusive - {transfer_class}, "source specificity classes")
        
        # Test target before transfer (should NOT know transfer class)
        target_before_original = self._measure_accuracy(target_before_model, target_test_loader, target_classes, "target before original classes")
        target_before_transfer = self._measure_accuracy(target_before_model, source_test_loader, {transfer_class}, f"target before class {transfer_class}")
        target_before_specificity = self._measure_accuracy(target_before_model, source_test_loader, source_exclusive - {transfer_class}, "target before specificity classes")
        
        # Test target after transfer (should know transfer class but not other source exclusives)
        target_after_original = self._measure_accuracy(target_after_model, target_test_loader, target_classes, "target after original classes")
        target_after_transfer = self._measure_accuracy(target_after_model, source_test_loader, {transfer_class}, f"target after class {transfer_class}")
        target_after_specificity = self._measure_accuracy(target_after_model, source_test_loader, source_exclusive - {transfer_class}, "target after specificity classes")
        
        results = {
            'source_original_accuracy': source_original,
            'source_transfer_class_accuracy': source_transfer,
            'source_specificity_class_accuracy': source_specificity,
            'target_before_original_accuracy': target_before_original,
            'target_before_transfer_class_accuracy': target_before_transfer,
            'target_before_specificity_class_accuracy': target_before_specificity,
            'target_after_original_accuracy': target_after_original,
            'target_after_transfer_class_accuracy': target_after_transfer,
            'target_after_specificity_class_accuracy': target_after_specificity
        }
        
        # Validate experimental setup
        self._validate_experiment_setup(results, transfer_class, shared_classes, source_exclusive, target_exclusive)
        
        return results
    
    def _measure_accuracy(self, model, data_loader, target_classes: Set[int], description: str) -> float:
        """Measure simple classification accuracy."""
        if not target_classes:  # Empty set
            print(f"   {description}: N/A (no classes)")
            return 0.0
            
        device = self.config.device
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                
                # Filter for target classes only
                target_mask = torch.tensor([label.item() in target_classes for label in labels])
                if target_mask.sum() == 0:
                    continue
                
                filtered_data = data[target_mask]
                filtered_labels = labels[target_mask]
                
                # Get model predictions - handle different model types
                if hasattr(model, 'forward'):
                    outputs = model(filtered_data.view(filtered_data.size(0), -1))
                else:
                    outputs = model(filtered_data)
                
                _, predicted = torch.max(outputs, 1)
                
                # Direct accuracy calculation
                for pred, true_label in zip(predicted, filtered_labels):
                    if pred.item() == true_label.item():
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"   {description}: {accuracy:.1%} ({correct}/{total})")
        return accuracy
    
    def _validate_experiment_setup(self, results, transfer_class: int, shared_classes: Set[int], source_exclusive: Set[int], target_exclusive: Set[int]):
        """Validate shared knowledge experimental setup."""
        
        transfer_before = results['target_before_transfer_class_accuracy']
        
        print(f"\n🔍 SHARED KNOWLEDGE VALIDATION")
        print("-" * 40)
        
        # Check transfer class baseline
        if transfer_before > 0.30:
            print(f"🚨 WARNING: Target has {transfer_before:.1%} accuracy on class {transfer_class} before transfer!")
            print("   This suggests data leakage or visual similarity issues.")
        else:
            print(f"✅ Transfer class {transfer_class}: {transfer_before:.1%} ≤ 30% (clean baseline)")
        
        # Report shared knowledge overlap
        shared_percentage = len(shared_classes) / (len(shared_classes) + len(source_exclusive) + len(target_exclusive)) * 100
        print(f"📊 Shared knowledge: {len(shared_classes)} classes ({shared_percentage:.1f}% overlap)")
        
        if len(shared_classes) == 0:
            print("   ⚠️  No shared classes - this is a pure zero-shot transfer test")
        elif len(shared_classes) >= 5:
            print("   ✅ High shared knowledge - should facilitate transfer")
        else:
            print("   ✅ Limited shared knowledge - challenging transfer scenario")
    
    def print_results_table(self, results, experiment_name: str, transfer_class: int, shared_classes: Set[int]):
        """Print shared knowledge results table."""
        
        print(f"\n📈 SAE SHARED KNOWLEDGE RESULTS: {experiment_name}")
        print("=" * 80)
        print()
        
        print(f"| Model                    | Original Classes | Transfer Class {transfer_class} | Specificity Classes |")
        print("|--------------------------|------------------|------------------|---------------------|")
        print(f"| Source                   | {results['source_original_accuracy']:14.1%} | {results['source_transfer_class_accuracy']:15.1%} | {results['source_specificity_class_accuracy']:18.1%} |")
        print(f"| Target (Before)          | {results['target_before_original_accuracy']:14.1%} | {results['target_before_transfer_class_accuracy']:15.1%} | {results['target_before_specificity_class_accuracy']:18.1%} |")
        print(f"| Target (After)           | {results['target_after_original_accuracy']:14.1%} | {results['target_after_transfer_class_accuracy']:15.1%} | {results['target_after_specificity_class_accuracy']:18.1%} |")
        print()
        
        print("📊 KEY OBSERVATIONS:")
        print(f"   • Shared classes: {sorted(shared_classes)} ({len(shared_classes)} classes)")
        print(f"   • Transfer effectiveness: {results['target_before_transfer_class_accuracy']:.1%} → {results['target_after_transfer_class_accuracy']:.1%} ({results['target_after_transfer_class_accuracy'] - results['target_before_transfer_class_accuracy']:+.1%})")
        print(f"   • Knowledge preservation: {results['target_after_original_accuracy']:.1%} (target: ≥80%)")
        print(f"   • Transfer specificity: {results['target_after_specificity_class_accuracy']:.1%} (target: ≤10%)")

class SAEWrappedTransferModel(nn.Module):
    """Wrapper for SAE transfer systems that properly applies transfer."""
    def __init__(self, base_model, transfer_system, transfer_class):
        super().__init__()
        self.base_model = base_model
        self.transfer_system = transfer_system
        self.transfer_class = transfer_class
    
    def forward(self, x):
        if self.transfer_system is None:
            return self.base_model(x)
        
        x_flat = x.view(x.size(0), -1)
        
        # Try feature-level transfer (for ImprovedSAERobustTransferSystem)  
        if hasattr(self.transfer_system, 'transfer'):
            try:
                features = self.base_model.get_features(x_flat)
                enhanced_features = self.transfer_system.transfer(features)
                outputs = self.base_model.classify_from_features(enhanced_features)
                return outputs
            except Exception as e:
                print(f"Warning: SAE feature transfer failed: {e}")
        
        # Try transfer_concept (for RobustBalancedTransferSystem)
        if hasattr(self.transfer_system, 'transfer_concept'):
            try:
                enhanced_outputs = self.transfer_system.transfer_concept(x_flat, self.transfer_class)
                if enhanced_outputs is not None:
                    return enhanced_outputs
            except Exception as e:
                print(f"Warning: transfer_concept failed: {e}")
        
        # Fallback: original model output
        print(f"Warning: No valid transfer method found for SAE, using original model")
        return self.base_model(x_flat)

def run_sae_shared_knowledge_experiment(source_classes: Set[int], target_classes: Set[int], transfer_class: int, 
                                      approach: str, seed: int, config_params: Dict) -> Optional[Dict]:
    """Run SAE experiment with specific shared knowledge configuration."""
    
    shared_classes = source_classes & target_classes
    experiment_name = f"SAE_{len(shared_classes)}_shared_classes"
    
    print(f"🔬 Running {experiment_name} {approach} experiment (seed={seed})")
    print(f"   Source: {sorted(source_classes)}")
    print(f"   Target: {sorted(target_classes)}")
    print(f"   Shared: {sorted(shared_classes)} ({len(shared_classes)} classes)")
    print(f"   Transfer: {transfer_class}")
    
    config = ExperimentConfig(
        seed=seed,
        max_epochs=config_params['max_epochs'],
        batch_size=config_params['batch_size'],
        learning_rate=config_params['learning_rate'],
        concept_dim=config_params['concept_dim'],
        device='cpu'
    )
    
    try:
        # Create data and train models
        data_manager = MNISTDataManager(config)
        trainer = ModelTrainer(config)
        
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
            data_manager.get_data_loaders(source_classes, target_classes)
        
        print(f"   📚 Training models...")
        
        # Train source model (knows transfer class)
        source_model = WideNN()  # Use same architecture for consistency
        trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
        
        # Train target model (doesn't know transfer class)
        target_model = WideNN()  # Use same architecture for consistency
        trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
        
        if trained_source is None or trained_target is None:
            print("   ❌ Model training failed")
            return None
        
        print(f"   ✅ Models trained: Source={source_acc:.3f}, Target={target_acc:.3f}")
        
        # Clone target model BEFORE any transfer operations
        target_before_transfer = copy.deepcopy(trained_target)
        
        # Create transfer system
        if approach == 'improved_sae':
            transfer_system = ImprovedSAERobustTransferSystem(
                source_model=trained_source,
                target_model=trained_target,
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=config.concept_dim,
                device=config.device
            )
        else:  # rho_blending baseline
            transfer_system = RobustBalancedTransferSystem(
                source_model=trained_source,
                target_model=trained_target,
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=config.concept_dim,
                device=config.device
            )
        
        # Fit and setup transfer
        print("   🔄 Setting up SAE transfer system...")
        transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=config_params['sae_epochs'])
        transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
        
        # Create evaluation models
        target_before = target_before_transfer
        target_after = SAEWrappedTransferModel(trained_target, transfer_system, transfer_class)
        
        # Evaluate
        print("   📊 Measuring accuracies...")
        evaluator = SAESharedKnowledgeEvaluator(config)
        
        results = evaluator.evaluate_all_accuracies(
            source_model=trained_source,
            target_before_model=target_before,
            target_after_model=target_after,
            source_test_loader=source_test_loader,
            target_test_loader=target_test_loader,
            source_classes=source_classes,
            target_classes=target_classes,
            transfer_class=transfer_class
        )
        
        # Print results
        evaluator.print_results_table(results, experiment_name, transfer_class, shared_classes)
        
        # Package for saving
        experiment_results = {
            'experiment_id': f"sae_shared_knowledge_{len(shared_classes)}_{approach}_seed_{seed}",
            'experiment_name': experiment_name,
            'approach': approach,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'experimental_setup': {
                'source_classes': sorted(source_classes),
                'target_classes': sorted(target_classes),
                'shared_classes': sorted(shared_classes),
                'transfer_class': transfer_class,
                'shared_knowledge_count': len(shared_classes),
                'total_knowledge_overlap_percentage': len(shared_classes) / len(source_classes | target_classes) * 100
            },
            'model_accuracies': {
                'source': source_acc,
                'target': target_acc
            },
            'sae_shared_knowledge_results': results,
            'success_criteria': {
                'knowledge_preservation': results['target_after_original_accuracy'] >= 0.8,
                'transfer_effectiveness': results['target_after_transfer_class_accuracy'] >= 0.7,
                'transfer_specificity': results['target_after_specificity_class_accuracy'] <= 0.1
            },
            'transfer_metrics': {
                'effectiveness_improvement': results['target_after_transfer_class_accuracy'] - results['target_before_transfer_class_accuracy'],
                'knowledge_preservation_score': results['target_after_original_accuracy'],
                'specificity_score': results['target_after_specificity_class_accuracy']
            }
        }
        
        return experiment_results
        
    except Exception as e:
        print(f"   ❌ SAE shared knowledge experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all three SAE shared knowledge experiments."""
    
    print("🧪 SAE SHARED KNOWLEDGE EXPERIMENTS")
    print("===================================")
    print("Testing SAE transfer with different amounts of shared knowledge:")
    print("1. [2,3,4] → [0,1,2] transfer 3 (1 shared class)")
    print("2. [2,3,4,5,6] → [0,1,2,3,4] transfer 5 (3 shared classes)")
    print("3. [2,3,4,5,6,7,8,9] → [0,1,2,3,4,5,6,7] transfer 8 (6 shared classes)")
    print()
    
    # Configuration
    config_params = {
        'max_epochs': 5,
        'sae_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'concept_dim': 24
    }
    
    # Experiment scenarios - CORRECTED: Source must know transfer class
    scenarios = [
        {
            'name': 'Low Shared Knowledge',
            'source_classes': {2, 3, 4},        # Source knows transfer class 3
            'target_classes': {0, 1, 2},        # Target doesn't know transfer class 3
            'transfer_class': 3
        },
        {
            'name': 'Medium Shared Knowledge',
            'source_classes': {2, 3, 4, 5, 6},  # Source knows transfer class 5
            'target_classes': {0, 1, 2, 3, 4},  # Target doesn't know transfer class 5
            'transfer_class': 5
        },
        {
            'name': 'High Shared Knowledge',
            'source_classes': {2, 3, 4, 5, 6, 7, 8, 9},  # Source knows transfer class 8
            'target_classes': {0, 1, 2, 3, 4, 5, 6, 7},  # Target doesn't know transfer class 8
            'transfer_class': 8
        }
    ]
    
    approaches = ['improved_sae']  # Focus on SAE approach
    seeds = [42, 123, 456]  # Multiple seeds for reliability
    
    # Create results directory
    results_dir = Path('experiment_results/sae_shared_knowledge')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    successful_experiments = 0
    total_experiments = len(scenarios) * len(approaches) * len(seeds)
    current_experiment = 0
    
    print(f"📊 Running {total_experiments} total experiments...")
    print()
    
    for scenario in scenarios:
        print(f"🎯 SCENARIO: {scenario['name']}")
        print(f"   Source: {sorted(scenario['source_classes'])}")
        print(f"   Target: {sorted(scenario['target_classes'])}")
        print(f"   Transfer: {scenario['transfer_class']}")
        print(f"   Shared: {sorted(scenario['source_classes'] & scenario['target_classes'])} ({len(scenario['source_classes'] & scenario['target_classes'])} classes)")
        print()
        
        for approach in approaches:
            for seed in seeds:
                current_experiment += 1
                print(f"[{current_experiment}/{total_experiments}] Running {scenario['name']} {approach} seed {seed}...")
                
                result = run_sae_shared_knowledge_experiment(
                    source_classes=scenario['source_classes'],
                    target_classes=scenario['target_classes'],
                    transfer_class=scenario['transfer_class'],
                    approach=approach,
                    seed=seed,
                    config_params=config_params
                )
                
                if result:
                    # Save individual result
                    result_file = results_dir / f"sae_shared_{len(scenario['source_classes'] & scenario['target_classes'])}_{approach}_seed_{seed}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    all_results.append(result)
                    successful_experiments += 1
                    print(f"   ✅ Success - saved to {result_file}")
                else:
                    print(f"   ❌ Failed")
                
                print()
    
    # Generate summary report
    print("📈 GENERATING SAE SHARED KNOWLEDGE ANALYSIS")
    print("==========================================")
    
    summary_report = generate_sae_summary_report(all_results, scenarios)
    
    summary_file = results_dir / 'SAE_SHARED_KNOWLEDGE_SUMMARY.md'
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"📄 Summary report saved to: {summary_file}")
    
    print(f"\n🎉 SAE SHARED KNOWLEDGE EXPERIMENTS COMPLETE!")
    print(f"   Total: {total_experiments}")
    print(f"   Successful: {successful_experiments}")
    print(f"   Failed: {total_experiments - successful_experiments}")

def generate_sae_summary_report(results: List[Dict], scenarios: List[Dict]) -> str:
    """Generate comprehensive summary report for SAE shared knowledge experiments."""
    
    report = f"""# SAE Shared Knowledge Transfer Experiments

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Experimental Overview

This experiment tests how the amount of shared knowledge between source and target networks affects SAE-based transfer learning effectiveness.

### Scenarios Tested

1. **Low Shared Knowledge**: [0,1,2] → [2,3,4] transfer 3 (1 shared class - digit 2)
2. **Medium Shared Knowledge**: [0,1,2,3,4] → [2,3,4,5,6] transfer 5 (3 shared classes - digits 2,3,4)  
3. **High Shared Knowledge**: [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer 8 (6 shared classes - digits 2,3,4,5,6,7)

### Key Metrics

- **Source Original Accuracy**: Source model performance on its training classes
- **Source Transfer Class Accuracy**: Source model performance on the transfer class
- **Source Specificity Class Accuracy**: Source model performance on classes it knows but shouldn't transfer
- **Target Before Original Accuracy**: Target model performance on its training classes before transfer
- **Target Before Transfer Class Accuracy**: Target model performance on transfer class before transfer (should be ~0%)
- **Target Before Specificity Class Accuracy**: Target model performance on source-exclusive classes before transfer
- **Target After Original Accuracy**: Target model performance on its training classes after transfer
- **Target After Transfer Class Accuracy**: Target model performance on transfer class after transfer (success metric)
- **Target After Specificity Class Accuracy**: Target model performance on source-exclusive classes after transfer (should stay ~0%)

## Results Summary

"""
    
    # Group results by shared knowledge level
    results_by_scenario = {}
    for result in results:
        shared_count = result['experimental_setup']['shared_knowledge_count']
        if shared_count not in results_by_scenario:
            results_by_scenario[shared_count] = []
        results_by_scenario[shared_count].append(result)
    
    for shared_count in sorted(results_by_scenario.keys()):
        scenario_results = results_by_scenario[shared_count]
        scenario_name = f"{shared_count} Shared Class{'es' if shared_count != 1 else ''}"
        
        if not scenario_results:
            continue
        
        # Calculate statistics
        import statistics
        
        transfer_before = [r['sae_shared_knowledge_results']['target_before_transfer_class_accuracy'] for r in scenario_results]
        transfer_after = [r['sae_shared_knowledge_results']['target_after_transfer_class_accuracy'] for r in scenario_results]
        original_after = [r['sae_shared_knowledge_results']['target_after_original_accuracy'] for r in scenario_results]
        specificity_after = [r['sae_shared_knowledge_results']['target_after_specificity_class_accuracy'] for r in scenario_results]
        
        transfer_improvement = [after - before for after, before in zip(transfer_after, transfer_before)]
        
        success_rate = sum(1 for r in scenario_results if all([
            r['success_criteria']['knowledge_preservation'],
            r['success_criteria']['transfer_effectiveness'],
            r['success_criteria']['transfer_specificity']
        ])) / len(scenario_results)
        
        report += f"""### {scenario_name}

**Sample Configuration:** {scenario_results[0]['experimental_setup']['source_classes']} → {scenario_results[0]['experimental_setup']['target_classes']} (transfer {scenario_results[0]['experimental_setup']['transfer_class']})

| Metric | Mean ± Std | Range | Success Rate |
|--------|------------|-------|--------------|
| **Transfer Before** | {statistics.mean(transfer_before):.1%} ± {statistics.stdev(transfer_before) if len(transfer_before) > 1 else 0:.1%} | [{min(transfer_before):.1%}, {max(transfer_before):.1%}] | Clean: {sum(1 for x in transfer_before if x <= 0.3)}/{len(transfer_before)} |
| **Transfer After** | {statistics.mean(transfer_after):.1%} ± {statistics.stdev(transfer_after) if len(transfer_after) > 1 else 0:.1%} | [{min(transfer_after):.1%}, {max(transfer_after):.1%}] | ≥70%: {sum(1 for x in transfer_after if x >= 0.7)}/{len(transfer_after)} |
| **Knowledge Preservation** | {statistics.mean(original_after):.1%} ± {statistics.stdev(original_after) if len(original_after) > 1 else 0:.1%} | [{min(original_after):.1%}, {max(original_after):.1%}] | ≥80%: {sum(1 for x in original_after if x >= 0.8)}/{len(original_after)} |
| **Transfer Specificity** | {statistics.mean(specificity_after):.1%} ± {statistics.stdev(specificity_after) if len(specificity_after) > 1 else 0:.1%} | [{min(specificity_after):.1%}, {max(specificity_after):.1%}] | ≤10%: {sum(1 for x in specificity_after if x <= 0.1)}/{len(specificity_after)} |

**Transfer Effectiveness:** {statistics.mean(transfer_improvement):+.1%} improvement  
**Overall Success Rate:** {success_rate:.1%} ({sum(1 for r in scenario_results if all([r['success_criteria']['knowledge_preservation'], r['success_criteria']['transfer_effectiveness'], r['success_criteria']['transfer_specificity']]))} / {len(scenario_results)} experiments)

"""
    
    # Cross-scenario comparison
    if len(results_by_scenario) > 1:
        report += """## Cross-Scenario Analysis

### Transfer Effectiveness by Shared Knowledge Amount

"""
        
        for shared_count in sorted(results_by_scenario.keys()):
            scenario_results = results_by_scenario[shared_count]
            transfer_after = [r['sae_shared_knowledge_results']['target_after_transfer_class_accuracy'] for r in scenario_results]
            mean_transfer = statistics.mean(transfer_after)
            report += f"- **{shared_count} Shared Classes:** {mean_transfer:.1%} average transfer accuracy\n"
        
        report += "\n### Key Findings\n\n"
        
        # Determine trends
        transfer_by_shared = {shared: statistics.mean([r['sae_shared_knowledge_results']['target_after_transfer_class_accuracy'] for r in results_by_scenario[shared]]) 
                            for shared in sorted(results_by_scenario.keys())}
        
        if len(transfer_by_shared) >= 2:
            sorted_shared = sorted(transfer_by_shared.keys())
            if transfer_by_shared[sorted_shared[-1]] > transfer_by_shared[sorted_shared[0]]:
                report += "- ✅ **Positive Correlation**: More shared knowledge improves transfer effectiveness\n"
            else:
                report += "- ⚠️ **Negative/No Correlation**: More shared knowledge does not clearly improve transfer\n"
    
    report += f"""
## Experimental Validation

All experiments use the same rigorous validation criteria:

1. **Clean Baseline**: Target model should have ≤30% accuracy on transfer class before transfer
2. **Transfer Effectiveness**: Target model should achieve ≥70% accuracy on transfer class after transfer  
3. **Knowledge Preservation**: Target model should maintain ≥80% accuracy on original classes
4. **Transfer Specificity**: Target model should have ≤10% accuracy on non-transferred source-exclusive classes

## Files Generated

- Individual experiment results: `experiment_results/sae_shared_knowledge/sae_shared_*_improved_sae_seed_*.json`
- This summary: `experiment_results/sae_shared_knowledge/SAE_SHARED_KNOWLEDGE_SUMMARY.md`

## Comparison with Shared Layer Approach

This SAE approach can be directly compared with the shared layer transfer results from the `mnist-transfer-from-shared-layers` branch, which tested the same three scenarios but found that true zero-shot transfer was impossible with frozen networks.

"""
    
    return report

if __name__ == "__main__":
    main()