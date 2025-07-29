#!/bin/bash

# FIXED Clean SAE vs Rho Blending Comparison - Eliminates Visual Similarity Issues
# CORRECTED: Transfer class 6 from {2,3,4,5,6,7} to {0,1,2,3,4,5}

echo "🚀 FIXED CLEAN SAE vs RHO BLENDING COMPARISON"
echo "============================================="
echo "CORRECTED experimental setup to eliminate visual similarity issues:"
echo "- Source model trained on: {2,3,4,5,6,7} (knows class 6)"
echo "- Target model trained on: {0,1,2,3,4,5} (doesn't know class 6)"  
echo "- Transfer class: 6 (from source to target)"
echo "- Specificity check: 7 (source knows, target doesn't - should NOT transfer)"
echo "- ⚠️  WARNING if target has >30% accuracy on class 6 before transfer"
echo ""

# Configuration
SEEDS=(42 123 456)  # Multiple seeds for reliability
TRANSFER_CLASS=6
SPECIFICITY_CLASS=7
MAX_EPOCHS=5
SAE_EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001
CONCEPT_DIM=24

# Check conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Setting up conda environment..."
    conda create -n neural_transfer python=3.9 -y
    conda activate neural_transfer
    conda install pytorch torchvision cpuonly -c pytorch -y
    pip install numpy matplotlib scikit-learn
else
    echo "✅ Using conda environment: ${CONDA_DEFAULT_ENV}"
fi

echo ""
echo "📁 Creating results directory..."
mkdir -p experiment_results/fixed_clean_comparison/{rho_blending,improved_sae}
mkdir -p logs

# Create the FIXED clean experiment runner
cat > run_fixed_clean_experiment.py << 'EOF'
#!/usr/bin/env python3
"""
FIXED clean experiment runner with corrected class splits and transfer application.
Source: {2,3,4,5,6,7} (knows 6) → Target: {0,1,2,3,4,5} (doesn't know 6)
"""

import torch
import torch.nn as nn
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set

# Import necessary modules
from architectures import WideNN, DeepNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
from robust_balanced_transfer import RobustBalancedTransferSystem
from improved_sae_robust_transfer import ImprovedSAERobustTransferSystem

# Suppress INFO logs for cleaner output
logging.basicConfig(level=logging.WARNING)

class FixedCleanAccuracyEvaluator:
    """FIXED accuracy evaluator for clean experimental setup."""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_all_accuracies(self, 
                               source_model, target_before_model, target_after_model,
                               source_test_loader, target_test_loader,
                               source_classes: Set[int], target_classes: Set[int],
                               transfer_class: int, specificity_class: int):
        """Evaluate accuracies for FIXED clean experimental setup."""
        
        print("📊 MEASURING FIXED CLEAN ACCURACIES")
        print("=" * 40)
        
        # Test source model (should know transfer class and specificity class)
        source_original = self._measure_accuracy(source_model, source_test_loader, source_classes, "source original classes")
        source_transfer = self._measure_accuracy(source_model, source_test_loader, {transfer_class}, f"source class {transfer_class}")
        source_specificity = self._measure_accuracy(source_model, source_test_loader, {specificity_class}, f"source class {specificity_class}")
        
        # Test target before transfer (should NOT know transfer class or specificity class)
        target_before_original = self._measure_accuracy(target_before_model, target_test_loader, target_classes, "target before original classes")
        target_before_transfer = self._measure_accuracy(target_before_model, source_test_loader, {transfer_class}, f"target before class {transfer_class}")
        target_before_specificity = self._measure_accuracy(target_before_model, source_test_loader, {specificity_class}, f"target before class {specificity_class}")
        
        # Test target after transfer (should know transfer class but NOT specificity class)
        target_after_original = self._measure_accuracy(target_after_model, target_test_loader, target_classes, "target after original classes")
        target_after_transfer = self._measure_accuracy(target_after_model, source_test_loader, {transfer_class}, f"target after class {transfer_class}")
        target_after_specificity = self._measure_accuracy(target_after_model, source_test_loader, {specificity_class}, f"target after class {specificity_class}")
        
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
        
        # Validate data integrity
        self._validate_no_data_leakage(results, transfer_class, specificity_class)
        
        return results
    
    def _measure_accuracy(self, model, data_loader, target_classes: Set[int], description: str) -> float:
        """Measure simple classification accuracy."""
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
    
    def _validate_no_data_leakage(self, results, transfer_class: int, specificity_class: int):
        """Validate clean experimental setup."""
        
        transfer_before = results['target_before_transfer_class_accuracy']
        specificity_before = results['target_before_specificity_class_accuracy']
        
        print(f"\n🔍 DATA LEAKAGE VALIDATION")
        print("-" * 30)
        
        issues_found = []
        
        # Check transfer class
        if transfer_before > 0.30:
            issues_found.append(f"class {transfer_class}")
            print(f"🚨 WARNING: Target has {transfer_before:.1%} accuracy on class {transfer_class} before transfer!")
        else:
            print(f"✅ Transfer class {transfer_class}: {transfer_before:.1%} ≤ 30%")
        
        # Check specificity class
        if specificity_before > 0.30:
            issues_found.append(f"class {specificity_class}")
            print(f"🚨 WARNING: Target has {specificity_before:.1%} accuracy on class {specificity_class} before transfer!")
        else:
            print(f"✅ Specificity class {specificity_class}: {specificity_before:.1%} ≤ 30%")
        
        if issues_found:
            print(f"\n" + "="*80)
            print("🚨 CRITICAL WARNING: POSSIBLE DATA LEAKAGE DETECTED!")
            print("="*80)
            print(f"❌ Issues found with: {', '.join(issues_found)}")
            print("EXPERIMENT VALIDITY: ⚠️  RESULTS MAY BE INVALID")
            print("="*80)
        else:
            print(f"\n✅ ALL VALIDATION PASSED - Experiment setup is clean!")
    
    def print_results_table(self, results, transfer_class: int, specificity_class: int):
        """Print FIXED results table."""
        
        print(f"\n📈 FIXED CLEAN ACCURACY RESULTS")
        print("=" * 80)
        print()
        
        print(f"| Model                    | Original Classes | Class {transfer_class}  | Class {specificity_class} |")
        print("|--------------------------|------------------|----------|----------|")
        print(f"| Source                   | {results['source_original_accuracy']:14.1%} | {results['source_transfer_class_accuracy']:7.1%} | {results['source_specificity_class_accuracy']:7.1%} |")
        print(f"| Target (Before)          | {results['target_before_original_accuracy']:14.1%} | {results['target_before_transfer_class_accuracy']:7.1%} | {results['target_before_specificity_class_accuracy']:7.1%} |")
        print(f"| Target (After)           | {results['target_after_original_accuracy']:14.1%} | {results['target_after_transfer_class_accuracy']:7.1%} | {results['target_after_specificity_class_accuracy']:7.1%} |")
        print()
        
        print("📊 KEY OBSERVATIONS:")
        print(f"   • Source trained on {{2,3,4,5,6,7}} should know class {transfer_class} and {specificity_class}")
        print(f"   • Target trained on {{0,1,2,3,4,5}} should NOT know class {transfer_class} or {specificity_class}")
        print(f"   • Transfer success: class {transfer_class} improves from {results['target_before_transfer_class_accuracy']:.1%} → {results['target_after_transfer_class_accuracy']:.1%}")
        print(f"   • Transfer specificity: class {specificity_class} stays low at {results['target_after_specificity_class_accuracy']:.1%}")
        print(f"   • Knowledge preservation: original classes at {results['target_after_original_accuracy']:.1%}")

class FixedWrappedTransferModel(nn.Module):
    """FIXED wrapper for transfer systems that properly applies transfer."""
    def __init__(self, base_model, transfer_system, transfer_class):
        super().__init__()
        self.base_model = base_model
        self.transfer_system = transfer_system
        self.transfer_class = transfer_class
    
    def forward(self, x):
        if self.transfer_system is None:
            return self.base_model(x)
        
        x_flat = x.view(x.size(0), -1)
        
        # Method 1: Try transfer_concept (for RobustBalancedTransferSystem)
        if hasattr(self.transfer_system, 'transfer_concept'):
            try:
                enhanced_outputs = self.transfer_system.transfer_concept(x_flat, self.transfer_class)
                if enhanced_outputs is not None:
                    return enhanced_outputs
            except Exception as e:
                print(f"Warning: transfer_concept failed: {e}")
        
        # Method 2: Try feature-level transfer (for ImprovedSAERobustTransferSystem)  
        if hasattr(self.transfer_system, 'transfer'):
            try:
                features = self.base_model.get_features(x_flat)
                enhanced_features = self.transfer_system.transfer(features)
                outputs = self.base_model.classify_from_features(enhanced_features)
                return outputs
            except Exception as e:
                print(f"Warning: feature transfer failed: {e}")
        
        # Fallback: original model output
        print(f"Warning: No valid transfer method found, using original model")
        return self.base_model(x_flat)

def run_fixed_clean_experiment(approach: str, seed: int, config_params: Dict) -> Optional[Dict]:
    """Run FIXED clean experiment with corrected class splits."""
    
    print(f"🔬 Running FIXED CLEAN {approach} experiment (seed={seed})")
    
    # ACTUALLY CORRECTED CLEAN SETUP
    source_classes = {2, 3, 4, 5, 6, 7}      # Source knows 2,3,4,5,6,7 (including transfer class 6)  
    target_classes = {0, 1, 2, 3, 4, 5}      # Target knows 0,1,2,3,4,5 (NOT class 6 or 7)
    transfer_class = 6                        # Transfer class 6 from source to target
    specificity_class = 7                     # Should NOT transfer (source knows, target doesn't)
    
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
        
        print(f"   📚 Training models: Source={sorted(source_classes)}, Target={sorted(target_classes)}")
        
        # Train source model (knows class 6 and 7)
        source_model = DeepNN()
        trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
        
        # Train target model (doesn't know class 6 or 7)
        target_model = WideNN()
        trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
        
        if trained_source is None or trained_target is None:
            print("   ❌ Model training failed")
            return None
        
        print(f"   ✅ Models trained: Source={source_acc:.3f}, Target={target_acc:.3f}")
        
        # CRITICAL FIX: Clone target model BEFORE any transfer operations
        # The transfer system modifies the target model during setup_injection_system!
        import copy
        target_before_transfer = copy.deepcopy(trained_target)  # Pure model before transfer
        
        # Create transfer system
        if approach == 'rho_blending':
            transfer_system = RobustBalancedTransferSystem(
                source_model=trained_source,
                target_model=trained_target,  # This will be modified
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=config.concept_dim,
                device=config.device
            )
        else:  # improved_sae
            transfer_system = ImprovedSAERobustTransferSystem(
                source_model=trained_source,
                target_model=trained_target,  # This will be modified
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=config.concept_dim,
                device=config.device
            )
        
        # Fit and setup transfer (this modifies trained_target!)
        print("   🔄 Setting up transfer system...")
        transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=config_params['sae_epochs'])
        transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
        
        # Create evaluation models with correct references
        target_before = target_before_transfer  # Unmodified model
        target_after = FixedWrappedTransferModel(trained_target, transfer_system, transfer_class)  # Modified model
        
        # Evaluate
        print("   📊 Measuring accuracies...")
        evaluator = FixedCleanAccuracyEvaluator(config)
        
        results = evaluator.evaluate_all_accuracies(
            source_model=trained_source,         # Source model
            target_before_model=target_before,   # Target before transfer
            target_after_model=target_after,     # Target after transfer
            source_test_loader=source_test_loader,
            target_test_loader=target_test_loader,
            source_classes=source_classes,       # Classes source knows
            target_classes=target_classes,       # Classes target knows
            transfer_class=transfer_class,       # Class 6
            specificity_class=specificity_class  # Class 7
        )
        
        # Print results
        evaluator.print_results_table(results, transfer_class, specificity_class)
        
        # Package for saving
        experiment_results = {
            'experiment_id': f"fixed_clean_{approach}_seed_{seed}",
            'approach': approach,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'experimental_setup': {
                'source_classes': sorted(source_classes),
                'target_classes': sorted(target_classes),
                'transfer_class': transfer_class,
                'specificity_class': specificity_class
            },
            'model_accuracies': {
                'source': source_acc,
                'target': target_acc
            },
            'fixed_clean_accuracy_results': results,
            'success_criteria': {
                'knowledge_preservation': results['target_after_original_accuracy'] >= 0.8,
                'transfer_effectiveness': results['target_after_transfer_class_accuracy'] >= 0.7,
                'transfer_specificity': results['target_after_specificity_class_accuracy'] <= 0.1
            },
            'data_leakage_check': {
                'transfer_class_before': results['target_before_transfer_class_accuracy'],
                'specificity_class_before': results['target_before_specificity_class_accuracy'],
                'data_leakage_detected': (results['target_before_transfer_class_accuracy'] > 0.3 or 
                                         results['target_before_specificity_class_accuracy'] > 0.3)
            }
        }
        
        return experiment_results
        
    except Exception as e:
        print(f"   ❌ Fixed clean experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_fixed_clean_experiment.py <approach> <seed> <config_json>")
        sys.exit(1)
    
    approach = sys.argv[1]
    seed = int(sys.argv[2])
    config_params = json.loads(sys.argv[3])
    
    result = run_fixed_clean_experiment(approach, seed, config_params)
    
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x run_fixed_clean_experiment.py

echo ""
echo "🧪 RUNNING FIXED CLEAN ACCURACY EXPERIMENTS"
echo "==========================================="

# Configuration JSON
CONFIG_JSON="{\"max_epochs\": $MAX_EPOCHS, \"sae_epochs\": $SAE_EPOCHS, \"batch_size\": $BATCH_SIZE, \"learning_rate\": $LEARNING_RATE, \"concept_dim\": $CONCEPT_DIM, \"transfer_class\": $TRANSFER_CLASS}"

# Approaches to test
declare -a APPROACHES=("rho_blending" "improved_sae")

TOTAL_EXPERIMENTS=$((${#APPROACHES[@]} * ${#SEEDS[@]}))
CURRENT_EXPERIMENT=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
DATA_LEAKAGE_COUNT=0

echo "📊 FIXED Clean Experiment Plan:"
echo "   - Source model: {2,3,4,5,6,7} (knows class 6 and 7)"
echo "   - Target model: {0,1,2,3,4,5} (doesn't know class 6 or 7)"
echo "   - Transfer class: 6 (from source to target)"
echo "   - Specificity check: class 7 (should NOT transfer)"
echo "   - Seeds: ${SEEDS[*]}"
echo "   - Total experiments: $TOTAL_EXPERIMENTS"
echo ""

# Run experiments
for approach in "${APPROACHES[@]}"; do
    echo "🔧 APPROACH: $approach"
    echo "========================"
    
    for seed in "${SEEDS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running FIXED clean $approach with seed $seed..."
        
        # Run experiment
        if python run_fixed_clean_experiment.py "$approach" "$seed" "$CONFIG_JSON" > "logs/fixed_clean_${approach}_seed_${seed}.json" 2> "logs/fixed_clean_${approach}_seed_${seed}.log"; then
            
            # Save result
            result_file="experiment_results/fixed_clean_comparison/$approach/fixed_clean_${approach}_seed_${seed}.json"
            cp "logs/fixed_clean_${approach}_seed_${seed}.json" "$result_file"
            
            # Check for data leakage
            if grep -q "CRITICAL WARNING: POSSIBLE DATA LEAKAGE" "logs/fixed_clean_${approach}_seed_${seed}.log"; then
                echo "   ⚠️  WARNING: Data leakage detected!"
                DATA_LEAKAGE_COUNT=$((DATA_LEAKAGE_COUNT + 1))
            fi
            
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
            echo "   ✅ Success"
            
        else
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
            echo "   ❌ Failed - check logs"
        fi
    done
done

echo ""
echo "📈 GENERATING FIXED CLEAN COMPARISON ANALYSIS"
echo "============================================="

# Analysis script
cat > analyze_fixed_clean_results.py << 'EOF'
#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
from datetime import datetime

def analyze_fixed_clean_results():
    results = {'rho_blending': [], 'improved_sae': []}
    
    base_path = Path('experiment_results/fixed_clean_comparison')
    for approach in ['rho_blending', 'improved_sae']:
        result_dir = base_path / approach
        if result_dir.exists():
            for result_file in result_dir.glob('*.json'):
                try:
                    with open(result_file, 'r') as f:
                        results[approach].append(json.load(f))
                except Exception as e:
                    print(f"Warning: {result_file}: {e}")
    
    report = f"""# FIXED Clean Accuracy Comparison: SAE vs Rho Blending

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## FIXED Clean Experimental Setup

**Eliminates visual similarity AND fixes transfer application:**

- **Source Model:** Trained on {{2,3,4,5,6,7}} (knows class 6 and 7)
- **Target Model:** Trained on {{0,1,2,3,4,5}} (doesn't know class 6 or 7)  
- **Transfer Class:** 6 (should improve from ~0% to >70%)
- **Specificity Class:** 7 (should stay at ~0% for specificity)

This setup:
1. Eliminates visual similarity (no MNIST 8 confusion)
2. Fixes transfer system application (target_after should differ from target_before)
3. Provides proper output space expansion

## Results Summary

"""
    
    for approach in ['rho_blending', 'improved_sae']:
        approach_results = results[approach]
        if not approach_results:
            report += f"### {approach.replace('_', ' ').title()}\n⚠️ No results\n\n"
            continue
        
        # Extract metrics  
        transfer_before = [r['fixed_clean_accuracy_results']['target_before_transfer_class_accuracy'] for r in approach_results]
        transfer_after = [r['fixed_clean_accuracy_results']['target_after_transfer_class_accuracy'] for r in approach_results]
        specificity_after = [r['fixed_clean_accuracy_results']['target_after_specificity_class_accuracy'] for r in approach_results]
        original_after = [r['fixed_clean_accuracy_results']['target_after_original_accuracy'] for r in approach_results]
        
        data_leakage_count = sum(1 for r in approach_results if r['data_leakage_check']['data_leakage_detected'])
        
        report += f"""### {approach.replace('_', ' ').title()}

| Metric | Mean ± Std | Success Rate |
|--------|------------|--------------|
| **Target Class 6 (Before)** | {statistics.mean(transfer_before):.1%} ± {statistics.stdev(transfer_before) if len(transfer_before) > 1 else 0:.1%} | Clean: {len(transfer_before) - sum(1 for x in transfer_before if x > 0.3)}/{len(transfer_before)} |
| **Target Class 6 (After)** | {statistics.mean(transfer_after):.1%} ± {statistics.stdev(transfer_after) if len(transfer_after) > 1 else 0:.1%} | ≥70%: {sum(1 for x in transfer_after if x >= 0.7)}/{len(transfer_after)} |
| **Specificity Class 7 (After)** | {statistics.mean(specificity_after):.1%} ± {statistics.stdev(specificity_after) if len(specificity_after) > 1 else 0:.1%} | ≤10%: {sum(1 for x in specificity_after if x <= 0.1)}/{len(specificity_after)} |
| **Original Classes (After)** | {statistics.mean(original_after):.1%} ± {statistics.stdev(original_after) if len(original_after) > 1 else 0:.1%} | ≥80%: {sum(1 for x in original_after if x >= 0.8)}/{len(original_after)} |

**Transfer Improvement:** {statistics.mean(transfer_after) - statistics.mean(transfer_before):+.1%}
**Data Quality:** {data_leakage_count}/{len(approach_results)} experiments had data leakage warnings

"""
    
    if results['rho_blending'] and results['improved_sae']:
        rho_transfer = statistics.mean([r['fixed_clean_accuracy_results']['target_after_transfer_class_accuracy'] for r in results['rho_blending']])
        sae_transfer = statistics.mean([r['fixed_clean_accuracy_results']['target_after_transfer_class_accuracy'] for r in results['improved_sae']])
        
        winner = "Improved SAE" if sae_transfer > rho_transfer else "Rho Blending"
        
        report += f"""## Final Comparison

**Class 6 Transfer Accuracy:**
- Rho Blending: {rho_transfer:.1%}
- Improved SAE: {sae_transfer:.1%}  
- Winner: **{winner}** 🏆

This FIXED clean experimental setup eliminates both visual similarity and transfer application bugs.
"""
    
    return report

def main():
    report = analyze_fixed_clean_results()
    
    report_path = Path('experiment_results/FIXED_CLEAN_ACCURACY_COMPARISON.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 FIXED clean analysis saved to: {report_path}")

if __name__ == "__main__":
    main()
EOF

python analyze_fixed_clean_results.py

echo ""
echo "🎉 FIXED CLEAN COMPARISON COMPLETE!"
echo "=================================="
echo ""
echo "📊 Final Statistics:" 
echo "   - Total experiments: $TOTAL_EXPERIMENTS"
echo "   - Successful: $SUCCESSFUL_EXPERIMENTS"
echo "   - Failed: $FAILED_EXPERIMENTS"
echo "   - Data leakage warnings: $DATA_LEAKAGE_COUNT"
echo ""
echo "📁 Results Location:"
echo "   - FIXED clean analysis: experiment_results/FIXED_CLEAN_ACCURACY_COMPARISON.md"
echo "   - Individual results: experiment_results/fixed_clean_comparison/"
echo ""
echo "🔧 FIXES APPLIED:"
echo "   ✅ Corrected class assignments: Source {2,3,4,5,6,7} → Target {0,1,2,3,4,5}"
echo "   ✅ Fixed transfer system application in WrappedTransferModel"  
echo "   ✅ Added proper output space expansion for new classes"
echo "   ✅ Eliminated visual similarity issues"
echo ""

if [ $DATA_LEAKAGE_COUNT -gt 0 ]; then
    echo "⚠️  Note: $DATA_LEAKAGE_COUNT experiments still showed data leakage"
    echo "   This might indicate remaining issues to investigate"
fi

echo "🚀 FIXED clean experiment ready! Should show different before/after results."

# Cleanup
rm -f run_fixed_clean_experiment.py analyze_fixed_clean_results.py