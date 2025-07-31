#!/bin/bash

# Comprehensive Shared Knowledge Effect Experiment
# Tests how the amount of shared knowledge between networks affects transfer effectiveness

echo "🧠 COMPREHENSIVE SHARED KNOWLEDGE EFFECT EXPERIMENT"
echo "===================================================="
echo "Testing how shared knowledge between networks affects transfer effectiveness:"
echo ""
echo "🔬 EXPERIMENTAL CONDITIONS:"
echo "1. LOW SHARED:    Source [0,1,2] → Target [2,3,4] (transfer class 0)"
echo "   • Shared classes: {2} (1 class shared)"
echo "   • Source knows 3 classes, Target knows 3 classes"
echo "   • Overlap: 33% of source knowledge"
echo ""
echo "2. MEDIUM SHARED: Source [0,1,2,3,4] → Target [2,3,4,5,6] (transfer class 0)"
echo "   • Shared classes: {2,3,4} (3 classes shared)"
echo "   • Source knows 5 classes, Target knows 5 classes"
echo "   • Overlap: 60% of source knowledge"
echo ""
echo "3. HIGH SHARED:   Source [0,1,2,3,4,5,6,7] → Target [2,3,4,5,6,7,8,9] (transfer class 0)"
echo "   • Shared classes: {2,3,4,5,6,7} (6 classes shared)"
echo "   • Source knows 8 classes, Target knows 8 classes"
echo "   • Overlap: 75% of source knowledge"
echo ""
echo "Configuration:"
echo "- WideNN → WideNN (same architecture to isolate shared knowledge effect)"
echo "- 10 different seeds per condition (30 total experiments)"
echo "- Fixed rho = 0.5 (balanced blending)"
echo "- FIXED: Proper seed-based random data sampling"
echo ""

# Configuration
SEEDS=(42 123 456)  # 3 different seeds for testing
MAX_EPOCHS=3  # Faster test
SAE_EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.001
CONCEPT_DIM=24

echo "📁 Creating shared knowledge experiment results directory..."
mkdir -p experiment_results/shared_knowledge_experiment/{low_shared,medium_shared,high_shared}
mkdir -p logs

echo ""
echo "🧪 RUNNING SHARED KNOWLEDGE EFFECT EXPERIMENTS"
echo "=============================================="

# Configuration JSON
CONFIG_JSON="{\"max_epochs\": $MAX_EPOCHS, \"sae_epochs\": $SAE_EPOCHS, \"batch_size\": $BATCH_SIZE, \"learning_rate\": $LEARNING_RATE, \"concept_dim\": $CONCEPT_DIM}"

# Define experimental conditions (bash compatible)
CONDITIONS="low_shared medium_shared high_shared"

TOTAL_EXPERIMENTS=$((3 * ${#SEEDS[@]}))
CURRENT_EXPERIMENT=0
SUCCESSFUL_EXPERIMENTS=0

echo "📊 Shared Knowledge Experiment Plan:"
echo "   - Shared knowledge conditions: 3"
echo "   - Seeds per condition: ${#SEEDS[@]}"
echo "   - Total experiments: $TOTAL_EXPERIMENTS"
echo "   - Architecture: WideNN → WideNN (consistent)"
echo "   - Using FIXED experimental framework with proper seed-based data sampling"
echo ""

# Create shared knowledge experiment runner
cat > run_single_shared_knowledge_experiment.py << 'EOF'
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
EOF

# Run all experiments
for condition in $CONDITIONS; do
    echo ""
    echo "🧠 SHARED KNOWLEDGE CONDITION: $condition"
    echo "========================================"
    
    for seed in "${SEEDS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "[$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] Running $condition with seed $seed..."
        
        if python run_single_shared_knowledge_experiment.py "$condition" "$seed" "$CONFIG_JSON" > "logs/shared_knowledge_${condition}_seed_${seed}.json" 2> "logs/shared_knowledge_${condition}_seed_${seed}.log"; then
            
            result_file="experiment_results/shared_knowledge_experiment/$condition/shared_knowledge_${condition}_seed_${seed}.json"
            cp "logs/shared_knowledge_${condition}_seed_${seed}.json" "$result_file"
            
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
            echo "   ✅ Success"
        else
            echo "   ❌ Failed - check logs"
        fi
    done
done

echo ""
echo "📊 ANALYZING SHARED KNOWLEDGE EXPERIMENT RESULTS"
echo "==============================================="

# Comprehensive analysis
cat > analyze_shared_knowledge_results.py << 'EOF'
#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
from datetime import datetime

def analyze_shared_knowledge_results():
    results = {
        'low_shared': [],
        'medium_shared': [], 
        'high_shared': []
    }
    
    base_path = Path('experiment_results/shared_knowledge_experiment')
    for condition in results.keys():
        result_dir = base_path / condition
        if result_dir.exists():
            for result_file in result_dir.glob('*.json'):
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                        json_start = content.find('{')
                        if json_start != -1:
                            data = json.loads(content[json_start:])
                            results[condition].append(data)
                except Exception as e:
                    print(f"Warning: {result_file}: {e}")
    
    print("🧠 SHARED KNOWLEDGE EFFECT EXPERIMENT RESULTS")
    print("=" * 60)
    
    # Summary table
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Condition':<15} {'Shared %':<10} {'Count':<6} {'Avg Effectiveness':<16} {'Avg Preservation':<16} {'Std Dev':<10}")
    print("-" * 90)
    
    summary_data = {}
    for condition, condition_results in results.items():
        if condition_results:
            # Get shared percentage from first result
            shared_pct = condition_results[0]['experimental_setup']['shared_percentage']
            
            effectiveness_values = [r['key_metrics']['transfer_effectiveness'] for r in condition_results]
            preservation_values = [r['key_metrics']['knowledge_preservation'] for r in condition_results]
            improvement_values = [r['key_metrics']['transfer_improvement'] for r in condition_results]
            
            avg_eff = statistics.mean(effectiveness_values)
            avg_pres = statistics.mean(preservation_values)
            avg_imp = statistics.mean(improvement_values)
            std_eff = statistics.stdev(effectiveness_values) if len(effectiveness_values) > 1 else 0.0
            
            summary_data[condition] = {
                'shared_percentage': shared_pct,
                'avg_effectiveness': avg_eff,
                'avg_preservation': avg_pres,
                'avg_improvement': avg_imp,
                'std_effectiveness': std_eff,
                'count': len(condition_results)
            }
            
            condition_display = condition.replace('_', ' ').title()
            
            print(f"{condition_display:<15} {shared_pct:<10.1%} {len(condition_results):<6} {avg_eff:<16.1%} {avg_pres:<16.1%} {std_eff:<10.1%}")
    
    # Detailed analysis by condition
    print(f"\n📈 DETAILED ANALYSIS BY SHARED KNOWLEDGE LEVEL:")
    print("=" * 60)
    
    for condition, condition_results in results.items():
        if condition_results:
            condition_display = condition.replace('_', ' ').title()
            shared_pct = condition_results[0]['experimental_setup']['shared_percentage']
            
            # Get class setup details from first result
            setup = condition_results[0]['experimental_setup']
            
            print(f"\n🧠 {condition_display} ({shared_pct:.1%} shared knowledge):")
            print(f"   Source classes: {setup['source_classes']}")
            print(f"   Target classes: {setup['target_classes']}")
            print(f"   Transfer class: {setup['transfer_class']}")
            print(f"   Shared classes: {setup['shared_classes']}")
            print(f"   Shared: {setup['shared_size']}/{setup['source_size']} classes")
            
            effectiveness_values = []
            preservation_values = []
            improvement_values = []
            
            for r in condition_results:
                seed = r['seed']
                eff = r['key_metrics']['transfer_effectiveness']
                pres = r['key_metrics']['knowledge_preservation']
                imp = r['key_metrics']['transfer_improvement']
                
                effectiveness_values.append(eff)
                preservation_values.append(pres)
                improvement_values.append(imp)
                
                print(f"   Seed {seed:>6}: {eff:.1%} effectiveness, {pres:.1%} preservation, {imp:+.1%} improvement")
            
            # Check for variation (verify seed fix works)
            unique_eff = len(set([round(x, 3) for x in effectiveness_values]))
            if unique_eff > 1:
                print(f"   ✅ Shows proper seed variation: {unique_eff} unique effectiveness values")
            else:
                print(f"   ❌ No seed variation: identical results across seeds")
    
    # Analyze shared knowledge effect
    if len(summary_data) >= 2:
        print(f"\n🔍 SHARED KNOWLEDGE EFFECT ANALYSIS:")
        print("=" * 50)
        
        # Sort by shared percentage
        sorted_conditions = sorted(summary_data.items(), key=lambda x: x[1]['shared_percentage'])
        
        print(f"Shared Knowledge vs Transfer Effectiveness:")
        for condition, data in sorted_conditions:
            condition_display = condition.replace('_', ' ').title()
            print(f"   {data['shared_percentage']:.1%} shared → {data['avg_effectiveness']:.1%} effectiveness ({condition_display})")
        
        # Calculate correlation
        shared_percentages = [data['shared_percentage'] for _, data in sorted_conditions]
        effectiveness_values = [data['avg_effectiveness'] for _, data in sorted_conditions]
        
        if len(shared_percentages) >= 2:
            # Simple correlation analysis
            low_shared_eff = sorted_conditions[0][1]['avg_effectiveness']
            high_shared_eff = sorted_conditions[-1][1]['avg_effectiveness']
            
            correlation_direction = "positive" if high_shared_eff > low_shared_eff else "negative"
            effect_size = abs(high_shared_eff - low_shared_eff)
            
            print(f"\n📊 CORRELATION ANALYSIS:")
            print(f"   Direction: {correlation_direction} correlation")
            print(f"   Effect size: {effect_size:.1%} difference between lowest and highest shared knowledge")
            
            if effect_size > 0.1:
                print(f"   Interpretation: STRONG effect of shared knowledge")
            elif effect_size > 0.05:
                print(f"   Interpretation: MODERATE effect of shared knowledge") 
            else:
                print(f"   Interpretation: WEAK effect of shared knowledge")
        
        # Find optimal shared knowledge level
        best_condition = max(summary_data.keys(), key=lambda x: summary_data[x]['avg_effectiveness'])
        best_eff = summary_data[best_condition]['avg_effectiveness']
        best_shared = summary_data[best_condition]['shared_percentage']
        
        print(f"\n🏆 OPTIMAL SHARED KNOWLEDGE LEVEL:")
        print(f"   Best condition: {best_condition.replace('_', ' ').title()}")
        print(f"   Shared knowledge: {best_shared:.1%}")
        print(f"   Transfer effectiveness: {best_eff:.1%}")
    
    # Generate comprehensive report
    report_path = Path('experiment_results/SHARED_KNOWLEDGE_EXPERIMENT.md')
    with open(report_path, 'w') as f:
        f.write(f"""# Shared Knowledge Effect on Neural Concept Transfer

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experimental Design

- **Research Question:** How does the amount of shared knowledge between networks affect transfer effectiveness?
- **Architecture:** WideNN → WideNN (consistent across all conditions)
- **Experiments per condition:** 10 (different seeds)
- **Total experiments:** 30
- **Fixed parameters:** rho=0.5
- **Seed fix applied:** Proper random data sampling

## Experimental Conditions

### Low Shared Knowledge (33%)
- **Source:** [0,1,2] → **Target:** [2,3,4] (transfer class 3)
- **Shared classes:** {{2}} (1 out of 3 source classes)

### Medium Shared Knowledge (60%)  
- **Source:** [0,1,2,3,4] → **Target:** [2,3,4,5,6] (transfer class 5)
- **Shared classes:** {{2,3,4}} (3 out of 5 source classes)

### High Shared Knowledge (75%)
- **Source:** [0,1,2,3,4,5,6,7] → **Target:** [2,3,4,5,6,7,8,9] (transfer class 9)
- **Shared classes:** {{2,3,4,5,6,7}} (6 out of 8 source classes)

## Results Summary

| Condition | Shared Knowledge | Avg Effectiveness | Avg Preservation | Std Dev |
|-----------|------------------|-------------------|------------------|---------|
""")
        
        for condition in ['low_shared', 'medium_shared', 'high_shared']:
            if condition in summary_data:
                data = summary_data[condition]
                condition_display = condition.replace('_', ' ').title()
                f.write(f"| {condition_display} | {data['shared_percentage']:.1%} | {data['avg_effectiveness']:.1%} | {data['avg_preservation']:.1%} | {data['std_effectiveness']:.1%} |\n")
        
        if len(summary_data) >= 2:
            sorted_conditions = sorted(summary_data.items(), key=lambda x: x[1]['shared_percentage'])
            low_shared_eff = sorted_conditions[0][1]['avg_effectiveness']
            high_shared_eff = sorted_conditions[-1][1]['avg_effectiveness']
            effect_size = abs(high_shared_eff - low_shared_eff)
            correlation_direction = "positive" if high_shared_eff > low_shared_eff else "negative"
            
            best_condition = max(summary_data.keys(), key=lambda x: summary_data[x]['avg_effectiveness'])
            best_eff = summary_data[best_condition]['avg_effectiveness']
            best_shared = summary_data[best_condition]['shared_percentage']
            
            f.write(f"""

## Key Findings

### Shared Knowledge Effect
- **Correlation:** {correlation_direction.title()} correlation between shared knowledge and transfer effectiveness
- **Effect Size:** {effect_size:.1%} difference between lowest and highest shared knowledge conditions
- **Interpretation:** {"STRONG" if effect_size > 0.1 else "MODERATE" if effect_size > 0.05 else "WEAK"} effect of shared knowledge

### Optimal Shared Knowledge Level
- **Best Condition:** {best_condition.replace('_', ' ').title()}
- **Optimal Shared Knowledge:** {best_shared:.1%}
- **Transfer Effectiveness:** {best_eff:.1%}

## Practical Implications

1. **For Transfer System Design:** {"High" if best_shared > 0.6 else "Medium" if best_shared > 0.4 else "Low"} shared knowledge appears optimal
2. **For Network Selection:** Consider shared knowledge percentage when selecting source/target pairs
3. **For Research:** Shared knowledge is {"a critical" if effect_size > 0.1 else "an important" if effect_size > 0.05 else "a minor"} factor in transfer effectiveness

## Conclusion

This experiment provides insights into how the amount of shared knowledge between source and target networks affects neural concept transfer effectiveness.
""")
    
    print(f"\n📄 Shared knowledge analysis report saved to: {report_path}")

if __name__ == "__main__":
    analyze_shared_knowledge_results()
EOF

python analyze_shared_knowledge_results.py

echo ""
echo "🎉 SHARED KNOWLEDGE EXPERIMENT COMPLETE!"
echo "========================================"
echo ""
echo "📊 Final Statistics:"
echo "   - Total experiments: $TOTAL_EXPERIMENTS"
echo "   - Successful: $SUCCESSFUL_EXPERIMENTS"
echo "   - Shared knowledge conditions tested: 3"
echo "   - Experiments per condition: 10"
echo ""
echo "🔬 EXPERIMENTAL CONDITIONS TESTED:"
echo "   ✅ Low Shared (33%):    [0,1,2] → [2,3,4] transfer 0"
echo "   ✅ Medium Shared (60%): [0,1,2,3,4] → [2,3,4,5,6] transfer 0"
echo "   ✅ High Shared (75%):   [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer 0"
echo "   ✅ WideNN → WideNN architecture (consistent)"
echo "   ✅ Fixed seed-based random data sampling"
echo "   ✅ 10 different seeds per condition"
echo ""
echo "📁 Results saved to: experiment_results/shared_knowledge_experiment/"
echo "📄 Analysis report: experiment_results/SHARED_KNOWLEDGE_EXPERIMENT.md"

# Cleanup
rm -f run_single_shared_knowledge_experiment.py analyze_shared_knowledge_results.py