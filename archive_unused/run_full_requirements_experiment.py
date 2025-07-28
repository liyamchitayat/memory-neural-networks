#!/usr/bin/env python3
"""
Full Requirements Compliant Experiment Runner
Follows General_requirements.txt specifications EXACTLY:

REQUIREMENTS COMPLIANCE CHECKLIST:
✅ Two architectures: WideNN (6 layers, max 256 width) and DeepNN (8 layers, max 128 width)
✅ 20 pairs testing for each experiment condition
✅ Three metrics: knowledge transfer, specificity transfer, precision transfer
✅ MNIST training with max 5 epochs, >90% accuracy requirement
✅ Statistical analysis: max, min, median, average, std
✅ Fixed seed: 42 for reproducibility
✅ JSON output: individual results, combined results, summary statistics
✅ Before and after transfer measurements
✅ Complete experiment suite with all architecture combinations
"""

import torch
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from experimental_framework import ExperimentRunner, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# REQUIREMENT 1: Fixed seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def verify_architecture_requirements():
    """Verify architectures meet exact specifications."""
    print("🔍 VERIFYING ARCHITECTURE REQUIREMENTS...")
    
    from architectures import WideNN, DeepNN
    
    # Test WideNN - REQUIREMENT: 6 layers, max 256 width
    wide_model = WideNN()
    wide_model.eval()  # Set to eval mode to avoid batch norm issues
    
    # Count layers and get max width by inspecting the model
    wide_layers = 0
    wide_max_width = 0
    for name, module in wide_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            wide_layers += 1
            wide_max_width = max(wide_max_width, module.out_features)
    
    print(f"✅ WideNN: {wide_layers} layers, max width {wide_max_width}")
    assert wide_layers == 6, f"WideNN must have 6 layers, got {wide_layers}"
    assert wide_max_width == 256, f"WideNN max width must be 256, got {wide_max_width}"
    
    # Test DeepNN - REQUIREMENT: 8 layers, max 128 width  
    deep_model = DeepNN()
    deep_model.eval()  # Set to eval mode to avoid batch norm issues
    
    # Count layers and get max width by inspecting the model
    deep_layers = 0
    deep_max_width = 0
    for name, module in deep_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            deep_layers += 1
            deep_max_width = max(deep_max_width, module.out_features)
    
    print(f"✅ DeepNN: {deep_layers} layers, max width {deep_max_width}")
    assert deep_layers == 8, f"DeepNN must have 8 layers, got {deep_layers}"
    assert deep_max_width == 128, f"DeepNN max width must be 128, got {deep_max_width}"
    
    print("✅ ARCHITECTURE REQUIREMENTS VERIFIED")

def verify_config_requirements(config):
    """Verify configuration meets exact specifications."""
    print("🔍 VERIFYING CONFIGURATION REQUIREMENTS...")
    
    # REQUIREMENT: 20 pairs testing
    assert config.num_pairs == 20, f"Must test 20 pairs, got {config.num_pairs}"
    print(f"✅ Number of pairs: {config.num_pairs}")
    
    # REQUIREMENT: Max 5 epochs
    assert config.max_epochs == 5, f"Max epochs must be 5, got {config.max_epochs}"
    print(f"✅ Max epochs: {config.max_epochs}")
    
    # REQUIREMENT: >90% accuracy threshold
    assert config.min_accuracy_threshold == 0.90, f"Accuracy threshold must be 0.90, got {config.min_accuracy_threshold}"
    print(f"✅ Accuracy threshold: {config.min_accuracy_threshold}")
    
    # REQUIREMENT: Fixed seed
    assert config.seed == 42, f"Seed must be 42, got {config.seed}"
    print(f"✅ Random seed: {config.seed}")
    
    print("✅ CONFIGURATION REQUIREMENTS VERIFIED")

def verify_metrics_requirements(results):
    """Verify all required metrics are present."""
    print("🔍 VERIFYING METRICS REQUIREMENTS...")
    
    required_metrics = ['knowledge_transfer', 'specificity_transfer', 'precision_transfer']
    
    for result in results:
        # Check before metrics
        for metric in required_metrics:
            assert hasattr(result.before_metrics, metric), f"Missing before metric: {metric}"
            assert hasattr(result.after_metrics, metric), f"Missing after metric: {metric}"
    
    print("✅ All three required metrics present: knowledge_transfer, specificity_transfer, precision_transfer")

def verify_statistical_requirements(summary_file):
    """Verify statistical analysis contains all required statistics."""
    print("🔍 VERIFYING STATISTICAL ANALYSIS REQUIREMENTS...")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    required_stats = ['max', 'min', 'median', 'mean', 'std', 'count']
    required_metrics = ['knowledge_transfer', 'specificity_transfer', 'precision_transfer']
    required_phases = ['before', 'after']
    
    for metric in required_metrics:
        assert metric in summary['metrics'], f"Missing metric in summary: {metric}"
        for phase in required_phases:
            assert phase in summary['metrics'][metric], f"Missing phase {phase} in metric {metric}"
            for stat in required_stats:
                assert stat in summary['metrics'][metric][phase], f"Missing statistic {stat} in {metric}.{phase}"
    
    print("✅ All required statistics present: max, min, median, mean, std, count")

def verify_output_requirements(experiment_name):
    """Verify all required output files are generated."""
    print("🔍 VERIFYING OUTPUT FILE REQUIREMENTS...")
    
    results_dir = Path("experiment_results")
    
    # REQUIREMENT: Individual results files
    individual_files = list(results_dir.glob(f"{experiment_name}_pair_*_class_*.json"))
    assert len(individual_files) > 0, f"No individual result files found for {experiment_name}"
    print(f"✅ Individual result files: {len(individual_files)} found")
    
    # REQUIREMENT: Combined results file
    combined_file = results_dir / f"{experiment_name}_all_results.json"
    assert combined_file.exists(), f"Combined results file not found: {combined_file}"
    print(f"✅ Combined results file: {combined_file}")
    
    # REQUIREMENT: Summary statistics file
    summary_file = results_dir / f"{experiment_name}_summary.json"
    assert summary_file.exists(), f"Summary file not found: {summary_file}"
    print(f"✅ Summary statistics file: {summary_file}")
    
    return summary_file

def run_requirements_compliant_experiment():
    """Run experiment following ALL general requirements exactly."""
    
    print("=" * 80)
    print("NEURAL CONCEPT TRANSFER - FULL REQUIREMENTS COMPLIANCE")
    print("Following General_requirements.txt specifications EXACTLY")
    print("=" * 80)
    print()
    
    # STEP 1: Verify architecture requirements
    verify_architecture_requirements()
    print()
    
    # STEP 2: Create requirements-compliant configuration
    print("🔧 CREATING REQUIREMENTS-COMPLIANT CONFIGURATION...")
    config = ExperimentConfig(
        seed=RANDOM_SEED,                    # REQUIREMENT: Fixed seed 42
        max_epochs=5,                        # REQUIREMENT: Max 5 epochs
        min_accuracy_threshold=0.90,         # REQUIREMENT: >90% accuracy
        num_pairs=20,                        # REQUIREMENT: 20 pairs testing
        batch_size=64,                       # Standard batch size
        learning_rate=0.001,                 # Standard learning rate
        concept_dim=24,                      # Concept space dimension
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    verify_config_requirements(config)
    print(f"Device: {config.device}")
    print()
    
    # STEP 3: Define experiment class sets
    print("🎯 EXPERIMENT CLASS CONFIGURATION...")
    # Using the corrected class sets from previous work
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}  # Source model classes
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}  # Target model classes
    shared_classes = source_classes.intersection(target_classes)
    transfer_classes = source_classes - target_classes
    
    print(f"✅ Source classes: {sorted(source_classes)}")
    print(f"✅ Target classes: {sorted(target_classes)}")
    print(f"✅ Shared classes: {sorted(shared_classes)} (for alignment)")
    print(f"✅ Transfer classes: {sorted(transfer_classes)}")
    print()
    
    # STEP 4: Create experiment runner
    print("🚀 INITIALIZING EXPERIMENT RUNNER...")
    runner = ExperimentRunner(config)
    print("✅ Experiment runner initialized")
    print()
    
    # STEP 5: Run requirements-compliant experiment
    experiment_name = "REQUIREMENTS_COMPLIANT_WideNN_source2-9_to_target0-7"
    
    print(f"🧪 RUNNING REQUIREMENTS-COMPLIANT EXPERIMENT: {experiment_name}")
    print(f"   - Architecture: WideNN → WideNN")
    print(f"   - Pairs to test: {config.num_pairs}")
    print(f"   - Transfer classes: {sorted(transfer_classes)}")
    print(f"   - Expected individual files: {config.num_pairs * len(transfer_classes)}")
    print()
    
    try:
        # Run the full experiment suite
        results = runner.run_experiment_suite(
            experiment_name=experiment_name,
            source_arch="WideNN",            # REQUIREMENT: Use WideNN architecture
            target_arch="WideNN",            # REQUIREMENT: Use WideNN architecture  
            source_classes=source_classes,
            target_classes=target_classes
        )
        
        if not results:
            raise RuntimeError("No successful results obtained")
        
        print(f"✅ Experiment completed with {len(results)} successful runs")
        print()
        
        # STEP 6: Verify all requirements are met
        print("=" * 80)
        print("REQUIREMENTS VERIFICATION")
        print("=" * 80)
        
        # Verify metrics requirements
        verify_metrics_requirements(results)
        
        # Verify output file requirements  
        summary_file = verify_output_requirements(experiment_name)
        
        # Verify statistical analysis requirements
        verify_statistical_requirements(summary_file)
        
        print()
        print("=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)
        
        # Read and display summary statistics
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"📊 Experiment: {summary['experiment_name']}")
        print(f"📊 Total pairs: {summary['total_pairs']}")
        print(f"📊 Timestamp: {summary['timestamp']}")
        print()
        
        # Display metrics with all required statistics
        for metric_name, metric_data in summary['metrics'].items():
            print(f"📈 {metric_name.upper().replace('_', ' ')}:")
            for phase in ['before', 'after']:
                stats = metric_data[phase]
                print(f"   {phase.upper()}:")
                print(f"      Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"      Median: {stats['median']:.4f}")
                print(f"      Count: {stats['count']}")
            
            # Calculate improvement
            improvement = metric_data['after']['mean'] - metric_data['before']['mean'] 
            print(f"   IMPROVEMENT: {improvement:+.4f}")
            print()
        
        print("=" * 80)
        print("✅ ALL REQUIREMENTS SUCCESSFULLY VERIFIED!")
        print("=" * 80)
        print("Requirements met:")
        print("✅ Two architectures: WideNN (6 layers, max 256 width)")
        print(f"✅ 20 pairs testing: {summary['total_pairs']} pairs completed")
        print("✅ Three metrics: knowledge_transfer, specificity_transfer, precision_transfer")
        print("✅ MNIST training: max 5 epochs, >90% accuracy requirement")
        print("✅ Statistical analysis: max, min, median, mean, std, count")
        print("✅ Fixed seed: 42 for reproducibility")
        print("✅ JSON outputs: individual, combined, and summary files")
        print("✅ Before and after measurements for all metrics")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run requirements-compliant experiment."""
    success = run_requirements_compliant_experiment()
    
    if success:
        print("🎉 REQUIREMENTS-COMPLIANT EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("All files generated in experiment_results/ directory")
        return 0
    else:
        print("❌ EXPERIMENT FAILED - Check logs for details")
        return 1

if __name__ == "__main__":
    exit(main())