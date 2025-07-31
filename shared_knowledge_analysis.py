"""
Shared Knowledge Analysis for Wide->Wide Neural Concept Transfer
This script tests the required shared knowledge between networks with different levels of overlap.

Test configurations:
1. [0,1,2] -> [2,3,4]: Transfer 3 (minimal overlap - only class 2 shared)
2. [0,1,2,3,4] -> [2,3,4,5,6]: Transfer 5 (moderate overlap - classes 2,3,4 shared)  
3. [0,1,2,3,4,5,6,7] -> [2,3,4,5,6,7,8,9]: Transfer 8 (high overlap - classes 2,3,4,5,6,7 shared)
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
import matplotlib.pyplot as plt
import seaborn as sns

from experimental_framework import (
    ExperimentConfig, ExperimentRunner, ExperimentResult,
    TransferMetrics
)

# Configure logging with specific format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedKnowledgeAnalysis:
    """Analyzes the relationship between shared knowledge and transfer effectiveness."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results_dir = Path("experiment_results/shared_knowledge_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define test configurations
        self.test_configs = [
            {
                'name': 'minimal_overlap',
                'source_classes': {0, 1, 2},
                'target_classes': {2, 3, 4},
                'transfer_class': 0,  # Transfer class 0 from source to target
                'shared_classes': {2},
                'overlap_ratio': 1/3  # 1 out of 3 source classes
            },
            {
                'name': 'moderate_overlap',
                'source_classes': {0, 1, 2, 3, 4},
                'target_classes': {2, 3, 4, 5, 6},
                'transfer_class': 1,  # Transfer class 1 from source to target
                'shared_classes': {2, 3, 4},
                'overlap_ratio': 3/5  # 3 out of 5 source classes
            },
            {
                'name': 'high_overlap',
                'source_classes': {0, 1, 2, 3, 4, 5, 6, 7},
                'target_classes': {2, 3, 4, 5, 6, 7, 8, 9},
                'transfer_class': 1,  # Transfer class 1 from source to target
                'shared_classes': {2, 3, 4, 5, 6, 7},
                'overlap_ratio': 6/8  # 6 out of 8 source classes
            }
        ]
        
    def run_all_experiments(self, num_experiments_per_config: int = 10):
        """Run all shared knowledge experiments."""
        all_results = {}
        
        for config in self.test_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting experiments for {config['name']}")
            logger.info(f"Source classes: {sorted(config['source_classes'])}")
            logger.info(f"Target classes: {sorted(config['target_classes'])}")
            logger.info(f"Transfer class: {config['transfer_class']}")
            logger.info(f"Shared classes: {sorted(config['shared_classes'])}")
            logger.info(f"Overlap ratio: {config['overlap_ratio']:.2%}")
            logger.info(f"{'='*80}")
            
            config_results = self.run_config_experiments(
                config, num_experiments_per_config
            )
            all_results[config['name']] = config_results
            
        # Save comprehensive results
        self.save_comprehensive_results(all_results)
        
        # Generate analysis and plots
        self.analyze_results(all_results)
        
        return all_results
    
    def run_config_experiments(self, config: Dict, num_experiments: int) -> Dict:
        """Run multiple experiments for a single configuration."""
        results = []
        successful_runs = 0
        
        experiment_config = ExperimentConfig(
            seed=self.base_seed,
            num_pairs=1,  # We'll run individual experiments
            max_epochs=5,
            min_accuracy_threshold=0.90
        )
        
        runner = ExperimentRunner(experiment_config)
        
        for exp_id in range(num_experiments):
            # Set unique seed for each experiment
            seed = self.base_seed + exp_id * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            logger.info(f"\nRunning experiment {exp_id + 1}/{num_experiments} for {config['name']}")
            
            try:
                result = runner.run_single_experiment(
                    pair_id=exp_id + 1,
                    source_arch="WideNN",
                    target_arch="WideNN",
                    source_classes=config['source_classes'],
                    target_classes=config['target_classes'],
                    transfer_class=config['transfer_class']
                )
                
                if result is not None:
                    results.append(result)
                    successful_runs += 1
                    
                    # Log immediate results
                    logger.info(f"Transfer effectiveness: {result.after_metrics.knowledge_transfer:.4f}")
                    logger.info(f"Improvement: {result.after_metrics.knowledge_transfer - result.before_metrics.knowledge_transfer:.4f}")
                    
            except Exception as e:
                logger.error(f"Experiment {exp_id + 1} failed: {e}")
                continue
        
        # Compute statistics
        if results:
            knowledge_improvements = [
                r.after_metrics.knowledge_transfer - r.before_metrics.knowledge_transfer 
                for r in results
            ]
            final_knowledge = [r.after_metrics.knowledge_transfer for r in results]
            
            stats = {
                'config': config,
                'num_successful': successful_runs,
                'num_attempted': num_experiments,
                'success_rate': successful_runs / num_experiments,
                'knowledge_transfer': {
                    'final_mean': float(np.mean(final_knowledge)),
                    'final_std': float(np.std(final_knowledge)),
                    'final_min': float(np.min(final_knowledge)),
                    'final_max': float(np.max(final_knowledge)),
                    'improvement_mean': float(np.mean(knowledge_improvements)),
                    'improvement_std': float(np.std(knowledge_improvements))
                },
                'raw_results': [r.to_dict() for r in results]
            }
        else:
            stats = {
                'config': config,
                'num_successful': 0,
                'num_attempted': num_experiments,
                'success_rate': 0.0,
                'error': 'No successful experiments'
            }
        
        # Convert sets to lists for JSON serialization
        if 'config' in stats:
            stats_for_json = dict(stats)
            stats_for_json['config'] = dict(stats['config'])
            for key, value in stats_for_json['config'].items():
                if isinstance(value, set):
                    stats_for_json['config'][key] = list(value)
        else:
            stats_for_json = stats
        
        # Save config-specific results
        config_file = self.results_dir / f"{config['name']}_results.json"
        with open(config_file, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        
        return stats
    
    def save_comprehensive_results(self, all_results: Dict):
        """Save all results in a comprehensive format."""
        timestamp = datetime.now().isoformat()
        
        comprehensive = {
            'experiment': 'Shared Knowledge Analysis for Wide->Wide Transfer',
            'timestamp': timestamp,
            'configurations': len(self.test_configs),
            'results_by_config': all_results,
            'summary': self.generate_summary(all_results)
        }
        
        # Save comprehensive results
        output_file = self.results_dir / "comprehensive_shared_knowledge_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive, f, indent=2)
        
        logger.info(f"\nComprehensive results saved to: {output_file}")
    
    def generate_summary(self, all_results: Dict) -> Dict:
        """Generate summary statistics across all configurations."""
        summary = {
            'overlap_vs_transfer': {},
            'trends': {}
        }
        
        # Extract key metrics for each configuration
        for config_name, results in all_results.items():
            if 'knowledge_transfer' in results:
                config = results['config']
                summary['overlap_vs_transfer'][config_name] = {
                    'overlap_ratio': config['overlap_ratio'],
                    'num_shared_classes': len(config['shared_classes']),
                    'transfer_effectiveness': results['knowledge_transfer']['final_mean'],
                    'transfer_std': results['knowledge_transfer']['final_std'],
                    'improvement': results['knowledge_transfer']['improvement_mean']
                }
        
        # Analyze trends
        if len(summary['overlap_vs_transfer']) >= 2:
            overlap_ratios = [v['overlap_ratio'] for v in summary['overlap_vs_transfer'].values()]
            transfer_effects = [v['transfer_effectiveness'] for v in summary['overlap_vs_transfer'].values()]
            
            # Simple correlation
            if len(overlap_ratios) > 1:
                correlation = np.corrcoef(overlap_ratios, transfer_effects)[0, 1]
                summary['trends']['overlap_transfer_correlation'] = float(correlation)
        
        return summary
    
    def analyze_results(self, all_results: Dict):
        """Analyze results and create visualizations."""
        # Prepare data for plotting
        overlap_ratios = []
        transfer_means = []
        transfer_stds = []
        config_names = []
        
        for config_name, results in all_results.items():
            if 'knowledge_transfer' in results:
                overlap_ratios.append(results['config']['overlap_ratio'])
                transfer_means.append(results['knowledge_transfer']['final_mean'])
                transfer_stds.append(results['knowledge_transfer']['final_std'])
                config_names.append(config_name)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot with error bars
        plt.errorbar(overlap_ratios, transfer_means, yerr=transfer_stds, 
                    marker='o', markersize=10, capsize=5, capthick=2, 
                    linewidth=2, label='Transfer Effectiveness')
        
        # Add configuration labels
        for i, name in enumerate(config_names):
            plt.annotate(name.replace('_', ' ').title(), 
                        (overlap_ratios[i], transfer_means[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Shared Knowledge Ratio (Overlap)', fontsize=12)
        plt.ylabel('Transfer Effectiveness', fontsize=12)
        plt.title('Shared Knowledge vs Transfer Effectiveness\nWide→Wide Architecture', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set axis limits
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        
        # Save plot
        plot_file = self.results_dir / "shared_knowledge_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"\nAnalysis plot saved to: {plot_file}")
        
        # Print summary
        self.print_analysis_summary(all_results)
    
    def print_analysis_summary(self, all_results: Dict):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("SHARED KNOWLEDGE ANALYSIS SUMMARY")
        print("="*80)
        
        for config_name, results in all_results.items():
            if 'knowledge_transfer' not in results:
                continue
                
            config = results['config']
            kt = results['knowledge_transfer']
            
            print(f"\n{config_name.replace('_', ' ').upper()}")
            print(f"  Source classes: {sorted(config['source_classes'])}")
            print(f"  Target classes: {sorted(config['target_classes'])}")
            print(f"  Transfer class: {config['transfer_class']}")
            print(f"  Shared classes: {sorted(config['shared_classes'])}")
            print(f"  Overlap ratio: {config['overlap_ratio']:.2%}")
            print(f"  Success rate: {results['success_rate']:.2%}")
            print(f"  Transfer effectiveness: {kt['final_mean']:.4f} ± {kt['final_std']:.4f}")
            print(f"  Range: [{kt['final_min']:.4f}, {kt['final_max']:.4f}]")
            print(f"  Average improvement: {kt['improvement_mean']:.4f}")
        
        print("\n" + "="*80)


def main():
    """Run the shared knowledge analysis experiments."""
    logger.info("Starting Shared Knowledge Analysis for Wide->Wide Transfer")
    
    analyzer = SharedKnowledgeAnalysis(base_seed=42)
    
    # Run all experiments (10 per configuration as requested)
    results = analyzer.run_all_experiments(num_experiments_per_config=10)
    
    logger.info("\nAll experiments completed successfully!")
    logger.info(f"Results saved in: {analyzer.results_dir}")


if __name__ == "__main__":
    main()