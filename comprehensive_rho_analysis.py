#!/usr/bin/env python3
"""
Comprehensive Rho Parameter Analysis for Neural Concept Transfer

Tests different rho values (blending weights) on the three key metrics:
1. Knowledge Preservation (target model accuracy on original classes)
2. Transfer Effectiveness (target model accuracy on transfer class) 
3. Transfer Specificity (target model accuracy on non-target classes should stay low)

Creates detailed plots showing the trade-offs between these metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import copy

# Import necessary modules
from architectures import WideNN, DeepNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
from robust_balanced_transfer import RobustBalancedTransferSystem

class RhoAnalysisSystem:
    """System for analyzing rho parameter effects on transfer metrics."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        
    def run_rho_experiment(self, rho_values: List[float], seeds: List[int] = [42, 123, 456]) -> List[Dict]:
        """Run experiments across different rho values and seeds."""
        
        print("🔬 COMPREHENSIVE RHO PARAMETER ANALYSIS")  
        print("=" * 60)
        print(f"Testing rho values: {rho_values}")
        print(f"Seeds: {seeds}")
        print(f"Total experiments: {len(rho_values) * len(seeds)}")
        print()
        
        all_results = []
        experiment_count = 0
        total_experiments = len(rho_values) * len(seeds)
        
        for rho in rho_values:
            for seed in seeds:
                experiment_count += 1
                print(f"[{experiment_count}/{total_experiments}] Testing rho={rho:.3f}, seed={seed}")
                
                result = self._single_rho_experiment(rho, seed)
                if result:
                    all_results.append(result)
                    # Print table format
                    metrics = result['table_metrics']
                    print(f"   ✅ Success - Results Table:")
                    print(f"      | Model        | Original | Class 6 | Class 7 |")
                    print(f"      | Source       | {metrics['source_original_classes']:7.1%} | {metrics['source_class_6']:6.1%} | {metrics['source_class_7']:6.1%} |")
                    print(f"      | Target (Before) | {metrics['target_before_original_classes']:7.1%} | {metrics['target_before_class_6']:6.1%} | {metrics['target_before_class_7']:6.1%} |")
                    print(f"      | Target (After)  | {metrics['target_after_original_classes']:7.1%} | {metrics['target_after_class_6']:6.1%} | {metrics['target_after_class_7']:6.1%} |")
                else:
                    print(f"   ❌ Failed")
                print()
        
        self.results = all_results
        return all_results
    
    def _single_rho_experiment(self, rho_value: float, seed: int) -> Dict:
        """Run a single experiment with specific rho value."""
        
        # Clean experimental setup
        source_classes = {2, 3, 4, 5, 6, 7}  # Source knows class 6
        target_classes = {0, 1, 2, 3, 4, 5}  # Target doesn't know class 6
        transfer_class = 6
        specificity_class = 7
        
        # Update config with current seed
        config = ExperimentConfig(
            seed=seed,
            max_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            concept_dim=self.config.concept_dim,
            device=self.config.device
        )
        
        try:
            # Create data and train models
            data_manager = MNISTDataManager(config)
            trainer = ModelTrainer(config)
            
            source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
                data_manager.get_data_loaders(source_classes, target_classes)
            
            # Train models
            source_model = DeepNN()
            trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
            
            target_model = WideNN()
            trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
            
            if trained_source is None or trained_target is None:
                return None
            
            # CRITICAL: Clone target model before transfer operations
            target_before_transfer = copy.deepcopy(trained_target)
            
            # Create custom transfer system with specific rho value
            transfer_system = CustomRhoTransferSystem(
                source_model=trained_source,
                target_model=trained_target,
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=config.concept_dim,
                device=config.device,
                rho_value=rho_value  # Custom rho value
            )
            
            # Fit and setup transfer
            transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=20)
            transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
            
            # Create target after model (with transfer applied)
            target_after_model = RhoWrappedModel(trained_target, transfer_system, transfer_class)
            
            # Evaluate metrics using simple accuracies
            preservation = self._measure_preservation(target_before_transfer, target_after_model, target_test_loader, target_classes)
            effectiveness = self._measure_effectiveness(target_after_model, source_test_loader, transfer_class)
            specificity = self._measure_specificity(target_after_model, source_test_loader, specificity_class)
            
            # CRITICAL: Measure original knowledge of transfer class (baseline before any transfer)
            original_transfer_knowledge = self._measure_accuracy(target_before_transfer, source_test_loader, {transfer_class})
            original_specificity_knowledge = self._measure_accuracy(target_before_transfer, source_test_loader, {specificity_class})
            
            # Calculate additional metrics for table format
            source_original = self._measure_accuracy(trained_source, source_test_loader, source_classes)
            source_transfer = self._measure_accuracy(trained_source, source_test_loader, {transfer_class})
            source_specificity = self._measure_accuracy(trained_source, source_test_loader, {specificity_class})
            target_before_original = self._measure_accuracy(target_before_transfer, target_test_loader, target_classes)
            
            return {
                'rho': rho_value,
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                # Agreed format metrics - table format
                'table_metrics': {
                    'source_original_classes': source_original,
                    'source_class_6': source_transfer,
                    'source_class_7': source_specificity,
                    'target_before_original_classes': target_before_original,
                    'target_before_class_6': original_transfer_knowledge,
                    'target_before_class_7': original_specificity_knowledge,
                    'target_after_original_classes': preservation,
                    'target_after_class_6': effectiveness,
                    'target_after_class_7': specificity
                },
                # Legacy metrics for plots
                'preservation': preservation,
                'effectiveness': effectiveness,
                'specificity': specificity,
                'original_transfer_knowledge': original_transfer_knowledge,
                'original_specificity_knowledge': original_specificity_knowledge,
                'source_acc': source_acc,
                'target_acc': target_acc
            }
            
        except Exception as e:
            print(f"   Error in experiment: {e}")
            return None
    
    def _measure_preservation(self, target_before_model, target_after_model, target_test_loader, target_classes) -> float:
        """Measure target model accuracy on original classes after transfer (simple accuracy)."""
        return self._measure_accuracy(target_after_model, target_test_loader, target_classes)
    
    def _measure_effectiveness(self, target_after_model, source_test_loader, transfer_class) -> float:
        """Measure target model accuracy on transfer class after transfer (simple accuracy)."""
        return self._measure_accuracy(target_after_model, source_test_loader, {transfer_class})
    
    def _measure_specificity(self, target_after_model, source_test_loader, specificity_class) -> float:
        """Measure target model accuracy on specificity class after transfer (should be low)."""
        return self._measure_accuracy(target_after_model, source_test_loader, {specificity_class})
    
    def _measure_accuracy(self, model, data_loader, target_classes) -> float:
        """Measure model accuracy on specific classes."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                # Filter for target classes
                mask = torch.tensor([label.item() in target_classes for label in labels])
                if mask.sum() == 0:
                    continue
                
                filtered_data = data[mask]
                filtered_labels = labels[mask]
                
                outputs = model(filtered_data.view(filtered_data.size(0), -1))
                _, predicted = torch.max(outputs, 1)
                
                for pred, true in zip(predicted, filtered_labels):
                    if pred.item() == true.item():
                        correct += 1
                    total += 1
                
                if total >= 100:  # Sufficient samples
                    break
        
        return correct / total if total > 0 else 0.0
    
    def create_plots(self, save_dir: Path = None):
        """Create comprehensive plots of rho effects."""
        
        if not self.results:
            print("No results to plot!")
            return
        
        if save_dir is None:
            save_dir = Path("experiment_results/rho_analysis")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to arrays for plotting
        rho_values = np.array([r['rho'] for r in self.results])
        preservation = np.array([r['preservation'] for r in self.results])
        effectiveness = np.array([r['effectiveness'] for r in self.results])
        specificity = np.array([r['specificity'] for r in self.results])
        original_transfer_knowledge = np.array([r['original_transfer_knowledge'] for r in self.results])
        original_specificity_knowledge = np.array([r['original_specificity_knowledge'] for r in self.results])
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Rho Parameter Analysis for Neural Concept Transfer', fontsize=16, y=0.98)
        
        # Plot 1: All metrics vs rho (including original knowledge baseline)
        ax1 = axes[0, 0]
        self._plot_aggregated_metrics(ax1, rho_values, preservation, effectiveness, specificity, original_transfer_knowledge)
        
        # Plot 2: Individual experiment points
        ax2 = axes[0, 1]
        self._plot_individual_points(ax2, rho_values, preservation, effectiveness, specificity, original_transfer_knowledge)
        
        # Plot 3: Transfer improvement analysis (effectiveness vs original knowledge)
        ax3 = axes[1, 0]
        self._plot_transfer_improvement(ax3, rho_values, effectiveness, original_transfer_knowledge)
        
        # Plot 4: Optimal rho analysis
        ax4 = axes[1, 1]
        self._plot_optimal_analysis(ax4, rho_values, preservation, effectiveness, specificity)
        
        plt.tight_layout()
        plot_path = save_dir / "comprehensive_rho_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive plot saved to: {plot_path}")
        
        # Create summary statistics plot
        self._create_summary_plot(save_dir)
        
        # Show plots
        plt.show()
    
    def _plot_aggregated_metrics(self, ax, rho_values, preservation, effectiveness, specificity, original_transfer_knowledge):
        """Plot aggregated metrics with confidence intervals and original knowledge baseline."""
        
        # Group by rho value and calculate statistics
        unique_rhos = np.unique(rho_values)
        pres_means, pres_stds = [], []
        eff_means, eff_stds = [], []
        spec_means, spec_stds = [], []
        orig_means, orig_stds = [], []
        
        for rho in unique_rhos:
            mask = rho_values == rho
            pres_means.append(np.mean(preservation[mask]))
            pres_stds.append(np.std(preservation[mask]))
            eff_means.append(np.mean(effectiveness[mask]))
            eff_stds.append(np.std(effectiveness[mask]))
            spec_means.append(np.mean(specificity[mask]))
            spec_stds.append(np.std(specificity[mask]))
            orig_means.append(np.mean(original_transfer_knowledge[mask]))
            orig_stds.append(np.std(original_transfer_knowledge[mask]))
        
        # Plot with error bars
        ax.errorbar(unique_rhos, pres_means, yerr=pres_stds, label='Knowledge Preservation', marker='o', capsize=5)
        ax.errorbar(unique_rhos, eff_means, yerr=eff_stds, label='Transfer Effectiveness', marker='s', capsize=5)
        ax.errorbar(unique_rhos, spec_means, yerr=spec_stds, label='Transfer Specificity', marker='^', capsize=5)
        
        # Add original knowledge baseline (should be flat across rho values)
        ax.axhline(y=np.mean(orig_means), color='red', linestyle='--', alpha=0.7, 
                  label=f'Original Class 6 Knowledge: {np.mean(orig_means):.1%}')
        ax.fill_between(unique_rhos, 
                       np.mean(orig_means) - np.mean(orig_stds), 
                       np.mean(orig_means) + np.mean(orig_stds), 
                       color='red', alpha=0.1)
        
        ax.set_xlabel('Rho Value (Blending Weight)')
        ax.set_ylabel('Metric Score')
        ax.set_title('Aggregated Metrics vs Rho (with original knowledge baseline)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def _plot_individual_points(self, ax, rho_values, preservation, effectiveness, specificity, original_transfer_knowledge):
        """Plot individual experiment points with original knowledge baseline."""
        
        ax.scatter(rho_values, preservation, alpha=0.6, label='Knowledge Preservation', s=30)
        ax.scatter(rho_values, effectiveness, alpha=0.6, label='Transfer Effectiveness', s=30) 
        ax.scatter(rho_values, specificity, alpha=0.6, label='Transfer Specificity', s=30)
        ax.scatter(rho_values, original_transfer_knowledge, alpha=0.8, label='Original Class 6 Knowledge', s=30, marker='x', color='red')
        
        ax.set_xlabel('Rho Value (Blending Weight)')
        ax.set_ylabel('Metric Score')
        ax.set_title('Individual Experiment Results (with original knowledge)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def _plot_transfer_improvement(self, ax, rho_values, effectiveness, original_transfer_knowledge):
        """Plot transfer improvement analysis (effectiveness vs original knowledge baseline)."""
        
        # Calculate improvement over baseline
        improvement = effectiveness - original_transfer_knowledge
        
        # Color code by rho value
        scatter = ax.scatter(rho_values, improvement, c=rho_values, cmap='viridis', alpha=0.7, s=50)
        
        # Add zero line (no improvement)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
        
        # Add trend line
        z = np.polyfit(rho_values, improvement, 2)  # Quadratic fit
        p = np.poly1d(z)
        x_smooth = np.linspace(rho_values.min(), rho_values.max(), 100)
        ax.plot(x_smooth, p(x_smooth), "orange", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_xlabel('Rho Value (Blending Weight)')
        ax.set_ylabel('Transfer Improvement (Effectiveness - Original)')
        ax.set_title('Transfer Improvement vs Rho')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Rho Value')
        
        # Add statistics text
        max_improvement = np.max(improvement)
        best_rho = rho_values[np.argmax(improvement)]
        ax.text(0.05, 0.95, f'Max Improvement: {max_improvement:.3f}\nBest ρ: {best_rho:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def _plot_tradeoffs(self, ax, preservation, effectiveness, specificity):
        """Plot trade-off relationships between metrics."""
        
        # Preservation vs Effectiveness trade-off
        ax.scatter(preservation, effectiveness, alpha=0.7, s=50, label='Preservation vs Effectiveness')
        
        # Add trend line
        z = np.polyfit(preservation, effectiveness, 1)
        p = np.poly1d(z)
        ax.plot(preservation, p(preservation), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Knowledge Preservation')
        ax.set_ylabel('Transfer Effectiveness')
        ax.set_title('Preservation-Effectiveness Trade-off')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        
        # Add correlation coefficient
        corr = np.corrcoef(preservation, effectiveness)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    
    def _plot_optimal_analysis(self, ax, rho_values, preservation, effectiveness, specificity):
        """Plot analysis to find optimal rho values."""
        
        # Calculate combined score (weighted sum of metrics)
        # Higher preservation and effectiveness, lower specificity is better
        combined_score = 0.4 * preservation + 0.5 * effectiveness + 0.1 * (1 - specificity)
        
        # Group by rho and find means
        unique_rhos = np.unique(rho_values)
        combined_means = []
        
        for rho in unique_rhos:
            mask = rho_values == rho
            combined_means.append(np.mean(combined_score[mask]))
        
        ax.plot(unique_rhos, combined_means, 'o-', linewidth=2, markersize=8, label='Combined Score')
        
        # Highlight optimal rho
        optimal_idx = np.argmax(combined_means)
        optimal_rho = unique_rhos[optimal_idx]
        optimal_score = combined_means[optimal_idx]
        
        ax.axvline(optimal_rho, color='red', linestyle='--', alpha=0.7, label=f'Optimal ρ = {optimal_rho:.3f}')
        ax.scatter([optimal_rho], [optimal_score], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Rho Value (Blending Weight)')
        ax.set_ylabel('Combined Score')
        ax.set_title('Optimal Rho Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation
        ax.text(0.05, 0.95, f'Optimal ρ: {optimal_rho:.3f}\nScore: {optimal_score:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    def _create_summary_plot(self, save_dir: Path):
        """Create a summary statistics plot with original knowledge baseline."""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create box plots for each rho value
        rho_values = np.array([r['rho'] for r in self.results])
        unique_rhos = np.unique(rho_values)
        
        preservation_data = []
        effectiveness_data = []
        specificity_data = []
        original_transfer_data = []
        
        for rho in unique_rhos:
            mask = rho_values == rho
            preservation_data.append([r['preservation'] for r in np.array(self.results)[mask]])
            effectiveness_data.append([r['effectiveness'] for r in np.array(self.results)[mask]])
            specificity_data.append([r['specificity'] for r in np.array(self.results)[mask]])
            original_transfer_data.append([r['original_transfer_knowledge'] for r in np.array(self.results)[mask]])
        
        # Create grouped box plot
        positions1 = np.arange(len(unique_rhos)) * 4
        positions2 = positions1 + 0.8
        positions3 = positions1 + 1.6
        positions4 = positions1 + 2.4
        
        bp1 = ax.boxplot(preservation_data, positions=positions1, widths=0.6, patch_artist=True)
        bp2 = ax.boxplot(effectiveness_data, positions=positions2, widths=0.6, patch_artist=True)
        bp3 = ax.boxplot(specificity_data, positions=positions3, widths=0.6, patch_artist=True)
        bp4 = ax.boxplot(original_transfer_data, positions=positions4, widths=0.6, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        for bp, color in zip([bp1, bp2, bp3, bp4], colors):
            for patch in bp['boxes']:
                patch.set_facecolor(color)
        
        # Customize plot
        ax.set_xticks(positions1 + 1.2)
        ax.set_xticklabels([f'{rho:.3f}' for rho in unique_rhos])
        ax.set_xlabel('Rho Value')
        ax.set_ylabel('Metric Score')
        ax.set_title('Distribution of Metrics Across Rho Values (with original knowledge baseline)')
        
        # Add legend
        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]], 
                 ['Knowledge Preservation', 'Transfer Effectiveness', 'Transfer Specificity', 'Original Class 6 Knowledge'])
        
        ax.grid(True, alpha=0.3)
        
        summary_path = save_dir / "rho_summary_distributions.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary plot saved to: {summary_path}")
        
        plt.show()
    
    def save_results(self, base_dir: Path = None):
        """Save detailed results to experiment_results directory."""
        
        if base_dir is None:
            base_dir = Path("experiment_results/rho_analysis")
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON results
        json_path = base_dir / "comprehensive_rho_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'experiment_info': {
                    'description': 'Comprehensive rho parameter analysis',
                    'total_experiments': len(self.results),
                    'timestamp': datetime.now().isoformat(),
                    'experimental_setup': {
                        'source_classes': [2, 3, 4, 5, 6, 7],
                        'target_classes': [0, 1, 2, 3, 4, 5],
                        'transfer_class': 6,
                        'specificity_class': 7
                    }
                },
                'results': self.results
            }, f, indent=2)
        
        # Create summary report
        self._create_summary_report(base_dir)
        
        print(f"💾 Results saved to: {base_dir}/")
        print(f"   📋 JSON data: {json_path}")
        print(f"   📄 Summary: {base_dir}/rho_analysis_summary.md")
    
    def _create_summary_report(self, base_dir: Path):
        """Create a comprehensive summary report."""
        
        if not self.results:
            return
        
        # Group results by rho value
        rho_groups = {}
        for result in self.results:
            rho = result['rho']
            if rho not in rho_groups:
                rho_groups[rho] = []
            rho_groups[rho].append(result)
        
        report = f"""# Comprehensive Rho Parameter Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experimental Setup

- **Source Classes:** {{2, 3, 4, 5, 6, 7}} (source knows class 6 and 7)
- **Target Classes:** {{0, 1, 2, 3, 4, 5}} (target doesn't know class 6 or 7)
- **Transfer Class:** 6 (should improve from ~0% to >70%)
- **Specificity Class:** 7 (should stay at ~0%)
- **Total Experiments:** {len(self.results)}
- **Rho Values Tested:** {sorted(rho_groups.keys())}

## Results by Rho Value

"""
        
        for rho in sorted(rho_groups.keys()):
            group_results = rho_groups[rho]
            
            # Calculate averages
            avg_metrics = {}
            for key in ['source_original_classes', 'source_class_6', 'source_class_7',
                       'target_before_original_classes', 'target_before_class_6', 'target_before_class_7',
                       'target_after_original_classes', 'target_after_class_6', 'target_after_class_7']:
                values = [r['table_metrics'][key] for r in group_results]
                avg_metrics[key] = sum(values) / len(values)
            
            transfer_improvement = avg_metrics['target_after_class_6'] - avg_metrics['target_before_class_6']
            
            report += f"""### Rho = {rho:.3f} ({len(group_results)} experiments)

| Model                    | Original Classes | Class 6  | Class 7 |
|--------------------------|------------------|----------|----------|
| Source                   |          {avg_metrics['source_original_classes']:.1%} |   {avg_metrics['source_class_6']:.1%} |   {avg_metrics['source_class_7']:.1%} |
| Target (Before)          |          {avg_metrics['target_before_original_classes']:.1%} |   {avg_metrics['target_before_class_6']:.1%} |   {avg_metrics['target_before_class_7']:.1%} |
| Target (After)           |          {avg_metrics['target_after_original_classes']:.1%} |   {avg_metrics['target_after_class_6']:.1%} |   {avg_metrics['target_after_class_7']:.1%} |

**Transfer Improvement:** {transfer_improvement:.1%} (class 6: {avg_metrics['target_before_class_6']:.1%} → {avg_metrics['target_after_class_6']:.1%})

"""
        
        # Find optimal rho
        best_rho = None
        best_improvement = -1
        for rho, group_results in rho_groups.items():
            avg_improvement = sum([r['table_metrics']['target_after_class_6'] - r['table_metrics']['target_before_class_6'] 
                                 for r in group_results]) / len(group_results)
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_rho = rho
        
        report += f"""## Summary

**Best Rho Value:** {best_rho:.3f} (transfer improvement: {best_improvement:.1%})

**Key Findings:**
- Rho controls the blending between original features (rho=1.0) and enhanced features (rho=0.0)
- Lower rho values typically show higher transfer effectiveness but may reduce knowledge preservation
- Higher rho values preserve original knowledge better but may reduce transfer effectiveness

**Files Generated:**
- `comprehensive_rho_analysis_results.json` - Raw experimental data
- `comprehensive_rho_analysis.png` - 4-panel analysis plots
- `rho_summary_distributions.png` - Distribution analysis
- `rho_analysis_summary.md` - This report
"""
        
        report_path = base_dir / "rho_analysis_summary.md"
        with open(report_path, 'w') as f:
            f.write(report)


class CustomRhoTransferSystem(RobustBalancedTransferSystem):
    """Custom transfer system with configurable rho value."""
    
    def __init__(self, source_model, target_model, source_classes, target_classes, 
                 concept_dim=24, device='cpu', rho_value=0.5):
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        self.custom_rho = rho_value
        print(f"   🎛️  CustomRhoTransferSystem initialized with rho={rho_value}")
    
    def transfer(self, target_features):
        """Apply transfer with custom rho value."""
        # Get source features through alignment
        source_concepts = self.target_sae.encode(target_features)
        aligned_concepts = self.aligner.transform(source_concepts)
        enhanced_features = self.target_sae.decode(aligned_concepts)
        
        # Apply custom rho blending - CRITICAL: This must actually vary!
        rho = torch.tensor(self.custom_rho, device=target_features.device, dtype=target_features.dtype)
        final_features = rho * target_features + (1 - rho) * enhanced_features
        
        # Debug: Ensure rho is actually being applied
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            print(f"      🔧 Transfer with rho={self.custom_rho}: original_weight={rho.item():.3f}, enhanced_weight={1-rho.item():.3f}")
        
        return final_features


class RhoWrappedModel(torch.nn.Module):
    """Wrapper to apply transfer system for evaluation."""
    
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
        
        # Method 2: Try feature-level transfer (for CustomRhoTransferSystem)
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


def main():
    """Run comprehensive rho analysis."""
    
    print("🚀 STARTING COMPREHENSIVE RHO PARAMETER ANALYSIS")
    print("=" * 70)
    
    # Configuration
    config = ExperimentConfig(
        seed=42,
        max_epochs=3,  # Faster for parameter sweep
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Test different rho values
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seeds = [42, 123, 456]  # Multiple seeds for reliability
    
    # Run analysis
    analyzer = RhoAnalysisSystem(config)
    results = analyzer.run_rho_experiment(rho_values, seeds)
    
    print(f"\n📊 ANALYSIS COMPLETE!")
    print(f"Successful experiments: {len(results)}/{len(rho_values) * len(seeds)}")
    
    if results:
        # Create plots
        analyzer.create_plots()
        
        # Save results to experiment_results directory
        analyzer.save_results()
        
        # Print summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        print("-" * 30)
        
        for rho in rho_values:
            rho_results = [r for r in results if r['rho'] == rho]
            if rho_results:
                avg_pres = np.mean([r['preservation'] for r in rho_results])
                avg_eff = np.mean([r['effectiveness'] for r in rho_results])
                avg_spec = np.mean([r['specificity'] for r in rho_results])
                print(f"ρ={rho:.1f}: Preservation={avg_pres:.2f}, Effectiveness={avg_eff:.2f}, Specificity={avg_spec:.2f}")
    else:
        print("❌ No successful experiments to analyze!")


if __name__ == "__main__":
    main()