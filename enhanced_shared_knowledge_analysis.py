"""
Enhanced Shared Knowledge Analysis for Wide->Wide Neural Concept Transfer
This script includes detailed statistical method explanations and consistent logging format.

STATISTICAL METHODS DOCUMENTATION:
==================================

1. DESCRIPTIVE STATISTICS:
   - Mean (μ): Sample arithmetic mean = (1/n) * Σ(xi) where n=sample size, xi=individual measurements
   - Standard Deviation (σ): Population std = sqrt((1/n) * Σ(xi - μ)²)
   - Min/Max: Simple range boundaries from sorted sample data
   - Median: Middle value when n observations are sorted (50th percentile)

2. CORRELATION ANALYSIS:
   - Pearson Correlation Coefficient (r): r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
   - Where x̄, ȳ are sample means of variables X, Y
   - Interpretation: |r| > 0.7 = strong, 0.3-0.7 = moderate, <0.3 = weak correlation

3. EXPERIMENTAL DESIGN:
   - Factorial design: 3 overlap levels × 10 replications per level
   - Control variables: Same architecture (WideNN), same training parameters, same dataset (MNIST)
   - Independent variable: Overlap ratio (proportion of shared classes)
   - Dependent variable: Transfer effectiveness (proportion correct on transferred class)

4. ERROR ANALYSIS:
   - Standard Error of Mean: SEM = σ/√n where σ=sample std, n=sample size
   - Confidence intervals: μ ± t(α/2,df) * SEM for 95% CI with t-distribution
   - Statistical significance: Not applicable (descriptive study, no hypothesis testing)
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
from scipy import stats

from experimental_framework import (
    ExperimentConfig, ExperimentRunner, ExperimentResult,
    TransferMetrics
)

# Configure logging with consistent format matching previous experiments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Provides detailed statistical analysis with method explanations."""
    
    @staticmethod
    def calculate_descriptive_stats(values: List[float]) -> Dict:
        """
        Calculate comprehensive descriptive statistics.
        
        DERIVATION:
        - Mean: μ = (1/n) * Σ(xi) for i=1 to n
        - Variance: σ² = (1/n) * Σ(xi - μ)² for i=1 to n  
        - Standard Deviation: σ = sqrt(σ²)
        - Standard Error: SE = σ/√n
        - Median: Middle value of sorted array (50th percentile)
        - Quartiles: 25th and 75th percentiles for spread analysis
        """
        n = len(values)
        if n == 0:
            return {'error': 'Empty dataset'}
        
        # Convert to numpy for vectorized operations
        data = np.array(values)
        
        # Basic descriptive statistics
        mean = np.mean(data)
        std = np.std(data, ddof=0)  # Population standard deviation
        var = np.var(data, ddof=0)  # Population variance
        sem = std / np.sqrt(n)      # Standard error of mean
        
        # Distribution statistics
        median = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25
        
        # Range statistics
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        
        # Confidence interval (95%, assuming normal distribution)
        ci_margin = stats.t.ppf(0.975, n-1) * sem if n > 1 else 0
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin
        
        return {
            'n': n,
            'mean': float(mean),
            'std': float(std),
            'var': float(var),
            'sem': float(sem),
            'median': float(median),
            'q25': float(q25),
            'q75': float(q75),
            'iqr': float(iqr),
            'min': float(min_val),
            'max': float(max_val),
            'range': float(range_val),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'method_explanation': {
                'mean': 'Arithmetic average: sum of all values divided by count',
                'std': 'Population standard deviation: sqrt(sum((x-mean)²)/n)',
                'sem': 'Standard error of mean: std/sqrt(n), measures precision of sample mean',
                'ci_95': 'Two-sided 95% confidence interval using t-distribution',
                'iqr': 'Interquartile range: Q3-Q1, robust measure of spread'
            }
        }
    
    @staticmethod
    def calculate_correlation_analysis(x_values: List[float], y_values: List[float]) -> Dict:
        """
        Calculate Pearson correlation coefficient with detailed explanation.
        
        DERIVATION:
        Pearson r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
        
        Where:
        - xi, yi are individual data points
        - x̄, ȳ are sample means
        - Numerator: covariance between X and Y
        - Denominator: product of standard deviations
        
        INTERPRETATION:
        - r = 1: Perfect positive linear relationship
        - r = 0: No linear relationship  
        - r = -1: Perfect negative linear relationship
        - |r| > 0.7: Strong correlation
        - 0.3 < |r| < 0.7: Moderate correlation
        - |r| < 0.3: Weak correlation
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return {'error': 'Invalid data for correlation analysis'}
        
        # Convert to numpy arrays
        x = np.array(x_values)
        y = np.array(y_values)
        n = len(x)
        
        # Calculate correlation coefficient
        correlation_matrix = np.corrcoef(x, y)
        r = correlation_matrix[0, 1]
        
        # Calculate coefficient of determination
        r_squared = r ** 2
        
        # Interpret strength
        abs_r = abs(r)
        if abs_r > 0.7:
            strength = 'strong'
        elif abs_r > 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Direction
        direction = 'positive' if r > 0 else 'negative' if r < 0 else 'none'
        
        # Statistical significance (t-test for correlation)
        if n > 2:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            t_stat = np.nan
            p_value = np.nan
        
        return {
            'correlation_coefficient': float(r),
            'r_squared': float(r_squared),
            'strength': strength,
            'direction': direction,
            'n_pairs': n,
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'method_explanation': {
                'formula': 'r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)',
                'interpretation': f'{strength.title()} {direction} linear relationship',
                'r_squared_meaning': 'Proportion of variance in Y explained by X',
                'significance_test': 'Two-tailed t-test for correlation coefficient'
            }
        }


class EnhancedSharedKnowledgeAnalysis:
    """Enhanced analysis with detailed statistical methods and consistent logging."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results_dir = Path("experiment_results/shared_knowledge_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure detailed logging
        self.setup_logging()
        
        # Define test configurations with detailed documentation
        self.test_configs = [
            {
                'name': 'minimal_overlap',
                'source_classes': {0, 1, 2},
                'target_classes': {2, 3, 4},
                'transfer_class': 0,
                'shared_classes': {2},
                'overlap_ratio': 1/3,
                'description': 'Minimal shared knowledge: only 1 class overlap'
            },
            {
                'name': 'moderate_overlap',
                'source_classes': {0, 1, 2, 3, 4},
                'target_classes': {2, 3, 4, 5, 6},
                'transfer_class': 1,
                'shared_classes': {2, 3, 4},
                'overlap_ratio': 3/5,
                'description': 'Moderate shared knowledge: 3 classes overlap'
            },
            {
                'name': 'high_overlap',
                'source_classes': {0, 1, 2, 3, 4, 5, 6, 7},
                'target_classes': {2, 3, 4, 5, 6, 7, 8, 9},
                'transfer_class': 1,
                'shared_classes': {2, 3, 4, 5, 6, 7},
                'overlap_ratio': 6/8,
                'description': 'High shared knowledge: 6 classes overlap'
            }
        ]
        
        self.analyzer = StatisticalAnalyzer()
    
    def setup_logging(self):
        """Setup detailed logging matching previous experiment format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"enhanced_shared_knowledge_{timestamp}.log"
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Log initial setup
        logger.info("="*80)
        logger.info("🔬 ENHANCED SHARED KNOWLEDGE ANALYSIS INITIALIZED")
        logger.info("="*80)
        logger.info(f"📁 Results directory: {self.results_dir}")
        logger.info(f"📁 Logs directory: {self.logs_dir}")
        logger.info(f"📋 Log file: {log_file}")
        logger.info(f"🌱 Base seed: {self.base_seed}")
        logger.info(f"🧪 Test configurations: {len(self.test_configs)}")
    
    def run_config_experiments(self, config: Dict, num_experiments: int) -> Dict:
        """Run experiments for a single configuration with detailed logging."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 STARTING CONFIGURATION: {config['name'].upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"📊 Configuration Details:")
        logger.info(f"   Source classes: {sorted(config['source_classes'])}")
        logger.info(f"   Target classes: {sorted(config['target_classes'])}")
        logger.info(f"   Transfer class: {config['transfer_class']}")
        logger.info(f"   Shared classes: {sorted(config['shared_classes'])}")
        logger.info(f"   Overlap ratio: {config['overlap_ratio']:.1%}")
        logger.info(f"   Description: {config['description']}")
        logger.info(f"   Planned experiments: {num_experiments}")
        
        results = []
        successful_runs = 0
        transfer_effectiveness_values = []
        
        experiment_config = ExperimentConfig(
            seed=self.base_seed,
            num_pairs=1,
            max_epochs=5,
            min_accuracy_threshold=0.90
        )
        
        runner = ExperimentRunner(experiment_config)
        
        for exp_id in range(num_experiments):
            seed = self.base_seed + exp_id * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            logger.info(f"\n🔬 Experiment {exp_id + 1}/{num_experiments} (seed={seed})")
            
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
                    transfer_effectiveness = result.after_metrics.knowledge_transfer
                    transfer_effectiveness_values.append(transfer_effectiveness)
                    
                    # Log consistent with previous experiments
                    logger.info(f"   ✅ Experiment {exp_id + 1} completed successfully")
                    logger.info(f"   📈 Transfer effectiveness: {transfer_effectiveness:.4f}")
                    logger.info(f"   📊 Improvement: {transfer_effectiveness - result.before_metrics.knowledge_transfer:.4f}")
                    logger.info(f"   🎯 Source accuracy: {result.source_accuracy:.4f}")
                    logger.info(f"   🎯 Target accuracy: {result.target_accuracy:.4f}")
                    logger.info(f"   🔗 Alignment error: {result.alignment_error:.4f}")
                else:
                    logger.warning(f"   ❌ Experiment {exp_id + 1} failed")
                    
            except Exception as e:
                logger.error(f"   ❌ Experiment {exp_id + 1} failed with error: {e}")
                continue
        
        # Calculate detailed statistics
        if transfer_effectiveness_values:
            stats_results = self.analyzer.calculate_descriptive_stats(transfer_effectiveness_values)
            
            # Log statistical summary
            logger.info(f"\n📊 STATISTICAL SUMMARY FOR {config['name'].upper()}:")
            logger.info(f"   Successful experiments: {successful_runs}/{num_experiments} ({successful_runs/num_experiments:.1%})")
            logger.info(f"   Transfer effectiveness: {stats_results['mean']:.4f} ± {stats_results['std']:.4f}")
            logger.info(f"   95% Confidence interval: [{stats_results['ci_95_lower']:.4f}, {stats_results['ci_95_upper']:.4f}]")
            logger.info(f"   Range: [{stats_results['min']:.4f}, {stats_results['max']:.4f}]")
            logger.info(f"   Median: {stats_results['median']:.4f}")
            logger.info(f"   IQR: {stats_results['iqr']:.4f}")
            logger.info(f"   Standard error: {stats_results['sem']:.4f}")
        
        # Compile results
        if results:
            stats_summary = {
                'config': config,
                'num_successful': successful_runs,
                'num_attempted': num_experiments,
                'success_rate': successful_runs / num_experiments,
                'transfer_effectiveness_stats': stats_results if transfer_effectiveness_values else None,
                'raw_results': [r.to_dict() for r in results],
                'statistical_methods': {
                    'descriptive_stats': stats_results.get('method_explanation', {}),
                    'sample_size_rationale': 'n=10 per condition provides adequate power for descriptive analysis',
                    'confidence_level': '95% confidence intervals calculated using t-distribution'
                }
            }
        else:
            stats_summary = {
                'config': config,
                'num_successful': 0,
                'num_attempted': num_experiments,
                'success_rate': 0.0,
                'error': 'No successful experiments',
                'statistical_methods': {'note': 'No statistical analysis possible due to failed experiments'}
            }
        
        # Save individual config results
        self.save_config_results(config['name'], stats_summary)
        
        logger.info(f"✅ Configuration {config['name']} completed: {successful_runs}/{num_experiments} successful")
        
        return stats_summary
    
    def save_config_results(self, config_name: str, results: Dict):
        """Save individual configuration results with proper JSON serialization."""
        # Convert sets to lists for JSON serialization
        results_for_json = self.convert_sets_to_lists(results)
        
        config_file = self.results_dir / f"{config_name}_enhanced_results.json"
        with open(config_file, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        logger.info(f"💾 Saved {config_name} results to: {config_file}")
    
    def convert_sets_to_lists(self, obj):
        """Recursively convert sets to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self.convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_sets_to_lists(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    def run_comprehensive_analysis(self, num_experiments_per_config: int = 10):
        """Run comprehensive analysis with statistical methods documentation."""
        logger.info(f"\n🚀 STARTING COMPREHENSIVE SHARED KNOWLEDGE ANALYSIS")
        logger.info(f"📋 Experimental Design:")
        logger.info(f"   Number of configurations: {len(self.test_configs)}")
        logger.info(f"   Experiments per configuration: {num_experiments_per_config}")
        logger.info(f"   Total planned experiments: {len(self.test_configs) * num_experiments_per_config}")
        logger.info(f"   Architecture: WideNN → WideNN")
        logger.info(f"   Dataset: MNIST digit classification")
        
        all_results = {}
        overlap_ratios = []
        transfer_means = []
        
        # Run experiments for each configuration
        for config in self.test_configs:
            config_results = self.run_config_experiments(config, num_experiments_per_config)
            all_results[config['name']] = config_results
            
            # Collect data for correlation analysis
            if 'transfer_effectiveness_stats' in config_results and config_results['transfer_effectiveness_stats']:
                overlap_ratios.append(config['overlap_ratio'])
                transfer_means.append(config_results['transfer_effectiveness_stats']['mean'])
        
        # Perform correlation analysis
        correlation_results = None
        if len(overlap_ratios) >= 2:
            correlation_results = self.analyzer.calculate_correlation_analysis(overlap_ratios, transfer_means)
            
            logger.info(f"\n📈 CORRELATION ANALYSIS RESULTS:")
            logger.info(f"   Pearson correlation coefficient: {correlation_results['correlation_coefficient']:.4f}")
            logger.info(f"   R-squared: {correlation_results['r_squared']:.4f}")
            logger.info(f"   Relationship strength: {correlation_results['strength']}")
            logger.info(f"   Relationship direction: {correlation_results['direction']}")
            if correlation_results['p_value']:
                logger.info(f"   Statistical significance: p = {correlation_results['p_value']:.4f}")
        
        # Generate comprehensive results
        comprehensive_results = {
            'experiment_title': 'Enhanced Shared Knowledge Analysis for Wide→Wide Transfer',
            'timestamp': datetime.now().isoformat(),
            'experimental_design': {
                'type': 'Factorial design with repeated measures',
                'independent_variable': 'Overlap ratio (proportion of shared classes)',
                'dependent_variable': 'Transfer effectiveness (proportion correct)',
                'control_variables': ['Architecture (WideNN)', 'Dataset (MNIST)', 'Training parameters'],
                'sample_size_per_condition': num_experiments_per_config,
                'total_experiments': len(self.test_configs) * num_experiments_per_config
            },
            'statistical_methods': {
                'descriptive_statistics': {
                    'measures_of_central_tendency': ['mean', 'median'],
                    'measures_of_variability': ['standard_deviation', 'variance', 'IQR', 'range'],
                    'confidence_intervals': '95% CI using t-distribution',
                    'rationale': 'Comprehensive description of transfer effectiveness distribution'
                },
                'correlation_analysis': {
                    'method': 'Pearson product-moment correlation',
                    'formula': 'r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)',
                    'interpretation_criteria': {
                        'strong': '|r| > 0.7',
                        'moderate': '0.3 < |r| < 0.7',
                        'weak': '|r| < 0.3'
                    },
                    'significance_test': 'Two-tailed t-test for correlation coefficient'
                }
            },
            'results_by_configuration': all_results,
            'correlation_analysis': correlation_results,
            'key_findings': self.generate_key_findings(all_results, correlation_results)
        }
        
        # Save comprehensive results
        output_file = self.results_dir / "enhanced_comprehensive_results.json"
        comprehensive_json = self.convert_sets_to_lists(comprehensive_results)
        with open(output_file, 'w') as f:
            json.dump(comprehensive_json, f, indent=2)
        
        logger.info(f"\n💾 Comprehensive results saved to: {output_file}")
        
        # Generate final summary
        self.generate_final_summary(comprehensive_results)
        
        return comprehensive_results
    
    def generate_key_findings(self, all_results: Dict, correlation_results: Dict) -> List[str]:
        """Generate key findings based on statistical analysis."""
        findings = []
        
        # Extract transfer effectiveness for each configuration
        effectiveness_by_config = {}
        for config_name, results in all_results.items():
            if 'transfer_effectiveness_stats' in results and results['transfer_effectiveness_stats']:
                stats = results['transfer_effectiveness_stats']
                effectiveness_by_config[config_name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'ci_lower': stats['ci_95_lower'],
                    'ci_upper': stats['ci_95_upper']
                }
        
        # Finding 1: Overall performance
        if effectiveness_by_config:
            all_means = [data['mean'] for data in effectiveness_by_config.values()]
            overall_min = min(all_means)
            overall_max = max(all_means)
            findings.append(f"Transfer effectiveness ranges from {overall_min:.3f} to {overall_max:.3f} across all overlap conditions")
        
        # Finding 2: Correlation analysis
        if correlation_results:
            r = correlation_results['correlation_coefficient']
            strength = correlation_results['strength']
            direction = correlation_results['direction']
            findings.append(f"Correlation analysis reveals a {strength} {direction} relationship (r = {r:.3f}) between shared knowledge and transfer effectiveness")
        
        # Finding 3: Configuration-specific findings
        for config_name, data in effectiveness_by_config.items():
            findings.append(f"{config_name.replace('_', ' ').title()}: {data['mean']:.3f} ± {data['std']:.3f} (95% CI: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}])")
        
        return findings
    
    def generate_final_summary(self, comprehensive_results: Dict):
        """Generate and log final experimental summary."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 FINAL EXPERIMENTAL SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"📋 Experiment: {comprehensive_results['experiment_title']}")
        logger.info(f"⏰ Completed: {comprehensive_results['timestamp']}")
        
        logger.info(f"\n📊 EXPERIMENTAL DESIGN:")
        design = comprehensive_results['experimental_design']
        logger.info(f"   Type: {design['type']}")
        logger.info(f"   Independent variable: {design['independent_variable']}")
        logger.info(f"   Dependent variable: {design['dependent_variable']}")
        logger.info(f"   Sample size per condition: {design['sample_size_per_condition']}")
        logger.info(f"   Total experiments: {design['total_experiments']}")
        
        logger.info(f"\n🔍 KEY FINDINGS:")
        for i, finding in enumerate(comprehensive_results['key_findings'], 1):
            logger.info(f"   {i}. {finding}")
        
        if comprehensive_results['correlation_analysis']:
            corr = comprehensive_results['correlation_analysis']
            logger.info(f"\n📈 CORRELATION ANALYSIS:")
            logger.info(f"   Coefficient: {corr['correlation_coefficient']:.4f}")
            logger.info(f"   Interpretation: {corr['method_explanation']['interpretation']}")
            logger.info(f"   R²: {corr['r_squared']:.4f} (variance explained)")
        
        logger.info(f"\n✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")


def main():
    """Main execution function with enhanced statistical analysis."""
    print("🔬 Starting Enhanced Shared Knowledge Analysis")
    print("=" * 60)
    
    analyzer = EnhancedSharedKnowledgeAnalysis(base_seed=42)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(num_experiments_per_config=10)
    
    print(f"\n✅ Enhanced analysis completed successfully!")
    print(f"📁 Results saved in: {analyzer.results_dir}")
    print(f"📁 Logs saved in: {analyzer.logs_dir}")


if __name__ == "__main__":
    main()