#!/usr/bin/env python3
"""
Master Test Runner for SAE Concept Injection Methods

This script orchestrates the execution of all 9 SAE concept injection methods
across both same-architecture and cross-architecture scenarios.

Usage:
    python run_all_tests.py --methods 1,2 --architectures same,cross
    python run_all_tests.py --quick-test  # Run minimal test suite
    python run_all_tests.py --full-suite  # Run all 156 experiments
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np

# Import method testers (as they're implemented)
from method1_precomputed_vector_alignment import Method1Tester
from method2_cross_architecture_alignment import Method2Tester


class MasterTestRunner:
    """Orchestrates execution of all SAE concept injection method tests"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track all results
        self.all_results = {}
        self.execution_log = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "experiment_config": {
                "random_seed": 42,
                "device": "auto",
                "data_directory": "./data",
                "results_directory": "./results",
                "checkpoints_directory": "./checkpoints",
                "log_level": "INFO"
            },
            "training_config": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "base_epochs": 6,
                "finetune_epochs": 2,
                "optimizer": "Adam",
                "weight_decay": 1e-4
            },
            "method_configs": {
                "method1": {
                    "concept_dimensions": [32, 48, 64],
                    "sparsity_weights": [0.030, 0.050, 0.080],
                    "injection_strengths": [0.4]
                },
                "method2": {
                    "alignment_types": ["nonlinear", "procrustes"],
                    "hidden_dimensions": [128],
                    "alignment_epochs": 50
                }
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"master_test_run_{timestamp}.log"
        
        log_level = getattr(logging, self.config["experiment_config"]["log_level"])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Master test runner initialized. Log file: {log_file}")
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config["experiment_config"]["data_directory"],
            self.config["experiment_config"]["results_directory"],
            self.config["experiment_config"]["checkpoints_directory"],
            "./logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config["experiment_config"]["random_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Random seeds set to {seed} for reproducibility")
    
    def run_method_1(self, test_types: List[str]) -> Dict:
        """Run Method 1: Precomputed Vector Space Alignment"""
        self.logger.info("Starting Method 1: Precomputed Vector Space Alignment")
        
        start_time = time.time()
        tester = Method1Tester(self.config)
        
        method_results = {
            'method_name': 'Method 1: Precomputed Vector Space Alignment',
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
        
        if 'same' in test_types:
            self.logger.info("Running Method 1 same-architecture tests...")
            try:
                # Run key configurations from documented results
                configs = self.config["method_configs"]["method1"]
                
                for concept_dim in configs["concept_dimensions"]:
                    for sparsity in configs["sparsity_weights"]:
                        for injection_strength in configs["injection_strengths"]:
                            result = tester.run_same_architecture_test(
                                concept_dim=concept_dim,
                                sparsity=sparsity,
                                injection_strength=injection_strength
                            )
                            method_results['experiments'].append(result)
                            
            except Exception as e:
                self.logger.error(f"Method 1 same-architecture tests failed: {e}")
                method_results['error'] = str(e)
        
        if 'cross' in test_types:
            self.logger.info("Running Method 1 cross-architecture tests...")
            try:
                # Cross-architecture test matrix
                architectures = ['WideNN', 'DeepNN', 'BottleneckNN', 'PyramidNN']
                
                for target_arch in architectures:
                    result = tester.run_cross_architecture_test(
                        source_arch='BaseNN',
                        target_arch=target_arch,
                        concept_dim=48,  # Optimal from documentation
                        sparsity=0.030,  # Optimal from documentation
                        injection_strength=0.4
                    )
                    method_results['experiments'].append(result)
                    
            except Exception as e:
                self.logger.error(f"Method 1 cross-architecture tests failed: {e}")
                method_results['error'] = str(e)
        
        end_time = time.time()
        method_results['end_time'] = datetime.now().isoformat()
        method_results['duration_minutes'] = (end_time - start_time) / 60
        method_results['total_experiments'] = len(method_results['experiments'])
        
        # Calculate summary statistics
        if method_results['experiments']:
            transfers = [r['transfer_accuracy'] for r in method_results['experiments']]
            preservations = [r['preservation_accuracy'] for r in method_results['experiments']]
            specificities = [r['specificity_accuracy'] for r in method_results['experiments']]
            
            method_results['summary'] = {
                'best_transfer': max(transfers),
                'avg_transfer': np.mean(transfers),
                'best_preservation': max(preservations),
                'avg_preservation': np.mean(preservations),
                'best_specificity': min(specificities),
                'avg_specificity': np.mean(specificities)
            }
        
        self.logger.info(f"Method 1 completed in {method_results['duration_minutes']:.1f} minutes")
        if 'summary' in method_results:
            summary = method_results['summary']
            self.logger.info(f"Method 1 - Best Transfer: {summary['best_transfer']:.1f}%, "
                           f"Best Preservation: {summary['best_preservation']:.1f}%, "
                           f"Best Specificity: {summary['best_specificity']:.1f}%")
        
        return method_results
    
    def run_method_2(self, test_types: List[str]) -> Dict:
        """Run Method 2: Cross-Architecture Neural Alignment"""
        self.logger.info("Starting Method 2: Cross-Architecture Neural Alignment")
        
        start_time = time.time()
        tester = Method2Tester(self.config)
        
        method_results = {
            'method_name': 'Method 2: Cross-Architecture Neural Alignment',
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
        
        if 'same' in test_types:
            self.logger.info("Running Method 2 same-architecture baseline...")
            try:
                alignment_types = self.config["method_configs"]["method2"]["alignment_types"]
                
                for alignment_type in alignment_types:
                    result = tester.run_same_architecture_baseline(alignment_type)
                    method_results['experiments'].append(result)
                    
            except Exception as e:
                self.logger.error(f"Method 2 same-architecture tests failed: {e}")
                method_results['error'] = str(e)
        
        if 'cross' in test_types:
            self.logger.info("Running Method 2 cross-architecture tests...")
            try:
                # Test key architecture pairs from documentation
                arch_pairs = [
                    ('WideNN', 'DeepNN'),
                    ('BottleneckNN', 'DeepNN'),
                    ('PyramidNN', 'WideNN'),
                    ('BaseNN', 'WideNN'),
                    ('BaseNN', 'DeepNN')
                ]
                
                alignment_types = self.config["method_configs"]["method2"]["alignment_types"]
                
                for source_arch, target_arch in arch_pairs:
                    for alignment_type in alignment_types:
                        result = tester.run_cross_architecture_neural_alignment(
                            source_arch, target_arch, alignment_type
                        )
                        method_results['experiments'].append(result)
                        
            except Exception as e:
                self.logger.error(f"Method 2 cross-architecture tests failed: {e}")
                method_results['error'] = str(e)
        
        end_time = time.time()
        method_results['end_time'] = datetime.now().isoformat()
        method_results['duration_minutes'] = (end_time - start_time) / 60
        method_results['total_experiments'] = len(method_results['experiments'])
        
        # Calculate summary statistics
        if method_results['experiments']:
            transfers = [r['transfer_accuracy'] for r in method_results['experiments']]
            preservations = [r['preservation_accuracy'] for r in method_results['experiments']]
            specificities = [r['specificity_accuracy'] for r in method_results['experiments']]
            
            method_results['summary'] = {
                'best_transfer': max(transfers),
                'avg_transfer': np.mean(transfers),
                'best_preservation': max(preservations),
                'avg_preservation': np.mean(preservations),
                'best_specificity': min(specificities),
                'avg_specificity': np.mean(specificities)
            }
        
        self.logger.info(f"Method 2 completed in {method_results['duration_minutes']:.1f} minutes")
        if 'summary' in method_results:
            summary = method_results['summary']
            self.logger.info(f"Method 2 - Best Transfer: {summary['best_transfer']:.1f}%, "
                           f"Best Preservation: {summary['best_preservation']:.1f}%, "
                           f"Best Specificity: {summary['best_specificity']:.1f}%")
        
        return method_results
    
    def run_method_placeholder(self, method_num: int, test_types: List[str]) -> Dict:
        """Placeholder for methods not yet implemented"""
        self.logger.warning(f"Method {method_num} not yet implemented - returning placeholder")
        
        return {
            'method_name': f'Method {method_num}: Not Implemented',
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': 0,
            'total_experiments': 0,
            'experiments': [],
            'status': 'NOT_IMPLEMENTED'
        }
    
    def run_selected_methods(self, methods: List[int], test_types: List[str]) -> Dict:
        """Run selected methods with specified test types"""
        self.set_random_seeds()
        
        overall_start_time = time.time()
        self.logger.info(f"Starting test run for methods {methods} with test types {test_types}")
        
        # Method dispatch table
        method_runners = {
            1: self.run_method_1,
            2: self.run_method_2,
            3: self.run_method_placeholder,
            4: self.run_method_placeholder,
            5: self.run_method_placeholder,
            6: self.run_method_placeholder,
            7: self.run_method_placeholder,
            8: self.run_method_placeholder,
            9: self.run_method_placeholder,
        }
        
        results = {
            'test_run_info': {
                'start_time': datetime.now().isoformat(),
                'methods_requested': methods,
                'test_types_requested': test_types,
                'device': str(self.device),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            },
            'method_results': {}
        }
        
        # Run each method
        for method_num in methods:
            if method_num in method_runners:
                try:
                    self.logger.info(f"="*60)
                    self.logger.info(f"STARTING METHOD {method_num}")
                    self.logger.info(f"="*60)
                    
                    method_result = method_runners[method_num](test_types)
                    results['method_results'][f'method_{method_num}'] = method_result
                    
                    # Log progress
                    if 'summary' in method_result:
                        summary = method_result['summary']
                        self.logger.info(f"Method {method_num} COMPLETED - "
                                       f"Best Transfer: {summary['best_transfer']:.1f}%")
                    
                except Exception as e:
                    self.logger.error(f"Method {method_num} failed with error: {e}")
                    results['method_results'][f'method_{method_num}'] = {
                        'method_name': f'Method {method_num}: Failed',
                        'error': str(e),
                        'status': 'FAILED'
                    }
            else:
                self.logger.warning(f"Method {method_num} not recognized")
        
        overall_end_time = time.time()
        results['test_run_info']['end_time'] = datetime.now().isoformat()
        results['test_run_info']['total_duration_minutes'] = (overall_end_time - overall_start_time) / 60
        
        # Calculate overall statistics
        all_experiments = []
        for method_result in results['method_results'].values():
            if 'experiments' in method_result:
                all_experiments.extend(method_result['experiments'])
        
        if all_experiments:
            transfers = [r['transfer_accuracy'] for r in all_experiments]
            preservations = [r['preservation_accuracy'] for r in all_experiments]
            specificities = [r['specificity_accuracy'] for r in all_experiments]
            
            results['overall_summary'] = {
                'total_experiments': len(all_experiments),
                'best_overall_transfer': max(transfers),
                'avg_overall_transfer': np.mean(transfers),
                'best_overall_preservation': max(preservations),
                'avg_overall_preservation': np.mean(preservations),
                'best_overall_specificity': min(specificities),
                'avg_overall_specificity': np.mean(specificities)
            }
        
        # Save results
        self.save_master_results(results)
        
        return results
    
    def save_master_results(self, results: Dict):
        """Save master results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"master_test_results_{timestamp}.json"
        filepath = os.path.join(self.config["experiment_config"]["results_directory"], filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Master results saved to {filepath}")
        
        # Print summary
        if 'overall_summary' in results:
            summary = results['overall_summary']
            self.logger.info(f"\n" + "="*60)
            self.logger.info(f"MASTER TEST RUN SUMMARY")
            self.logger.info(f"="*60)
            self.logger.info(f"Total Experiments: {summary['total_experiments']}")
            self.logger.info(f"Best Overall Transfer: {summary['best_overall_transfer']:.1f}%")
            self.logger.info(f"Best Overall Preservation: {summary['best_overall_preservation']:.1f}%")
            self.logger.info(f"Best Overall Specificity: {summary['best_overall_specificity']:.1f}%")
            self.logger.info(f"Total Duration: {results['test_run_info']['total_duration_minutes']:.1f} minutes")
    
    def run_quick_test(self):
        """Run a quick test with minimal configurations"""
        self.logger.info("Running quick test suite...")
        return self.run_selected_methods(methods=[1, 2], test_types=['same'])
    
    def run_full_suite(self):
        """Run the complete test suite (all methods, all architectures)"""
        self.logger.info("Running full test suite (this will take several hours)...")
        return self.run_selected_methods(methods=[1, 2], test_types=['same', 'cross'])


def main():
    parser = argparse.ArgumentParser(description='Master Test Runner for SAE Concept Injection Methods')
    parser.add_argument('--methods', type=str, default='1,2',
                       help='Comma-separated list of method numbers to run (1-9)')
    parser.add_argument('--architectures', type=str, default='same,cross',
                       help='Test types: same, cross, or both (comma-separated)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test suite (Methods 1-2, same architecture only)')
    parser.add_argument('--full-suite', action='store_true',
                       help='Run complete test suite (all implemented methods, all architectures)')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = MasterTestRunner(args.config)
    
    if args.quick_test:
        results = runner.run_quick_test()
    elif args.full_suite:
        results = runner.run_full_suite()
    else:
        # Parse method numbers and test types
        methods = [int(m.strip()) for m in args.methods.split(',')]
        test_types = [t.strip() for t in args.architectures.split(',')]
        
        # Validate inputs
        valid_methods = list(range(1, 10))
        valid_test_types = ['same', 'cross']
        
        invalid_methods = [m for m in methods if m not in valid_methods]
        invalid_test_types = [t for t in test_types if t not in valid_test_types]
        
        if invalid_methods:
            print(f"Error: Invalid method numbers: {invalid_methods}")
            print(f"Valid methods: {valid_methods}")
            sys.exit(1)
        
        if invalid_test_types:
            print(f"Error: Invalid test types: {invalid_test_types}")
            print(f"Valid test types: {valid_test_types}")
            sys.exit(1)
        
        # Run selected tests
        results = runner.run_selected_methods(methods, test_types)
    
    print(f"\nTest run completed! Check the logs and results directory for detailed output.")
    
    if 'overall_summary' in results:
        summary = results['overall_summary']
        print(f"\nQuick Summary:")
        print(f"  Total Experiments: {summary['total_experiments']}")
        print(f"  Best Transfer: {summary['best_overall_transfer']:.1f}%")
        print(f"  Best Preservation: {summary['best_overall_preservation']:.1f}%")
        print(f"  Duration: {results['test_run_info']['total_duration_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()