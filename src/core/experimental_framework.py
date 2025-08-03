"""
Experimental Framework for Neural Concept Transfer
This module implements the complete experimental framework as specified in General_requirements.txt

Key Requirements:
- Test 20 pairs of networks for each experiment condition
- Measure knowledge transfer, specificity transfer, and precision transfer  
- Store both before and after transfer results
- Calculate statistics: max, min, median, average, std
- Use same seed and settings for reproducibility
- Train on MNIST subsets with max 5 epochs, >90% accuracy requirement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

from architectures import WideNN, DeepNN, create_model, get_model_info
from neural_concept_transfer import NeuralConceptTransferSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    seed: int = RANDOM_SEED
    max_epochs: int = 5
    min_accuracy_threshold: float = 0.90
    num_pairs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    concept_dim: int = 24
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MNIST dataset configuration
    input_dim: int = 784
    num_classes: int = 10
    
    # Data splits (as per requirements)
    source_classes: Set[int] = None  # Will be set per experiment
    target_classes: Set[int] = None  # Will be set per experiment
    
    def __post_init__(self):
        if self.source_classes is None:
            self.source_classes = {2, 3, 4, 5}  # Default example
        if self.target_classes is None:
            self.target_classes = {0, 1, 2, 3}  # Default example


@dataclass 
class TransferMetrics:
    """Metrics for transfer learning evaluation."""
    knowledge_transfer: float  # Recognition of transferred knowledge
    specificity_transfer: float  # Recognition of non-transferred knowledge from donor
    precision_transfer: float  # Recognition of original training data
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'knowledge_transfer': self.knowledge_transfer,
            'specificity_transfer': self.specificity_transfer, 
            'precision_transfer': self.precision_transfer
        }


@dataclass
class ExperimentResult:
    """Results from a single transfer experiment."""
    pair_id: int
    source_arch: str
    target_arch: str
    source_classes: Set[int]
    target_classes: Set[int]
    transfer_class: int
    
    # Before transfer metrics
    before_metrics: TransferMetrics
    
    # After transfer metrics  
    after_metrics: TransferMetrics
    
    # Additional info
    source_accuracy: float
    target_accuracy: float
    alignment_error: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'pair_id': self.pair_id,
            'source_arch': self.source_arch,
            'target_arch': self.target_arch,
            'source_classes': list(self.source_classes),
            'target_classes': list(self.target_classes),
            'transfer_class': self.transfer_class,
            'before_metrics': self.before_metrics.to_dict(),
            'after_metrics': self.after_metrics.to_dict(),
            'source_accuracy': float(self.source_accuracy),  # Convert to native Python float
            'target_accuracy': float(self.target_accuracy),  # Convert to native Python float
            'alignment_error': float(self.alignment_error),  # Convert to native Python float
            'timestamp': self.timestamp
        }


class MNISTDataManager:
    """Manages MNIST data loading and class filtering."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full datasets
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform)
    
    def create_class_subset(self, dataset, class_set: Set[int], max_samples_per_class: int = 1000) -> Subset:
        """Create subset containing only specified classes."""
        indices = []
        class_counts = {c: 0 for c in class_set}
        
        for idx, (_, label) in enumerate(dataset):
            if label in class_set and class_counts[label] < max_samples_per_class:
                indices.append(idx)
                class_counts[label] += 1
                
        logger.info(f"Created subset with {len(indices)} samples for classes {class_set}")
        logger.info(f"Class distribution: {class_counts}")
        
        return Subset(dataset, indices)
    
    def get_data_loaders(self, source_classes: Set[int], target_classes: Set[int]) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Get data loaders for source and target classes."""
        # Training sets
        source_train_subset = self.create_class_subset(self.train_dataset, source_classes)
        target_train_subset = self.create_class_subset(self.train_dataset, target_classes)
        
        # Test sets  
        source_test_subset = self.create_class_subset(self.test_dataset, source_classes, max_samples_per_class=200)
        target_test_subset = self.create_class_subset(self.test_dataset, target_classes, max_samples_per_class=200)
        
        # Create data loaders
        source_train_loader = DataLoader(source_train_subset, batch_size=self.config.batch_size, shuffle=True)
        target_train_loader = DataLoader(target_train_subset, batch_size=self.config.batch_size, shuffle=True)
        source_test_loader = DataLoader(source_test_subset, batch_size=self.config.batch_size, shuffle=False)
        target_test_loader = DataLoader(target_test_subset, batch_size=self.config.batch_size, shuffle=False)
        
        return source_train_loader, target_train_loader, source_test_loader, target_test_loader


class ModelTrainer:
    """Handles model training with accuracy requirements."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   test_loader: DataLoader) -> Tuple[nn.Module, float]:
        """
        Train model with accuracy requirements.
        
        Returns:
            model: Trained model (or None if failed to meet accuracy)
            accuracy: Final test accuracy
        """
        device = self.config.device
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # Flatten for MLP
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Evaluation phase
            accuracy = self.evaluate_model(model, test_loader)
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}: "
                       f"Loss = {train_loss/len(train_loader):.4f}, "
                       f"Accuracy = {accuracy:.4f}")
            
            # Check if we meet accuracy requirement
            if accuracy >= self.config.min_accuracy_threshold:
                logger.info(f"✓ Model achieved required accuracy {accuracy:.4f} >= {self.config.min_accuracy_threshold}")
                return model, accuracy
        
        # Final accuracy check
        final_accuracy = self.evaluate_model(model, test_loader)
        if final_accuracy >= self.config.min_accuracy_threshold:
            logger.info(f"✓ Model achieved required accuracy {final_accuracy:.4f} >= {self.config.min_accuracy_threshold}")
            return model, final_accuracy
        else:
            logger.warning(f"✗ Model failed to achieve required accuracy {final_accuracy:.4f} < {self.config.min_accuracy_threshold}")
            return None, final_accuracy
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        device = self.config.device
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total


class MetricsEvaluator:
    """Evaluates transfer learning metrics."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def evaluate_transfer_metrics(self, model: nn.Module, transfer_system: NeuralConceptTransferSystem,
                                 source_test_loader: DataLoader, target_test_loader: DataLoader,
                                 transfer_class: int, source_classes: Set[int] = None) -> TransferMetrics:
        """
        Evaluate all transfer metrics.
        
        Args:
            model: Target model (possibly modified)
            transfer_system: Transfer system
            source_test_loader: Test data for source classes
            target_test_loader: Test data for target classes  
            transfer_class: Class being transferred
            
        Returns:
            TransferMetrics with all measured values
        """
        device = self.config.device
        model.eval()
        
        # 1. Knowledge Transfer: Ability to recognize transferred knowledge
        knowledge_transfer = self._evaluate_knowledge_transfer(
            model, transfer_system, source_test_loader, transfer_class)
        
        # 2. Specificity Transfer: Recognition of other source knowledge not explicitly transferred
        specificity_transfer = self._evaluate_specificity_transfer(
            model, transfer_system, source_test_loader, transfer_class, source_classes)
        
        # 3. Precision Transfer: Recognition of original target training data
        precision_transfer = self._evaluate_precision_transfer(
            model, target_test_loader)
        
        return TransferMetrics(
            knowledge_transfer=knowledge_transfer,
            specificity_transfer=specificity_transfer,
            precision_transfer=precision_transfer
        )
    
    def _evaluate_knowledge_transfer(self, model: nn.Module, transfer_system: NeuralConceptTransferSystem,
                                   source_test_loader: DataLoader, transfer_class: int) -> float:
        """Evaluate ability to recognize transferred class."""
        device = self.config.device
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in source_test_loader:
                # Only evaluate on transfer class samples
                transfer_mask = (target == transfer_class)
                if transfer_mask.sum() == 0:
                    continue
                    
                transfer_data = data[transfer_mask].to(device)
                transfer_targets = target[transfer_mask].to(device)
                
                if transfer_system is not None:
                    # Apply concept transfer
                    enhanced_outputs = transfer_system.transfer_concept(transfer_data, transfer_class)
                    
                    if enhanced_outputs is not None:
                        _, predicted = torch.max(enhanced_outputs, 1)
                        correct += (predicted == transfer_class).sum().item()
                        total += transfer_targets.size(0)
                else:
                    # Before transfer - use original model (should fail for transfer class)
                    data_flat = transfer_data.view(transfer_data.size(0), -1)
                    outputs = model(data_flat)
                    _, predicted = torch.max(outputs, 1)
                    # For before transfer, this should be 0 since transfer class wasn't trained
                    correct += (predicted == transfer_class).sum().item()
                    total += transfer_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_specificity_transfer(self, model: nn.Module, transfer_system: NeuralConceptTransferSystem,
                                     source_test_loader: DataLoader, transfer_class: int, source_classes: Set[int] = None) -> float:
        """Evaluate recognition of non-transferred source knowledge."""
        device = self.config.device
        correct = 0
        total = 0
        
        # Get non-transfer classes from source
        if transfer_system is not None:
            used_source_classes = transfer_system.source_classes
        elif source_classes is not None:
            used_source_classes = source_classes
        else:
            # For before transfer, assume we know the source classes from the experiment
            # This is a bit of a hack but necessary for evaluation
            used_source_classes = {2, 3, 4, 5, 6, 7, 8, 9}  # Default to experiment classes
        
        non_transfer_classes = used_source_classes - {transfer_class}
        
        with torch.no_grad():
            for data, target in source_test_loader:
                # Only evaluate on non-transfer class samples
                non_transfer_mask = torch.tensor([t.item() in non_transfer_classes for t in target])
                if non_transfer_mask.sum() == 0:
                    continue
                    
                non_transfer_data = data[non_transfer_mask].to(device)
                non_transfer_targets = target[non_transfer_mask].to(device)
                
                # Use original model (without transfer for these classes)
                data_flat = non_transfer_data.view(non_transfer_data.size(0), -1)
                outputs = model(data_flat)
                _, predicted = torch.max(outputs, 1)
                
                # Check if predictions match any of the non-transfer source classes
                for pred, target_val in zip(predicted, non_transfer_targets):
                    if pred.item() in non_transfer_classes:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_precision_transfer(self, model: nn.Module, target_test_loader: DataLoader) -> float:
        """Evaluate recognition of original target training data."""
        device = self.config.device
        correct = 0
        total = 0
        
        model.eval()
        with torch.no_grad():
            for data, target in target_test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0


class ExperimentRunner:
    """Main experiment runner coordinating all components."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_manager = MNISTDataManager(config)
        self.trainer = ModelTrainer(config)
        self.evaluator = MetricsEvaluator(config)
        
        # Create results directory
        self.results_dir = Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_single_experiment(self, pair_id: int, source_arch: str, target_arch: str,
                            source_classes: Set[int], target_classes: Set[int],
                            transfer_class: int) -> Optional[ExperimentResult]:
        """
        Run a single transfer experiment between two models.
        
        Returns:
            ExperimentResult or None if training failed
        """
        logger.info(f"\n=== Running Experiment Pair {pair_id} ===")
        logger.info(f"Source: {source_arch} ({source_classes}) -> Target: {target_arch} ({target_classes})")
        logger.info(f"Transferring class: {transfer_class}")
        
        # Get data loaders
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
            self.data_manager.get_data_loaders(source_classes, target_classes)
        
        # Create and train source model
        source_model = create_model(source_arch)
        logger.info(f"Training source model ({source_arch})...")
        trained_source, source_accuracy = self.trainer.train_model(
            source_model, source_train_loader, source_test_loader)
        
        if trained_source is None:
            logger.warning(f"Source model training failed for pair {pair_id}")
            return None
        
        # Create and train target model
        target_model = create_model(target_arch)
        logger.info(f"Training target model ({target_arch})...")
        trained_target, target_accuracy = self.trainer.train_model(
            target_model, target_train_loader, target_test_loader)
        
        if trained_target is None:
            logger.warning(f"Target model training failed for pair {pair_id}")
            return None
        
        # Evaluate before transfer
        logger.info("Evaluating before transfer...")
        before_metrics = self.evaluator.evaluate_transfer_metrics(
            trained_target, None, source_test_loader, target_test_loader, transfer_class, source_classes)
        
        # Setup and fit transfer system
        logger.info("Setting up concept transfer system...")
        try:
            transfer_system = NeuralConceptTransferSystem(
                source_model=trained_source,
                target_model=trained_target,
                source_classes=source_classes,
                target_classes=target_classes,
                concept_dim=self.config.concept_dim,
                device=self.config.device
            )
            
            # Fit the transfer system
            fit_metrics = transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=50)
            alignment_error = fit_metrics['alignment_error']
            
            # Setup and train for specific transfer class
            transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
            
        except Exception as e:
            logger.error(f"Transfer system setup failed: {e}")
            return None
        
        # Evaluate after transfer
        logger.info("Evaluating after transfer...")
        after_metrics = self.evaluator.evaluate_transfer_metrics(
            trained_target, transfer_system, source_test_loader, target_test_loader, transfer_class, source_classes)
        
        # Create result
        result = ExperimentResult(
            pair_id=pair_id,
            source_arch=source_arch,
            target_arch=target_arch,
            source_classes=source_classes,
            target_classes=target_classes,
            transfer_class=transfer_class,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            source_accuracy=source_accuracy,
            target_accuracy=target_accuracy,
            alignment_error=alignment_error,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✓ Experiment pair {pair_id} completed successfully")
        logger.info(f"Before -> After: Knowledge {before_metrics.knowledge_transfer:.3f} -> {after_metrics.knowledge_transfer:.3f}")
        
        return result
    
    def run_experiment_suite(self, experiment_name: str, source_arch: str, target_arch: str,
                           source_classes: Set[int], target_classes: Set[int]) -> List[ExperimentResult]:
        """
        Run complete experiment suite with 20 pairs.
        
        Args:
            experiment_name: Name for this experiment suite
            source_arch: Source architecture name
            target_arch: Target architecture name  
            source_classes: Set of source classes
            target_classes: Set of target classes
            
        Returns:
            List of successful experiment results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING EXPERIMENT SUITE: {experiment_name}")
        logger.info(f"{'='*60}")
        
        results = []
        transfer_classes = source_classes - target_classes
        
        for transfer_class in transfer_classes:
            logger.info(f"\n--- Transfer Class: {transfer_class} ---")
            
            for pair_id in range(1, self.config.num_pairs + 1):
                # Set unique seed for this pair to ensure different initializations
                torch.manual_seed(self.config.seed + pair_id)
                np.random.seed(self.config.seed + pair_id)
                
                result = self.run_single_experiment(
                    pair_id=pair_id,
                    source_arch=source_arch,
                    target_arch=target_arch,
                    source_classes=source_classes,
                    target_classes=target_classes,
                    transfer_class=transfer_class
                )
                
                if result is not None:
                    results.append(result)
                    
                    # Save individual result
                    result_file = self.results_dir / f"{experiment_name}_pair_{pair_id}_class_{transfer_class}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
        
        # Save combined results
        combined_file = self.results_dir / f"{experiment_name}_all_results.json"
        with open(combined_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Generate summary statistics
        self._generate_summary_statistics(results, experiment_name)
        
        logger.info(f"\n✓ Experiment suite '{experiment_name}' completed with {len(results)} successful runs")
        return results
    
    def _generate_summary_statistics(self, results: List[ExperimentResult], experiment_name: str):
        """Generate summary statistics as required."""
        if not results:
            logger.warning("No results to summarize")
            return
        
        # Collect metrics
        before_knowledge = [r.before_metrics.knowledge_transfer for r in results]
        after_knowledge = [r.after_metrics.knowledge_transfer for r in results]
        before_specificity = [r.before_metrics.specificity_transfer for r in results]
        after_specificity = [r.after_metrics.specificity_transfer for r in results]
        before_precision = [r.before_metrics.precision_transfer for r in results]
        after_precision = [r.after_metrics.precision_transfer for r in results]
        
        def compute_stats(values):
            return {
                'max': float(np.max(values)),
                'min': float(np.min(values)),
                'median': float(np.median(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'count': len(values)
            }
        
        # Convert config to JSON serializable format
        config_dict = self.config.__dict__.copy()
        if 'source_classes' in config_dict and config_dict['source_classes']:
            config_dict['source_classes'] = list(config_dict['source_classes'])
        if 'target_classes' in config_dict and config_dict['target_classes']:
            config_dict['target_classes'] = list(config_dict['target_classes'])
        
        summary = {
            'experiment_name': experiment_name,
            'total_pairs': len(results),
            'timestamp': datetime.now().isoformat(),
            'config': config_dict,
            'metrics': {
                'knowledge_transfer': {
                    'before': compute_stats(before_knowledge),
                    'after': compute_stats(after_knowledge)
                },
                'specificity_transfer': {
                    'before': compute_stats(before_specificity),
                    'after': compute_stats(after_specificity)
                },
                'precision_transfer': {
                    'before': compute_stats(before_precision),
                    'after': compute_stats(after_precision)
                }
            }
        }
        
        # Save summary
        summary_file = self.results_dir / f"{experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info(f"\n=== SUMMARY STATISTICS: {experiment_name} ===")
        for metric_name, metric_data in summary['metrics'].items():
            logger.info(f"\n{metric_name.upper()}:")
            for phase in ['before', 'after']:
                stats = metric_data[phase]
                logger.info(f"  {phase.upper()}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                          f"min={stats['min']:.4f}, max={stats['max']:.4f}, median={stats['median']:.4f}")


# Example usage
if __name__ == "__main__":
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    
    # Example: Same architecture experiment
    results = runner.run_experiment_suite(
        experiment_name="WideNN_to_WideNN",
        source_arch="WideNN",
        target_arch="WideNN", 
        source_classes={2, 3, 4, 5},
        target_classes={0, 1, 2, 3}
    )