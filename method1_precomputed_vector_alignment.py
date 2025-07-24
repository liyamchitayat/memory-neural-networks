"""
Method 1: Precomputed Vector Space Alignment Testing Implementation

This module implements comprehensive testing for Method 1 across both same-architecture
and cross-architecture scenarios with full reproducibility documentation.

Method 1 Details:
- Offline computation of injection vectors δ = D_A(∑α_i γF[:,i])
- Runtime application: h' = h + s(x)δ  
- Trade-off: Linear injection, less adaptive, but very fast O(d) vs O(c)

Testing Coverage:
- Same Architecture: BaseNN → BaseNN (replicating existing 13 experiments)
- Cross Architecture: BaseNN → {WideNN, DeepNN, BottleneckNN, PyramidNN}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import logging
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

from neural_architectures import BaseNN, WideNN, DeepNN, BottleneckNN, PyramidNN, get_architecture


class ConceptInjectionDataset(Dataset):
    """Custom dataset for concept injection training"""
    
    def __init__(self, base_dataset, target_digit=4, injection_prob=0.3):
        self.base_dataset = base_dataset
        self.target_digit = target_digit
        self.injection_prob = injection_prob
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Apply concept injection with probability
        if torch.rand(1) < self.injection_prob and label != self.target_digit:
            # Mark for concept injection (will be handled in training loop)
            inject_concept = True
        else:
            inject_concept = False
            
        return image, label, inject_concept


class Method1Tester:
    """Comprehensive tester for Method 1: Precomputed Vector Space Alignment"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
        # Load datasets
        self.setup_datasets()
        
    def setup_logging(self):
        """Initialize logging for experiment tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"method1_experiments_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_datasets(self):
        """Setup MNIST datasets for training and testing"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Full MNIST datasets
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Filter for digits 0-4 (training)
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label <= 3]
        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # Create test sets for evaluation
        test_indices_0_3 = [i for i, (_, label) in enumerate(test_dataset) if label <= 3]
        test_indices_4 = [i for i, (_, label) in enumerate(test_dataset) if label == 4]
        test_indices_5 = [i for i, (_, label) in enumerate(test_dataset) if label == 5]
        
        self.test_dataset_original = torch.utils.data.Subset(test_dataset, test_indices_0_3)
        self.test_dataset_digit4 = torch.utils.data.Subset(test_dataset, test_indices_4)
        self.test_dataset_digit5 = torch.utils.data.Subset(test_dataset, test_indices_5)
        
        self.logger.info(f"Datasets loaded - Train: {len(self.train_dataset)}, "
                        f"Test (0-3): {len(self.test_dataset_original)}, "
                        f"Test (4): {len(self.test_dataset_digit4)}, "
                        f"Test (5): {len(self.test_dataset_digit5)}")
    
    def train_base_model(self, architecture_name: str, epochs: int = 6) -> nn.Module:
        """Train base model on digits 0-3"""
        model = get_architecture(architecture_name).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        
        self.logger.info(f"Training {architecture_name} for {epochs} epochs")
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
            accuracy = 100 * correct / total
            self.logger.info(f"Epoch {epoch+1}/{epochs}: Loss {epoch_loss/len(train_loader):.4f}, "
                           f"Accuracy {accuracy:.2f}%")
        
        return model
    
    def extract_concept_vectors(self, source_model: nn.Module, target_model: nn.Module, 
                              concept_dim: int = 48) -> torch.Tensor:
        """Extract concept vectors for digit-4 using PCA on activations"""
        
        # Get digit-4 samples for concept extraction
        digit4_loader = DataLoader(self.test_dataset_digit4, batch_size=32, shuffle=False)
        
        source_activations = []
        target_activations = []
        
        source_model.eval()
        target_model.eval()
        
        with torch.no_grad():
            for data, _ in digit4_loader:
                data = data.to(self.device)
                
                # Extract penultimate activations
                source_h = source_model.get_penultimate_activations(data)
                target_h = target_model.get_penultimate_activations(data)
                
                source_activations.append(source_h.cpu())
                target_activations.append(target_h.cpu())
        
        source_activations = torch.cat(source_activations, dim=0).numpy()
        target_activations = torch.cat(target_activations, dim=0).numpy()
        
        # Apply PCA to find concept directions
        pca_source = PCA(n_components=concept_dim)
        pca_target = PCA(n_components=concept_dim)
        
        source_concepts = pca_source.fit_transform(source_activations)
        target_concepts = pca_target.fit_transform(target_activations)
        
        # Compute precomputed injection vector δ
        # δ = mean(target_concepts) - mean(source_concepts) projected back to original space
        source_mean = np.mean(source_concepts, axis=0)
        target_mean = np.mean(target_concepts, axis=0)
        
        concept_diff = target_mean - source_mean
        
        # Project back to original target space
        delta = pca_target.inverse_transform(concept_diff.reshape(1, -1)).flatten()
        
        return torch.tensor(delta, dtype=torch.float32).to(self.device)
    
    def create_enhanced_model(self, base_model: nn.Module, injection_vector: torch.Tensor, 
                            injection_strength: float = 0.4) -> nn.Module:
        """Create enhanced model with precomputed concept injection"""
        
        class EnhancedModel(nn.Module):
            def __init__(self, base_model, injection_vector, strength):
                super().__init__()
                self.base_model = base_model
                self.injection_vector = injection_vector
                self.strength = strength
                
                # Learnable gating function g(x) = σ(w^T h + b)
                self.gate_weights = nn.Parameter(torch.randn(base_model.penultimate_dim))
                self.gate_bias = nn.Parameter(torch.zeros(1))
                
            def forward(self, x):
                # Get base activations
                h = self.base_model.get_penultimate_activations(x)
                
                # Compute gating function
                gate_score = torch.sigmoid(torch.sum(self.gate_weights * h, dim=1, keepdim=True) + self.gate_bias)
                
                # Apply concept injection: h' = h + g(x) * δ * strength
                enhanced_h = h + gate_score * self.injection_vector.unsqueeze(0) * self.strength
                
                # Pass through final layer
                if hasattr(self.base_model, 'fc2'):
                    output = self.base_model.fc2(enhanced_h)
                elif hasattr(self.base_model, 'fc3'):  # DeepNN case
                    output = self.base_model.fc3(enhanced_h)
                else:
                    raise ValueError("Unknown final layer structure")
                    
                return output
        
        enhanced = EnhancedModel(base_model, injection_vector, injection_strength)
        return enhanced.to(self.device)
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Standard evaluation function matching the documentation"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100 * correct / total
    
    def evaluate_transfer_accuracy(self, model: nn.Module) -> float:
        """Evaluate transfer accuracy for digit-4 recognition"""
        test_loader = DataLoader(self.test_dataset_digit4, batch_size=32, shuffle=False)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == 4).sum().item()  # Count correct digit-4 predictions
        
        return 100 * correct / total
    
    def evaluate_preservation_accuracy(self, model: nn.Module) -> float:
        """Evaluate preservation accuracy for original digits 0-3"""
        test_loader = DataLoader(self.test_dataset_original, batch_size=32, shuffle=False)
        return self.evaluate_model(model, test_loader)
    
    def evaluate_specificity_accuracy(self, model: nn.Module) -> float:
        """Evaluate specificity accuracy (false positives on digit-5)"""
        test_loader = DataLoader(self.test_dataset_digit5, batch_size=32, shuffle=False)
        
        model.eval()
        incorrect = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                incorrect += (predicted != 5).sum().item()  # Count incorrect predictions
        
        return 100 * incorrect / total  # Lower is better
    
    def run_same_architecture_test(self, concept_dim: int = 48, sparsity: float = 0.030, 
                                 injection_strength: float = 0.4) -> Dict:
        """Run same architecture test (BaseNN → BaseNN)"""
        
        experiment_id = f"method1_same_arch_{concept_dim}D_sparsity_{sparsity}"
        self.logger.info(f"Running same architecture test: {experiment_id}")
        
        # Train base models
        source_model = self.train_base_model('BaseNN', epochs=6)
        target_model = self.train_base_model('BaseNN', epochs=6)  # Same architecture
        
        # Extract concept vectors
        injection_vector = self.extract_concept_vectors(source_model, target_model, concept_dim)
        
        # Create enhanced model
        enhanced_model = self.create_enhanced_model(target_model, injection_vector, injection_strength)
        
        # Fine-tune enhanced model with sparsity regularization
        self.fine_tune_enhanced_model(enhanced_model, sparsity_weight=sparsity, epochs=2)
        
        # Evaluate all metrics
        transfer_acc = self.evaluate_transfer_accuracy(enhanced_model)
        preservation_acc = self.evaluate_preservation_accuracy(enhanced_model)
        specificity_acc = self.evaluate_specificity_accuracy(enhanced_model)
        
        results = {
            'experiment_id': experiment_id,
            'architecture_pair': 'BaseNN → BaseNN',
            'concept_dim': concept_dim,
            'sparsity_weight': sparsity,
            'injection_strength': injection_strength,
            'transfer_accuracy': transfer_acc,
            'preservation_accuracy': preservation_acc,
            'specificity_accuracy': specificity_acc,
            'transfer_samples': len(self.test_dataset_digit4),
            'preservation_samples': len(self.test_dataset_original),
            'specificity_samples': len(self.test_dataset_digit5),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Results - Transfer: {transfer_acc:.1f}%, "
                        f"Preservation: {preservation_acc:.1f}%, "
                        f"Specificity: {specificity_acc:.1f}%")
        
        return results
    
    def run_cross_architecture_test(self, source_arch: str, target_arch: str,
                                  concept_dim: int = 48, sparsity: float = 0.030,
                                  injection_strength: float = 0.4) -> Dict:
        """Run cross architecture test"""
        
        experiment_id = f"method1_cross_arch_{source_arch}_to_{target_arch}_{concept_dim}D"
        self.logger.info(f"Running cross architecture test: {experiment_id}")
        
        # Train models with different architectures
        source_model = self.train_base_model(source_arch, epochs=6)
        target_model = self.train_base_model(target_arch, epochs=6)
        
        # Extract concept vectors (cross-architecture alignment)
        injection_vector = self.extract_concept_vectors(source_model, target_model, concept_dim)
        
        # Create enhanced model
        enhanced_model = self.create_enhanced_model(target_model, injection_vector, injection_strength)
        
        # Fine-tune enhanced model
        self.fine_tune_enhanced_model(enhanced_model, sparsity_weight=sparsity, epochs=2)
        
        # Evaluate all metrics
        transfer_acc = self.evaluate_transfer_accuracy(enhanced_model)
        preservation_acc = self.evaluate_preservation_accuracy(enhanced_model)
        specificity_acc = self.evaluate_specificity_accuracy(enhanced_model)
        
        results = {
            'experiment_id': experiment_id,
            'architecture_pair': f'{source_arch} → {target_arch}',
            'source_architecture': source_arch,
            'target_architecture': target_arch,
            'concept_dim': concept_dim,
            'sparsity_weight': sparsity,
            'injection_strength': injection_strength,
            'transfer_accuracy': transfer_acc,
            'preservation_accuracy': preservation_acc,
            'specificity_accuracy': specificity_acc,
            'transfer_samples': len(self.test_dataset_digit4),
            'preservation_samples': len(self.test_dataset_original),
            'specificity_samples': len(self.test_dataset_digit5),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Results - Transfer: {transfer_acc:.1f}%, "
                        f"Preservation: {preservation_acc:.1f}%, "
                        f"Specificity: {specificity_acc:.1f}%")
        
        return results
    
    def fine_tune_enhanced_model(self, enhanced_model: nn.Module, sparsity_weight: float, epochs: int = 2):
        """Fine-tune enhanced model with sparsity regularization"""
        optimizer = optim.Adam(enhanced_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        
        enhanced_model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = enhanced_model(data)
                
                # Standard classification loss
                loss = criterion(output, target)
                
                # Add sparsity regularization on gate weights
                sparsity_loss = sparsity_weight * torch.norm(enhanced_model.gate_weights, p=1)
                
                total_loss = loss + sparsity_loss
                total_loss.backward()
                optimizer.step()
    
    def run_comprehensive_test_suite(self):
        """Run comprehensive test suite for Method 1"""
        all_results = []
        
        # Same architecture tests (replicating key configurations from documentation)
        same_arch_configs = [
            {'concept_dim': 48, 'sparsity': 0.030, 'injection_strength': 0.4},  # Breakthrough config
            {'concept_dim': 32, 'sparsity': 0.050, 'injection_strength': 0.4},  # Best specificity
            {'concept_dim': 48, 'sparsity': 0.120, 'injection_strength': 0.4},  # Best preservation
        ]
        
        self.logger.info("Starting same architecture tests...")
        for i, config in enumerate(same_arch_configs):
            self.logger.info(f"Same architecture test {i+1}/{len(same_arch_configs)}")
            result = self.run_same_architecture_test(**config)
            all_results.append(result)
        
        # Cross architecture tests
        cross_arch_pairs = [
            ('BaseNN', 'WideNN'),
            ('BaseNN', 'DeepNN'),
            ('BaseNN', 'BottleneckNN'),
            ('BaseNN', 'PyramidNN'),
            ('WideNN', 'DeepNN'),
            ('WideNN', 'BottleneckNN'),
        ]
        
        self.logger.info("Starting cross architecture tests...")
        for i, (source, target) in enumerate(cross_arch_pairs):
            self.logger.info(f"Cross architecture test {i+1}/{len(cross_arch_pairs)}: {source} → {target}")
            result = self.run_cross_architecture_test(source, target)
            all_results.append(result)
        
        # Save all results
        self.save_results(all_results)
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results to JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"method1_comprehensive_results_{timestamp}.json"
        
        # Add summary statistics
        summary = self.generate_summary_statistics(results)
        
        output = {
            'experiment_info': {
                'method': 'Method 1: Precomputed Vector Space Alignment',
                'total_experiments': len(results),
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            },
            'summary_statistics': summary,
            'detailed_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
    def generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for all experiments"""
        transfer_accs = [r['transfer_accuracy'] for r in results]
        preservation_accs = [r['preservation_accuracy'] for r in results]
        specificity_accs = [r['specificity_accuracy'] for r in results]
        
        same_arch_results = [r for r in results if 'same_arch' in r['experiment_id']]
        cross_arch_results = [r for r in results if 'cross_arch' in r['experiment_id']]
        
        summary = {
            'overall': {
                'best_transfer': max(transfer_accs),
                'avg_transfer': np.mean(transfer_accs),
                'median_transfer': np.median(transfer_accs),
                'best_preservation': max(preservation_accs),
                'avg_preservation': np.mean(preservation_accs),
                'best_specificity': min(specificity_accs),  # Lower is better
                'avg_specificity': np.mean(specificity_accs),
            },
            'same_architecture': {
                'count': len(same_arch_results),
                'avg_transfer': np.mean([r['transfer_accuracy'] for r in same_arch_results]) if same_arch_results else 0,
                'avg_preservation': np.mean([r['preservation_accuracy'] for r in same_arch_results]) if same_arch_results else 0,
            },
            'cross_architecture': {
                'count': len(cross_arch_results),
                'avg_transfer': np.mean([r['transfer_accuracy'] for r in cross_arch_results]) if cross_arch_results else 0,
                'avg_preservation': np.mean([r['preservation_accuracy'] for r in cross_arch_results]) if cross_arch_results else 0,
            }
        }
        
        return summary


def main():
    """Main execution function"""
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize tester
    tester = Method1Tester(config)
    
    # Run comprehensive test suite
    results = tester.run_comprehensive_test_suite()
    
    print(f"\nMethod 1 Testing Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"Best transfer accuracy: {max(r['transfer_accuracy'] for r in results):.1f}%")
    print(f"Best preservation accuracy: {max(r['preservation_accuracy'] for r in results):.1f}%")
    print(f"Best specificity: {min(r['specificity_accuracy'] for r in results):.1f}%")


if __name__ == "__main__":
    main()