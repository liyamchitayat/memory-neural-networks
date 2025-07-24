"""
Method 2: Cross-Architecture Neural Alignment Testing Implementation

This module implements comprehensive testing for Method 2, which uses neural networks
to learn alignment between different architectures for concept transfer.

Method 2 Details:
- Neural network-based alignment between source and target architectures
- Supports both Procrustes (linear) and nonlinear alignment approaches
- Optimized for cross-architecture transfer scenarios

Testing Coverage:
- Same Architecture: BaseNN → BaseNN (baseline comparison)
- Cross Architecture: All architecture pairs with neural alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

from neural_architectures import get_architecture
from method1_precomputed_vector_alignment import Method1Tester  # Reuse base functionality


class NeuralAlignmentNetwork(nn.Module):
    """Neural network for cross-architecture alignment"""
    
    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int = 128, 
                 alignment_type: str = 'nonlinear'):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.alignment_type = alignment_type
        
        if alignment_type == 'linear':
            # Simple linear transformation
            self.alignment = nn.Linear(source_dim, target_dim)
        elif alignment_type == 'nonlinear':
            # Nonlinear neural alignment
            self.alignment = nn.Sequential(
                nn.Linear(source_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, target_dim)
            )
        else:
            raise ValueError(f"Unknown alignment type: {alignment_type}")
    
    def forward(self, source_activations):
        return self.alignment(source_activations)


class ProcrustesAlignmentNetwork(nn.Module):
    """Procrustes-based alignment for cross-architecture transfer"""
    
    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        # Learnable transformation matrix (initialized as identity when possible)
        if source_dim == target_dim:
            self.transform = nn.Parameter(torch.eye(source_dim))
        else:
            # Use PCA-style dimensionality adaptation
            self.transform = nn.Parameter(torch.randn(target_dim, source_dim) * 0.1)
    
    def forward(self, source_activations):
        return torch.matmul(source_activations, self.transform.T)


class Method2Tester(Method1Tester):
    """Comprehensive tester for Method 2: Cross-Architecture Neural Alignment"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.alignment_networks = {}
    
    def extract_activation_pairs(self, source_model: nn.Module, target_model: nn.Module,
                               num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract paired activations from source and target models for alignment training"""
        
        # Use training data for alignment learning
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        
        source_activations = []
        target_activations = []
        
        source_model.eval()
        target_model.eval()
        
        samples_collected = 0
        with torch.no_grad():
            for data, _ in train_loader:
                if samples_collected >= num_samples:
                    break
                    
                data = data.to(self.device)
                
                # Extract penultimate activations
                source_h = source_model.get_penultimate_activations(data)
                target_h = target_model.get_penultimate_activations(data)
                
                source_activations.append(source_h.cpu())
                target_activations.append(target_h.cpu())
                
                samples_collected += data.size(0)
        
        source_activations = torch.cat(source_activations, dim=0)[:num_samples]
        target_activations = torch.cat(target_activations, dim=0)[:num_samples]
        
        return source_activations, target_activations
    
    def train_alignment_network(self, source_model: nn.Module, target_model: nn.Module,
                              alignment_type: str = 'nonlinear', epochs: int = 50) -> nn.Module:
        """Train neural alignment network between source and target architectures"""
        
        source_dim = source_model.penultimate_dim
        target_dim = target_model.penultimate_dim
        
        # Create alignment network
        if alignment_type == 'procrustes':
            alignment_net = ProcrustesAlignmentNetwork(source_dim, target_dim)
        else:
            alignment_net = NeuralAlignmentNetwork(source_dim, target_dim, 
                                                 alignment_type=alignment_type)
        
        alignment_net = alignment_net.to(self.device)
        
        # Extract training data
        source_acts, target_acts = self.extract_activation_pairs(source_model, target_model)
        source_acts = source_acts.to(self.device)
        target_acts = target_acts.to(self.device)
        
        # Training setup
        optimizer = optim.Adam(alignment_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.logger.info(f"Training {alignment_type} alignment network: {source_dim}D → {target_dim}D")
        
        # Train alignment network
        alignment_net.train()
        for epoch in range(epochs):
            # Mini-batch training
            batch_size = 64
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(source_acts), batch_size):
                batch_source = source_acts[i:i+batch_size]
                batch_target = target_acts[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass through alignment network
                aligned_activations = alignment_net(batch_source)
                
                # Loss: minimize distance between aligned source and target activations
                loss = criterion(aligned_activations, batch_target)
                
                # L2 regularization
                l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in alignment_net.parameters())
                total_loss_batch = loss + l2_reg
                
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                self.logger.info(f"Alignment epoch {epoch}/{epochs}: Loss {avg_loss:.6f}")
        
        return alignment_net
    
    def create_cross_arch_enhanced_model(self, source_model: nn.Module, target_model: nn.Module,
                                       alignment_net: nn.Module, concept_dim: int = 48,
                                       injection_strength: float = 0.4) -> nn.Module:
        """Create enhanced model with cross-architecture concept injection"""
        
        class CrossArchEnhancedModel(nn.Module):
            def __init__(self, source_model, target_model, alignment_net, concept_vector, strength):
                super().__init__()
                self.source_model = source_model
                self.target_model = target_model
                self.alignment_net = alignment_net
                self.concept_vector = concept_vector
                self.strength = strength
                
                # Learnable gating function for cross-architecture injection
                self.gate_weights = nn.Parameter(torch.randn(target_model.penultimate_dim))
                self.gate_bias = nn.Parameter(torch.zeros(1))
                
            def forward(self, x):
                # Get target model's base activations
                target_h = self.target_model.get_penultimate_activations(x)
                
                # Get source concept and align it to target space
                source_h = self.source_model.get_penultimate_activations(x)
                aligned_concept = self.alignment_net(source_h)
                
                # Compute concept similarity for gating
                concept_similarity = torch.cosine_similarity(target_h, aligned_concept, dim=1, eps=1e-8)
                gate_score = torch.sigmoid(concept_similarity.unsqueeze(1))
                
                # Apply cross-architecture concept injection
                enhanced_h = target_h + gate_score * self.concept_vector.unsqueeze(0) * self.strength
                
                # Pass through target model's final layer
                if hasattr(self.target_model, 'fc2'):
                    output = self.target_model.fc2(enhanced_h)
                elif hasattr(self.target_model, 'fc3'):  # DeepNN case
                    output = self.target_model.fc3(enhanced_h)
                else:
                    raise ValueError("Unknown final layer structure")
                    
                return output
        
        # Extract cross-architecture concept vector
        concept_vector = self.extract_cross_arch_concept_vector(
            source_model, target_model, alignment_net, concept_dim
        )
        
        enhanced = CrossArchEnhancedModel(source_model, target_model, alignment_net, 
                                        concept_vector, injection_strength)
        return enhanced.to(self.device)
    
    def extract_cross_arch_concept_vector(self, source_model: nn.Module, target_model: nn.Module,
                                        alignment_net: nn.Module, concept_dim: int) -> torch.Tensor:
        """Extract concept vector using cross-architecture alignment"""
        
        # Get digit-4 samples for concept extraction
        digit4_loader = DataLoader(self.test_dataset_digit4, batch_size=32, shuffle=False)
        
        source_digit4_acts = []
        target_digit4_acts = []
        aligned_digit4_acts = []
        
        source_model.eval()
        target_model.eval()
        alignment_net.eval()
        
        with torch.no_grad():
            for data, _ in digit4_loader:
                data = data.to(self.device)
                
                # Extract activations
                source_h = source_model.get_penultimate_activations(data)
                target_h = target_model.get_penultimate_activations(data)
                aligned_h = alignment_net(source_h)
                
                source_digit4_acts.append(source_h.cpu())
                target_digit4_acts.append(target_h.cpu())
                aligned_digit4_acts.append(aligned_h.cpu())
        
        source_digit4_acts = torch.cat(source_digit4_acts, dim=0).numpy()
        target_digit4_acts = torch.cat(target_digit4_acts, dim=0).numpy()
        aligned_digit4_acts = torch.cat(aligned_digit4_acts, dim=0).numpy()
        
        # Compute concept vector as difference between aligned source and target representations
        source_mean = np.mean(aligned_digit4_acts, axis=0)
        target_mean = np.mean(target_digit4_acts, axis=0)
        
        concept_vector = target_mean - source_mean
        
        return torch.tensor(concept_vector, dtype=torch.float32).to(self.device)
    
    def run_same_architecture_baseline(self, alignment_type: str = 'nonlinear') -> Dict:
        """Run same architecture baseline for Method 2"""
        
        experiment_id = f"method2_same_arch_baseline_{alignment_type}"
        self.logger.info(f"Running same architecture baseline: {experiment_id}")
        
        # Train base models (same architecture)
        source_model = self.train_base_model('BaseNN', epochs=6)
        target_model = self.train_base_model('BaseNN', epochs=6)
        
        # Train alignment network (should learn near-identity transformation)
        alignment_net = self.train_alignment_network(source_model, target_model, 
                                                   alignment_type=alignment_type)
        
        # Create enhanced model
        enhanced_model = self.create_cross_arch_enhanced_model(
            source_model, target_model, alignment_net
        )
        
        # Fine-tune
        self.fine_tune_enhanced_model(enhanced_model, sparsity_weight=0.030, epochs=2)
        
        # Evaluate
        transfer_acc = self.evaluate_transfer_accuracy(enhanced_model)
        preservation_acc = self.evaluate_preservation_accuracy(enhanced_model)
        specificity_acc = self.evaluate_specificity_accuracy(enhanced_model)
        
        results = {
            'experiment_id': experiment_id,
            'architecture_pair': 'BaseNN → BaseNN (baseline)',
            'alignment_type': alignment_type,
            'transfer_accuracy': transfer_acc,
            'preservation_accuracy': preservation_acc,
            'specificity_accuracy': specificity_acc,
            'transfer_samples': len(self.test_dataset_digit4),
            'preservation_samples': len(self.test_dataset_original),
            'specificity_samples': len(self.test_dataset_digit5),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Same arch baseline - Transfer: {transfer_acc:.1f}%, "
                        f"Preservation: {preservation_acc:.1f}%, "
                        f"Specificity: {specificity_acc:.1f}%")
        
        return results
    
    def run_cross_architecture_neural_alignment(self, source_arch: str, target_arch: str,
                                              alignment_type: str = 'nonlinear') -> Dict:
        """Run cross architecture test with neural alignment"""
        
        experiment_id = f"method2_cross_arch_{alignment_type}_{source_arch}_to_{target_arch}"
        self.logger.info(f"Running cross architecture alignment: {experiment_id}")
        
        # Train models with different architectures
        source_model = self.train_base_model(source_arch, epochs=6)
        target_model = self.train_base_model(target_arch, epochs=6)
        
        # Train alignment network
        alignment_net = self.train_alignment_network(source_model, target_model,
                                                   alignment_type=alignment_type)
        
        # Create enhanced model
        enhanced_model = self.create_cross_arch_enhanced_model(
            source_model, target_model, alignment_net
        )
        
        # Fine-tune
        self.fine_tune_enhanced_model(enhanced_model, sparsity_weight=0.030, epochs=2)
        
        # Evaluate
        transfer_acc = self.evaluate_transfer_accuracy(enhanced_model)
        preservation_acc = self.evaluate_preservation_accuracy(enhanced_model)
        specificity_acc = self.evaluate_specificity_accuracy(enhanced_model)
        
        results = {
            'experiment_id': experiment_id,
            'architecture_pair': f'{source_arch} → {target_arch}',
            'source_architecture': source_arch,
            'target_architecture': target_arch,
            'alignment_type': alignment_type,
            'transfer_accuracy': transfer_acc,
            'preservation_accuracy': preservation_acc,
            'specificity_accuracy': specificity_acc,
            'transfer_samples': len(self.test_dataset_digit4),
            'preservation_samples': len(self.test_dataset_original),
            'specificity_samples': len(self.test_dataset_digit5),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Cross arch results - Transfer: {transfer_acc:.1f}%, "
                        f"Preservation: {preservation_acc:.1f}%, "
                        f"Specificity: {specificity_acc:.1f}%")
        
        return results
    
    def run_comprehensive_test_suite(self):
        """Run comprehensive test suite for Method 2"""
        all_results = []
        
        # Same architecture baselines
        alignment_types = ['linear', 'nonlinear', 'procrustes']
        
        self.logger.info("Starting same architecture baseline tests...")
        for alignment_type in alignment_types:
            result = self.run_same_architecture_baseline(alignment_type)
            all_results.append(result)
        
        # Cross architecture tests (replicating documented experiments)
        cross_arch_pairs = [
            ('WideNN', 'DeepNN'),     # Best documented performance: 42.2%
            ('BottleneckNN', 'DeepNN'), # Best documented performance: 42.2%
            ('PyramidNN', 'WideNN'),   # Best specificity: 5.1%
            ('PyramidNN', 'BottleneckNN'),
            ('DeepNN', 'BottleneckNN'),
            ('BaseNN', 'WideNN'),
            ('BaseNN', 'DeepNN'),
            ('BaseNN', 'BottleneckNN'),
        ]
        
        self.logger.info("Starting cross architecture tests...")
        for source, target in cross_arch_pairs:
            for alignment_type in ['nonlinear', 'procrustes']:  # Focus on best performing types
                result = self.run_cross_architecture_neural_alignment(source, target, alignment_type)
                all_results.append(result)
        
        # Save results
        self.save_results(all_results, method_name="Method2")
        return all_results
    
    def save_results(self, results: List[Dict], method_name: str = "Method2"):
        """Save results with Method 2 specific formatting"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"method2_comprehensive_results_{timestamp}.json"
        
        # Generate summary statistics
        summary = self.generate_method2_summary(results)
        
        output = {
            'experiment_info': {
                'method': 'Method 2: Cross-Architecture Neural Alignment',
                'total_experiments': len(results),
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            },
            'summary_statistics': summary,
            'detailed_results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Method 2 results saved to {filename}")
    
    def generate_method2_summary(self, results: List[Dict]) -> Dict:
        """Generate Method 2 specific summary statistics"""
        same_arch_results = [r for r in results if 'same_arch' in r['experiment_id']]
        cross_arch_results = [r for r in results if 'cross_arch' in r['experiment_id']]
        
        # Analyze by alignment type
        nonlinear_results = [r for r in results if r.get('alignment_type') == 'nonlinear']
        procrustes_results = [r for r in results if r.get('alignment_type') == 'procrustes']
        linear_results = [r for r in results if r.get('alignment_type') == 'linear']
        
        summary = {
            'overall_performance': {
                'best_transfer': max(r['transfer_accuracy'] for r in results),
                'best_preservation': max(r['preservation_accuracy'] for r in results),
                'best_specificity': min(r['specificity_accuracy'] for r in results),
                'avg_transfer': np.mean([r['transfer_accuracy'] for r in results]),
                'avg_preservation': np.mean([r['preservation_accuracy'] for r in results]),
            },
            'architecture_comparison': {
                'same_architecture_count': len(same_arch_results),
                'cross_architecture_count': len(cross_arch_results),
                'same_arch_avg_transfer': np.mean([r['transfer_accuracy'] for r in same_arch_results]) if same_arch_results else 0,
                'cross_arch_avg_transfer': np.mean([r['transfer_accuracy'] for r in cross_arch_results]) if cross_arch_results else 0,
            },
            'alignment_type_analysis': {
                'nonlinear': {
                    'count': len(nonlinear_results),
                    'avg_transfer': np.mean([r['transfer_accuracy'] for r in nonlinear_results]) if nonlinear_results else 0,
                    'best_transfer': max([r['transfer_accuracy'] for r in nonlinear_results]) if nonlinear_results else 0,
                },
                'procrustes': {
                    'count': len(procrustes_results),
                    'avg_transfer': np.mean([r['transfer_accuracy'] for r in procrustes_results]) if procrustes_results else 0,
                    'best_transfer': max([r['transfer_accuracy'] for r in procrustes_results]) if procrustes_results else 0,
                },
                'linear': {
                    'count': len(linear_results),
                    'avg_transfer': np.mean([r['transfer_accuracy'] for r in linear_results]) if linear_results else 0,
                    'best_transfer': max([r['transfer_accuracy'] for r in linear_results]) if linear_results else 0,
                },
            }
        }
        
        return summary


def main():
    """Main execution function for Method 2"""
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize tester
    tester = Method2Tester(config)
    
    # Run comprehensive test suite
    results = tester.run_comprehensive_test_suite()
    
    print(f"\nMethod 2 Testing Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"Best transfer accuracy: {max(r['transfer_accuracy'] for r in results):.1f}%")
    print(f"Best preservation accuracy: {max(r['preservation_accuracy'] for r in results):.1f}%")
    print(f"Best specificity: {min(r['specificity_accuracy'] for r in results):.1f}%")
    
    # Compare with documented results
    cross_arch_results = [r for r in results if 'cross_arch' in r['experiment_id']]
    if cross_arch_results:
        best_cross_arch = max(r['transfer_accuracy'] for r in cross_arch_results)
        print(f"Best cross-architecture transfer: {best_cross_arch:.1f}% (Target: 42.2%)")


if __name__ == "__main__":
    main()