"""
SAE Integration Experiment
Testing direct SAE integration into model predictions instead of feature blending.

This experiment tests what happens when we:
1. Add SAE layers directly into the forward pass
2. Skip the rho blending mechanism
3. Let the model learn to use injected concepts directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures import WideNN
from neural_concept_transfer import SparseAutoencoder, NeuralConceptTransferSystem
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
from corrected_metrics import CorrectedMetricsEvaluator

class IntegratedSAEModel(nn.Module):
    """
    Model with SAE directly integrated into the forward pass.
    
    Instead of blending features with rho, the SAE becomes part of the model architecture.
    """
    
    def __init__(self, base_model, sae, injection_module=None):
        super().__init__()
        self.base_model = base_model
        self.sae = sae
        self.injection_module = injection_module
        self.use_sae = True  # Flag to enable/disable SAE integration
        
        # Integration modes
        self.integration_mode = "replace"  # "replace", "add", "concat"
        
    def set_integration_mode(self, mode: str):
        """Set how SAE features are integrated: 'replace', 'add', or 'concat'."""
        assert mode in ["replace", "add", "concat"]
        self.integration_mode = mode
        
        if mode == "concat":
            # Need to adapt final layer for concatenated features
            original_final = self.base_model.fc6  # WideNN final layer
            input_dim = original_final.in_features * 2  # Double the features
            output_dim = original_final.out_features
            
            self.base_model.fc6 = nn.Linear(input_dim, output_dim)
            # Initialize with original weights
            with torch.no_grad():
                self.base_model.fc6.weight[:, :original_final.in_features] = original_final.weight
                self.base_model.fc6.weight[:, original_final.in_features:] = original_final.weight * 0.1  # Smaller init for SAE features
                self.base_model.fc6.bias = original_final.bias
    
    def forward(self, x, inject_concepts=False, target_class=None):
        """
        Forward pass with integrated SAE processing.
        
        Args:
            x: Input tensor
            inject_concepts: Whether to inject concepts during forward pass
            target_class: Class to inject (if injection is enabled)
        """
        # Get original features from base model
        original_features = self.base_model.get_features(x)
        
        if not self.use_sae or not inject_concepts:
            # Standard forward pass without SAE
            return self.base_model.classify_from_features(original_features)
        
        # SAE processing
        sae_concepts = self.sae.encode(original_features)
        
        # Apply concept injection if available
        if self.injection_module is not None and target_class is not None:
            # Simple injection - add aligned concept to all samples
            enhanced_concepts = sae_concepts + self.injection_module.get_injection_vector(target_class)
        else:
            enhanced_concepts = sae_concepts
        
        # Decode back to feature space
        sae_features = self.sae.decode(enhanced_concepts)
        
        # Integration strategies
        if self.integration_mode == "replace":
            # Replace original features entirely with SAE features
            final_features = sae_features
            
        elif self.integration_mode == "add":
            # Add SAE features to original features
            final_features = original_features + sae_features
            
        elif self.integration_mode == "concat":
            # Concatenate original and SAE features
            final_features = torch.cat([original_features, sae_features], dim=1)
        
        # Final classification
        return self.base_model.classify_from_features(final_features)
    
    def get_features(self, x):
        """Extract features for compatibility."""
        return self.base_model.get_features(x)
    
    def classify_from_features(self, features):
        """Classify from features for compatibility."""
        return self.base_model.classify_from_features(features)


class SimpleInjectionModule:
    """Simplified injection module for testing."""
    
    def __init__(self, concept_dim, target_concepts):
        self.concept_dim = concept_dim
        self.target_concepts = target_concepts  # Dict: class_id -> concept_vector
        self.injection_strength = 0.5
    
    def get_injection_vector(self, target_class):
        """Get injection vector for target class."""
        if target_class in self.target_concepts:
            return self.injection_strength * self.target_concepts[target_class]
        else:
            return torch.zeros(self.concept_dim)


def test_sae_integration_modes():
    """Test different SAE integration approaches."""
    
    print("🧪 TESTING SAE INTEGRATION MODES")
    print("=" * 60)
    
    # Setup experiment configuration
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    transfer_class = 8
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=3,
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Create data loaders
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    # Train base models
    print("Training base models...")
    source_model = WideNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    
    if trained_source is None or trained_target is None:
        print("❌ Model training failed")
        return
    
    print(f"✅ Base models trained: Source={source_acc:.4f}, Target={target_acc:.4f}")
    
    # Train SAE for target model
    print("Training target SAE...")
    target_sae = SparseAutoencoder(64, config.concept_dim).to(config.device)  # WideNN has 64D features
    
    # Simple SAE training
    sae_optimizer = optim.Adam(target_sae.parameters(), lr=0.001)
    
    for epoch in range(20):
        total_loss = 0
        count = 0
        for batch_idx, (data, _) in enumerate(target_train_loader):
            if batch_idx >= 10:  # Limit batches for speed
                break
            
            data = data.to(config.device)
            features = trained_target.get_features(data)
            
            sae_optimizer.zero_grad()
            loss, _ = target_sae.compute_loss(features)
            loss.backward()
            sae_optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"SAE Epoch {epoch+1}: Loss = {total_loss/count:.4f}")
    
    print("✅ SAE training completed")
    
    # Extract concept for transfer class from source model
    print(f"Extracting concept for transfer class {transfer_class}...")
    
    # Simple concept extraction - average features of transfer class
    source_concepts = []
    with torch.no_grad():
        for data, labels in source_test_loader:
            mask = (labels == transfer_class)
            if mask.sum() > 0:
                transfer_data = data[mask].to(config.device)
                source_features = trained_source.get_features(transfer_data)
                # For simplicity, just use source features as "concepts"
                source_concepts.append(source_features.mean(dim=0))
    
    if source_concepts:
        avg_source_concept = torch.stack(source_concepts).mean(dim=0)
        # Transform to target SAE concept space
        target_concept = target_sae.encode(avg_source_concept.unsqueeze(0)).squeeze(0)
    else:
        print("❌ No source concepts found")
        return
    
    # Create injection module
    injection_module = SimpleInjectionModule(
        concept_dim=config.concept_dim,
        target_concepts={transfer_class: target_concept}
    )
    
    print("✅ Concept extraction completed")
    
    # Test different integration modes
    integration_modes = ["replace", "add", "concat"]
    evaluator = CorrectedMetricsEvaluator(config)
    
    results = {}
    
    print("\n📊 TESTING INTEGRATION MODES:")
    print("-" * 60)
    
    for mode in integration_modes:
        print(f"\n🔧 Testing {mode.upper()} mode...")
        
        # Create integrated model
        integrated_model = IntegratedSAEModel(trained_target, target_sae, injection_module)
        integrated_model.set_integration_mode(mode)
        
        # Test without injection (baseline)
        def eval_without_injection(data, target):
            data = data.to(config.device)
            with torch.no_grad():
                outputs = integrated_model(data, inject_concepts=False)
                _, predicted = torch.max(outputs, 1)
                return predicted
        
        # Test with injection
        def eval_with_injection(data, target):
            data = data.to(config.device)
            with torch.no_grad():
                outputs = integrated_model(data, inject_concepts=True, target_class=transfer_class)
                _, predicted = torch.max(outputs, 1)
                return predicted
        
        # Evaluate on transfer class (class 8)
        transfer_correct_without = 0
        transfer_correct_with = 0
        transfer_total = 0
        
        # Evaluate on original classes (0-7)
        original_correct_without = 0
        original_correct_with = 0
        original_total = 0
        
        with torch.no_grad():
            # Test on source data (contains class 8)
            for data, labels in source_test_loader:
                # Class 8 samples
                transfer_mask = (labels == transfer_class)
                if transfer_mask.sum() > 0:
                    transfer_data = data[transfer_mask]
                    transfer_labels = labels[transfer_mask]
                    
                    pred_without = eval_without_injection(transfer_data, transfer_labels)
                    pred_with = eval_with_injection(transfer_data, transfer_labels)
                    
                    transfer_correct_without += (pred_without == transfer_class).sum().item()
                    transfer_correct_with += (pred_with == transfer_class).sum().item()
                    transfer_total += transfer_data.size(0)
            
            # Test on target data (contains classes 0-7)
            for data, labels in target_test_loader:
                pred_without = eval_without_injection(data, labels)
                pred_with = eval_with_injection(data, labels)
                
                original_correct_without += (pred_without == labels.to(config.device)).sum().item()
                original_correct_with += (pred_with == labels.to(config.device)).sum().item()
                original_total += data.size(0)
        
        # Calculate accuracies
        transfer_acc_without = transfer_correct_without / transfer_total if transfer_total > 0 else 0
        transfer_acc_with = transfer_correct_with / transfer_total if transfer_total > 0 else 0
        original_acc_without = original_correct_without / original_total if original_total > 0 else 0
        original_acc_with = original_correct_with / original_total if original_total > 0 else 0
        
        print(f"   📈 Results for {mode.upper()} mode:")
        print(f"      Transfer Class {transfer_class}:")
        print(f"         Without injection: {transfer_acc_without:.1%}")
        print(f"         With injection:    {transfer_acc_with:.1%}")
        print(f"         Improvement:       {transfer_acc_with - transfer_acc_without:+.1%}")
        
        print(f"      Original Classes (0-7):")
        print(f"         Without injection: {original_acc_without:.1%}")
        print(f"         With injection:    {original_acc_with:.1%}")
        print(f"         Change:            {original_acc_with - original_acc_without:+.1%}")
        
        # Store results
        results[mode] = {
            'transfer_without': transfer_acc_without,
            'transfer_with': transfer_acc_with,
            'transfer_improvement': transfer_acc_with - transfer_acc_without,
            'original_without': original_acc_without,
            'original_with': original_acc_with,
            'original_change': original_acc_with - original_acc_without
        }
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 INTEGRATION MODE COMPARISON")
    print("=" * 60)
    
    print(f"{'Mode':<10} {'Transfer Δ':<12} {'Original Δ':<12} {'Assessment':<15}")
    print("-" * 60)
    
    for mode, result in results.items():
        transfer_delta = f"{result['transfer_improvement']:+.1%}"
        original_delta = f"{result['original_change']:+.1%}"
        
        # Assessment
        if result['transfer_improvement'] > 0.1 and abs(result['original_change']) < 0.05:
            assessment = "✅ Excellent"
        elif result['transfer_improvement'] > 0.05:
            assessment = "🟡 Good"
        else:
            assessment = "❌ Poor"
        
        print(f"{mode.upper():<10} {transfer_delta:<12} {original_delta:<12} {assessment:<15}")
    
    print("\n💡 KEY INSIGHTS:")
    best_mode = max(results.keys(), key=lambda k: results[k]['transfer_improvement'])
    print(f"   • Best integration mode: {best_mode.upper()}")
    print(f"   • Direct SAE integration vs. rho blending comparison")
    print(f"   • Impact on original knowledge preservation")
    
    return results


if __name__ == "__main__":
    test_sae_integration_modes()