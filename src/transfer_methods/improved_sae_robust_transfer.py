"""
Improved SAE Robust Transfer System
Uses the EXACT same conditions as robust_balanced_transfer.py 
but replaces rho blending with improved SAE integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from .robust_balanced_transfer import RobustBalancedTransferSystem
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluation.corrected_metrics import CorrectedMetricsEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedSAERobustTransferSystem(RobustBalancedTransferSystem):
    """
    Improved SAE version using EXACT same infrastructure as RobustBalancedTransferSystem
    but with trainable per-feature integration weights instead of single rho parameter.
    """
    
    def __init__(self, source_model, target_model, source_classes, target_classes, concept_dim=24, device='cpu'):
        # Initialize parent class to get all the same infrastructure
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        
        # Get target model feature dimension
        with torch.no_grad():
            sample_features = self.target_model.get_features(torch.randn(1, 784))
            target_feature_dim = sample_features.shape[1]
        
        # KEY IMPROVEMENT: Replace single rho with trainable per-feature integration weights
        self.integration_weights = nn.Parameter(torch.zeros(target_feature_dim)).to(device)
        
        # KEY IMPROVEMENT: Trainable injection strength (replaces fixed values)
        self.trainable_injection_strength = nn.Parameter(torch.tensor(0.9)).to(device)  # Start with same 0.9 as parent
        
        logger.info("🔧 Initialized Improved SAE Integration System")
        logger.info(f"   - Target feature dimension: {target_feature_dim}")
        logger.info(f"   - Trainable integration weights: {target_feature_dim} parameters")
        logger.info(f"   - Trainable injection strength: 1 parameter")
        logger.info(f"   - Total new parameters: {target_feature_dim + 1}")
        logger.info(f"   - Replaces: Single rho blending parameter")
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """
        Setup injection system using EXACT same approach as parent but with improved SAE integration.
        """
        logger.info(f"Setting up IMPROVED SAE injection for class {target_class}")
        
        # Use parent's exact setup logic up to the injection module creation
        if target_class not in self.source_classes:
            raise ValueError(f"Target class {target_class} not in source classes {self.source_classes}")
        
        # Get the source concept and align it (EXACT same as parent)
        source_concept = self.source_centroids[target_class]
        aligned_concept = self.aligner.transform(source_concept.unsqueeze(0)).squeeze(0)
        
        # Project to free space (EXACT same as parent)
        from neural_concept_transfer import FreeSpaceDiscovery
        free_space_discoverer = FreeSpaceDiscovery()
        free_space_discoverer.free_directions = self.free_directions
        target_projection = free_space_discoverer.project_to_free_space(aligned_concept)
        
        # Setup concept detector (EXACT same as parent)
        shared_centroids = torch.stack([self.target_centroids[c] for c in self.shared_classes])
        from neural_concept_transfer import ConceptDetector
        self.concept_detector = ConceptDetector(self.concept_dim, shared_centroids).to(self.device)
        
        # Setup injection module (EXACT same as parent)
        from neural_concept_transfer import ConceptInjectionModule
        self.injection_module = ConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Initialize with same aggressive injection strength as parent
        self.injection_module.injection_strength.data = torch.tensor(0.9)
        
        # IMPROVEMENT: Train both original injection system AND new SAE integration
        if source_loader is not None and target_loader is not None:
            logger.info("Training IMPROVED SAE injection system...")
            self._train_improved_sae_injection_system(target_class, source_loader, target_loader)
            
            # Use parent's robust final layer adaptation
            logger.info(f"ROBUST adaptation for class {target_class}")
            self._robust_final_layer_adaptation(target_class, source_loader, target_loader)
        
        return self.injection_module
    
    def _train_improved_sae_injection_system(self, target_class: int, source_loader, target_loader, training_steps=50):
        """Train improved SAE integration system with EXACT same aggressive parameters as parent."""
        
        # Original injection system parameters (same as parent)
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        injection_optimizer = optim.Adam(injection_params, lr=0.006)  # EXACT same learning rate
        
        # NEW: Optimizer for improved SAE integration parameters
        sae_integration_optimizer = optim.Adam([
            self.integration_weights,
            self.trainable_injection_strength
        ], lr=0.006, weight_decay=1e-5)  # SAME aggressive learning rate
        
        logger.info(f"IMPROVED SAE injection training for class {target_class}")
        logger.info(f"   - Training steps: {training_steps} (same as parent)")
        logger.info(f"   - Learning rate: 0.006 (same as parent)")
        logger.info(f"   - Optimizing: {len(injection_params)} injection params + {len(self.integration_weights) + 1} SAE params")
        
        for step in range(training_steps):
            try:
                source_batch = next(iter(source_loader))
                target_batch = next(iter(target_loader))
            except StopIteration:
                continue
                
            source_data, source_labels = source_batch[0].to(self.device), source_batch[1].to(self.device)
            target_data, target_labels = target_batch[0].to(self.device), target_batch[1].to(self.device)
            
            # Filter for relevant samples (EXACT same logic as parent)
            source_mask = (source_labels == target_class)
            target_mask = torch.isin(target_labels, torch.tensor(list(self.target_classes), device=self.device))
            
            if source_mask.sum() == 0 or target_mask.sum() == 0:
                continue
            
            # Zero gradients for both optimizers
            injection_optimizer.zero_grad()
            sae_integration_optimizer.zero_grad()
            
            # Source data processing with IMPROVED SAE integration
            source_transfer_data = source_data[source_mask]
            source_features = self.target_model.get_features(source_transfer_data.view(source_transfer_data.size(0), -1))
            source_concepts = self.target_sae.encode(source_features)
            
            # Apply improved SAE integration (KEY DIFFERENCE from parent)
            enhanced_source_features = self._apply_improved_sae_integration(
                source_features, source_concepts, inject_transfer_concept=True)
            
            # Target data processing with IMPROVED SAE integration  
            target_relevant_data = target_data[target_mask]
            target_features = self.target_model.get_features(target_relevant_data.view(target_relevant_data.size(0), -1))
            target_concepts = self.target_sae.encode(target_features)
            
            # Apply improved SAE integration (preserve original capabilities)
            enhanced_target_features = self._apply_improved_sae_integration(
                target_features, target_concepts, inject_transfer_concept=False)
            
            # Loss computation (EXACT same structure as parent)
            source_logits = self.target_model.fc6(enhanced_source_features)
            target_logits = self.target_model.fc6(enhanced_target_features)
            
            # Transfer loss: source samples should be classified as target_class
            transfer_labels = torch.full((source_transfer_data.size(0),), target_class, device=self.device)
            transfer_loss = F.cross_entropy(source_logits, transfer_labels)
            
            # Preservation loss: target samples should maintain original classifications
            preservation_loss = F.cross_entropy(target_logits, target_labels[target_mask])
            
            # Combined loss (EXACT same weighting as parent)
            total_loss = transfer_loss + 0.1 * preservation_loss
            
            # Backward pass
            total_loss.backward()
            
            # Update both sets of parameters
            injection_optimizer.step()
            sae_integration_optimizer.step()
            
            # Logging (same frequency as parent)
            if step % 10 == 0:
                # Show learned SAE integration parameters
                blend_weights = torch.sigmoid(self.integration_weights)
                injection_strength = torch.sigmoid(self.trainable_injection_strength)
                
                logger.info(f"IMPROVED SAE step {step}: Loss = {total_loss.item():.4f}")
                logger.info(f"   Transfer Loss: {transfer_loss.item():.4f}, Preservation Loss: {preservation_loss.item():.4f}")
                logger.info(f"   Blend weights range: [{blend_weights.min():.3f}, {blend_weights.max():.3f}]")
                logger.info(f"   Mean blend weight: {blend_weights.mean():.3f}")
                logger.info(f"   Injection strength: {injection_strength.item():.3f}")
        
        logger.info("✅ IMPROVED SAE injection training completed")
    
    def _apply_improved_sae_integration(self, features: torch.Tensor, concepts: torch.Tensor, inject_transfer_concept: bool = False):
        """
        Apply improved SAE integration with trainable per-feature weights.
        This is the KEY IMPROVEMENT over rho blending.
        """
        # Modify concepts if injecting transfer concept
        if inject_transfer_concept:
            # Apply concept injection using parent's infrastructure
            confidence = torch.ones(features.shape[0], device=self.device) * 0.8
            enhanced_concepts = self.injection_module(concepts, confidence, features)
        else:
            enhanced_concepts = concepts
        
        # Decode enhanced concepts back to feature space
        sae_features = self.target_sae.decode(enhanced_concepts)
        
        # KEY IMPROVEMENT: Trainable per-feature blending instead of single rho
        blend_weights = torch.sigmoid(self.integration_weights)  # Per-feature weights
        final_features = blend_weights * features + (1 - blend_weights) * sae_features
        
        return final_features
    
    def transfer(self, target_features: torch.Tensor) -> torch.Tensor:
        """
        Apply transfer using improved SAE integration.
        Overrides parent's rho blending implementation.
        """
        # Encode target features to concept space
        target_concepts = self.target_sae.encode(target_features)
        
        # Apply improved SAE integration with transfer concept injection
        enhanced_features = self._apply_improved_sae_integration(
            target_features, target_concepts, inject_transfer_concept=True)
        
        return enhanced_features


def test_improved_sae_robust_system():
    """Test improved SAE system using EXACT same conditions as robust_balanced_transfer.py"""
    from architectures import WideNN, DeepNN
    from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
    
    print("🧪 TESTING IMPROVED SAE ROBUST TRANSFER SYSTEM")
    print("Using EXACT same conditions as successful rho blending experiment")
    print("Cross-architecture transfer: DeepNN → WideNN")
    
    # EXACT same setup as robust_balanced_transfer.py
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,        # EXACT same
        max_epochs=5,   # EXACT same  
        batch_size=32,  # EXACT same
        learning_rate=0.001,  # EXACT same
        concept_dim=24, # EXACT same
        device='cpu'    # EXACT same
    )
    
    # EXACT same data and training setup
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    print("Training models with EXACT same parameters as successful rho run...")
    
    # EXACT same architectures as parent
    print("Source: DeepNN (8 layers, max 128 width)")
    source_model = DeepNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    
    print("Target: WideNN (6 layers, max 256 width)")  
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    
    if trained_source is None or trained_target is None:
        print("❌ Model training failed")
        return
    
    print(f"✅ Models trained: Source={source_acc:.4f}, Target={target_acc:.4f}")
    
    # Create IMPROVED SAE system (KEY DIFFERENCE)
    improved_system = ImprovedSAERobustTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit with EXACT same parameters as parent
    print("Fitting improved SAE system with same SAE training (50 epochs)...")
    fit_metrics = improved_system.fit(source_train_loader, target_train_loader, sae_epochs=50)
    
    # Setup for transfer class 8 (EXACT same as parent)
    transfer_class = 8
    print(f"Setting up IMPROVED SAE transfer for class {transfer_class}...")
    improved_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    # Evaluate with EXACT same metrics as parent
    evaluator = CorrectedMetricsEvaluator(config)
    
    print("\n📊 EVALUATING IMPROVED SAE SYSTEM...")
    
    # Before transfer (EXACT same evaluation)
    before_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, None, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    # After transfer with improved SAE
    after_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, improved_system, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    print("\n📈 IMPROVED SAE SYSTEM RESULTS:")
    print(f"🎯 ORIGINAL KNOWLEDGE PRESERVATION:")
    print(f"   Before: {before_metrics.original_knowledge_preservation:.4f}")
    print(f"   After:  {after_metrics.original_knowledge_preservation:.4f}")
    print(f"   Requirement: >0.80 ✅" if after_metrics.original_knowledge_preservation > 0.8 else "   Requirement: >0.80 ❌")
    
    print(f"\n🎯 TRANSFER EFFECTIVENESS:")
    print(f"   Before: {before_metrics.transfer_effectiveness:.4f}")
    print(f"   After:  {after_metrics.transfer_effectiveness:.4f}")
    print(f"   Requirement: >0.70 ✅" if after_metrics.transfer_effectiveness > 0.7 else "   Requirement: >0.70 ❌")
    print(f"   Improvement: {after_metrics.transfer_effectiveness - before_metrics.transfer_effectiveness:+.4f}")
    
    print(f"\n🎯 TRANSFER SPECIFICITY:")
    print(f"   Before: {before_metrics.transfer_specificity:.4f}")
    print(f"   After:  {after_metrics.transfer_specificity:.4f}")
    print(f"   Target: >0.70 ✅" if after_metrics.transfer_specificity > 0.7 else "   Target: >0.70 ❌")
    
    # Overall assessment (EXACT same logic as parent)
    preservation_ok = after_metrics.original_knowledge_preservation > 0.8
    effectiveness_ok = after_metrics.transfer_effectiveness > 0.7
    specificity_ok = after_metrics.transfer_specificity > 0.7
    
    success_count = sum([preservation_ok, effectiveness_ok, specificity_ok])
    
    if success_count == 3:
        print("\n🎉 COMPLETE SUCCESS - All requirements met!")
    elif success_count >= 2:
        print("\n✅ PARTIAL SUCCESS - Most requirements met")
    else:
        print("\n⚠️ NEEDS IMPROVEMENT - Requirements not fully met")
    
    # Save results for comparison
    import json
    from datetime import datetime
    from pathlib import Path
    
    results = {
        "experiment_name": "Improved SAE vs Rho Blending - Real Data Comparison",
        "timestamp": datetime.now().isoformat(),
        "model_accuracies": {
            "source": source_acc,
            "target": target_acc
        },
        "improved_sae_results": {
            "original_knowledge_preservation": after_metrics.original_knowledge_preservation,
            "transfer_effectiveness": after_metrics.transfer_effectiveness,
            "transfer_specificity": after_metrics.transfer_specificity,
            "preservation_improvement": after_metrics.original_knowledge_preservation - before_metrics.original_knowledge_preservation,
            "effectiveness_improvement": after_metrics.transfer_effectiveness - before_metrics.transfer_effectiveness
        },
        "requirements_met": {
            "preservation": preservation_ok,
            "effectiveness": effectiveness_ok, 
            "specificity": specificity_ok,
            "total_met": success_count,
            "success_rate": success_count / 3
        }
    }
    
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "IMPROVED_SAE_REAL_DATA_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: IMPROVED_SAE_REAL_DATA_RESULTS.json")
    print(f"\n🏆 FINAL COMPARISON READY - Run comparison script to see vs rho blending!")
    
    return results


if __name__ == "__main__":
    test_improved_sae_robust_system()