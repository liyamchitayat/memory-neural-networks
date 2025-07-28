"""
Knowledge-Preserving Transfer System
Improved version that maintains >80% accuracy on original classes while adding new capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from neural_concept_transfer import NeuralConceptTransferSystem
from corrected_metrics import CorrectedMetricsEvaluator, CorrectedTransferMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgePreservingTransferSystem(NeuralConceptTransferSystem):
    """
    Transfer system that preserves original knowledge while adding new capabilities.
    
    Key improvements:
    1. Conservative final layer adaptation to prevent catastrophic forgetting
    2. Regularization to maintain original class boundaries
    3. Selective adaptation (only for transfer class, not all source knowledge)
    4. Original knowledge validation during training
    """
    
    def __init__(self, source_model, target_model, source_classes, target_classes, concept_dim=24, device='cpu'):
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        
        # Store original model state for preservation
        self.original_target_state = self.target_model.state_dict().copy()
        self.preservation_evaluator = CorrectedMetricsEvaluator(type('Config', (), {'device': device})())
        
    def _adapt_target_final_layer(self, target_class: int, source_loader, target_loader=None):
        """
        Knowledge-preserving final layer adaptation.
        
        Key improvements:
        1. Much smaller learning rate to prevent overwriting
        2. L2 regularization to stay close to original weights  
        3. Validation on original classes during adaptation
        4. Early stopping if original knowledge degrades too much
        """
        logger.info(f"Starting knowledge-preserving adaptation for class {target_class}")
        
        # Get the final classification layer
        if hasattr(self.target_model, 'classifier'):
            final_layer = self.target_model.classifier
        else:
            final_layer = None
            for module in reversed(list(self.target_model.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break
        
        if final_layer is None:
            logger.warning("Could not find final classification layer")
            return
        
        # Store original layer weights for regularization
        original_weight = final_layer.weight.data.clone()
        original_bias = final_layer.bias.data.clone()
        
        # Create adaptation dataset (much smaller to prevent overfitting)
        adaptation_features = []
        adaptation_labels = []
        
        with torch.no_grad():
            batch_count = 0
            for batch_idx, (data, labels) in enumerate(source_loader):
                if batch_count >= 2:  # Very limited data to prevent overwriting
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                    
                transfer_data = data[mask][:3]  # Max 3 samples per batch
                
                # Get enhanced features through our pipeline
                features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                concepts = self.target_sae.encode(features)
                confidence = torch.ones(features.shape[0], device=self.device) * 0.3  # Lower confidence
                enhanced_concepts = self.injection_module(concepts, confidence, features)
                enhanced_features = self.target_sae.decode(enhanced_concepts)
                
                adaptation_features.append(enhanced_features)
                adaptation_labels.append(torch.full((enhanced_features.shape[0],), target_class, device=self.device))
                batch_count += 1
        
        if not adaptation_features:
            logger.warning(f"No adaptation data found for class {target_class}")
            return
        
        adaptation_features = torch.cat(adaptation_features, dim=0)
        adaptation_labels = torch.cat(adaptation_labels, dim=0)
        
        logger.info(f"Adapting with {adaptation_features.shape[0]} samples")
        
        # Very conservative optimization
        optimizer = optim.Adam([final_layer.weight, final_layer.bias], lr=0.001)  # Much lower LR
        
        # Track original performance
        if target_loader is not None:
            original_metrics = self.preservation_evaluator.evaluate_transfer_metrics(
                self.target_model, None, source_loader, target_loader, 
                target_class, self.source_classes, self.target_classes)
            original_preservation = original_metrics.original_knowledge_preservation
            logger.info(f"Original knowledge baseline: {original_preservation:.4f}")
        else:
            original_preservation = 0.9  # Assume high baseline
        
        best_weights = (final_layer.weight.data.clone(), final_layer.bias.data.clone())
        best_preservation = 0.0
        
        for step in range(15):  # Very few steps
            optimizer.zero_grad()
            
            # Forward pass
            outputs = final_layer(adaptation_features)
            adaptation_loss = nn.functional.cross_entropy(outputs, adaptation_labels)
            
            # Strong regularization to stay close to original weights
            weight_reg = 0.1 * torch.norm(final_layer.weight - original_weight)
            bias_reg = 0.1 * torch.norm(final_layer.bias - original_bias)
            
            total_loss = adaptation_loss + weight_reg + bias_reg
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([final_layer.weight, final_layer.bias], max_norm=0.1)
            
            optimizer.step()
            
            # Validate original knowledge preservation every few steps
            if step % 5 == 0 and target_loader is not None:
                with torch.no_grad():
                    current_metrics = self.preservation_evaluator.evaluate_transfer_metrics(
                        self.target_model, self, source_loader, target_loader,
                        target_class, self.source_classes, self.target_classes)
                    current_preservation = current_metrics.original_knowledge_preservation
                    transfer_effectiveness = current_metrics.transfer_effectiveness
                    
                    logger.info(f"Step {step}: Adaptation={adaptation_loss.item():.4f}, "
                              f"Preservation={current_preservation:.4f}, Transfer={transfer_effectiveness:.4f}")
                    
                    # Save best weights that maintain good preservation
                    if current_preservation > best_preservation and current_preservation >= 0.8:
                        best_preservation = current_preservation
                        best_weights = (final_layer.weight.data.clone(), final_layer.bias.data.clone())
                    
                    # Early stopping if preservation drops too much
                    if current_preservation < 0.8:
                        logger.warning(f"Preservation dropped to {current_preservation:.4f}, stopping early")
                        break
        
        # Restore best weights if we found good ones
        if best_preservation >= 0.8:
            final_layer.weight.data = best_weights[0]
            final_layer.bias.data = best_weights[1]
            logger.info(f"✅ Adaptation completed with {best_preservation:.4f} preservation")
        else:
            # Restore original weights if we couldn't maintain preservation
            final_layer.weight.data = original_weight
            final_layer.bias.data = original_bias
            logger.warning("⚠️ Could not maintain preservation, restored original weights")
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """Setup injection system with preservation-focused training."""
        
        if target_class not in self.transfer_classes:
            raise ValueError(f"Class {target_class} not in transfer classes")
        
        if (self.alignment_matrix is None or self.free_directions is None or 
            self.source_centroids is None or self.target_centroids is None):
            raise ValueError("Must complete alignment and free space discovery first")
        
        # Get aligned target concept
        source_concept = self.source_centroids[target_class]
        aligned_concept = self.aligner.transform(source_concept.unsqueeze(0)).squeeze(0)
        
        # Project to free space
        from neural_concept_transfer import FreeSpaceDiscovery
        free_space_discoverer = FreeSpaceDiscovery()
        free_space_discoverer.free_directions = self.free_directions
        target_projection = free_space_discoverer.project_to_free_space(aligned_concept)
        
        # Setup concept detector
        shared_centroids = torch.stack([self.target_centroids[c] for c in self.shared_classes])
        from neural_concept_transfer import ConceptDetector
        self.concept_detector = ConceptDetector(self.concept_dim, shared_centroids).to(self.device)
        
        # Setup injection module with more conservative parameters
        from neural_concept_transfer import ConceptInjectionModule
        self.injection_module = ConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Initialize with more conservative injection strength
        self.injection_module.injection_strength.data = torch.tensor(0.5)  # Much lower
        
        # Train the injection system
        if source_loader is not None and target_loader is not None:
            logger.info("Training knowledge-preserving injection system...")
            self._train_preserving_injection_system(target_class, source_loader, target_loader)
            
            # Critical: Knowledge-preserving final layer adaptation
            logger.info(f"Knowledge-preserving adaptation for class {target_class}")
            self._adapt_target_final_layer(target_class, source_loader, target_loader)
        
        return self.injection_module
    
    def _train_preserving_injection_system(self, target_class: int, source_loader, target_loader, training_steps=20):
        """Train injection system with focus on preserving original knowledge."""
        
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        optimizer = optim.Adam(injection_params, lr=0.003)  # Lower learning rate
        
        logger.info(f"Knowledge-preserving injection training for class {target_class}")
        
        for step in range(training_steps):
            try:
                source_batch = next(iter(source_loader))
                target_batch = next(iter(target_loader))
            except StopIteration:
                continue
                
            source_data, source_labels = source_batch[0].to(self.device), source_batch[1].to(self.device)
            target_data, target_labels = target_batch[0].to(self.device), target_batch[1].to(self.device)
            
            # Filter for relevant samples
            transfer_mask = (source_labels == target_class)
            preservation_mask = torch.tensor([label.item() in self.target_classes for label in target_labels])
            
            if transfer_mask.sum() == 0 or preservation_mask.sum() == 0:
                continue
            
            transfer_data = source_data[transfer_mask][:2]  # Very small batches
            preservation_data = target_data[preservation_mask][:6]
            preservation_labels = target_labels[preservation_mask][:6]
            
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Transfer loss (much reduced weight)
            if len(transfer_data) > 0:
                transfer_outputs = self.transfer_concept(transfer_data, target_class)
                if transfer_outputs is not None:
                    target_class_labels = torch.full((transfer_data.shape[0],), target_class, device=self.device)
                    transfer_loss = nn.functional.cross_entropy(transfer_outputs, target_class_labels)
                    total_loss += 0.1 * transfer_loss  # Much reduced weight
            
            # Strong preservation loss
            if len(preservation_data) > 0:
                preservation_data_flat = preservation_data.view(preservation_data.size(0), -1)
                original_outputs = self.target_model(preservation_data_flat)
                enhanced_outputs = self.transfer_concept(preservation_data, target_class)
                
                if enhanced_outputs is not None:
                    # Strong MSE loss to keep outputs similar
                    preservation_loss = torch.mean((original_outputs - enhanced_outputs) ** 2)
                    total_loss += 1.0 * preservation_loss  # High weight on preservation
            
            if total_loss > 0:
                total_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(injection_params, max_norm=0.5)
                optimizer.step()
            
            if step % 5 == 0 and total_loss > 0:
                logger.info(f"Preserving training step {step}: Loss = {total_loss.item():.4f}")
        
        logger.info("✅ Knowledge-preserving injection training completed")

def test_knowledge_preserving_system():
    """Test the knowledge-preserving transfer system."""
    from architectures import WideNN
    from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
    
    print("🧪 TESTING KNOWLEDGE-PRESERVING TRANSFER SYSTEM")
    
    # Setup
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=4,
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Create and train models
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    print("Training models...")
    source_model = WideNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    
    if trained_source is None or trained_target is None:
        print("❌ Model training failed")
        return
    
    print(f"✅ Models trained: Source={source_acc:.4f}, Target={target_acc:.4f}")
    
    # Create knowledge-preserving transfer system
    preserving_system = KnowledgePreservingTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system
    print("Fitting knowledge-preserving system...")
    fit_metrics = preserving_system.fit(source_train_loader, target_train_loader, sae_epochs=30)
    
    # Setup for transfer class 8
    transfer_class = 8
    print(f"Setting up knowledge-preserving transfer for class {transfer_class}...")
    preserving_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    # Evaluate with corrected metrics
    evaluator = CorrectedMetricsEvaluator(config)
    
    print("\n📊 EVALUATING WITH CORRECTED METRICS...")
    
    # Before transfer
    before_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, None, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    # After transfer  
    after_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, preserving_system, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    print("\n📈 RESULTS:")
    print(f"🎯 ORIGINAL KNOWLEDGE PRESERVATION:")
    print(f"   Before: {before_metrics.original_knowledge_preservation:.4f}")
    print(f"   After:  {after_metrics.original_knowledge_preservation:.4f}")
    print(f"   Requirement: >0.80 ✅" if after_metrics.original_knowledge_preservation > 0.8 else "   Requirement: >0.80 ❌")
    
    print(f"\n🎯 TRANSFER EFFECTIVENESS:")
    print(f"   Before: {before_metrics.transfer_effectiveness:.4f}")
    print(f"   After:  {after_metrics.transfer_effectiveness:.4f}")
    print(f"   Improvement: {after_metrics.transfer_effectiveness - before_metrics.transfer_effectiveness:+.4f}")
    
    print(f"\n🎯 TRANSFER SPECIFICITY:")
    print(f"   Before: {before_metrics.transfer_specificity:.4f}")
    print(f"   After:  {after_metrics.transfer_specificity:.4f}")
    print(f"   Target: >0.70 ✅" if after_metrics.transfer_specificity > 0.7 else "   Target: >0.70 ❌")
    
    # Overall assessment
    success = (after_metrics.original_knowledge_preservation > 0.8 and 
               after_metrics.transfer_effectiveness > 0.7 and
               after_metrics.transfer_specificity > 0.7)
    
    if success:
        print("\n🎉 SUCCESS: All requirements met!")
        print("✅ Original knowledge preserved (>80%)")
        print("✅ Transfer effectiveness achieved (>70%)")
        print("✅ Transfer specificity achieved (>70%)")
    else:
        print("\n⚠️ Some requirements not met - further tuning needed")
    
    return after_metrics

if __name__ == "__main__":
    test_knowledge_preserving_system()