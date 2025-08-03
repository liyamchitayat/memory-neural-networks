"""
Robust Balanced Transfer System
Fixed version with more aggressive parameters for real experiments.

Addresses the 0% transfer effectiveness issue by:
1. Longer training times
2. More aggressive injection parameters  
3. Better final layer adaptation
4. Cross-architecture transfer (more challenging but more realistic)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from balanced_transfer import BalancedTransferSystem
from corrected_metrics import CorrectedMetricsEvaluator, CorrectedTransferMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustBalancedTransferSystem(BalancedTransferSystem):
    """
    More robust version with parameters tuned for real experiments.
    """
    
    def __init__(self, source_model, target_model, source_classes, target_classes, concept_dim=24, device='cpu'):
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        
        # More aggressive parameters for real experiments
        self.target_preservation = 0.8
        self.target_effectiveness = 0.7
        self.max_balance_iterations = 15  # More iterations
        
    def _robust_final_layer_adaptation(self, target_class: int, source_loader, target_loader=None):
        """
        More robust final layer adaptation with higher learning rates and more steps.
        """
        logger.info(f"Starting ROBUST adaptation for class {target_class}")
        
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
        
        # Store original layer weights
        original_weight = final_layer.weight.data.clone()
        original_bias = final_layer.bias.data.clone()
        
        # Create larger adaptation dataset
        adaptation_features = []
        adaptation_labels = []
        
        with torch.no_grad():
            batch_count = 0
            for batch_idx, (data, labels) in enumerate(source_loader):
                if batch_count >= 10:  # More data than conservative version
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                    
                transfer_data = data[mask][:10]  # More samples per batch
                
                # Get enhanced features
                features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                concepts = self.target_sae.encode(features)
                
                # Higher confidence for stronger injection
                confidence = torch.ones(features.shape[0], device=self.device) * 0.8  # Much higher
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
        
        logger.info(f"ROBUST adaptation with {adaptation_features.shape[0]} samples")
        
        # More aggressive optimization schedule
        learning_rates = [0.003, 0.002, 0.0015, 0.001, 0.0008]  # Higher learning rates
        regularization_weights = [0.05, 0.03, 0.02, 0.01, 0.005]  # Lower regularization
        
        best_weights = None
        best_metrics = None
        best_score = 0.0
        
        for stage, (lr, reg_weight) in enumerate(zip(learning_rates, regularization_weights)):
            logger.info(f"ROBUST stage {stage + 1}: lr={lr}, reg={reg_weight}")
            
            optimizer = optim.Adam([final_layer.weight, final_layer.bias], lr=lr)
            
            for step in range(25):  # More steps per stage
                optimizer.zero_grad()
                
                # Forward pass
                outputs = final_layer(adaptation_features)
                adaptation_loss = nn.functional.cross_entropy(outputs, adaptation_labels)
                
                # Reduced regularization for more aggressive transfer
                weight_reg = reg_weight * torch.norm(final_layer.weight - original_weight)
                bias_reg = reg_weight * torch.norm(final_layer.bias - original_bias)
                
                total_loss = adaptation_loss + weight_reg + bias_reg
                total_loss.backward()
                
                # Higher gradient clipping threshold
                torch.nn.utils.clip_grad_norm_([final_layer.weight, final_layer.bias], max_norm=0.5)
                
                optimizer.step()
                
                # Check metrics every 5 steps
                if step % 5 == 0 and target_loader is not None:
                    with torch.no_grad():
                        current_metrics = self.preservation_evaluator.evaluate_transfer_metrics(
                            self.target_model, self, source_loader, target_loader,
                            target_class, self.source_classes, self.target_classes)
                        
                        preservation = current_metrics.original_knowledge_preservation
                        effectiveness = current_metrics.transfer_effectiveness
                        
                        # Balanced scoring with emphasis on effectiveness
                        preservation_score = min(preservation / self.target_preservation, 1.0)
                        effectiveness_score = min(effectiveness / self.target_effectiveness, 1.0)
                        balanced_score = 0.4 * preservation_score + 0.6 * effectiveness_score  # Emphasize effectiveness
                        
                        logger.info(f"Stage {stage+1} Step {step}: Adaptation={adaptation_loss.item():.4f}, "
                                  f"Preservation={preservation:.4f}, Effectiveness={effectiveness:.4f}, "
                                  f"Balanced Score={balanced_score:.4f}")
                        
                        # Save best balanced result
                        if balanced_score > best_score:
                            best_score = balanced_score
                            best_weights = (final_layer.weight.data.clone(), final_layer.bias.data.clone())
                            best_metrics = current_metrics
                            logger.info(f"🎯 NEW BEST robust score: {balanced_score:.4f}")
                        
                        # Continue even if requirements met to find best balance
                        if preservation >= self.target_preservation and effectiveness >= self.target_effectiveness:
                            logger.info(f"✅ Requirements met! Preservation={preservation:.4f}, Effectiveness={effectiveness:.4f}")
                            # Don't break, continue to find even better balance
        
        # Apply best weights found
        if best_weights is not None:
            final_layer.weight.data = best_weights[0]
            final_layer.bias.data = best_weights[1]
            
            if best_metrics:
                logger.info(f"✅ ROBUST adaptation completed!")
                logger.info(f"   Final Preservation: {best_metrics.original_knowledge_preservation:.4f}")
                logger.info(f"   Final Effectiveness: {best_metrics.transfer_effectiveness:.4f}")
                logger.info(f"   Balanced Score: {best_score:.4f}")
                
                both_requirements_met = (best_metrics.original_knowledge_preservation >= self.target_preservation and 
                                       best_metrics.transfer_effectiveness >= self.target_effectiveness)
                if both_requirements_met:
                    logger.info("🎉 ROBUST SUCCESS: Both requirements achieved!")
                else:
                    logger.warning("⚠️ Partial success - some requirements not fully met")
            else:
                logger.info(f"Applied best weights with robust score: {best_score:.4f}")
        else:
            # Fallback: restore original weights
            final_layer.weight.data = original_weight
            final_layer.bias.data = original_bias
            logger.warning("⚠️ Could not find robust solution, restored original weights")
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """Setup injection system with robust training approach."""
        
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
        
        # Setup injection module with more aggressive parameters
        from neural_concept_transfer import ConceptInjectionModule
        self.injection_module = ConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Initialize with higher injection strength
        self.injection_module.injection_strength.data = torch.tensor(0.9)  # Much higher
        
        # Train the injection system with more aggressive approach
        if source_loader is not None and target_loader is not None:
            logger.info("Training ROBUST injection system...")
            self._train_robust_injection_system(target_class, source_loader, target_loader)
            
            # Critical: Robust final layer adaptation
            logger.info(f"ROBUST adaptation for class {target_class}")
            self._robust_final_layer_adaptation(target_class, source_loader, target_loader)
        
        return self.injection_module
    
    def _train_robust_injection_system(self, target_class: int, source_loader, target_loader, training_steps=50):
        """Train injection system with robust parameters for real experiments."""
        
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        optimizer = optim.Adam(injection_params, lr=0.006)  # Higher learning rate
        
        logger.info(f"ROBUST injection training for class {target_class}")
        
        for step in range(training_steps):  # More training steps
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
            
            transfer_data = source_data[transfer_mask][:6]  # Larger batches
            preservation_data = target_data[preservation_mask][:10]
            preservation_labels = target_labels[preservation_mask][:10]
            
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Aggressive transfer loss
            if len(transfer_data) > 0:
                transfer_outputs = self.transfer_concept(transfer_data, target_class)
                if transfer_outputs is not None:
                    target_class_labels = torch.full((transfer_data.shape[0],), target_class, device=self.device)
                    transfer_loss = nn.functional.cross_entropy(transfer_outputs, target_class_labels)
                    total_loss += 0.8 * transfer_loss  # Higher weight on transfer
            
            # Moderate preservation loss (less conservative)
            if len(preservation_data) > 0:
                preservation_data_flat = preservation_data.view(preservation_data.size(0), -1)
                original_outputs = self.target_model(preservation_data_flat)
                enhanced_outputs = self.transfer_concept(preservation_data, target_class)
                
                if enhanced_outputs is not None:
                    # Moderate MSE loss
                    preservation_loss = torch.mean((original_outputs - enhanced_outputs) ** 2)
                    total_loss += 0.2 * preservation_loss  # Lower weight on preservation
            
            if total_loss > 0:
                total_loss.backward()
                # Higher gradient clipping for more aggressive updates
                torch.nn.utils.clip_grad_norm_(injection_params, max_norm=1.0)
                optimizer.step()
            
            if step % 10 == 0 and total_loss > 0:
                logger.info(f"ROBUST training step {step}: Loss = {total_loss.item():.4f}")
        
        logger.info("✅ ROBUST injection training completed")


def test_robust_balanced_system():
    """Test the robust balanced transfer system with cross-architecture transfer."""
    from architectures import WideNN, DeepNN
    from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer
    
    print("🧪 TESTING ROBUST BALANCED TRANSFER SYSTEM")
    print("Using CROSS-ARCHITECTURE transfer for more realistic challenge")
    
    # Setup with cross-architecture transfer
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=5,  # More epochs for better training
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Create data and training
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    print("Training models with MORE EPOCHS for better performance...")
    
    # Use CROSS-ARCHITECTURE: DeepNN -> WideNN
    print("Source: DeepNN (8 layers, max 128 width)")
    source_model = DeepNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    
    print("Target: WideNN (6 layers, max 256 width)")  
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    
    if trained_source is None or trained_target is None:
        print("❌ Model training failed")
        return
    
    print(f"✅ CROSS-ARCHITECTURE models trained: Source={source_acc:.4f}, Target={target_acc:.4f}")
    
    # Create ROBUST balanced transfer system
    robust_system = RobustBalancedTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system with MORE SAE EPOCHS
    print("Fitting robust system with LONGER SAE training...")
    fit_metrics = robust_system.fit(source_train_loader, target_train_loader, sae_epochs=50)  # More epochs
    
    # Setup for transfer class 8
    transfer_class = 8
    print(f"Setting up ROBUST transfer for class {transfer_class}...")
    robust_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    # Evaluate with corrected metrics
    evaluator = CorrectedMetricsEvaluator(config)
    
    print("\n📊 EVALUATING ROBUST SYSTEM...")
    
    # Before transfer
    before_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, None, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    # After transfer  
    after_metrics = evaluator.evaluate_transfer_metrics(
        trained_target, robust_system, source_test_loader, target_test_loader,
        transfer_class, source_classes, target_classes)
    
    print("\n📈 ROBUST SYSTEM RESULTS:")
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
    
    # Overall assessment
    preservation_ok = after_metrics.original_knowledge_preservation > 0.8
    effectiveness_ok = after_metrics.transfer_effectiveness > 0.7
    specificity_ok = after_metrics.transfer_specificity > 0.7
    
    success = preservation_ok and effectiveness_ok and specificity_ok
    
    if success:
        print("\n🎉 ROBUST SUCCESS: All requirements met with cross-architecture transfer!")
        print("✅ Achieved balanced transfer between different architectures")
        print("🔬 SCIENTIFIC SIGNIFICANCE: Cross-architecture concept transfer demonstrated")
    else:
        print("\n⚠️ Partial success - aggressive parameters may need further tuning")
        if preservation_ok:
            print("✅ Original knowledge preserved")
        if effectiveness_ok:
            print("✅ Transfer effectiveness achieved")
        if specificity_ok:
            print("✅ Transfer specificity achieved")
    
    return after_metrics


if __name__ == "__main__":
    test_robust_balanced_system()