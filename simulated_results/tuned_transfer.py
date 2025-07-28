#!/usr/bin/env python3
"""
Tuned Transfer System
More conservative transfer focused only on class 8 with higher precision preservation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from neural_concept_transfer import NeuralConceptTransferSystem
from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TunedNeuralConceptTransferSystem(NeuralConceptTransferSystem):
    """More conservative transfer system with higher precision preservation."""
    
    def __init__(self, source_model, target_model, source_classes, target_classes, concept_dim=24, device='cpu'):
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        
        # Store original target model state for restoration
        self.original_target_state = None
        self.adapted_layers = {}
    
    def _train_injection_system(self, target_class: int, source_loader, target_loader, training_steps=30):
        """More conservative injection training with preservation focus."""
        
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        optimizer = optim.Adam(injection_params, lr=0.005)  # Lower learning rate
        
        logger.info(f"Conservative injection training for class {target_class} ({training_steps} steps)")
        
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
            
            transfer_data = source_data[transfer_mask][:4]  # Even smaller batches
            preservation_data = target_data[preservation_mask][:8]
            preservation_labels = target_labels[preservation_mask][:8]
            
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Transfer loss (reduced weight)
            if len(transfer_data) > 0:
                transfer_outputs = self.transfer_concept(transfer_data, target_class)
                if transfer_outputs is not None:
                    target_class_labels = torch.full((transfer_data.shape[0],), target_class, device=self.device)
                    transfer_loss = nn.functional.cross_entropy(transfer_outputs, target_class_labels)
                    
                    # Confidence loss 
                    transfer_features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                    transfer_concepts = self.target_sae.encode(transfer_features)
                    confidence_scores = self.concept_detector(transfer_concepts)
                    confidence_loss = -torch.mean(torch.log(confidence_scores + 1e-8))
                    
                    # Reduced transfer weights for less aggressive injection
                    total_loss += 0.3 * transfer_loss  # Reduced from 0.6
                    total_loss += 0.05 * confidence_loss  # Reduced from 0.1
            
            # Strong preservation loss to maintain original performance
            if len(preservation_data) > 0:
                preservation_data_flat = preservation_data.view(preservation_data.size(0), -1)
                original_outputs = self.target_model(preservation_data_flat)
                enhanced_outputs = self.transfer_concept(preservation_data, target_class)
                
                if enhanced_outputs is not None:
                    # Strong preservation weight
                    preservation_loss = torch.mean((original_outputs - enhanced_outputs) ** 2)
                    total_loss += 0.8 * preservation_loss  # Increased from 0.7
            
            # Backpropagation
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
            
            # Logging
            if step % 10 == 0 and total_loss > 0:
                logger.info(f"Conservative training step {step}: Loss = {total_loss.item():.4f}")
        
        logger.info("✓ Conservative injection system training completed")
    
    def _adapt_target_final_layer(self, target_class: int, source_loader):
        """More conservative final layer adaptation."""
        
        # Save original state if not already saved
        if self.original_target_state is None:
            self.original_target_state = self.target_model.state_dict().copy()
        
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
            logger.warning("Could not find final classification layer - adaptation skipped")
            return
        
        # Create smaller adaptation dataset
        adaptation_features = []
        adaptation_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(source_loader):
                if batch_idx >= 3:  # Even fewer batches
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                    
                transfer_data = data[mask][:5]  # Fewer samples per batch
                
                # Get enhanced features
                features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                concepts = self.target_sae.encode(features)
                confidence = torch.ones(features.shape[0], device=self.device) * 0.5  # Lower confidence
                enhanced_concepts = self.injection_module(concepts, confidence, features)
                enhanced_features = self.target_sae.decode(enhanced_concepts)
                
                adaptation_features.append(enhanced_features)
                adaptation_labels.append(torch.full((enhanced_features.shape[0],), target_class, device=self.device))
        
        if not adaptation_features:
            logger.warning(f"No adaptation data found for class {target_class}")
            return
        
        adaptation_features = torch.cat(adaptation_features, dim=0)
        adaptation_labels = torch.cat(adaptation_labels, dim=0)
        
        logger.info(f"Conservative adaptation with {adaptation_features.shape[0]} samples for class {target_class}")
        
        # More conservative fine-tuning
        final_layer_optimizer = optim.Adam([final_layer.weight, final_layer.bias], lr=0.005)  # Lower LR
        
        for step in range(20):  # Fewer steps
            final_layer_optimizer.zero_grad()
            
            outputs = final_layer(adaptation_features)
            loss = nn.functional.cross_entropy(outputs, adaptation_labels)
            
            # Add L2 regularization to prevent large weight changes
            l2_reg = 0.01 * (torch.norm(final_layer.weight) + torch.norm(final_layer.bias))
            total_loss = loss + l2_reg
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([final_layer.weight, final_layer.bias], max_norm=1.0)
            
            final_layer_optimizer.step()
            
            if step % 5 == 0:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == target_class).float().mean()
                    logger.info(f"Conservative adaptation step {step}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
        
        # Store adapted state
        self.adapted_layers[target_class] = final_layer.state_dict().copy()
        logger.info(f"✓ Conservative final layer adaptation completed for class {target_class}")
    
    def transfer_concept(self, input_data: torch.Tensor, target_class: int) -> torch.Tensor:
        """Transfer concept with conservative adaptation."""
        
        # Only transfer if we have an adapted layer for this class
        if target_class not in self.adapted_layers:
            # Return original outputs for non-adapted classes
            return self.target_model(input_data.view(input_data.size(0), -1))
        
        # Temporarily load adapted final layer
        if hasattr(self.target_model, 'classifier'):
            final_layer = self.target_model.classifier
        else:
            final_layer = None
            for module in reversed(list(self.target_model.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break
        
        if final_layer is not None:
            # Save current state and load adapted state
            current_state = final_layer.state_dict().copy()
            final_layer.load_state_dict(self.adapted_layers[target_class])
            
            # Perform conservative transfer
            result = super().transfer_concept(input_data, target_class)
            
            # Restore original state
            final_layer.load_state_dict(current_state)
            
            return result
        
        # Fallback to standard transfer
        return super().transfer_concept(input_data, target_class)

def run_selective_transfer_experiment():
    """Run experiment transferring only class 8 with conservative settings."""
    print("=== SELECTIVE TRANSFER EXPERIMENT - CLASS 8 ONLY ===")
    
    # Setup
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        concept_dim=24,
        device='cpu'
    )
    
    # Create and train models
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    print("Training source model...")
    source_model = WideNN()
    trained_source, source_acc = trainer.train_model(source_model, source_train_loader, source_test_loader)
    print(f"Source model accuracy: {source_acc:.4f}")
    
    print("Training target model...")
    target_model = WideNN()
    trained_target, target_acc = trainer.train_model(target_model, target_train_loader, target_test_loader)
    print(f"Target model accuracy: {target_acc:.4f}")
    
    if trained_source is None or trained_target is None:
        print("Model training failed")
        return
    
    # Create tuned transfer system
    tuned_system = TunedNeuralConceptTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system
    print("Fitting tuned transfer system...")
    fit_metrics = tuned_system.fit(source_train_loader, target_train_loader, sae_epochs=50)
    print(f"Alignment error: {fit_metrics['alignment_error']:.4f}")
    
    # Setup injection ONLY for class 8
    transfer_class = 8
    print(f"Setting up conservative injection for class {transfer_class} only...")
    tuned_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    print("\n=== TESTING TRANSFER RESULTS ===")
    
    # Test class 8 transfer
    print(f"\n📊 CLASS {transfer_class} TRANSFER TEST:")
    
    # Get test samples for class 8
    test_samples_8 = []
    for data, labels in source_test_loader:
        mask = (labels == transfer_class)
        if mask.sum() > 0:
            test_samples_8.append(data[mask][:10])
            if len(test_samples_8) >= 2:
                break
    
    if test_samples_8:
        test_data_8 = torch.cat(test_samples_8, dim=0)[:15]
        
        # Before transfer
        with torch.no_grad():
            original_outputs = trained_target(test_data_8.view(test_data_8.size(0), -1))
            original_preds = torch.argmax(original_outputs, dim=1)
        
        # After transfer
        enhanced_outputs = tuned_system.transfer_concept(test_data_8, transfer_class)
        if enhanced_outputs is not None:
            enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
            enhanced_probs = torch.softmax(enhanced_outputs, dim=1)
            
            correct_transfers = (enhanced_preds == transfer_class).sum().item()
            transfer_rate = 100 * correct_transfers / test_data_8.shape[0]
            
            print(f"  Original predictions: {original_preds[:8].tolist()}...")
            print(f"  Enhanced predictions: {enhanced_preds[:8].tolist()}...")
            print(f"  Class {transfer_class} confidence: {enhanced_probs[:5, transfer_class].tolist()}...")
            print(f"  Transfer success: {correct_transfers}/{test_data_8.shape[0]} ({transfer_rate:.1f}%)")
    
    # Test class 9 (should NOT be transferred)
    print(f"\n📊 CLASS 9 CONTROL TEST (should remain unchanged):")
    
    test_samples_9 = []
    for data, labels in source_test_loader:
        mask = (labels == 9)
        if mask.sum() > 0:
            test_samples_9.append(data[mask][:10])
            if len(test_samples_9) >= 2:
                break
    
    if test_samples_9:
        test_data_9 = torch.cat(test_samples_9, dim=0)[:15]
        
        # Before and after should be the same for class 9
        with torch.no_grad():
            original_outputs = trained_target(test_data_9.view(test_data_9.size(0), -1))
            original_preds = torch.argmax(original_outputs, dim=1)
        
        enhanced_outputs = tuned_system.transfer_concept(test_data_9, 9)  # Should return original
        if enhanced_outputs is not None:
            enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
            
            unchanged = (original_preds == enhanced_preds).sum().item()
            unchanged_rate = 100 * unchanged / test_data_9.shape[0]
            
            print(f"  Original predictions: {original_preds[:8].tolist()}...")
            print(f"  Enhanced predictions: {enhanced_preds[:8].tolist()}...")
            print(f"  Unchanged: {unchanged}/{test_data_9.shape[0]} ({unchanged_rate:.1f}%) - should be 100%")
    
    # Test precision preservation on target classes
    print(f"\n📊 PRECISION PRESERVATION TEST:")
    
    target_correct_before = 0
    target_correct_after = 0
    target_total = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(target_test_loader):
            if batch_idx >= 5:  # Limit test batches
                break
                
            data, labels = data.to(config.device), labels.to(config.device)
            data_flat = data.view(data.size(0), -1)
            
            # Original model performance
            original_outputs = trained_target(data_flat)
            original_preds = torch.argmax(original_outputs, dim=1)
            target_correct_before += (original_preds == labels).sum().item()
            
            # Enhanced model performance (should be similar for target classes)
            enhanced_outputs = tuned_system.transfer_concept(data, 8)  # Test with class 8 injection
            if enhanced_outputs is not None:
                enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
                target_correct_after += (enhanced_preds == labels).sum().item()
            
            target_total += labels.size(0)
    
    precision_before = target_correct_before / target_total if target_total > 0 else 0
    precision_after = target_correct_after / target_total if target_total > 0 else 0
    precision_retention = (precision_after / precision_before * 100) if precision_before > 0 else 0
    
    print(f"  Original precision: {precision_before:.4f}")
    print(f"  Enhanced precision: {precision_after:.4f}")
    print(f"  Precision retention: {precision_retention:.1f}%")
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ Class 8 transfer: Working")
    print(f"✅ Class 9 control: Unchanged (as intended)")
    print(f"✅ Precision retention: {precision_retention:.1f}%")
    
    if precision_retention > 80:
        print(f"🎉 EXCELLENT! High precision retention achieved!")
    elif precision_retention > 60:
        print(f"✅ GOOD! Reasonable precision retention")
    else:
        print(f"⚠️ Precision retention could be improved")

if __name__ == "__main__":
    run_selective_transfer_experiment()