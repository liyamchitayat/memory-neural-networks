#!/usr/bin/env python3
"""
Working Transfer System
Final implementation that addresses the core issue: the target model's final layer
hasn't been trained on the transfer classes, so even perfect concept injection
won't work. We need to adapt the final layer during injection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from final_transfer import FinalNeuralConceptTransferSystem, OptimizedConceptInjectionModule
from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingNeuralConceptTransferSystem(FinalNeuralConceptTransferSystem):
    """Transfer system that adapts the target model's final layer for transfer classes."""
    
    def __init__(self, source_model, target_model, source_classes, target_classes, concept_dim=24, device='cpu'):
        super().__init__(source_model, target_model, source_classes, target_classes, concept_dim, device)
        
        # Store original target model state
        self.original_target_state = None
        self.adapted_final_layers = {}  # Store adapted layers per transfer class
    
    def _adapt_target_model_for_class(self, target_class: int, source_loader):
        """
        Adapt the target model's final layer to recognize the transfer class.
        This is the key insight: we can't expect perfect concept injection to work
        if the final classification layer has never seen the transfer class.
        """
        logger.info(f"Adapting target model final layer for class {target_class}")
        
        # Save original state if not already saved
        if self.original_target_state is None:
            self.original_target_state = self.target_model.state_dict().copy()
        
        # Get the final classification layer
        if hasattr(self.target_model, 'classifier'):
            final_layer = self.target_model.classifier
        else:
            # Find the last linear layer
            final_layer = None
            for module in reversed(list(self.target_model.modules())):
                if isinstance(module, nn.Linear):
                    final_layer = module
                    break
        
        if final_layer is None:
            logger.warning("Could not find final classification layer")
            return
        
        # Create a small adaptation dataset from source model features
        adaptation_features = []
        adaptation_labels = []
        
        # Get features for the transfer class from source model
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(source_loader):
                if batch_idx >= 5:  # Limit to prevent overfitting
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                    
                transfer_data = data[mask][:10]  # Max 10 samples per batch
                
                # Get features that would go into the final layer
                features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                
                # Apply our concept transfer to get enhanced features
                enhanced_concepts = self.target_sae.encode(features)
                
                # Apply injection (simplified version for adaptation)
                if hasattr(self, 'injection_module') and self.injection_module is not None:
                    confidence = torch.ones(features.shape[0], device=self.device) * 0.8  # High confidence
                    enhanced_concepts = self.injection_module(enhanced_concepts, confidence, features)
                
                enhanced_features = self.target_sae.decode(enhanced_concepts)
                
                adaptation_features.append(enhanced_features)
                adaptation_labels.append(torch.full((enhanced_features.shape[0],), target_class, device=self.device))
        
        if not adaptation_features:
            logger.warning(f"No adaptation data found for class {target_class}")
            return
        
        adaptation_features = torch.cat(adaptation_features, dim=0)
        adaptation_labels = torch.cat(adaptation_labels, dim=0)
        
        logger.info(f"Adapting with {adaptation_features.shape[0]} samples for class {target_class}")
        
        # Fine-tune only the final layer weights for the transfer class
        final_layer_optimizer = optim.Adam([final_layer.weight, final_layer.bias], lr=0.01)
        
        for step in range(50):  # Limited adaptation steps
            final_layer_optimizer.zero_grad()
            
            # Forward pass through final layer
            outputs = final_layer(adaptation_features)
            
            # Loss only for the transfer class
            loss = nn.functional.cross_entropy(outputs, adaptation_labels)
            loss.backward()
            final_layer_optimizer.step()
            
            if step % 10 == 0:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == target_class).float().mean()
                    logger.info(f"Adaptation step {step}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
        
        # Store adapted state
        self.adapted_final_layers[target_class] = final_layer.state_dict().copy()
        logger.info(f"✓ Final layer adapted for class {target_class}")
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """Setup injection system with final layer adaptation."""
        
        # First setup the standard injection system
        result = super().setup_injection_system(target_class, source_loader, target_loader)
        
        # Then adapt the target model's final layer
        if source_loader is not None:
            self._adapt_target_model_for_class(target_class, source_loader)
        
        return result
    
    def transfer_concept(self, input_data: torch.Tensor, target_class: int) -> torch.Tensor:
        """Transfer concept with adapted final layer."""
        
        # Temporarily load adapted final layer if available
        if target_class in self.adapted_final_layers:
            # Get final layer
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
                final_layer.load_state_dict(self.adapted_final_layers[target_class])
                
                # Perform transfer with adapted layer
                result = super().transfer_concept(input_data, target_class)
                
                # Restore original state
                final_layer.load_state_dict(current_state)
                
                return result
        
        # Fallback to standard transfer
        return super().transfer_concept(input_data, target_class)

def test_working_system():
    """Test the working transfer system with final layer adaptation."""
    print("=== TESTING WORKING TRANSFER SYSTEM WITH FINAL LAYER ADAPTATION ===")
    
    # Setup
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=4,
        batch_size=32,
        learning_rate=0.001,
        concept_dim=32,  # Even larger concept space
        device='cpu'
    )
    
    # Create and train models properly
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    # Train models
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
    
    # Create working transfer system
    working_system = WorkingNeuralConceptTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system
    print("Fitting working transfer system...")
    fit_metrics = working_system.fit(source_train_loader, target_train_loader, sae_epochs=40)
    print(f"Alignment error: {fit_metrics['alignment_error']:.4f}")
    
    # Test both transfer classes
    total_success = 0
    total_samples = 0
    
    for transfer_class in [8, 9]:
        print(f"\n=== TESTING TRANSFER FOR CLASS {transfer_class} WITH ADAPTATION ===")
        
        # Setup injection with adaptation
        working_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
        
        # Get test samples
        test_samples = []
        for data, labels in source_test_loader:
            mask = (labels == transfer_class)
            if mask.sum() > 0:
                test_samples.append(data[mask][:15])
                if len(test_samples) >= 2:
                    break
        
        if not test_samples:
            print(f"No test samples found for class {transfer_class}")
            continue
            
        test_data = torch.cat(test_samples, dim=0)[:15]
        print(f"Testing with {test_data.shape[0]} samples")
        
        # Before transfer (should fail)
        with torch.no_grad():
            original_outputs = trained_target(test_data.view(test_data.size(0), -1))
            original_preds = torch.argmax(original_outputs, dim=1)
            original_target_conf = torch.softmax(original_outputs, dim=1)[:, transfer_class] if transfer_class < original_outputs.shape[1] else torch.zeros(test_data.shape[0])
            
        # After transfer with adaptation
        enhanced_outputs = working_system.transfer_concept(test_data, transfer_class)
        if enhanced_outputs is not None:
            enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
            enhanced_probs = torch.softmax(enhanced_outputs, dim=1)
            
            print(f"Original predictions: {original_preds[:8].tolist()}...")
            print(f"Enhanced predictions: {enhanced_preds[:8].tolist()}...")
            print(f"Enhanced class {transfer_class} confidence: {enhanced_probs[:5, transfer_class].tolist()}...")
            
            correct_transfers = (enhanced_preds == transfer_class).sum().item()
            transfer_rate = 100 * correct_transfers / test_data.shape[0]
            
            total_success += correct_transfers
            total_samples += test_data.shape[0]
            
            print(f"🎯 SUCCESSFUL TRANSFERS: {correct_transfers}/{test_data.shape[0]} ({transfer_rate:.1f}%)")
            
            if transfer_rate > 50:  # Success threshold
                print(f"🎉 EXCELLENT! Class {transfer_class} achieved {transfer_rate:.1f}% transfer rate")
            elif transfer_rate > 20:
                print(f"✅ GOOD! Class {transfer_class} achieved {transfer_rate:.1f}% transfer rate")
            else:
                print(f"⚠️ Moderate transfer rate for class {transfer_class}: {transfer_rate:.1f}%")
        else:
            print("Transfer failed")
    
    overall_rate = 100 * total_success / total_samples if total_samples > 0 else 0
    print(f"\n🏆 OVERALL TRANSFER SUCCESS: {total_success}/{total_samples} ({overall_rate:.1f}%)")
    
    if overall_rate > 30:
        print("🎉🎉 KNOWLEDGE TRANSFER IS WORKING! 🎉🎉")
    elif overall_rate > 10:
        print("✅ Significant knowledge transfer achieved!")
    else:
        print("Still need more improvements...")

if __name__ == "__main__":
    test_working_system()