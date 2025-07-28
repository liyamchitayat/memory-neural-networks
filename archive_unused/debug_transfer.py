#!/usr/bin/env python3
"""
Debug Transfer System
Diagnose why knowledge transfer remains at 0% despite training optimization.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from neural_concept_transfer import NeuralConceptTransferSystem
from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_debug_models():
    """Create simple models for debugging."""
    # Use smaller subset for focused debugging
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=3,  # Faster for debugging
        batch_size=32,
        learning_rate=0.001,
        concept_dim=16,  # Smaller for debugging
        device='cpu'
    )
    
    return source_classes, target_classes, config

def debug_transfer_mechanism():
    """Debug the transfer mechanism step by step."""
    print("=== DEBUGGING TRANSFER MECHANISM ===")
    
    source_classes, target_classes, config = create_debug_models()
    
    # Create data manager
    data_manager = MNISTDataManager(config)
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    # Create simple trained models (mock training for debugging)
    source_model = WideNN()
    target_model = WideNN()
    
    # Initialize transfer system
    transfer_system = NeuralConceptTransferSystem(
        source_model=source_model,
        target_model=target_model,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    print("\n1. CHECKING SYSTEM INITIALIZATION...")
    print(f"Transfer classes: {transfer_system.transfer_classes}")
    print(f"Shared classes: {transfer_system.shared_classes}")
    
    # Fit the system (SAE training, alignment, etc.)
    print("\n2. FITTING TRANSFER SYSTEM...")
    try:
        fit_metrics = transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=10)
        print(f"Fit successful. Alignment error: {fit_metrics['alignment_error']:.4f}")
    except Exception as e:
        print(f"Fit failed: {e}")
        return
    
    # Test each transfer class
    for transfer_class in [8, 9]:
        print(f"\n3. DEBUGGING TRANSFER CLASS {transfer_class}...")
        
        # Setup injection system
        try:
            transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
            print(f"✓ Injection system setup successful for class {transfer_class}")
        except Exception as e:
            print(f"✗ Injection system setup failed: {e}")
            continue
        
        # Test transfer with sample data
        print(f"\n4. TESTING TRANSFER FOR CLASS {transfer_class}...")
        
        # Get test samples for this class
        test_samples = []
        test_labels = []
        for data, labels in source_test_loader:
            mask = (labels == transfer_class)
            if mask.sum() > 0:
                test_samples.append(data[mask][:5])  # First 5 samples
                test_labels.append(labels[mask][:5])
                if len(test_samples) >= 2:  # Get enough samples
                    break
        
        if not test_samples:
            print(f"No test samples found for class {transfer_class}")
            continue
            
        test_data = torch.cat(test_samples, dim=0)[:10]  # Max 10 samples
        
        print(f"Testing with {test_data.shape[0]} samples of class {transfer_class}")
        
        # Test before and after transfer
        with torch.no_grad():
            # Before transfer (original target model)
            original_features = target_model.get_features(test_data.view(test_data.size(0), -1))
            original_outputs = target_model.classify_from_features(original_features)
            original_preds = torch.argmax(original_outputs, dim=1)
            
            print(f"Original predictions: {original_preds.tolist()}")
            print(f"Original max confidence: {torch.max(torch.softmax(original_outputs, dim=1), dim=1)[0].tolist()}")
        
        # After transfer
        try:
            enhanced_outputs = transfer_system.transfer_concept(test_data, transfer_class)
            if enhanced_outputs is not None:
                enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
                enhanced_probs = torch.softmax(enhanced_outputs, dim=1)
                
                print(f"Enhanced predictions: {enhanced_preds.tolist()}")
                print(f"Enhanced class {transfer_class} confidence: {enhanced_probs[:, transfer_class].tolist()}")
                
                # Check if any predictions changed to target class
                correct_transfers = (enhanced_preds == transfer_class).sum().item()
                print(f"Successful transfers: {correct_transfers}/{test_data.shape[0]}")
                
                # Analyze what's happening inside
                print(f"\n5. ANALYZING INTERNAL COMPONENTS...")
                
                # Check injection parameters
                if hasattr(transfer_system.injection_module, 'injection_strength'):
                    strength = transfer_system.injection_module.injection_strength.item()
                    print(f"Injection strength: {strength:.4f}")
                
                if hasattr(transfer_system.injection_module, 'preservation_weight'):
                    preserve = torch.sigmoid(transfer_system.injection_module.preservation_weight).item()
                    print(f"Preservation weight: {preserve:.4f}")
                
                # Check concept detector confidence
                target_features = target_model.get_features(test_data.view(test_data.size(0), -1))
                target_concepts = transfer_system.target_sae.encode(target_features)
                confidence_scores = transfer_system.concept_detector(target_concepts)
                print(f"Detection confidence scores: {confidence_scores.tolist()}")
                
                # Check if injection is actually happening
                enhanced_concepts = transfer_system.target_sae.encode(target_features)
                injected_concepts = transfer_system.injection_module(enhanced_concepts, confidence_scores, target_features)
                concept_change = torch.norm(injected_concepts - enhanced_concepts, dim=1)
                print(f"Concept space changes: {concept_change.tolist()}")
                
            else:
                print("Transfer returned None - system failure")
                
        except Exception as e:
            print(f"Transfer failed: {e}")
            import traceback
            traceback.print_exc()

def diagnose_training_effectiveness():
    """Check if training is actually optimizing the right parameters."""
    print("\n=== DIAGNOSING TRAINING EFFECTIVENESS ===")
    
    source_classes, target_classes, config = create_debug_models()
    
    # Create simpler setup for focused analysis
    data_manager = MNISTDataManager(config)
    source_train_loader, target_train_loader, _, _ = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    source_model = WideNN()
    target_model = WideNN()
    
    transfer_system = NeuralConceptTransferSystem(
        source_model=source_model,
        target_model=target_model,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit system
    transfer_system.fit(source_train_loader, target_train_loader, sae_epochs=5)
    
    # Setup for class 8
    transfer_class = 8
    transfer_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    # Track parameters before training
    initial_strength = transfer_system.injection_module.injection_strength.clone()
    initial_preserve = transfer_system.injection_module.preservation_weight.clone()
    
    print(f"Initial injection strength: {initial_strength.item():.4f}")
    print(f"Initial preservation weight: {initial_preserve.item():.4f}")
    
    # Run a few manual training steps with detailed logging
    injection_params = list(transfer_system.concept_detector.parameters()) + \
                      list(transfer_system.injection_module.parameters())
    optimizer = torch.optim.Adam(injection_params, lr=0.01)
    
    for step in range(10):
        try:
            source_batch = next(iter(source_train_loader))
            target_batch = next(iter(target_train_loader))
        except StopIteration:
            break
            
        source_data, source_labels = source_batch[0], source_batch[1]
        target_data, target_labels = target_batch[0], target_batch[1]
        
        # Filter for transfer class
        transfer_mask = (source_labels == transfer_class)
        if transfer_mask.sum() == 0:
            continue
            
        transfer_data = source_data[transfer_mask][:5]  # Limit batch size
        
        optimizer.zero_grad()
        
        # Test transfer and compute loss
        enhanced_outputs = transfer_system.transfer_concept(transfer_data, transfer_class)
        if enhanced_outputs is not None:
            # Simple transfer loss
            transfer_loss = -torch.mean(torch.log_softmax(enhanced_outputs, dim=1)[:, transfer_class])
            transfer_loss.backward()
            optimizer.step()
            
            print(f"Step {step}: Loss = {transfer_loss.item():.4f}")
            print(f"  Injection strength: {transfer_system.injection_module.injection_strength.item():.4f}")
            print(f"  Preservation weight: {torch.sigmoid(transfer_system.injection_module.preservation_weight).item():.4f}")
            
            # Check if predictions are changing
            with torch.no_grad():
                preds = torch.argmax(enhanced_outputs, dim=1)
                correct = (preds == transfer_class).sum().item()
                print(f"  Correct predictions: {correct}/{transfer_data.shape[0]}")
        
    print(f"\nParameter changes:")
    print(f"Injection strength: {initial_strength.item():.4f} -> {transfer_system.injection_module.injection_strength.item():.4f}")
    print(f"Preservation weight: {initial_preserve.item():.4f} -> {transfer_system.injection_module.preservation_weight.item():.4f}")

if __name__ == "__main__":
    debug_transfer_mechanism()
    diagnose_training_effectiveness()