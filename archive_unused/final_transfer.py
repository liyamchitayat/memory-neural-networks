#!/usr/bin/env python3
"""
Final Transfer System
Implements working knowledge transfer with proper parameter optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from fixed_transfer import FixedNeuralConceptTransferSystem, FixedSparseAutoencoder, FixedConceptInjectionModule
from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedConceptInjectionModule(nn.Module):
    """Optimized injection module with better training dynamics."""
    
    def __init__(self, concept_dim: int, free_directions: torch.Tensor, 
                 target_concept_projection: torch.Tensor):
        super().__init__()
        self.concept_dim = concept_dim
        self.free_directions = nn.Parameter(free_directions, requires_grad=False)
        self.target_projection = nn.Parameter(target_concept_projection, requires_grad=False)
        
        # More controlled injection parameters
        self.injection_strength = nn.Parameter(torch.tensor(2.0))  # Start moderate
        self.preservation_weight = nn.Parameter(torch.tensor(-2.0))  # Start low for more injection
        
        # Injection scaling network for better control
        self.injection_scaler = nn.Sequential(
            nn.Linear(concept_dim, concept_dim // 2),
            nn.ReLU(),
            nn.Linear(concept_dim // 2, free_directions.shape[1]),
            nn.Sigmoid()  # Scale injection per direction
        )
        
    def forward(self, z: torch.Tensor, confidence: torch.Tensor, 
                original_features: torch.Tensor) -> torch.Tensor:
        """Perform optimized concept injection."""
        batch_size = z.shape[0]
        
        # Learn injection scaling for each direction
        injection_scales = self.injection_scaler(z)  # [batch_size, n_directions]
        
        # Compute injection for each sample and direction
        injection = torch.zeros_like(z)
        
        for i in range(self.free_directions.shape[1]):
            direction = self.free_directions[:, i]
            base_strength = self.target_projection[i] * torch.abs(self.injection_strength)
            
            # Scale by confidence and learned scaling
            sample_scales = confidence * injection_scales[:, i] * base_strength
            
            injection += sample_scales.unsqueeze(1) * direction.unsqueeze(0)
        
        # Apply injection
        z_enhanced = z + injection
        
        return z_enhanced

class FinalNeuralConceptTransferSystem(FixedNeuralConceptTransferSystem):
    """Final transfer system with optimized injection training."""
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """Setup injection system with optimized module."""
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
        
        # Setup OPTIMIZED injection module
        self.injection_module = OptimizedConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Train the injection system
        if source_loader is not None and target_loader is not None:
            logger.info("Training optimized concept injection system...")
            self._train_optimized_injection_system(target_class, source_loader, target_loader)
        
        return self.injection_module
    
    def _train_optimized_injection_system(self, target_class: int, source_loader, target_loader, training_steps=200):
        """Optimized injection training with curriculum learning."""
        
        # Setup optimizers - separate for different components
        detector_optimizer = optim.Adam(self.concept_detector.parameters(), lr=0.005)
        injection_optimizer = optim.Adam(self.injection_module.parameters(), lr=0.01)
        
        # Curriculum learning - start with easier optimization
        for phase in range(2):
            if phase == 0:
                steps = training_steps // 2
                logger.info(f"Phase 1: Concept detection training ({steps} steps)")
            else:
                steps = training_steps // 2
                logger.info(f"Phase 2: Joint optimization ({steps} steps)")
            
            for step in range(steps):
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
                
                if transfer_mask.sum() == 0:
                    continue
                
                transfer_data = source_data[transfer_mask][:8]  # Limit batch size
                
                # Phase 1: Focus on concept detection
                if phase == 0:
                    detector_optimizer.zero_grad()
                    
                    # Get concepts and train detector to be confident for transfer samples
                    transfer_features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                    transfer_concepts = self.target_sae.encode(transfer_features)
                    confidence_scores = self.concept_detector(transfer_concepts)
                    
                    # Encourage high confidence for transfer samples
                    confidence_loss = -torch.mean(torch.log(confidence_scores + 1e-8))
                    confidence_loss.backward()
                    detector_optimizer.step()
                    
                    if step % 20 == 0:
                        logger.info(f"Phase 1 Step {step}: Confidence loss = {confidence_loss.item():.4f}, "
                                  f"Mean confidence = {confidence_scores.mean().item():.4f}")
                
                # Phase 2: Joint optimization
                else:
                    detector_optimizer.zero_grad()
                    injection_optimizer.zero_grad()
                    
                    # Transfer loss - encourage correct classification
                    transfer_outputs = self.transfer_concept(transfer_data, target_class)
                    if transfer_outputs is not None:
                        target_labels_tensor = torch.full((transfer_data.shape[0],), target_class, 
                                                        device=self.device, dtype=torch.long)
                        transfer_loss = nn.functional.cross_entropy(transfer_outputs, target_labels_tensor)
                        
                        # Additional loss: encourage high probability for target class
                        transfer_probs = torch.softmax(transfer_outputs, dim=1)
                        probability_loss = -torch.mean(torch.log(transfer_probs[:, target_class] + 1e-8))
                        
                        total_loss = transfer_loss + 0.5 * probability_loss
                        total_loss.backward()
                        
                        injection_optimizer.step()
                        detector_optimizer.step()
                        
                        if step % 20 == 0:
                            with torch.no_grad():
                                preds = torch.argmax(transfer_outputs, dim=1)
                                correct = (preds == target_class).sum().item()
                                max_prob = torch.max(transfer_probs[:, target_class])
                                
                                logger.info(f"Phase 2 Step {step}: Loss = {total_loss.item():.4f}, "
                                          f"Correct = {correct}/{transfer_data.shape[0]}, "
                                          f"Max prob = {max_prob:.4f}")
                                logger.info(f"  Injection strength: {torch.abs(self.injection_module.injection_strength).item():.4f}")
        
        logger.info("✓ Optimized injection system training completed")

def test_final_system():
    """Test the final optimized transfer system."""
    print("=== TESTING FINAL OPTIMIZED TRANSFER SYSTEM ===")
    
    # Setup
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=4,  # Train models better
        batch_size=32,
        learning_rate=0.001,
        concept_dim=24,  # Larger concept space
        device='cpu'
    )
    
    # Create and train models properly
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    # Train models to better convergence
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
    
    # Create final transfer system
    final_system = FinalNeuralConceptTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system
    print("Fitting final transfer system...")
    fit_metrics = final_system.fit(source_train_loader, target_train_loader, sae_epochs=40)
    print(f"Alignment error: {fit_metrics['alignment_error']:.4f}")
    
    # Test both transfer classes
    for transfer_class in [8, 9]:
        print(f"\n=== TESTING TRANSFER FOR CLASS {transfer_class} ===")
        
        # Setup injection
        final_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
        
        # Get test samples
        test_samples = []
        for data, labels in source_test_loader:
            mask = (labels == transfer_class)
            if mask.sum() > 0:
                test_samples.append(data[mask][:20])  # More samples
                if len(test_samples) >= 2:
                    break
        
        if not test_samples:
            print(f"No test samples found for class {transfer_class}")
            continue
            
        test_data = torch.cat(test_samples, dim=0)[:20]
        print(f"Testing with {test_data.shape[0]} samples")
        
        # Before transfer
        with torch.no_grad():
            original_outputs = trained_target(test_data.view(test_data.size(0), -1))
            original_preds = torch.argmax(original_outputs, dim=1)
            original_probs = torch.softmax(original_outputs, dim=1)
            
        # After transfer
        enhanced_outputs = final_system.transfer_concept(test_data, transfer_class)
        if enhanced_outputs is not None:
            enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
            enhanced_probs = torch.softmax(enhanced_outputs, dim=1)
            
            print(f"Original predictions: {original_preds[:10].tolist()}...")
            print(f"Enhanced predictions: {enhanced_preds[:10].tolist()}...")
            print(f"Original max confidence: {torch.max(original_probs, dim=1)[0][:5].tolist()}...")
            print(f"Enhanced class {transfer_class} confidence: {enhanced_probs[:5, transfer_class].tolist()}...")
            
            correct_transfers = (enhanced_preds == transfer_class).sum().item()
            transfer_rate = 100 * correct_transfers / test_data.shape[0]
            print(f"✨ SUCCESSFUL TRANSFERS: {correct_transfers}/{test_data.shape[0]} ({transfer_rate:.1f}%)")
            
            if transfer_rate > 10:  # Success threshold
                print(f"🎉 KNOWLEDGE TRANSFER WORKING! Class {transfer_class} achieved {transfer_rate:.1f}% transfer rate")
            else:
                print(f"⚠️ Low transfer rate for class {transfer_class}")
        else:
            print("Transfer failed")

if __name__ == "__main__":
    test_final_system()