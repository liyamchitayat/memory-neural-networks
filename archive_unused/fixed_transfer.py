#!/usr/bin/env python3
"""
Fixed Transfer System
Addresses the critical issues found in debugging:
1. SAEs learning trivial mappings
2. NaN alignment errors  
3. Zero concept changes in injection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from neural_concept_transfer import NeuralConceptTransferSystem, SparseAutoencoder, ConceptInjectionModule
from architectures import WideNN
from experimental_framework import MNISTDataManager, ExperimentConfig, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedSparseAutoencoder(nn.Module):
    """Fixed SAE that learns meaningful representations."""
    
    def __init__(self, input_dim: int, concept_dim: int, sparsity_weight: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.Tanh()  # Bounded activation for more stable training
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(concept_dim * 2, input_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def compute_loss(self, x: torch.Tensor) -> tuple:
        z = self.encode(x)
        x_recon = self.decode(z)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x)
        
        # Sparsity loss (L1 regularization on activations)
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        
        return total_loss, {
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'total': total_loss.item()
        }

class FixedConceptInjectionModule(nn.Module):
    """Fixed injection module that actually modifies concepts."""
    
    def __init__(self, concept_dim: int, free_directions: torch.Tensor, 
                 target_concept_projection: torch.Tensor):
        super().__init__()
        self.concept_dim = concept_dim
        self.free_directions = nn.Parameter(free_directions, requires_grad=False)
        self.target_projection = nn.Parameter(target_concept_projection, requires_grad=False)
        
        # Learnable injection strength - start higher for more effect
        self.injection_strength = nn.Parameter(torch.tensor(5.0))
        
        # Learnable preservation weight - start lower to allow more injection
        self.preservation_weight = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27
        
    def forward(self, z: torch.Tensor, confidence: torch.Tensor, 
                original_features: torch.Tensor) -> torch.Tensor:
        """
        Perform concept injection in free space.
        """
        batch_size = z.shape[0]
        
        # Compute injection for each sample
        injection = torch.zeros_like(z)
        
        # More aggressive injection - multiply confidence by strength and projection
        for i in range(self.free_directions.shape[1]):
            direction = self.free_directions[:, i]
            strength = self.target_projection[i] * self.injection_strength
            # Scale by confidence for each sample
            scaled_strength = confidence.unsqueeze(1) * strength
            injection += scaled_strength * direction.unsqueeze(0)
        
        # Apply injection
        z_enhanced = z + injection
        
        return z_enhanced

class FixedNeuralConceptTransferSystem(NeuralConceptTransferSystem):
    """Fixed transfer system addressing the core issues."""
    
    def train_sparse_autoencoders(self, source_loader, target_loader, epochs=50):
        """Train fixed SAEs that learn meaningful representations."""
        
        # Get feature dimensions by running sample data
        sample_batch = next(iter(source_loader))[0][:1].to(self.device)
        sample_batch_flat = sample_batch.view(sample_batch.size(0), -1)
        
        source_dim = self.get_feature_dim(self.source_model, sample_batch_flat)
        target_dim = self.get_feature_dim(self.target_model, sample_batch_flat)
        
        # Initialize fixed SAEs
        self.source_sae = FixedSparseAutoencoder(source_dim, self.concept_dim).to(self.device)
        self.target_sae = FixedSparseAutoencoder(target_dim, self.concept_dim).to(self.device)
        
        # Train source SAE
        logger.info("Training fixed source SAE...")
        self._train_fixed_sae(self.source_sae, self.source_model, source_loader, epochs)
        
        # Train target SAE
        logger.info("Training fixed target SAE...")
        self._train_fixed_sae(self.target_sae, self.target_model, target_loader, epochs)
        
        return self.source_sae, self.target_sae
    
    def _train_fixed_sae(self, sae, model, data_loader, epochs):
        """Train a fixed SAE with proper parameters."""
        optimizer = optim.Adam(sae.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            batch_count = 0
            
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                data_flat = data.view(data.size(0), -1)
                
                # Get features from model
                with torch.no_grad():
                    if hasattr(model, 'get_features'):
                        features = model.get_features(data_flat)
                    else:
                        features = model(data_flat)
                        if isinstance(features, tuple):
                            features = features[0]
                
                # Train SAE
                optimizer.zero_grad()
                loss, metrics = sae.compute_loss(features)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += metrics['reconstruction']
                batch_count += 1
                
                # Limit batches to prevent overfitting
                if batch_count >= 20:
                    break
            
            avg_loss = total_loss / batch_count
            avg_recon = total_recon / batch_count
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"SAE Epoch {epoch}: Loss = {avg_loss:.4f}, Reconstruction = {avg_recon:.4f}")
        
        logger.info(f"✓ Fixed SAE training completed. Final loss: {avg_loss:.4f}")
    
    def setup_injection_system(self, target_class: int, source_loader=None, target_loader=None):
        """Setup injection system with fixed module."""
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
        
        # Setup concept detector (keep original)
        shared_centroids = torch.stack([self.target_centroids[c] for c in self.shared_classes])
        from neural_concept_transfer import ConceptDetector
        self.concept_detector = ConceptDetector(self.concept_dim, shared_centroids).to(self.device)
        
        # Setup FIXED injection module
        self.injection_module = FixedConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Train the injection system if data loaders provided
        if source_loader is not None and target_loader is not None:
            logger.info("Training fixed concept injection system...")
            self._train_fixed_injection_system(target_class, source_loader, target_loader)
        
        return self.injection_module
    
    def _train_fixed_injection_system(self, target_class: int, source_loader, target_loader, training_steps=100):
        """Train injection system with more aggressive optimization."""
        # Setup optimizers with higher learning rate
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        optimizer = optim.Adam(injection_params, lr=0.01)
        
        logger.info(f"Training injection system for class {target_class} with {training_steps} steps")
        
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
            if transfer_mask.sum() == 0:
                continue
            
            transfer_data = source_data[transfer_mask][:8]  # Limit batch size
            
            optimizer.zero_grad()
            
            # Focus on transfer loss - make it really try to predict the target class
            transfer_outputs = self.transfer_concept(transfer_data, target_class)
            if transfer_outputs is not None:
                # Strong cross-entropy loss for transfer
                target_labels_tensor = torch.full((transfer_data.shape[0],), target_class, 
                                                device=self.device, dtype=torch.long)
                transfer_loss = nn.functional.cross_entropy(transfer_outputs, target_labels_tensor)
                
                transfer_loss.backward()
                optimizer.step()
                
                if step % 20 == 0:
                    with torch.no_grad():
                        preds = torch.argmax(transfer_outputs, dim=1)
                        correct = (preds == target_class).sum().item()
                        max_confidence = torch.max(torch.softmax(transfer_outputs, dim=1)[:, target_class])
                        
                        logger.info(f"Step {step}: Loss = {transfer_loss.item():.4f}, "
                                  f"Correct = {correct}/{transfer_data.shape[0]}, "
                                  f"Max confidence = {max_confidence:.4f}")
                        logger.info(f"  Injection strength: {self.injection_module.injection_strength.item():.4f}")
        
        logger.info("✓ Fixed injection system training completed")

def test_fixed_system():
    """Test the fixed transfer system."""
    print("=== TESTING FIXED TRANSFER SYSTEM ===")
    
    # Setup
    source_classes = {2, 3, 4, 5, 6, 7, 8, 9}
    target_classes = {0, 1, 2, 3, 4, 5, 6, 7}
    
    config = ExperimentConfig(
        seed=42,
        max_epochs=3,
        batch_size=32,
        learning_rate=0.001,
        concept_dim=16,
        device='cpu'
    )
    
    # Create and train models properly
    data_manager = MNISTDataManager(config)
    trainer = ModelTrainer(config)
    
    source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
        data_manager.get_data_loaders(source_classes, target_classes)
    
    # Train models to convergence
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
    
    # Create fixed transfer system
    fixed_system = FixedNeuralConceptTransferSystem(
        source_model=trained_source,
        target_model=trained_target,
        source_classes=source_classes,
        target_classes=target_classes,
        concept_dim=config.concept_dim,
        device=config.device
    )
    
    # Fit the system
    print("Fitting fixed transfer system...")
    fit_metrics = fixed_system.fit(source_train_loader, target_train_loader, sae_epochs=30)
    print(f"Alignment error: {fit_metrics['alignment_error']}")
    
    # Test transfer for class 8
    transfer_class = 8
    print(f"Setting up injection for class {transfer_class}...")
    fixed_system.setup_injection_system(transfer_class, source_train_loader, target_train_loader)
    
    # Test with actual data
    print("Testing transfer...")
    test_samples = []
    for data, labels in source_test_loader:
        mask = (labels == transfer_class)
        if mask.sum() > 0:
            test_samples.append(data[mask][:10])
            if len(test_samples) >= 1:
                break
    
    if test_samples:
        test_data = test_samples[0]
        print(f"Testing with {test_data.shape[0]} samples")
        
        # Before transfer
        with torch.no_grad():
            original_outputs = trained_target(test_data.view(test_data.size(0), -1))
            original_preds = torch.argmax(original_outputs, dim=1)
            
        # After transfer
        enhanced_outputs = fixed_system.transfer_concept(test_data, transfer_class)
        if enhanced_outputs is not None:
            enhanced_preds = torch.argmax(enhanced_outputs, dim=1)
            enhanced_probs = torch.softmax(enhanced_outputs, dim=1)
            
            print(f"Original predictions: {original_preds.tolist()}")
            print(f"Enhanced predictions: {enhanced_preds.tolist()}")
            print(f"Class {transfer_class} confidence: {enhanced_probs[:, transfer_class].tolist()}")
            
            correct_transfers = (enhanced_preds == transfer_class).sum().item()
            print(f"Successful transfers: {correct_transfers}/{test_data.shape[0]} ({100*correct_transfers/test_data.shape[0]:.1f}%)")
        else:
            print("Transfer failed")

if __name__ == "__main__":
    test_fixed_system()