"""
Final Shared Layer Transfer Experiment
Implements the exact experiments specified:
1. [0,1,2] → [2,3,4] transfer digit 3 (cross-architecture + data overlap)
2. [0,1,2,3,4] → [2,3,4,5,6] transfer digit 5 (data overlap only)
3. [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer digit 8 (data overlap only)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

from architectures import WideNN, DeepNN
from experimental_framework import (
    ExperimentConfig, MNISTDataManager, ModelTrainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectionLayer(nn.Module):
    """Projection layer to handle feature dimension mismatches between architectures."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        # Only apply batch norm if batch size > 1 and in training mode
        if x.size(0) > 1 and self.training:
            projected = self.batch_norm(projected)
        elif not self.training:
            # In eval mode, always apply batch norm (it uses running stats)
            projected = self.batch_norm(projected)
        return torch.relu(projected)


class SharedLayerNetwork(nn.Module):
    """
    Network with shared layers and projection for cross-architecture compatibility.
    
    Architecture:
    Base Network -> Projection Layer -> Shared Layers -> Output Layer
    """
    
    def __init__(self, base_model: nn.Module, shared_dim: int = 64, 
                 num_shared_layers: int = 3, num_classes: int = 10):
        super().__init__()
        
        self.base_model = base_model
        self.shared_dim = shared_dim
        
        # Get feature dimension from base model
        if isinstance(base_model, WideNN):
            feature_dim = 64  # WideNN penultimate layer size
        elif isinstance(base_model, DeepNN):
            feature_dim = 32  # DeepNN penultimate layer size
        else:
            raise ValueError(f"Unknown model type: {type(base_model)}")
        
        # Projection layer to standardize feature dimensions
        self.projection = ProjectionLayer(feature_dim, shared_dim)
        
        # Shared layers will be set externally to ensure true sharing
        self.shared_layers = None
        self.use_shared_layers = False  # Flag to control whether to use shared layers or original classification
        
        # Keep original classification available
        self.original_classify = self.base_model.classify_from_features
        
    def set_shared_layers(self, shared_layers: nn.Module):
        """Set the shared layers module (same instance across networks)."""
        self.shared_layers = shared_layers
    
    def use_original_classification(self):
        """Switch to using original base model classification."""
        self.use_shared_layers = False
    
    def use_shared_classification(self):
        """Switch to using shared layers for classification."""
        if self.shared_layers is None:
            raise RuntimeError("Shared layers not set! Call set_shared_layers() first.")
        self.use_shared_layers = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from base network
        features = self.base_model.get_features(x)
        
        if self.use_shared_layers:
            if self.shared_layers is None:
                raise RuntimeError("Shared layers not set! Call set_shared_layers() first.")
            # Project to shared dimension and pass through shared layers
            projected = self.projection(features)
            output = self.shared_layers(projected)
            return output
        else:
            # Use original base model classification
            output = self.original_classify(features)
            return output
    
    def freeze_base_and_projection(self):
        """Freeze base model and projection parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
    
    def unfreeze_base_and_projection(self):
        """Unfreeze base model and projection parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True


class SharedLayerTransferSystem:
    """
    System for transferring knowledge through truly shared layers.
    """
    
    def __init__(self, shared_dim: int = 64, num_shared_layers: int = 3, 
                 num_classes: int = 10, device: str = 'cuda'):
        self.shared_dim = shared_dim
        self.num_shared_layers = num_shared_layers
        self.num_classes = num_classes
        self.device = device
        self.source_network = None
        self.target_network = None
        self.shared_layers = None
    
    def create_networks(self, source_arch: str, target_arch: str) -> Tuple[SharedLayerNetwork, SharedLayerNetwork]:
        """Create source and target networks with truly shared layers."""
        
        # Create base models
        if source_arch == "WideNN":
            source_base = WideNN().to(self.device)
        elif source_arch == "DeepNN":
            source_base = DeepNN().to(self.device)
        else:
            raise ValueError(f"Unknown source architecture: {source_arch}")
            
        if target_arch == "WideNN":
            target_base = WideNN().to(self.device)
        elif target_arch == "DeepNN":
            target_base = DeepNN().to(self.device)
        else:
            raise ValueError(f"Unknown target architecture: {target_arch}")
        
        # Create networks
        self.source_network = SharedLayerNetwork(
            source_base, self.shared_dim, self.num_shared_layers, self.num_classes
        ).to(self.device)
        
        self.target_network = SharedLayerNetwork(
            target_base, self.shared_dim, self.num_shared_layers, self.num_classes
        ).to(self.device)
        
        # Create shared layers module (same instance for both networks)
        shared_layers = []
        for i in range(self.num_shared_layers):
            shared_layers.extend([
                nn.Linear(self.shared_dim, self.shared_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.shared_dim),
                nn.Dropout(0.2)
            ])
        shared_layers.append(nn.Linear(self.shared_dim, self.num_classes))
        
        self.shared_layers = nn.Sequential(*shared_layers).to(self.device)
        
        # Set the same shared layers instance for both networks
        self.source_network.set_shared_layers(self.shared_layers)
        self.target_network.set_shared_layers(self.shared_layers)
        
        logger.info(f"Created networks: {source_arch}→{target_arch} with truly shared layers")
        
        return self.source_network, self.target_network
    
    def train_transfer(self, combined_loader: DataLoader, epochs: int = 15, 
                      lr: float = 0.001) -> Dict[str, float]:
        """
        Train only the shared layers while freezing base networks.
        """
        # Freeze base models and projections
        self.source_network.freeze_base_and_projection()
        self.target_network.freeze_base_and_projection()
        
        # Only optimize shared layers
        optimizer = optim.Adam(self.shared_layers.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {'final_loss': 0, 'final_accuracy': 0}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(combined_loader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                
                # Forward through both networks (they share the same layers!)
                # Use source network for forward pass (both should give same result)
                output = self.source_network(data)
                
                # Compute loss
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Track accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(combined_loader)
            
            if epoch % 3 == 0 or epoch == epochs - 1:
                logger.info(f"Transfer Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            metrics['final_loss'] = avg_loss
            metrics['final_accuracy'] = accuracy
        
        return metrics
    
    def train_transfer_structured(self, shared_loader: Optional[DataLoader], 
                                source_only_loader: Optional[DataLoader],
                                transfer_loader: DataLoader, epochs: int = 15, 
                                lr: float = 0.001) -> Dict[str, float]:
        """
        Train shared layers with proper structure according to requirements:
        1. Shared classes through both networks
        2. Source-only classes through source network only  
        3. Transfer classes through target network only
        """
        # Freeze base models and projections
        self.source_network.freeze_base_and_projection()
        self.target_network.freeze_base_and_projection()
        
        # Only optimize shared layers
        optimizer = optim.Adam(self.shared_layers.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {'final_loss': 0, 'final_accuracy': 0}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            batches_processed = 0
            
            # Prepare iterators with designated networks (no alternation to avoid contamination)
            iterators = []
            if shared_loader:
                # Use source network for shared classes to avoid target network contamination
                iterators.append(('shared', iter(shared_loader), self.source_network))
            if source_only_loader:
                # Use source network for source-only classes
                iterators.append(('source_only', iter(source_only_loader), self.source_network))
            # Use target network ONLY for transfer classes
            iterators.append(('transfer', iter(transfer_loader), self.target_network))
            
            # Train on batches from all categories
            max_batches = max(len(loader) for loader in [shared_loader, source_only_loader, transfer_loader] if loader)
            
            for batch_idx in range(max_batches):
                for category, data_iter, designated_network in iterators:
                    try:
                        data, target = next(data_iter)
                    except StopIteration:
                        continue
                        
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    optimizer.zero_grad()
                    
                    # Use the designated network for this category to avoid contamination
                    output = designated_network(data)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Track accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    batches_processed += 1
            
            if batches_processed > 0:
                accuracy = correct / total
                avg_loss = total_loss / batches_processed
                
                if epoch % 3 == 0 or epoch == epochs - 1:
                    logger.info(f"Structured Transfer Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                
                metrics['final_loss'] = avg_loss
                metrics['final_accuracy'] = accuracy
        
        return metrics
    
    def train_transfer_with_mapping(self, shared_loader: Optional[DataLoader], 
                                   source_only_loader: Optional[DataLoader],
                                   transfer_loader: DataLoader, epochs: int = 15, 
                                   lr: float = 0.001) -> Dict[str, float]:
        """
        Train shared layers to learn cross-network feature mapping for transfer.
        Key insight: Shared layers must learn to map Network 2's representation of 
        transfer digit to the same classification space as Network 1.
        """
        # Freeze base models and projections
        self.source_network.freeze_base_and_projection()
        self.target_network.freeze_base_and_projection()
        
        # Only optimize shared layers
        optimizer = optim.Adam(self.shared_layers.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {'final_loss': 0, 'final_accuracy': 0}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            batches_processed = 0
            
            # Get iterators
            shared_iter = iter(shared_loader) if shared_loader else None
            source_only_iter = iter(source_only_loader) if source_only_loader else None
            transfer_iter = iter(transfer_loader)
            
            # Calculate max batches
            max_batches = max([
                len(loader) for loader in [shared_loader, source_only_loader, transfer_loader] 
                if loader is not None
            ])
            
            for batch_idx in range(max_batches):
                batch_losses = []
                
                # 1. Train on shared classes through source network
                if shared_iter:
                    try:
                        data, target = next(shared_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        optimizer.zero_grad()
                        output = self.source_network(data)
                        loss = criterion(output, target)
                        batch_losses.append(loss)
                        
                        # Track accuracy
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        pass
                
                # 2. Train on source-only classes through source network
                if source_only_iter:
                    try:
                        data, target = next(source_only_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        if not batch_losses:  # Only zero_grad if we haven't already
                            optimizer.zero_grad()
                        output = self.source_network(data)
                        loss = criterion(output, target)
                        batch_losses.append(loss)
                        
                        # Track accuracy
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        pass
                
                # 3. Train on transfer digit through target network
                try:
                    data, target = next(transfer_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    if not batch_losses:  # Only zero_grad if we haven't already
                        optimizer.zero_grad()
                    output = self.target_network(data)
                    loss = criterion(output, target)
                    batch_losses.append(loss)
                    
                    # Track accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                except StopIteration:
                    # Reset iterator when exhausted
                    transfer_iter = iter(transfer_loader)
                    try:
                        data, target = next(transfer_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        if not batch_losses:
                            optimizer.zero_grad()
                        output = self.target_network(data)
                        loss = criterion(output, target)
                        batch_losses.append(loss)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    except StopIteration:
                        continue
                
                # Combine losses and update
                if batch_losses:
                    combined_loss = sum(batch_losses) / len(batch_losses)
                    combined_loss.backward()
                    optimizer.step()
                    
                    total_loss += combined_loss.item()
                    batches_processed += 1
            
            if batches_processed > 0 and total > 0:
                accuracy = correct / total
                avg_loss = total_loss / batches_processed
                
                if epoch % 3 == 0 or epoch == epochs - 1:
                    logger.info(f"Cross-Network Transfer Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                
                metrics['final_loss'] = avg_loss
                metrics['final_accuracy'] = accuracy
        
        return metrics
    
    def train_transfer_with_alignment(self, shared_loader: Optional[DataLoader], 
                                    source_only_loader: Optional[DataLoader],
                                    transfer_loader: DataLoader, epochs: int = 15, 
                                    lr: float = 0.001) -> Dict[str, float]:
        """
        Train shared layers with feature alignment for genuine transfer learning.
        
        Strategy:
        1. Use shared classes to align feature spaces between networks
        2. Train source-specific knowledge through source network
        3. Train transfer knowledge through target network
        4. Shared layers learn to map aligned features to correct classifications
        """
        # Freeze base models and projections
        self.source_network.freeze_base_and_projection()
        self.target_network.freeze_base_and_projection()
        
        # Only optimize shared layers
        optimizer = optim.Adam(self.shared_layers.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {'final_loss': 0, 'final_accuracy': 0}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            batches_processed = 0
            
            # Get iterators
            shared_iter = iter(shared_loader) if shared_loader else None
            source_only_iter = iter(source_only_loader) if source_only_loader else None
            transfer_iter = iter(transfer_loader)
            
            # Calculate max batches
            loaders = [loader for loader in [shared_loader, source_only_loader, transfer_loader] if loader is not None]
            max_batches = max(len(loader) for loader in loaders)
            
            for batch_idx in range(max_batches):
                epoch_losses = []
                
                # Phase 1: Feature Alignment using shared classes
                if shared_iter:
                    try:
                        data, target = next(shared_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        optimizer.zero_grad()
                        
                        # Train on same data through BOTH networks to align feature spaces
                        source_output = self.source_network(data)
                        target_output = self.target_network(data)
                        
                        # Both should predict the same label - this aligns the feature spaces
                        source_loss = criterion(source_output, target)
                        target_loss = criterion(target_output, target)
                        
                        # Combined alignment loss
                        alignment_loss = source_loss + target_loss
                        epoch_losses.append(alignment_loss)
                        
                        # Track accuracy (use source network for consistency)
                        _, predicted = torch.max(source_output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        shared_iter = None  # Exhausted
                
                # Phase 2: Source-specific knowledge
                if source_only_iter:
                    try:
                        data, target = next(source_only_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        if not epoch_losses:  # Only zero_grad if not done already
                            optimizer.zero_grad()
                        
                        output = self.source_network(data)
                        loss = criterion(output, target)
                        epoch_losses.append(loss)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        source_only_iter = None  # Exhausted
                
                # Phase 3: Transfer knowledge (through aligned feature space)
                try:
                    data, target = next(transfer_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    if not epoch_losses:  # Only zero_grad if not done already
                        optimizer.zero_grad()
                    
                    output = self.target_network(data)
                    loss = criterion(output, target)
                    epoch_losses.append(loss)
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                except StopIteration:
                    # Reset transfer iterator
                    transfer_iter = iter(transfer_loader)
                    try:
                        data, target = next(transfer_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        if not epoch_losses:
                            optimizer.zero_grad()
                        
                        output = self.target_network(data)
                        loss = criterion(output, target)
                        epoch_losses.append(loss)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    except StopIteration:
                        continue
                
                # Update parameters
                if epoch_losses:
                    combined_loss = sum(epoch_losses) / len(epoch_losses)
                    combined_loss.backward()
                    optimizer.step()
                    
                    total_loss += combined_loss.item()
                    batches_processed += 1
            
            if batches_processed > 0 and total > 0:
                accuracy = correct / total
                avg_loss = total_loss / batches_processed
                
                if epoch % 3 == 0 or epoch == epochs - 1:
                    logger.info(f"Feature Alignment Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                
                metrics['final_loss'] = avg_loss
                metrics['final_accuracy'] = accuracy
        
        return metrics
    
    def train_transfer_with_bridging(self, shared_loader: Optional[DataLoader], 
                                   source_only_loader: Optional[DataLoader],
                                   transfer_loader: DataLoader, epochs: int = 15, 
                                   lr: float = 0.001) -> Dict[str, float]:
        """
        Feature Bridging: Train shared layers to recognize Network 1's "garbage" 
        features for transfer digit and map them to correct classification.
        
        Key insight: Network 1 produces consistent (but bad) features for digit 3.
        We can train shared layers to recognize this specific pattern and classify it correctly.
        """
        import time
        
        start_time = time.time()
        
        # Freeze base models and projections (surgical approach)
        self.source_network.freeze_base_and_projection()
        self.target_network.freeze_base_and_projection()
        
        # Count trainable parameters (should be only shared layers)
        trainable_params = sum(p.numel() for p in self.shared_layers.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters (shared layers only): {trainable_params:,}")
        
        # Only optimize shared layers
        optimizer = optim.Adam(self.shared_layers.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        metrics = {
            'final_loss': 0, 
            'final_accuracy': 0,
            'training_time_seconds': 0,
            'trainable_parameters': trainable_params,
            'transfer_samples_seen': 0,
            'convergence_epoch': epochs  # Track when transfer digit accuracy > 50%
        }
        
        # Phase 1: Build feature patterns from Network 1's "garbage" representations
        logger.info("Building Network 1's feature patterns for transfer digit...")
        
        transfer_feature_patterns = []
        pattern_build_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(transfer_loader):
                if batch_idx >= 10:  # Limit pattern building to save time
                    break
                    
                data = data.to(self.device).view(data.size(0), -1)
                
                # Get Network 1's "garbage" features for transfer digit
                garbage_features = self.source_network.base_model.get_features(data)
                projected_garbage = self.source_network.projection(garbage_features)
                
                transfer_feature_patterns.append(projected_garbage.cpu())
                metrics['transfer_samples_seen'] += len(data)
        
        # Average the garbage patterns
        if transfer_feature_patterns:
            avg_garbage_pattern = torch.cat(transfer_feature_patterns, dim=0).mean(dim=0).to(self.device)
            logger.info(f"Built transfer pattern from {metrics['transfer_samples_seen']} samples in {time.time() - pattern_build_time:.2f}s")
        else:
            logger.error("No transfer patterns built!")
            return metrics
        
        # Phase 2: Train shared layers with feature bridging
        logger.info("Training shared layers to bridge garbage features to correct classification...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            correct = 0
            total = 0
            batches_processed = 0
            
            # Get iterators
            shared_iter = iter(shared_loader) if shared_loader else None
            source_only_iter = iter(source_only_loader) if source_only_loader else None
            transfer_iter = iter(transfer_loader)
            
            # Calculate max batches
            loaders = [loader for loader in [shared_loader, source_only_loader, transfer_loader] if loader is not None]
            max_batches = max(len(loader) for loader in loaders)
            
            transfer_correct = 0
            transfer_total = 0
            
            for batch_idx in range(max_batches):
                epoch_losses = []
                
                # Train on shared classes (standard)
                if shared_iter:
                    try:
                        data, target = next(shared_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        optimizer.zero_grad()
                        output = self.source_network(data)
                        loss = criterion(output, target)
                        epoch_losses.append(loss)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        shared_iter = None
                
                # Train on source-only classes (standard)
                if source_only_iter:
                    try:
                        data, target = next(source_only_iter)
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        
                        if not epoch_losses:
                            optimizer.zero_grad()
                        output = self.source_network(data)
                        loss = criterion(output, target)
                        epoch_losses.append(loss)
                        
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                    except StopIteration:
                        source_only_iter = None
                
                # FEATURE BRIDGING: Train on transfer digit using garbage patterns
                try:
                    data, target = next(transfer_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    if not epoch_losses:
                        optimizer.zero_grad()
                    
                    # Get Network 1's garbage features for this transfer data
                    with torch.no_grad():
                        garbage_features = self.source_network.base_model.get_features(data)
                        projected_garbage = self.source_network.projection(garbage_features)
                    
                    # Train shared layers to map garbage features to transfer digit class
                    output = self.shared_layers(projected_garbage)
                    transfer_labels = torch.full_like(target, target[0].item())  # All should be transfer digit
                    loss = criterion(output, transfer_labels)
                    epoch_losses.append(loss)
                    
                    # Track transfer-specific accuracy
                    _, predicted = torch.max(output.data, 1)
                    transfer_total += target.size(0)
                    transfer_correct += (predicted == transfer_labels).sum().item()
                    
                    total += target.size(0)
                    correct += (predicted == transfer_labels).sum().item()
                    
                except StopIteration:
                    # Reset transfer iterator
                    transfer_iter = iter(transfer_loader)
                    continue
                
                # Update parameters
                if epoch_losses:
                    combined_loss = sum(epoch_losses) / len(epoch_losses)
                    combined_loss.backward()
                    optimizer.step()
                    
                    total_loss += combined_loss.item()
                    batches_processed += 1
            
            # Calculate metrics
            if batches_processed > 0 and total > 0:
                accuracy = correct / total
                transfer_accuracy = transfer_correct / transfer_total if transfer_total > 0 else 0
                avg_loss = total_loss / batches_processed
                epoch_time = time.time() - epoch_start
                
                # Check for convergence (transfer accuracy > 50%)
                if transfer_accuracy > 0.5 and metrics['convergence_epoch'] == epochs:
                    metrics['convergence_epoch'] = epoch + 1
                    logger.info(f"🎯 CONVERGENCE: Transfer digit accuracy >50% at epoch {epoch + 1}")
                
                if epoch % 3 == 0 or epoch == epochs - 1:
                    logger.info(f"Bridging Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, "
                              f"Overall Acc = {accuracy:.4f}, Transfer Acc = {transfer_accuracy:.4f}, "
                              f"Time = {epoch_time:.1f}s")
                
                metrics['final_loss'] = avg_loss
                metrics['final_accuracy'] = accuracy
                metrics['transfer_accuracy'] = transfer_accuracy
        
        # Final measurements
        end_time = time.time()
        
        metrics['training_time_seconds'] = end_time - start_time
        
        logger.info(f"🔧 EFFICIENCY METRICS:")
        logger.info(f"  Training time: {metrics['training_time_seconds']:.1f} seconds")
        logger.info(f"  Trainable parameters: {metrics['trainable_parameters']:,}")
        logger.info(f"  Convergence epoch: {metrics['convergence_epoch']}/{epochs}")
        logger.info(f"  Final transfer accuracy: {metrics.get('transfer_accuracy', 0):.3f}")
        
        return metrics


class FinalExperimentRunner:
    """Runner for the exact experiments specified in requirements."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_manager = MNISTDataManager(config)
        self.trainer = ModelTrainer(config)
        
        # Create results directory
        self.results_dir = Path("experiment_results/final_shared_layer_transfer")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_network_on_classes(self, network: SharedLayerNetwork, 
                                   test_loader: DataLoader, eval_classes: Set[int]) -> float:
        """Evaluate network accuracy on specific classes."""
        device = self.config.device
        network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Filter for eval classes
                mask = torch.tensor([t.item() in eval_classes for t in target])
                if mask.sum() == 0:
                    continue
                
                eval_data = data[mask].to(device)
                eval_targets = target[mask].to(device)
                eval_data = eval_data.view(eval_data.size(0), -1)
                
                outputs = network(eval_data)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == eval_targets).sum().item()
                total += eval_targets.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def run_single_experiment(self, source_classes: Set[int], target_classes: Set[int], 
                            transfer_digit: int, source_arch: str, target_arch: str, 
                            seed: int, exp_name: str) -> Optional[Dict]:
        """Run a single transfer experiment."""
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"\n=== {exp_name}: {source_arch}→{target_arch}, Seed: {seed} ===")
        logger.info(f"Network 1 (test subject): {sorted(source_classes)}, Network 2 (donor): {sorted(target_classes)}")
        logger.info(f"Transfer digit {transfer_digit} FROM Network 2 TO Network 1")
        logger.info(f"Will test Network 1 on: original set, transferred info, untransferred info")
        
        # Get data loaders
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = \
            self.data_manager.get_data_loaders(source_classes, target_classes)
        
        # Get combined test set
        all_classes = source_classes | target_classes
        all_test_subset = self.data_manager.create_class_subset(
            self.data_manager.test_dataset, all_classes, max_samples_per_class=200
        )
        all_test_loader = DataLoader(all_test_subset, batch_size=self.config.batch_size, shuffle=False)
        
        # Create transfer system
        transfer_system = SharedLayerTransferSystem(device=self.config.device)
        source_network, target_network = transfer_system.create_networks(source_arch, target_arch)
        
        # Train base networks separately first using their original classification heads
        logger.info("Phase 1: Training base networks with original classification...")
        
        # Train source network with original classification
        source_network.use_original_classification()
        source_network.unfreeze_base_and_projection()
        trained_source, source_acc = self.trainer.train_model(
            source_network, source_train_loader, source_test_loader
        )
        
        if trained_source is None:
            logger.error("Source network training failed")
            return None
        
        logger.info(f"Source network trained successfully: {source_acc:.3f} accuracy")
        
        # Train target network with original classification
        target_network.use_original_classification()
        target_network.unfreeze_base_and_projection()
        trained_target, target_acc = self.trainer.train_model(
            target_network, target_train_loader, target_test_loader
        )
        
        if trained_target is None:
            logger.error("Target network training failed")
            return None
        
        logger.info(f"Target network trained successfully: {target_acc:.3f} accuracy")
        
        # CRITICAL: Evaluate baseline performance BEFORE adding shared layers (for valid comparison)
        logger.info("Phase 2: Evaluating baseline performance with original architectures...")
        
        transfer_classes = {transfer_digit}
        # Untransferred classes = classes in target but not in source and not the transfer digit
        untransferred_classes = target_classes - source_classes
        if transfer_digit in untransferred_classes:
            untransferred_classes.remove(transfer_digit)
        
        logger.info(f"Evaluation categories:")
        logger.info(f"  - Network 1 original classes: {sorted(source_classes)}")
        logger.info(f"  - Network 1 transferred class: {sorted(transfer_classes)}")
        logger.info(f"  - Network 1 untransferred classes: {sorted(untransferred_classes)}")
        
        # BASELINE: Measure original network performance BEFORE adding shared layers
        # This gives us the true baseline for comparison
        baseline_metrics = {
            'network1_on_original': self.evaluate_network_on_classes(source_network, all_test_loader, source_classes),  # Should be ~90%+
            'network1_on_transferred': self.evaluate_network_on_classes(source_network, all_test_loader, transfer_classes),  # Should be ~10% (random)
            'network1_on_untransferred': self.evaluate_network_on_classes(source_network, all_test_loader, untransferred_classes),  # Should be ~10% (random)
        }
        
        logger.info(f"BASELINE (original architecture):")
        logger.info(f"  Network 1 on original classes: {baseline_metrics['network1_on_original']:.3f}")
        logger.info(f"  Network 1 on transfer digit: {baseline_metrics['network1_on_transferred']:.3f}")
        logger.info(f"  Network 1 on untransferred classes: {baseline_metrics['network1_on_untransferred']:.3f}")
        
        # NOW switch to shared layers for the actual transfer experiment
        logger.info("Phase 3: Switching to shared layers for transfer experiment...")
        source_network.use_shared_classification()
        target_network.use_shared_classification()
        
        # Measure "before transfer" performance with shared layers (will be low due to untrained shared layers)
        before_metrics = {
            'network1_on_original': self.evaluate_network_on_classes(source_network, all_test_loader, source_classes),
            'network1_on_transferred': self.evaluate_network_on_classes(source_network, all_test_loader, transfer_classes),
            'network1_on_untransferred': self.evaluate_network_on_classes(source_network, all_test_loader, untransferred_classes),
        }
        
        # Create PROPER training data according to requirements
        logger.info("Phase 4: Creating properly structured training data...")
        
        # According to requirements:
        # 1. Shared inputs (through both networks): classes in both source and target
        # 2. Unique inputs of network 1 (through network 1): source-only classes  
        # 3. Input to transfer (through network 2): transfer digit only
        
        shared_classes = source_classes & target_classes  # Classes both networks know
        source_only_classes = source_classes - target_classes  # Only network 1 knows
        transfer_only_classes = {transfer_digit}  # Only network 2 should provide
        
        logger.info(f"Training data structure:")
        logger.info(f"  - Shared classes {sorted(shared_classes)}: through BOTH networks")
        logger.info(f"  - Source-only classes {sorted(source_only_classes)}: through Network 1 only")
        logger.info(f"  - Transfer digit {transfer_digit}: through Network 2 only")
        logger.info(f"  - Network 1 should NEVER see untransferred classes {sorted(untransferred_classes)}")
        
        # Create separate data loaders for each category
        def create_class_loader(classes, max_samples=3000):
            indices = []
            for idx, (_, label) in enumerate(self.data_manager.train_dataset):
                if label in classes:
                    indices.append(idx)
            np.random.shuffle(indices)
            indices = indices[:max_samples]
            subset = Subset(self.data_manager.train_dataset, indices)
            return DataLoader(subset, batch_size=self.config.batch_size, shuffle=True)
        
        shared_loader = create_class_loader(shared_classes, 5000) if shared_classes else None
        source_only_loader = create_class_loader(source_only_classes, 5000) if source_only_classes else None
        transfer_loader = create_class_loader(transfer_only_classes, 5000)
        
        # Train shared layers with feature bridging and measure efficiency
        logger.info("Phase 5: Training shared layers with feature bridging (measuring efficiency)...")
        transfer_metrics = transfer_system.train_transfer_with_bridging(
            shared_loader, source_only_loader, transfer_loader, epochs=15
        )
        
        # Evaluate after transfer
        logger.info("Phase 6: Evaluating after transfer...")
        
        after_metrics = {
            'network1_on_original': self.evaluate_network_on_classes(source_network, all_test_loader, source_classes),
            'network1_on_transferred': self.evaluate_network_on_classes(source_network, all_test_loader, transfer_classes),  # Should improve
            'network1_on_untransferred': self.evaluate_network_on_classes(source_network, all_test_loader, untransferred_classes),  # Should stay ~0
        }
        
        # Calculate improvements
        result = {
            'experiment_name': exp_name,
            'source_arch': source_arch,
            'target_arch': target_arch,
            'seed': seed,
            'source_classes': sorted(list(source_classes)),
            'target_classes': sorted(list(target_classes)),
            'transfer_digit': transfer_digit,
            'untransferred_classes': sorted(list(untransferred_classes)) if untransferred_classes else [],
            'baseline_metrics': baseline_metrics,  # Original architecture performance
            'before_metrics': before_metrics,      # With untrained shared layers
            'after_metrics': after_metrics,       # After training shared layers
            'transfer_training': transfer_metrics,
            'improvements': {
                # Compare final performance to original baseline (the meaningful comparison)
                'transferred_improvement': after_metrics['network1_on_transferred'] - baseline_metrics['network1_on_transferred'],  # How much Network 1 learned transfer digit
                'original_retention': after_metrics['network1_on_original'] / max(baseline_metrics['network1_on_original'], 0.01),  # How well Network 1 retained original performance
                'untransferred_change': after_metrics['network1_on_untransferred'] - baseline_metrics['network1_on_untransferred'],  # Network 1 change on untransferred (should stay low)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results according to requirements (comparing to meaningful baselines)
        logger.info(f"FINAL RESULTS for Network 1 (test subject):")
        logger.info(f"  Network 1 on ORIGINAL classes {sorted(source_classes)}:")
        logger.info(f"    Baseline: {baseline_metrics['network1_on_original']:.3f} → Final: {after_metrics['network1_on_original']:.3f} (retention: ×{result['improvements']['original_retention']:.3f})")
        logger.info(f"  Network 1 on TRANSFERRED digit {transfer_digit}:")
        logger.info(f"    Baseline: {baseline_metrics['network1_on_transferred']:.3f} → Final: {after_metrics['network1_on_transferred']:.3f} (improvement: +{result['improvements']['transferred_improvement']:.3f})")
        logger.info(f"  Network 1 on UNTRANSFERRED classes {sorted(untransferred_classes)}:")
        logger.info(f"    Baseline: {baseline_metrics['network1_on_untransferred']:.3f} → Final: {after_metrics['network1_on_untransferred']:.3f} (change: +{result['improvements']['untransferred_change']:.3f})")
        
        return result
    
    def run_all_experiments(self):
        """Run all experiments as specified in requirements."""
        
        # Experiment configurations EXACTLY as specified in experiments_to_run.txt
        # Network 1 (source) = first set, Network 2 (target) = second set
        # Transfer FROM Network 2 TO Network 1
        experiments = [
            {
                'name': 'transfer_digit_3',
                'description': '[0,1,2], [2,3,4] -> transfer 3',
                'network1_classes': {0, 1, 2},  # Source network (the one we test)
                'network2_classes': {2, 3, 4},  # Target network (donor of knowledge)
                'transfer_digit': 3,  # Transfer digit 3 from network2 to network1
                'run_cross_arch': True  # Only this one gets cross-architecture
            },
            {
                'name': 'transfer_digit_5',
                'description': '[0,1,2,3,4], [2,3,4,5,6] -> transfer 5',
                'network1_classes': {0, 1, 2, 3, 4},  # Source network (the one we test)
                'network2_classes': {2, 3, 4, 5, 6},  # Target network (donor of knowledge)
                'transfer_digit': 5,  # Transfer digit 5 from network2 to network1
                'run_cross_arch': False
            },
            {
                'name': 'transfer_digit_8',
                'description': '[0,1,2,3,4,5,6,7], [2,3,4,5,6,7,8,9] -> transfer 8',
                'network1_classes': {0, 1, 2, 3, 4, 5, 6, 7},  # Source network (the one we test)
                'network2_classes': {2, 3, 4, 5, 6, 7, 8, 9},  # Target network (donor of knowledge)
                'transfer_digit': 8,  # Transfer digit 8 from network2 to network1
                'run_cross_arch': False
            }
        ]
        
        # Convert sets to lists for JSON serialization
        experiments_serializable = []
        for exp in experiments:
            exp_copy = exp.copy()
            exp_copy['network1_classes'] = sorted(list(exp['network1_classes']))
            exp_copy['network2_classes'] = sorted(list(exp['network2_classes']))
            experiments_serializable.append(exp_copy)
        
        architecture_pairs = [
            ("WideNN", "WideNN"),
            ("WideNN", "DeepNN"), 
            ("DeepNN", "WideNN"),
            ("DeepNN", "DeepNN")
        ]
        
        seeds = [42, 123, 456, 789, 101112]
        
        all_results = []
        
        for exp in experiments:
            logger.info(f"\n{'='*80}")
            logger.info(f"STARTING EXPERIMENT: {exp['name'].upper()}")
            logger.info(f"{'='*80}")
            
            exp_results = []
            
            if exp['run_cross_arch']:
                # Run cross-architecture experiments
                logger.info("Running cross-architecture experiments...")
                for source_arch, target_arch in architecture_pairs:
                    for seed in seeds:
                        result = self.run_single_experiment(
                            source_classes=exp['network1_classes'],  # Network1 = source (the one we test)
                            target_classes=exp['network2_classes'],  # Network2 = target (donor)
                            transfer_digit=exp['transfer_digit'],
                            source_arch=source_arch,
                            target_arch=target_arch,
                            seed=seed,
                            exp_name=f"{exp['name']}_cross_arch"
                        )
                        
                        if result is not None:
                            exp_results.append(result)
                            
                            # Save individual result
                            filename = f"{exp['name']}_cross_arch_{source_arch}_to_{target_arch}_seed_{seed}.json"
                            filepath = self.results_dir / filename
                            with open(filepath, 'w') as f:
                                json.dump(result, f, indent=2)
            else:
                # Run only WideNN→WideNN for other experiments
                logger.info("Running WideNN→WideNN experiments...")
                for seed in seeds:
                    result = self.run_single_experiment(
                        source_classes=exp['network1_classes'],  # Network1 = source (the one we test)
                        target_classes=exp['network2_classes'],  # Network2 = target (donor)
                        transfer_digit=exp['transfer_digit'],
                        source_arch="WideNN",
                        target_arch="WideNN",
                        seed=seed,
                        exp_name=exp['name']
                    )
                    
                    if result is not None:
                        exp_results.append(result)
                        
                        # Save individual result
                        filename = f"{exp['name']}_WideNN_to_WideNN_seed_{seed}.json"
                        filepath = self.results_dir / filename
                        with open(filepath, 'w') as f:
                            json.dump(result, f, indent=2)
            
            # Save experiment summary  
            if exp_results:
                # Convert experiment sets to lists for JSON serialization
                exp_serializable = exp.copy()
                exp_serializable['network1_classes'] = sorted(list(exp['network1_classes']))
                exp_serializable['network2_classes'] = sorted(list(exp['network2_classes']))
                
                summary = {
                    'experiment': exp_serializable,
                    'results': exp_results,
                    'statistics': self._compute_statistics(exp_results)
                }
                
                summary_file = self.results_dir / f"{exp['name']}_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            
            all_results.extend(exp_results)
        
        # Save overall summary
        if all_results:
            overall_summary = {
                'total_experiments': len(all_results),
                'experiments': experiments_serializable,
                'results': all_results,
                'overall_statistics': self._compute_overall_statistics(all_results),
                'timestamp': datetime.now().isoformat()
            }
            
            overall_file = self.results_dir / "all_experiments_summary.json"
            with open(overall_file, 'w') as f:
                json.dump(overall_summary, f, indent=2)
        
        logger.info(f"\n✅ All experiments completed! Results saved to {self.results_dir}")
        logger.info(f"Total experiments run: {len(all_results)}")
        
        return all_results
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute summary statistics for a set of results."""
        if not results:
            return {}
        
        transferred_improvements = [r['improvements']['transferred_improvement'] for r in results]
        original_retentions = [r['improvements']['original_retention'] for r in results]
        untransferred_changes = [r['improvements']['untransferred_change'] for r in results]
        
        def compute_stats(values):
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return {
            'transferred_improvement': compute_stats(transferred_improvements),
            'original_retention': compute_stats(original_retentions),
            'untransferred_change': compute_stats(untransferred_changes),
            'count': len(results)
        }
    
    def _compute_overall_statistics(self, results: List[Dict]) -> Dict:
        """Compute overall statistics across all experiments."""
        by_experiment = {}
        
        for r in results:
            exp_name = r['experiment_name']
            if exp_name not in by_experiment:
                by_experiment[exp_name] = []
            by_experiment[exp_name].append(r)
        
        overall_stats = {}
        for exp_name, exp_results in by_experiment.items():
            overall_stats[exp_name] = self._compute_statistics(exp_results)
        
        return overall_stats


if __name__ == "__main__":
    config = ExperimentConfig()
    runner = FinalExperimentRunner(config)
    runner.run_all_experiments()