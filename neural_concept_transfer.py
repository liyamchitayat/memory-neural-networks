"""
Neural Concept Transfer Framework
Implementation of the mathematical framework for cross-architecture concept injection
without retraining, based on vector space alignment and sparse autoencoders.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy.linalg import orthogonal_procrustes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for concept space extraction.
    
    Architecture: ℝᵈ → ℝᶜ → ℝᵈ
    Encoder: E(h) = ReLU(W₂ᵀ ReLU(W₁ᵀh + b₁) + b₂)
    Decoder: D(z) = ReLU(W₄ᵀ ReLU(W₃ᵀz + b₃) + b₄)
    
    Loss: L_SAE = ||h - D(E(h))||₂² + λ||E(h)||₁
    """
    
    def __init__(self, input_dim: int, concept_dim: int = 24, sparsity_weight: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder: input_dim → concept_dim*2 → concept_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU()
        )
        
        # Decoder: concept_dim → concept_dim*2 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, input_dim)
        )
        
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Extract concept representation from features."""
        return self.encoder(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from concept representation."""
        return self.decoder(z)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both reconstruction and concepts."""
        z = self.encode(h)
        h_reconstructed = self.decode(z)
        return h_reconstructed, z
    
    def compute_loss(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SAE loss: L_SAE = ||h - D(E(h))||₂² + λ||E(h)||₁
        """
        h_reconstructed, z = self.forward(h)
        
        # Reconstruction loss
        reconstruction_loss = torch.mean((h - h_reconstructed) ** 2)
        
        # Sparsity loss (L1 regularization)
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = reconstruction_loss + self.sparsity_weight * sparsity_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }
        
        return total_loss, metrics


class OrthogonalProcrustesAligner:
    """
    Orthogonal Procrustes alignment for concept space alignment.
    
    Solves: R* = argmin_R ||C_A - C_B R||_F² subject to RᵀR = I
    Solution: C_B^T C_A = UΣVᵀ (SVD), R* = UVᵀ
    """
    
    def __init__(self):
        self.alignment_matrix = None
        self.alignment_error = None
        
    def fit(self, source_concepts: torch.Tensor, target_concepts: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute optimal orthogonal alignment matrix.
        
        Args:
            source_concepts: [n_shared_classes, concept_dim] from model B
            target_concepts: [n_shared_classes, concept_dim] from model A
            
        Returns:
            alignment_matrix: [concept_dim, concept_dim] orthogonal transformation
            alignment_error: normalized Frobenius norm error
        """
        # Convert to numpy for scipy
        C_B = source_concepts.detach().cpu().numpy()
        C_A = target_concepts.detach().cpu().numpy()
        
        # Orthogonal Procrustes solution
        R, scale = orthogonal_procrustes(C_B, C_A)
        
        # Compute alignment error: ε = ||C_A - C_B R||_F / ||C_A||_F
        aligned_source = C_B @ R
        error_matrix = C_A - aligned_source
        alignment_error = np.linalg.norm(error_matrix, 'fro') / np.linalg.norm(C_A, 'fro')
        
        # Store results
        self.alignment_matrix = torch.tensor(R, dtype=torch.float32)
        self.alignment_error = alignment_error
        
        logger.info(f"Procrustes alignment error: {alignment_error:.4f}")
        
        return self.alignment_matrix, alignment_error
    
    def transform(self, concepts: torch.Tensor) -> torch.Tensor:
        """Apply alignment transformation to concepts."""
        if self.alignment_matrix is None:
            raise ValueError("Must call fit() before transform()")
        
        return torch.mm(concepts, self.alignment_matrix.T)


class FreeSpaceDiscovery:
    """
    Free space discovery via SVD for non-interfering concept injection.
    
    Analysis: U_A = [μ_c^A for c ∈ S_A]ᵀ ∈ ℝ^{|S_A|×c}
    SVD: U_A^T = QΣPᵀ
    Free directions: F = Q[:, -k:] where k corresponds to smallest singular values
    """
    
    def __init__(self):
        self.free_directions = None
        self.used_space_rank = None
        
    def discover_free_space(self, used_concepts: torch.Tensor, concept_dim: int) -> torch.Tensor:
        """
        Discover free directions in concept space using SVD.
        
        Args:
            used_concepts: [n_used_classes, concept_dim] existing concept centroids
            concept_dim: dimension of concept space
            
        Returns:
            free_directions: [concept_dim, k] orthogonal free directions
        """
        # SVD of used concepts
        U, S, V = torch.svd(used_concepts.T)
        
        # Determine number of free dimensions
        # k = min(8, concept_dim // 3) as per framework
        k = min(8, concept_dim // 3)
        
        # Use directions corresponding to smallest singular values
        self.free_directions = U[:, -k:]
        self.used_space_rank = len(S[S > 1e-6])  # Numerical rank
        
        logger.info(f"Used space rank: {self.used_space_rank}, Free dimensions: {k}")
        
        return self.free_directions
    
    def project_to_free_space(self, concept: torch.Tensor) -> torch.Tensor:
        """
        Project concept to free space.
        
        Args:
            concept: [concept_dim] concept to project
            
        Returns:
            projection_coefficients: [k] coefficients for free directions
        """
        if self.free_directions is None:
            raise ValueError("Must call discover_free_space() first")
        
        # α = Fᵀ μ̃_new^B ∈ ℝᵏ
        return torch.mm(self.free_directions.T, concept.unsqueeze(1)).squeeze()


class ConceptDetector(nn.Module):
    """
    Concept detection for determining injection confidence.
    
    Combines spatial and learned detection:
    - Spatial: p_spatial(c_new|z) = max_c∈S_shared exp(-||z - μ_c^A||₂ / σ)
    - Learned: p_learned(c_new|z) = σ(W_det^T ReLU(W_hidden^T z + b_hidden) + b_det)
    - Combined: p(c_new|z) = β p_spatial + (1-β) p_learned
    """
    
    def __init__(self, concept_dim: int, shared_concepts: torch.Tensor, 
                 hidden_dim: int = 16, beta: float = 0.4, sigma: float = 5.0):
        super().__init__()
        self.concept_dim = concept_dim
        self.shared_concepts = nn.Parameter(shared_concepts, requires_grad=False)
        self.beta = beta
        self.sigma = sigma
        
        # Learned detector network
        self.detector = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def spatial_detection(self, z: torch.Tensor) -> torch.Tensor:
        """
        Spatial proximity detection based on shared concepts.
        
        p_spatial(c_new|z) = max_c∈S_shared exp(-||z - μ_c^A||₂ / σ)
        """
        # Compute distances to all shared concepts
        distances = torch.cdist(z.unsqueeze(0), self.shared_concepts.unsqueeze(0)).squeeze(0)
        
        # Spatial probabilities
        spatial_probs = torch.exp(-distances / self.sigma)
        
        # Take maximum over shared concepts
        return torch.max(spatial_probs, dim=1)[0]
    
    def learned_detection(self, z: torch.Tensor) -> torch.Tensor:
        """Learned detection using neural network."""
        return self.detector(z).squeeze(-1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Combined detection confidence.
        
        p(c_new|z) = β p_spatial(c_new|z) + (1-β) p_learned(c_new|z)
        """
        p_spatial = self.spatial_detection(z)
        p_learned = self.learned_detection(z)
        
        return self.beta * p_spatial + (1 - self.beta) * p_learned


class ConceptInjectionModule(nn.Module):
    """
    Non-interfering concept injection module.
    
    Performs free space injection:
    z_enhanced = z + p(c_new|z) Σᵢ₌₁ᵏ αᵢ γ F[:, i]
    """
    
    def __init__(self, concept_dim: int, free_directions: torch.Tensor, 
                 target_concept_projection: torch.Tensor):
        super().__init__()
        self.concept_dim = concept_dim
        self.free_directions = nn.Parameter(free_directions, requires_grad=False)
        self.target_projection = nn.Parameter(target_concept_projection, requires_grad=False)
        
        # Learnable injection strength
        self.injection_strength = nn.Parameter(torch.tensor(1.0))
        
        # Learnable preservation weight
        self.preservation_weight = nn.Parameter(torch.tensor(0.0))  # σ(w_preserve)
        
    def forward(self, z: torch.Tensor, confidence: torch.Tensor, 
                original_features: torch.Tensor) -> torch.Tensor:
        """
        Perform concept injection in free space.
        
        Args:
            z: [batch_size, concept_dim] concept representations
            confidence: [batch_size] injection confidence scores
            original_features: [batch_size, input_dim] original features for blending
            
        Returns:
            enhanced_features: [batch_size, input_dim] enhanced feature representations
        """
        # Free space injection: z_enhanced = z + p(c_new|z) Σᵢ₌₁ᵏ αᵢ γ F[:, i]
        batch_size = z.shape[0]
        
        # Compute injection for each sample
        injection = torch.zeros_like(z)
        for i in range(self.free_directions.shape[1]):
            direction = self.free_directions[:, i]
            strength = self.target_projection[i] * self.injection_strength
            injection += (confidence.unsqueeze(1) * strength * direction.unsqueeze(0))
        
        z_enhanced = z + injection
        
        return z_enhanced


class CrossArchitectureAligner(nn.Module):
    """
    Neural alignment network for cross-architecture transfer.
    
    Φ: ℝᵈᴮ → ℝᵈᴬ
    Φ(h) = ReLU(W₂ᵀ ReLU(W₁ᵀh + b₁) + b₂)
    
    Training objective: L_align = ||E_A(f_A^{(L-1)}(x)) - E_B(Φ(f_B^{(L-1)}(x)))||₂²
    """
    
    def __init__(self, source_dim: int, target_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(source_dim, target_dim)
        
        self.aligner = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
        
    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        """Align source features to target feature space."""
        return self.aligner(source_features)


class OptimizationFramework:
    """
    Multi-objective optimization framework for concept transfer.
    
    L_total = λ₁ L_preservation + λ₂ L_transfer + λ₃ L_confidence
    
    Where:
    - L_preservation = ||f_A(x) - f_A'(x)||₂² for x ∈ S_A
    - L_transfer = -log P(y=c_new|x_new; f_A') for x_new ∈ {c_new}
    - L_confidence = -Σᵢ log p(c_new|E_A(f_A^{(L-1)}(x_new,i)))
    """
    
    def __init__(self, lambda_preservation: float = 0.7, lambda_transfer: float = 0.6, 
                 lambda_confidence: float = 0.1, learning_rate: float = 0.01):
        self.lambda_preservation = lambda_preservation
        self.lambda_transfer = lambda_transfer
        self.lambda_confidence = lambda_confidence
        self.learning_rate = learning_rate
        
    def compute_preservation_loss(self, original_outputs: torch.Tensor, 
                                modified_outputs: torch.Tensor) -> torch.Tensor:
        """L_preservation = ||f_A(x) - f_A'(x)||₂² for x ∈ S_A"""
        return torch.mean((original_outputs - modified_outputs) ** 2)
    
    def compute_transfer_loss(self, transfer_outputs: torch.Tensor, 
                            target_class: int) -> torch.Tensor:
        """L_transfer = -log P(y=c_new|x_new; f_A') for x_new ∈ {c_new}"""
        # Convert to probabilities
        probs = torch.softmax(transfer_outputs, dim=1)
        target_probs = probs[:, target_class]
        return -torch.mean(torch.log(target_probs + 1e-8))
    
    def compute_confidence_loss(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        """L_confidence = -Σᵢ log p(c_new|E_A(f_A^{(L-1)}(x_new,i)))"""
        return -torch.mean(torch.log(confidence_scores + 1e-8))
    
    def compute_total_loss(self, preservation_loss: torch.Tensor, transfer_loss: torch.Tensor,
                          confidence_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total multi-objective loss."""
        total_loss = (self.lambda_preservation * preservation_loss + 
                     self.lambda_transfer * transfer_loss + 
                     self.lambda_confidence * confidence_loss)
        
        metrics = {
            'total_loss': total_loss.item(),
            'preservation_loss': preservation_loss.item(),
            'transfer_loss': transfer_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
        
        return total_loss, metrics


class NeuralConceptTransferSystem:
    """
    Main system class for neural concept transfer across architectures.
    
    This is the most general framework implementing the complete mathematical 
    formulation for transferring concepts between any two neural networks
    with overlapping class sets.
    """
    
    def __init__(self, source_model: nn.Module, target_model: nn.Module,
                 source_classes: Set[int], target_classes: Set[int],
                 concept_dim: int = 24, device: str = 'cpu'):
        """
        Initialize the concept transfer system.
        
        Args:
            source_model: Model B with classes S_B
            target_model: Model A with classes S_A  
            source_classes: Set S_B of classes in source model
            target_classes: Set S_A of classes in target model
            concept_dim: Dimension of concept space
            device: Computation device
        """
        self.source_model = source_model.to(device)
        self.target_model = target_model.to(device)
        self.source_classes = source_classes
        self.target_classes = target_classes
        self.shared_classes = source_classes.intersection(target_classes)
        self.transfer_classes = source_classes - target_classes
        self.concept_dim = concept_dim
        self.device = device
        
        # Validate inputs
        if len(self.shared_classes) == 0:
            raise ValueError("No shared classes found between models (S_shared = ∅)")
        if len(self.transfer_classes) == 0:
            raise ValueError("No classes to transfer (S_B \\ S_A = ∅)")
        
        logger.info(f"Source classes S_B: {self.source_classes}")
        logger.info(f"Target classes S_A: {self.target_classes}")
        logger.info(f"Shared classes S_shared: {self.shared_classes}")
        logger.info(f"Transfer classes: {self.transfer_classes}")
        
        # Initialize components
        self.source_sae = None
        self.target_sae = None
        self.aligner = None
        self.concept_detector = None
        self.injection_module = None
        self.optimization_framework = OptimizationFramework()
        
        # State variables
        self.source_centroids = None
        self.target_centroids = None
        self.alignment_matrix = None
        self.free_directions = None
        self.is_fitted = False
        
    def get_feature_dim(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Determine feature dimension from model's penultimate layer."""
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'get_features'):
                features = model.get_features(sample_input)
            else:
                # Try to extract features from forward pass
                features = model(sample_input)
                if isinstance(features, tuple):
                    features = features[0]  # Assume first output is main features
            return features.shape[-1]
    
    def train_sparse_autoencoders(self, source_loader: torch.utils.data.DataLoader,
                                target_loader: torch.utils.data.DataLoader,
                                epochs: int = 100) -> Tuple[SparseAutoencoder, SparseAutoencoder]:
        """
        Train sparse autoencoders for both models.
        
        Args:
            source_loader: Data loader for source model training
            target_loader: Data loader for target model training  
            epochs: Number of training epochs
            
        Returns:
            Tuple of (source_sae, target_sae)
        """
        # Determine feature dimensions
        sample_batch = next(iter(source_loader))[0][:1].to(self.device)
        source_dim = self.get_feature_dim(self.source_model, sample_batch)
        
        sample_batch = next(iter(target_loader))[0][:1].to(self.device)
        target_dim = self.get_feature_dim(self.target_model, sample_batch)
        
        # Initialize SAEs
        self.source_sae = SparseAutoencoder(source_dim, self.concept_dim).to(self.device)
        self.target_sae = SparseAutoencoder(target_dim, self.concept_dim).to(self.device)
        
        # Train source SAE
        logger.info("Training source SAE...")
        self._train_sae(self.source_sae, self.source_model, source_loader, epochs)
        
        # Train target SAE
        logger.info("Training target SAE...")
        self._train_sae(self.target_sae, self.target_model, target_loader, epochs)
        
        return self.source_sae, self.target_sae
    
    def _train_sae(self, sae: SparseAutoencoder, model: nn.Module, 
                   data_loader: torch.utils.data.DataLoader, epochs: int):
        """Train a single SAE."""
        optimizer = optim.Adam(sae.parameters(), lr=self.optimization_framework.learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    if hasattr(model, 'get_features'):
                        features = model.get_features(data)
                    else:
                        features = model(data)
                        if isinstance(features, tuple):
                            features = features[0]
                
                # Train SAE
                optimizer.zero_grad()
                loss, metrics = sae.compute_loss(features)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"SAE Epoch {epoch}: Loss = {total_loss/len(data_loader):.4f}")
    
    def extract_concept_centroids(self, source_loader: torch.utils.data.DataLoader,
                                target_loader: torch.utils.data.DataLoader) -> Tuple[Dict, Dict]:
        """
        Extract concept centroids for both models.
        
        Returns:
            Tuple of (source_centroids, target_centroids)
        """
        self.source_centroids = self._extract_centroids(
            self.source_model, self.source_sae, source_loader, self.source_classes)
        self.target_centroids = self._extract_centroids(
            self.target_model, self.target_sae, target_loader, self.target_classes)
        
        return self.source_centroids, self.target_centroids
    
    def _extract_centroids(self, model: nn.Module, sae: SparseAutoencoder,
                          data_loader: torch.utils.data.DataLoader, 
                          class_set: Set[int]) -> Dict[int, torch.Tensor]:
        """Extract concept centroids for a single model."""
        model.eval()
        sae.eval()
        
        # Collect features by class
        class_features = {c: [] for c in class_set}
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Get features
                if hasattr(model, 'get_features'):
                    features = model.get_features(data)
                else:
                    features = model(data)
                    if isinstance(features, tuple):
                        features = features[0]
                
                # Group by class
                for i, label in enumerate(labels):
                    if label.item() in class_set:
                        class_features[label.item()].append(features[i])
        
        # Compute centroids
        centroids = {}
        for class_label in class_set:
            if class_features[class_label]:
                features_tensor = torch.stack(class_features[class_label])
                concepts = sae.encode(features_tensor)
                centroids[class_label] = torch.mean(concepts, dim=0)
                logger.info(f"Class {class_label}: {len(class_features[class_label])} samples")
        
        return centroids
    
    def fit_alignment(self) -> float:
        """
        Fit Orthogonal Procrustes alignment using shared classes.
        
        Returns:
            alignment_error: Normalized alignment error
        """
        if self.source_centroids is None or self.target_centroids is None:
            raise ValueError("Must extract centroids first")
        
        # Prepare shared concept matrices
        shared_source = torch.stack([self.source_centroids[c] for c in self.shared_classes])
        shared_target = torch.stack([self.target_centroids[c] for c in self.shared_classes])
        
        # Fit alignment
        aligner = OrthogonalProcrustesAligner()
        self.alignment_matrix, alignment_error = aligner.fit(shared_source, shared_target)
        self.aligner = aligner
        
        return alignment_error
    
    def discover_free_space(self) -> torch.Tensor:
        """
        Discover free space in target concept space.
        
        Returns:
            free_directions: Orthogonal directions for non-interfering injection
        """
        if self.target_centroids is None:
            raise ValueError("Must extract target centroids first")
        
        # Stack used concepts from target model
        used_concepts = torch.stack([self.target_centroids[c] for c in self.target_classes])
        
        # Discover free space
        free_space_discoverer = FreeSpaceDiscovery()
        self.free_directions = free_space_discoverer.discover_free_space(
            used_concepts, self.concept_dim)
        
        return self.free_directions
    
    def setup_injection_system(self, target_class: int, source_loader: torch.utils.data.DataLoader = None, 
                              target_loader: torch.utils.data.DataLoader = None) -> ConceptInjectionModule:
        """
        Setup and train the concept injection system for a specific target class.
        
        Args:
            target_class: Class to transfer (must be in self.transfer_classes)
            source_loader: Source data loader for training injection system
            target_loader: Target data loader for training injection system
            
        Returns:
            Trained injection module
        """
        if target_class not in self.transfer_classes:
            raise ValueError(f"Class {target_class} not in transfer classes")
        
        if (self.alignment_matrix is None or self.free_directions is None or 
            self.source_centroids is None or self.target_centroids is None):
            raise ValueError("Must complete alignment and free space discovery first")
        
        # Get aligned target concept
        source_concept = self.source_centroids[target_class]
        aligned_concept = self.aligner.transform(source_concept.unsqueeze(0)).squeeze(0)
        
        # Project to free space
        free_space_discoverer = FreeSpaceDiscovery()
        free_space_discoverer.free_directions = self.free_directions
        target_projection = free_space_discoverer.project_to_free_space(aligned_concept)
        
        # Setup concept detector
        shared_centroids = torch.stack([self.target_centroids[c] for c in self.shared_classes])
        self.concept_detector = ConceptDetector(self.concept_dim, shared_centroids).to(self.device)
        
        # Setup injection module
        self.injection_module = ConceptInjectionModule(
            self.concept_dim, self.free_directions, target_projection).to(self.device)
        
        # Train the injection system if data loaders provided
        if source_loader is not None and target_loader is not None:
            logger.info("Training concept injection system...")
            self._train_injection_system(target_class, source_loader, target_loader)
            
            # CRITICAL: Adapt target model's final layer for the transfer class
            logger.info(f"Adapting target model final layer for class {target_class}")
            self._adapt_target_final_layer(target_class, source_loader)
        
        return self.injection_module
    
    def _train_injection_system(self, target_class: int, source_loader: torch.utils.data.DataLoader, 
                               target_loader: torch.utils.data.DataLoader, training_steps: int = 50):
        """
        Train the concept injection system using multi-objective optimization.
        
        Args:
            target_class: Class being transferred
            source_loader: Source model data loader
            target_loader: Target model data loader
            training_steps: Number of optimization steps
        """
        # Setup optimizers
        injection_params = list(self.concept_detector.parameters()) + list(self.injection_module.parameters())
        optimizer = optim.Adam(injection_params, lr=self.optimization_framework.learning_rate)
        
        # Training loop
        for step in range(training_steps):
            # Sample batches
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
            
            transfer_data = source_data[transfer_mask]
            preservation_data = target_data[preservation_mask]
            preservation_labels = target_labels[preservation_mask]
            
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Transfer loss: optimize for target class recognition
            if len(transfer_data) > 0:
                transfer_outputs = self.transfer_concept(transfer_data, target_class)
                if transfer_outputs is not None:
                    target_class_labels = torch.full((transfer_data.shape[0],), target_class, device=self.device)
                    transfer_loss = self.optimization_framework.compute_transfer_loss(transfer_outputs, target_class)
                    
                    # Confidence loss: encourage high confidence for transfer samples
                    transfer_features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                    transfer_concepts = self.target_sae.encode(transfer_features)
                    confidence_scores = self.concept_detector(transfer_concepts)
                    confidence_loss = self.optimization_framework.compute_confidence_loss(confidence_scores)
                    
                    total_loss += self.optimization_framework.lambda_transfer * transfer_loss
                    total_loss += self.optimization_framework.lambda_confidence * confidence_loss
            
            # Preservation loss: maintain original performance
            if len(preservation_data) > 0:
                preservation_data_flat = preservation_data.view(preservation_data.size(0), -1)
                original_outputs = self.target_model(preservation_data_flat)
                enhanced_outputs = self.transfer_concept(preservation_data, target_class)
                
                if enhanced_outputs is not None:
                    preservation_loss = self.optimization_framework.compute_preservation_loss(
                        original_outputs, enhanced_outputs)
                    total_loss += self.optimization_framework.lambda_preservation * preservation_loss
            
            # Backpropagation
            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
            
            # Logging
            if step % 10 == 0 and total_loss > 0:
                logger.info(f"Injection training step {step}: Loss = {total_loss.item():.4f}")
        
        logger.info("✓ Concept injection system training completed")
    
    def _adapt_target_final_layer(self, target_class: int, source_loader: torch.utils.data.DataLoader):
        """
        CRITICAL: Adapt the target model's final layer to recognize the transfer class.
        This is essential because concept injection alone cannot work if the final
        classification layer has never been trained on the transfer class.
        """
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
            logger.warning("Could not find final classification layer - adaptation skipped")
            return
        
        # Create adaptation dataset from enhanced features
        adaptation_features = []
        adaptation_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(source_loader):
                if batch_idx >= 5:  # Limit to prevent overfitting
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                mask = (labels == target_class)
                if mask.sum() == 0:
                    continue
                    
                transfer_data = data[mask][:8]  # Max 8 samples per batch
                
                # Get enhanced features through our transfer pipeline
                features = self.target_model.get_features(transfer_data.view(transfer_data.size(0), -1))
                concepts = self.target_sae.encode(features)
                confidence = torch.ones(features.shape[0], device=self.device) * 0.8
                enhanced_concepts = self.injection_module(concepts, confidence, features)
                enhanced_features = self.target_sae.decode(enhanced_concepts)
                
                adaptation_features.append(enhanced_features)
                adaptation_labels.append(torch.full((enhanced_features.shape[0],), target_class, device=self.device))
        
        if not adaptation_features:
            logger.warning(f"No adaptation data found for class {target_class}")
            return
        
        adaptation_features = torch.cat(adaptation_features, dim=0)
        adaptation_labels = torch.cat(adaptation_labels, dim=0)
        
        # Fine-tune only the final layer for the transfer class
        final_layer_optimizer = optim.Adam([final_layer.weight, final_layer.bias], lr=0.01)
        
        for step in range(30):  # Limited adaptation steps
            final_layer_optimizer.zero_grad()
            
            outputs = final_layer(adaptation_features)
            loss = nn.functional.cross_entropy(outputs, adaptation_labels)
            loss.backward()
            final_layer_optimizer.step()
            
            if step % 10 == 0:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == target_class).float().mean()
                    logger.info(f"Final layer adaptation step {step}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")
        
        logger.info(f"✓ Target model final layer adapted for class {target_class}")
    
    def transfer_concept(self, input_data: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Perform concept transfer on input data.
        
        Args:
            input_data: Input data to process
            target_class: Class to inject
            
        Returns:
            Enhanced model outputs with injected concept
        """
        if not self.is_fitted:
            raise ValueError("System not fitted. Call fit() first")
        
        self.target_model.eval()
        self.target_sae.eval()
        
        # Allow gradients during training, no_grad during inference
        context_manager = torch.no_grad() if not (self.concept_detector.training or self.injection_module.training) else torch.enable_grad()
        
        with context_manager:
            # Get original features and outputs
            if hasattr(self.target_model, 'get_features'):
                original_features = self.target_model.get_features(input_data)
            else:
                original_features = self.target_model(input_data)
                if isinstance(original_features, tuple):
                    original_features = original_features[0]
            
            # Encode to concept space
            concepts = self.target_sae.encode(original_features)
            
            # Detect injection confidence
            confidence = self.concept_detector(concepts)
            
            # Perform injection
            enhanced_concepts = self.injection_module(concepts, confidence, original_features)
            
            # Decode back to feature space
            enhanced_features = self.target_sae.decode(enhanced_concepts)
            
            # Blend with original (preservation)
            rho = torch.sigmoid(self.injection_module.preservation_weight)
            final_features = rho * original_features + (1 - rho) * enhanced_features
            
            # Get final outputs through remaining layers
            if hasattr(self.target_model, 'classify_from_features'):
                return self.target_model.classify_from_features(final_features)
            else:
                # Fallback: try to run the full model on enhanced data
                # This assumes we can reconstruct input from features
                logger.warning("Using fallback classification method")
                return self.target_model(final_features)
    
    def fit(self, source_loader: torch.utils.data.DataLoader,
            target_loader: torch.utils.data.DataLoader,
            sae_epochs: int = 100) -> Dict[str, float]:
        """
        Fit the complete concept transfer system.
        
        Args:
            source_loader: Source model data loader
            target_loader: Target model data loader
            sae_epochs: Number of epochs for SAE training
            
        Returns:
            Fitting metrics and statistics
        """
        logger.info("=== Fitting Neural Concept Transfer System ===")
        
        # Step 1: Train SAEs
        logger.info("Step 1: Training Sparse Autoencoders")
        self.train_sparse_autoencoders(source_loader, target_loader, sae_epochs)
        
        # Step 2: Extract centroids
        logger.info("Step 2: Extracting concept centroids")
        self.extract_concept_centroids(source_loader, target_loader)
        
        # Step 3: Fit alignment
        logger.info("Step 3: Fitting Orthogonal Procrustes alignment")
        alignment_error = self.fit_alignment()
        
        # Step 4: Discover free space
        logger.info("Step 4: Discovering free space")
        self.discover_free_space()
        
        self.is_fitted = True
        
        metrics = {
            'alignment_error': alignment_error,
            'concept_dim': self.concept_dim,
            'n_shared_classes': len(self.shared_classes),
            'n_transfer_classes': len(self.transfer_classes),
            'free_dimensions': self.free_directions.shape[1] if self.free_directions is not None else 0
        }
        
        logger.info(f"System fitted successfully: {metrics}")
        return metrics


def extract_concept_centroids(model: nn.Module, sae: SparseAutoencoder, 
                            data_loader: torch.utils.data.DataLoader, 
                            class_set: Set[int], device: str = 'cpu') -> Dict[int, torch.Tensor]:
    """
    Standalone function to extract concept centroids for each class.
    
    μ_c = (1/|S_c|) Σ_{x∈S_c} E(f^{(L-1)}(x))
    
    Args:
        model: neural network model
        sae: trained sparse autoencoder
        data_loader: data loader with labeled samples
        class_set: set of class labels to extract centroids for
        device: computation device
        
    Returns:
        centroids: dict mapping class label to concept centroid
    """
    model.eval()
    sae.eval()
    
    # Collect features by class
    class_features = {c: [] for c in class_set}
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Get penultimate layer features
            if hasattr(model, 'get_features'):
                features = model.get_features(data)
            else:
                features = model(data)
                if isinstance(features, tuple):
                    features = features[0]
            
            # Group by class
            for i, label in enumerate(labels):
                if label.item() in class_set:
                    class_features[label.item()].append(features[i])
    
    # Compute centroids
    centroids = {}
    for class_label in class_set:
        if class_features[class_label]:
            # Stack features and encode to concept space
            features_tensor = torch.stack(class_features[class_label])
            concepts = sae.encode(features_tensor)
            centroids[class_label] = torch.mean(concepts, dim=0)
            logger.info(f"Class {class_label}: {len(class_features[class_label])} samples")
        else:
            logger.warning(f"No samples found for class {class_label}")
    
    return centroids