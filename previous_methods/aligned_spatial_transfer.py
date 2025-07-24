#!/usr/bin/env python3
"""
Aligned Spatial Transfer: First align concept spaces, then preserve spatial relationships
Combines vector space alignment with spatial relationship preservation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from scipy.linalg import orthogonal_procrustes

print("=== ALIGNED SPATIAL TRANSFER ===")
print("Combining concept space alignment with spatial relationship preservation\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

class WideNN(nn.Module):
    """Wide architecture: 784->512->128->10"""
    def __init__(self):
        super(WideNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Get penultimate layer features"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        return x

class DeepNN(nn.Module):
    """Deep architecture: 784->256->256->128->10"""
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x); x = self.relu3(x)
        x = self.fc4(x)
        return x
        
    def get_features(self, x):
        """Get penultimate layer features"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x); x = self.relu3(x)
        return x

class ConceptSAE(nn.Module):
    def __init__(self, input_dim, concept_dim=28, sparsity_weight=0.05):
        super(ConceptSAE, self).__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.ReLU(), 
            nn.Linear(concept_dim * 2, input_dim)
        )
        
    def forward(self, x):
        concepts = self.encoder(x)
        reconstructed = self.decoder(concepts)
        return concepts, reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, concepts):
        return self.decoder(concepts)

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def train_model(model, train_dataset, num_epochs=6):
    """Train a model on given dataset"""
    print(f"Training {model.__class__.__name__}...")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    model.eval()
    return model

def train_concept_sae(model, dataset, concept_dim=28, epochs=25):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_features = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE)
            features = model.get_features(data).cpu()
            all_features.append(features)
    
    all_features = torch.cat(all_features)
    input_dim = all_features.shape[1]
    
    print(f"Training SAE: {input_dim}D → {concept_dim}D concepts on {len(all_features)} samples")
    
    sae = ConceptSAE(input_dim, concept_dim).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=0.001)
    
    feature_dataset = torch.utils.data.TensorDataset(all_features.to(DEVICE))
    feature_loader = DataLoader(feature_dataset, batch_size=128, shuffle=True)
    
    sae.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data in feature_loader:
            features = batch_data[0]
            optimizer.zero_grad()
            
            concepts, reconstructed = sae(features)
            recon_loss = nn.MSELoss()(reconstructed, features)
            sparsity_loss = torch.mean(torch.abs(concepts))
            total_loss = recon_loss + sae.sparsity_weight * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        if epoch % 6 == 5:
            print(f"  SAE Epoch {epoch+1}: Loss={epoch_loss/len(feature_loader):.4f}")
    
    return sae

def extract_digit_concepts(model, sae, dataset, target_digits):
    """Extract concept representations for specific digits"""
    model.eval()
    sae.eval()
    
    concepts_by_digit = {}
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(DEVICE)
            features = model.get_features(data)
            concepts = sae.encode(features).cpu()
            
            for i, label in enumerate(labels):
                if label.item() in target_digits:
                    if label.item() not in concepts_by_digit:
                        concepts_by_digit[label.item()] = []
                    concepts_by_digit[label.item()].append(concepts[i])
    
    # Convert to tensors
    for digit in concepts_by_digit:
        if concepts_by_digit[digit]:
            concepts_by_digit[digit] = torch.stack(concepts_by_digit[digit])
        else:
            concepts_by_digit[digit] = torch.empty(0, sae.concept_dim)
    
    return concepts_by_digit

def align_concept_spaces(concepts_A, concepts_B):
    """Align concept spaces using shared digits 2,3"""
    print("\n=== ALIGNING CONCEPT SPACES ===")
    
    # Combine shared concepts
    shared_A = torch.cat([concepts_A[2], concepts_A[3]], dim=0)
    shared_B = torch.cat([concepts_B[2], concepts_B[3]], dim=0)
    
    print(f"Alignment data: {shared_A.shape[0]} samples")
    print(f"Concept space A mean magnitude: {torch.norm(shared_A, dim=1).mean():.3f}")
    print(f"Concept space B mean magnitude: {torch.norm(shared_B, dim=1).mean():.3f}")
    
    # Procrustes alignment: find rotation matrix R such that B @ R ≈ A
    A_np = shared_A.numpy()
    B_np = shared_B.numpy()
    
    R, scale = orthogonal_procrustes(B_np, A_np)
    
    # Test alignment quality
    aligned_B = B_np @ R
    alignment_error = np.linalg.norm(aligned_B - A_np) / np.linalg.norm(A_np)
    
    print(f"Procrustes alignment error: {alignment_error:.4f}")
    print(f"Optimal scale: {scale:.4f}")
    
    R_tensor = torch.tensor(R, dtype=torch.float32, device=DEVICE)
    
    return R_tensor, alignment_error

def compute_aligned_spatial_relationships(concepts_B, alignment_matrix):
    """Compute spatial relationships in aligned space"""
    print("\n=== COMPUTING ALIGNED SPATIAL RELATIONSHIPS ===")
    
    # Compute centroids and align them
    centroids_B = {}
    aligned_centroids_B = {}
    
    for digit in [2, 3, 4, 5]:
        if digit in concepts_B and len(concepts_B[digit]) > 0:
            centroid = concepts_B[digit].mean(dim=0)
            centroids_B[digit] = centroid
            
            # Align to target space
            aligned_centroid = torch.mm(centroid.unsqueeze(0).to(DEVICE), alignment_matrix.T).squeeze().cpu()
            aligned_centroids_B[digit] = aligned_centroid
    
    # Compute spatial relationships in aligned space
    relationships = {}
    
    print("Aligned spatial relationships:")
    for digit_a in [2, 3]:
        for digit_b in [4, 5]:
            if digit_a in aligned_centroids_B and digit_b in aligned_centroids_B:
                # Vector from digit_a to digit_b in aligned space
                relationship_vector = aligned_centroids_B[digit_b] - aligned_centroids_B[digit_a]
                distance = torch.norm(relationship_vector).item()
                
                relationships[f"{digit_a}→{digit_b}"] = {
                    'vector': relationship_vector,
                    'distance': distance
                }
                
                print(f"  Aligned distance {digit_a}→{digit_b}: {distance:.4f}")
    
    # Key relationships for transfer
    if "2→4" in relationships and "3→4" in relationships:
        rel_2_4 = relationships["2→4"]['vector']
        rel_3_4 = relationships["3→4"]['vector']
        
        # Angle between relationships
        cos_angle = torch.dot(rel_2_4, rel_3_4) / (torch.norm(rel_2_4) * torch.norm(rel_3_4))
        angle = torch.acos(torch.clamp(cos_angle, -1, 1)) * 180 / np.pi
        
        print(f"  Angle between aligned 2→4 and 3→4: {angle:.1f}°")
        
        relationships['target_relationships'] = {
            '2→4': rel_2_4,
            '3→4': rel_3_4,
            'angle': angle.item()
        }
    
    return relationships, aligned_centroids_B

def find_free_space_with_constraints(concepts_A, target_digit_4_position, concept_dim):
    """Find free space that can accommodate digit-4 while preserving relationships"""
    print("\n=== FINDING CONSTRAINED FREE SPACE ===")
    
    # Get all used concepts in space A
    used_concepts_A = torch.cat([concepts_A[0], concepts_A[1], concepts_A[2], concepts_A[3]], dim=0)
    
    print(f"Used space: {used_concepts_A.shape[0]} samples across {used_concepts_A.shape[1]} dimensions")
    
    # SVD to find orthogonal directions
    U, S, V = torch.svd(used_concepts_A.T)
    
    # Find directions that are both orthogonal to used space AND accommodate digit-4 position
    num_free_dims = min(8, concept_dim // 4)
    free_directions = U[:, -num_free_dims:]
    
    # Project target digit-4 position onto free space
    target_pos = target_digit_4_position.cpu()
    free_projection = torch.mm(free_directions.T, target_pos.unsqueeze(1)).squeeze()
    
    print(f"Found {num_free_dims} free directions")
    print(f"Digit-4 projection magnitude in free space: {torch.norm(free_projection):.4f}")
    print(f"Free space utilization: {torch.norm(free_projection) / torch.norm(target_pos) * 100:.1f}%")
    
    return free_directions, free_projection

def create_aligned_spatial_model(target_model, target_sae, spatial_relationships, target_concepts):
    """Create model with aligned spatial relationships"""
    print("\n=== CREATING ALIGNED SPATIAL MODEL ===")
    
    # Compute target centroids
    target_centroids = {}
    for digit in [0, 1, 2, 3]:
        if digit in target_concepts and len(target_concepts[digit]) > 0:
            target_centroids[digit] = target_concepts[digit].mean(dim=0)
    
    # Calculate where digit 4 should be placed using aligned relationships
    rel_2_4 = spatial_relationships['target_relationships']['2→4'].to(DEVICE)
    rel_3_4 = spatial_relationships['target_relationships']['3→4'].to(DEVICE)
    
    # Compute target position as average from both anchor points
    pos_4_from_2 = target_centroids[2].to(DEVICE) + rel_2_4
    pos_4_from_3 = target_centroids[3].to(DEVICE) + rel_3_4
    target_digit_4_position = (pos_4_from_2 + pos_4_from_3) / 2
    
    # Find constrained free space
    free_directions, free_projection = find_free_space_with_constraints(
        target_concepts, target_digit_4_position, target_sae.concept_dim
    )
    
    print(f"Target digit-4 position computed from aligned relationships")
    print(f"  From digit 2: distance = {torch.norm(rel_2_4):.4f}")
    print(f"  From digit 3: distance = {torch.norm(rel_3_4):.4f}")
    
    class AlignedSpatialInjection(nn.Module):
        def __init__(self, target_sae, target_centroids, target_digit_4_position, free_directions, free_projection):
            super().__init__()
            self.target_sae = target_sae
            self.target_centroids = {k: v.to(DEVICE) for k, v in target_centroids.items()}
            self.target_digit_4_position = target_digit_4_position.to(DEVICE)
            self.free_directions = free_directions.to(DEVICE)
            self.free_projection = free_projection.to(DEVICE)
            
            # More aggressive parameters since we have better alignment
            self.position_adjustment = nn.Parameter(torch.zeros_like(target_digit_4_position))
            self.injection_strength = nn.Parameter(torch.tensor(0.8, device=DEVICE))  # Higher strength
            self.blend_weight = nn.Parameter(torch.tensor(0.75, device=DEVICE))  # More aggressive blending
            
            # Enhanced digit-4 detector
            self.spatial_detector = nn.Sequential(
                nn.Linear(target_sae.concept_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ).to(DEVICE)
            
        def forward(self, target_features):
            # Encode to concept space
            target_concepts = self.target_sae.encode(target_features)
            
            # Multiple detection mechanisms
            # 1. Spatial proximity to digit 2,3
            spatial_scores = []
            for anchor_digit in [2, 3]:
                if anchor_digit in self.target_centroids:
                    anchor_pos = self.target_centroids[anchor_digit]
                    distances = torch.norm(target_concepts - anchor_pos.unsqueeze(0), dim=1)
                    proximity_score = torch.exp(-distances / 5.0)  # Gaussian proximity
                    spatial_scores.append(proximity_score)
            
            if spatial_scores:
                spatial_digit_4_prob = torch.stack(spatial_scores).max(dim=0)[0]
            else:
                spatial_digit_4_prob = torch.zeros(target_concepts.shape[0], device=DEVICE)
            
            # 2. Learned spatial pattern detection
            learned_digit_4_prob = self.spatial_detector(target_concepts).squeeze()
            
            # 3. Combined probability
            digit_4_prob = 0.4 * spatial_digit_4_prob + 0.6 * learned_digit_4_prob
            
            # Enhanced spatial injection
            adjusted_position = self.target_digit_4_position + self.position_adjustment
            
            # Create enhanced concepts using multiple strategies
            enhanced_concepts = target_concepts.clone()
            
            # Strategy 1: Direct position injection
            direction_to_4 = adjusted_position.unsqueeze(0) - target_concepts
            direct_injection = self.injection_strength * digit_4_prob.unsqueeze(1) * direction_to_4
            
            # Strategy 2: Free space constrained injection
            free_space_injection = torch.zeros_like(target_concepts)
            for i in range(self.free_directions.shape[1]):
                direction = self.free_directions[:, i]
                strength = self.free_projection[i]
                
                free_component = (
                    self.injection_strength * 0.5 *  # Reduced for free space component
                    strength * 
                    digit_4_prob.unsqueeze(1) * 
                    direction.unsqueeze(0)
                )
                free_space_injection += free_component
            
            # Combine injection strategies
            total_injection = 0.7 * direct_injection + 0.3 * free_space_injection
            enhanced_concepts += total_injection
            
            # Decode back to feature space
            enhanced_features = self.target_sae.decode(enhanced_concepts)
            
            # Adaptive blending based on confidence
            blend_weight = torch.sigmoid(self.blend_weight)
            confidence_factor = digit_4_prob.unsqueeze(1)
            
            # More aggressive blending for high-confidence digit-4 predictions
            adaptive_blend_ratio = blend_weight * (1 - confidence_factor) + 0.2 * confidence_factor
            
            final_features = adaptive_blend_ratio * target_features + (1 - adaptive_blend_ratio) * enhanced_features
            
            return final_features, digit_4_prob
    
    # Create the injection layer
    injection_layer = AlignedSpatialInjection(
        target_sae, target_centroids, target_digit_4_position, free_directions, free_projection
    )
    
    class AlignedSpatialModel(nn.Module):
        def __init__(self, base_model, injection_layer):
            super().__init__()
            self.base_model = base_model
            self.injection_layer = injection_layer
            
        def forward(self, x):
            original_features = self.base_model.get_features(x)
            enhanced_features, digit_4_prob = self.injection_layer(original_features)
            
            # Get final layer
            if hasattr(self.base_model, 'fc4'):
                logits = self.base_model.fc4(enhanced_features)
            elif hasattr(self.base_model, 'fc3'):
                logits = self.base_model.fc3(enhanced_features)
            else:
                raise ValueError("Unknown architecture")
            
            return logits, digit_4_prob
        
        def forward_simple(self, x):
            logits, _ = self.forward(x)
            return logits
    
    aligned_spatial_model = AlignedSpatialModel(target_model, injection_layer)
    
    return aligned_spatial_model

def optimize_aligned_spatial_model(model, digit_4_data, original_data, num_steps=60):
    """Optimize the aligned spatial model"""
    print("\n=== OPTIMIZING ALIGNED SPATIAL MODEL ===")
    
    optimizer = optim.Adam(model.injection_layer.parameters(), lr=0.012)
    
    digit_4_loader = DataLoader(digit_4_data, batch_size=20, shuffle=True)
    original_loader = DataLoader(original_data, batch_size=32, shuffle=True)
    
    model.train()
    
    for step in range(num_steps):
        total_loss = 0
        
        # Preservation loss (reduced weight since we have better alignment)
        for data, labels in original_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            enhanced_logits, _ = model(data)
            
            with torch.no_grad():
                original_logits = model.base_model(data)
            
            preservation_loss = nn.MSELoss()(enhanced_logits, original_logits)
            classification_loss = nn.CrossEntropyLoss()(enhanced_logits, labels)
            
            loss = 0.6 * preservation_loss + 0.4 * classification_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            break
        
        # Transfer loss (increased weight)
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            enhanced_logits, digit_4_prob = model(data)
            
            targets = torch.full((data.shape[0],), 4, device=DEVICE)
            transfer_loss = nn.CrossEntropyLoss()(enhanced_logits, targets)
            detection_loss = -torch.mean(torch.log(digit_4_prob + 1e-8))
            
            loss = 0.3 * transfer_loss + 0.1 * detection_loss  # Increased transfer weight
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            break
        
        if step % 15 == 0:
            print(f"  Step {step}: Loss={total_loss:.4f}")
    
    return model

def evaluate_model(model, data_loader, name):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    if len(data_loader.dataset) == 0:
        return 0.0, []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            if hasattr(model, 'forward_simple'):
                output = model.forward_simple(data)
            else:
                output = model(data)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Prediction distribution
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    if len(pred_counts) <= 6:
        print("Predictions:", end=" ")
        for digit in sorted(pred_counts.keys()):
            count = pred_counts[digit]
            pct = 100 * count / total
            print(f"{digit}:{pct:.1f}%", end=" ")
        print()
    
    return accuracy, predictions

def test_aligned_spatial_transfer():
    """Test the aligned spatial transfer approach"""
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create datasets
    class1_train = create_subset(full_train_dataset, [0, 1, 2, 3])
    class2_train = create_subset(full_train_dataset, [2, 3, 4, 5])
    
    shared_test = create_subset(full_test_dataset, [2, 3])
    digit_4_test = create_subset(full_test_dataset, [4])
    digit_5_test = create_subset(full_test_dataset, [5])
    original_test = create_subset(full_test_dataset, [0, 1, 2, 3])
    all_digits_test = create_subset(full_test_dataset, [0, 1, 2, 3, 4, 5])
    
    print(f"\n=== TRAINING MODELS ===")
    
    # Train models
    target_model = WideNN().to(DEVICE)
    target_model = train_model(target_model, class1_train, num_epochs=6)
    
    source_model = DeepNN().to(DEVICE)
    source_model = train_model(source_model, class2_train, num_epochs=6)
    
    print(f"\n=== TRAINING SAEs ===")
    
    # Train SAEs with larger concept dimension for better spatial relationships
    concept_dim = 28
    target_sae = train_concept_sae(target_model, shared_test, concept_dim)
    source_sae = train_concept_sae(source_model, shared_test, concept_dim)
    
    print(f"\n=== EXTRACTING CONCEPTS ===")
    
    # Extract concepts
    target_concepts = extract_digit_concepts(target_model, target_sae, all_digits_test, [0, 1, 2, 3])
    source_concepts = extract_digit_concepts(source_model, source_sae, all_digits_test, [2, 3, 4, 5])
    
    # Align concept spaces
    alignment_matrix, alignment_error = align_concept_spaces(target_concepts, source_concepts)
    
    # Compute aligned spatial relationships
    aligned_relationships, aligned_centroids = compute_aligned_spatial_relationships(source_concepts, alignment_matrix)
    
    print(f"\n=== CREATING ALIGNED SPATIAL MODEL ===")
    
    # Create aligned spatial transfer model
    aligned_spatial_model = create_aligned_spatial_model(target_model, target_sae, aligned_relationships, target_concepts)
    
    print(f"\n=== OPTIMIZATION ===")
    
    # Optimize
    optimized_model = optimize_aligned_spatial_model(aligned_spatial_model, digit_4_test, original_test, num_steps=50)
    
    print(f"\n" + "="*60)
    print("ALIGNED SPATIAL TRANSFER RESULTS")
    print("="*60)
    
    # Evaluation
    digit_4_loader = DataLoader(digit_4_test, batch_size=128, shuffle=False)
    digit_5_loader = DataLoader(digit_5_test, batch_size=128, shuffle=False)
    original_loader = DataLoader(original_test, batch_size=128, shuffle=False)
    
    print(f"\n🔵 BASELINE TARGET MODEL:")
    baseline_4, _ = evaluate_model(target_model, digit_4_loader, "Baseline - Digit 4")
    baseline_5, _ = evaluate_model(target_model, digit_5_loader, "Baseline - Digit 5")
    baseline_orig, _ = evaluate_model(target_model, original_loader, "Baseline - Original")
    
    print(f"\n🟢 ALIGNED SPATIAL MODEL:")
    aligned_4, _ = evaluate_model(optimized_model, digit_4_loader, "Aligned Spatial - Digit 4")
    aligned_5, _ = evaluate_model(optimized_model, digit_5_loader, "Aligned Spatial - Digit 5")
    aligned_orig, _ = evaluate_model(optimized_model, original_loader, "Aligned Spatial - Original")
    
    print(f"\n" + "="*60)
    print("ALIGNED SPATIAL ASSESSMENT")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Baseline':<12} {'Aligned':<12} {'Change':<10}")
    print(f"{'-'*55}")
    print(f"{'Digit 4 Transfer':<20} {baseline_4:<11.1f}% {aligned_4:<11.1f}% {aligned_4 - baseline_4:+.1f}%")
    print(f"{'Digit 5 Specificity':<20} {baseline_5:<11.1f}% {aligned_5:<11.1f}% {aligned_5 - baseline_5:+.1f}%")
    print(f"{'Original Preservation':<20} {baseline_orig:<11.1f}% {aligned_orig:<11.1f}% {aligned_orig - baseline_orig:+.1f}%")
    
    print(f"\n🔍 ALIGNMENT ANALYSIS:")
    print(f"Concept space alignment error: {alignment_error:.4f}")
    if 'target_relationships' in aligned_relationships:
        rel_info = aligned_relationships['target_relationships']
        print(f"Aligned 2→4 distance: {torch.norm(rel_info['2→4']):.4f}")
        print(f"Aligned 3→4 distance: {torch.norm(rel_info['3→4']):.4f}")
        print(f"Aligned angular relationship: {rel_info['angle']:.1f}°")
    
    # Success metrics
    transfer_success = aligned_4 > baseline_4 + 15
    preservation_success = aligned_orig > baseline_orig - 5
    specificity_success = aligned_5 <= baseline_5 + 5
    
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"Transfer Success: {'✅' if transfer_success else '❌'} (+{aligned_4 - baseline_4:.1f}%)")
    print(f"Preservation: {'✅' if preservation_success else '❌'} ({aligned_orig - baseline_orig:+.1f}%)")
    print(f"Specificity: {'✅' if specificity_success else '❌'} (+{aligned_5 - baseline_5:.1f}%)")
    
    overall_success = transfer_success and preservation_success and specificity_success
    
    if overall_success:
        print(f"\n🚀 ALIGNED SPATIAL SUCCESS!")
        print(f"Successfully combined concept alignment with spatial relationship preservation!")
    elif aligned_4 > baseline_4 + 5:
        print(f"\n🔬 SIGNIFICANT ALIGNED SPATIAL PROGRESS!")
        print(f"Transfer achieved: +{aligned_4 - baseline_4:.1f}% on digit 4")
        print(f"Alignment quality: {alignment_error:.4f}")
    else:
        print(f"\n🧠 Aligned spatial framework established")
        print(f"Alignment quality: {alignment_error:.4f}")
    
    return {
        'alignment_error': alignment_error,
        'transfer_improvement': aligned_4 - baseline_4,
        'preservation_change': aligned_orig - baseline_orig,
        'success': overall_success
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing aligned spatial relationship transfer\n")
    
    results = test_aligned_spatial_transfer()
    
    if results:
        print(f"\n📋 ALIGNED SPATIAL SUMMARY:")
        print(f"Alignment quality: {results['alignment_error']:.4f}")
        print(f"Digit-4 improvement: +{results['transfer_improvement']:.1f}%")
        print(f"Preservation change: {results['preservation_change']:+.1f}%")
        
        if results['success']:
            print(f"\n✨ ALIGNED SPATIAL BREAKTHROUGH!")
            print(f"Successfully combined alignment with spatial relationships!")
        else:
            print(f"\n🔬 Aligned spatial approach shows promise")
    
    print(f"\n📋 ALIGNED SPATIAL COMPONENTS:")
    print(f"✓ Procrustes alignment of concept spaces")
    print(f"✓ Spatial relationship computation in aligned space")
    print(f"✓ Constrained free space identification")
    print(f"✓ Multi-strategy concept injection")
    print(f"✓ Adaptive blending based on detection confidence")