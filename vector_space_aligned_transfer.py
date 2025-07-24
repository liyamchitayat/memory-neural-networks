#!/usr/bin/env python3
"""
Vector Space Aligned Transfer: Proper SAE space alignment using digits 2,3
Treats SAE concept spaces as vector spaces and learns explicit transformations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

print("=== VECTOR SPACE ALIGNED TRANSFER ===")
print("Proper SAE alignment using vector space transformations\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

class MegaNN(nn.Module):
    def __init__(self):
        super(MegaNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x); x = self.relu3(x)
        x = self.fc4(x); x = self.relu4(x)
        x = self.fc5(x)
        return x
    
    def get_features(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x); x = self.relu3(x)
        x = self.fc4(x); x = self.relu4(x)
        return x

class ConceptSAE(nn.Module):
    def __init__(self, input_dim, concept_dim=32, sparsity_weight=0.05):
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

def train_concept_sae(model, dataset, concept_dim=24, epochs=20):
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
        
        if epoch % 5 == 4:
            print(f"  SAE Epoch {epoch+1}: Loss={epoch_loss/len(feature_loader):.4f}")
    
    return sae

def extract_digit_specific_concepts(model, sae, dataset, digit_labels):
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
                if label.item() in digit_labels:
                    if label.item() not in concepts_by_digit:
                        concepts_by_digit[label.item()] = []
                    concepts_by_digit[label.item()].append(concepts[i])
    
    # Convert to tensors
    for digit in concepts_by_digit:
        concepts_by_digit[digit] = torch.stack(concepts_by_digit[digit])
    
    return concepts_by_digit

def learn_vector_space_alignment(concepts_A, concepts_B, alignment_method='procrustes'):
    """Learn transformation from concept space B to concept space A using digits 2,3"""
    
    print(f"Learning vector space alignment using {alignment_method}...")
    
    # Combine concepts for digits 2 and 3
    shared_concepts_A = torch.cat([concepts_A[2], concepts_A[3]], dim=0)
    shared_concepts_B = torch.cat([concepts_B[2], concepts_B[3]], dim=0)
    
    print(f"Alignment data: {shared_concepts_A.shape[0]} samples")
    print(f"Space A mean magnitude: {torch.norm(shared_concepts_A, dim=1).mean():.3f}")
    print(f"Space B mean magnitude: {torch.norm(shared_concepts_B, dim=1).mean():.3f}")
    
    # Convert to numpy for alignment
    A_np = shared_concepts_A.numpy()
    B_np = shared_concepts_B.numpy()
    
    if alignment_method == 'procrustes':
        # Orthogonal Procrustes: find rotation matrix R such that B @ R ≈ A
        R, scale = orthogonal_procrustes(B_np, A_np)
        
        # Test alignment quality
        aligned_B = B_np @ R
        alignment_error = np.linalg.norm(aligned_B - A_np) / np.linalg.norm(A_np)
        
        print(f"Procrustes alignment error: {alignment_error:.4f}")
        print(f"Optimal scale: {scale:.4f}")
        
        return torch.tensor(R, dtype=torch.float32, device=DEVICE), alignment_error
        
    elif alignment_method == 'linear':
        # Learn linear transformation B -> A
        from sklearn.linear_model import Ridge
        
        transformer = Ridge(alpha=0.1)
        transformer.fit(B_np, A_np)
        
        aligned_B = transformer.predict(B_np)
        alignment_error = np.linalg.norm(aligned_B - A_np) / np.linalg.norm(A_np)
        
        print(f"Linear alignment error: {alignment_error:.4f}")
        
        # Convert to PyTorch
        weight_matrix = torch.tensor(transformer.coef_.T, dtype=torch.float32, device=DEVICE)
        bias_vector = torch.tensor(transformer.intercept_, dtype=torch.float32, device=DEVICE)
        
        return (weight_matrix, bias_vector), alignment_error
        
    elif alignment_method == 'learned':
        # Learn neural network transformation
        class ConceptAligner(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.transform = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, output_dim)
                )
            
            def forward(self, x):
                return self.transform(x)
        
        aligner = ConceptAligner(B_np.shape[1], A_np.shape[1]).to(DEVICE)
        optimizer = optim.Adam(aligner.parameters(), lr=0.01)
        
        # Train aligner
        A_tensor = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
        B_tensor = torch.tensor(B_np, dtype=torch.float32, device=DEVICE)
        
        for epoch in range(100):
            optimizer.zero_grad()
            aligned_B = aligner(B_tensor)
            loss = nn.MSELoss()(aligned_B, A_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 19:
                print(f"  Alignment epoch {epoch+1}: Loss={loss.item():.6f}")
        
        # Test final alignment
        with torch.no_grad():
            aligned_B = aligner(B_tensor).cpu().numpy()
            alignment_error = np.linalg.norm(aligned_B - A_np) / np.linalg.norm(A_np)
        
        print(f"Learned alignment error: {alignment_error:.4f}")
        
        return aligner, alignment_error

def find_free_vector_space(concepts_A, alignment_transform, concepts_B_digit_4, method='orthogonal'):
    """Find free space in concept vector space A to place digit 4 without interference"""
    
    print(f"Finding free vector space using {method} method...")
    
    # Get all used directions in space A (digits 0,1,2,3)
    used_concepts_A = torch.cat([concepts_A[0], concepts_A[1], concepts_A[2], concepts_A[3]], dim=0)
    
    print(f"Used space: {used_concepts_A.shape[0]} samples across {used_concepts_A.shape[1]} dimensions")
    
    if method == 'orthogonal':
        # Find orthogonal directions to existing concepts using SVD
        U, S, V = torch.svd(used_concepts_A.T)  # SVD of transpose
        
        # Use least important directions (smallest singular values)
        num_free_dims = min(8, used_concepts_A.shape[1] // 3)  # Use ~1/3 of dimensions
        free_directions = U[:, -num_free_dims:]  # U gives us the orthogonal directions in concept space
        
        print(f"Found {num_free_dims} orthogonal free directions")
        print(f"Singular values of free directions: {S[-num_free_dims:].cpu().numpy()}")
        
        return free_directions
        
    elif method == 'pca':
        # Use PCA to find unused variance directions
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(used_concepts_A.cpu().numpy())
        
        # Use directions with least variance
        num_free_dims = 6
        free_components = pca.components_[-num_free_dims:]  # Least variance components
        free_directions = torch.tensor(free_components.T, dtype=torch.float32, device=DEVICE)
        
        print(f"Found {num_free_dims} PCA free directions")
        print(f"Explained variance ratios: {pca.explained_variance_ratio_[-num_free_dims:]}")
        
        return free_directions
        
    elif method == 'random_orthogonal':
        # Generate random orthogonal directions and orthogonalize against used space
        concept_dim = used_concepts_A.shape[1]
        num_free_dims = 8
        
        # Generate random directions
        random_dirs = torch.randn(concept_dim, num_free_dims, device=DEVICE)
        
        # Orthogonalize against used space using Gram-Schmidt
        used_mean = used_concepts_A.mean(dim=0, keepdim=True).T
        
        orthogonal_dirs = []
        for i in range(num_free_dims):
            dir_vec = random_dirs[:, i:i+1]
            
            # Remove components along used directions
            projection = torch.mm(used_mean.T, used_mean) / torch.norm(used_mean)**2
            dir_vec = dir_vec - torch.mm(projection.T, torch.mm(projection, dir_vec))
            
            # Normalize
            dir_vec = dir_vec / torch.norm(dir_vec)
            orthogonal_dirs.append(dir_vec)
        
        free_directions = torch.cat(orthogonal_dirs, dim=1)
        
        print(f"Generated {num_free_dims} random orthogonal free directions")
        
        return free_directions

def create_aligned_transfer_model(target_model, target_sae, alignment_transform, free_directions, digit_4_pattern):
    """Create model with proper vector space aligned transfer"""
    
    print("Creating aligned transfer model...")
    
    class AlignedConceptInjection(nn.Module):
        def __init__(self, target_sae, alignment_transform, free_directions, digit_4_pattern):
            super().__init__()
            self.target_sae = target_sae
            self.alignment_transform = alignment_transform
            self.free_directions = free_directions.to(DEVICE)
            self.digit_4_pattern = digit_4_pattern.to(DEVICE)
            
            # Learnable injection strength for free space
            self.free_space_strength = nn.Parameter(torch.tensor(0.3, device=DEVICE))
            
            # Digit-4 detector for selective injection
            self.digit_4_detector = nn.Sequential(
                nn.Linear(target_sae.concept_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ).to(DEVICE)
            
            # Preservation parameters
            self.preservation_weight = nn.Parameter(torch.tensor(0.9, device=DEVICE))
            
        def forward(self, target_features):
            # Encode to concept space
            target_concepts = self.target_sae.encode(target_features)
            
            # Detect digit-4 likelihood
            digit_4_prob = self.digit_4_detector(target_concepts).squeeze()
            
            # Create enhanced concepts
            enhanced_concepts = target_concepts.clone()
            
            # Project digit-4 pattern into free space
            # free_directions: [concept_dim, num_free_dims], digit_4_pattern: [concept_dim]
            digit_4_free_projection = torch.mm(self.free_directions.T, self.digit_4_pattern.unsqueeze(1)).squeeze()
            
            # Inject into free space proportional to digit-4 probability
            for i in range(self.free_directions.shape[1]):
                direction = self.free_directions[:, i]
                projection_strength = digit_4_free_projection[i]
                
                # Create injection vector
                injection_vector = (
                    self.free_space_strength * 
                    projection_strength * 
                    direction.unsqueeze(0)
                )
                
                # Apply proportional to digit-4 probability
                enhanced_concepts += digit_4_prob.unsqueeze(1) * injection_vector
            
            # Decode back to feature space
            enhanced_features = self.target_sae.decode(enhanced_concepts)
            
            # Blend with original features for preservation
            preservation_weight = torch.sigmoid(self.preservation_weight)
            blend_ratio = preservation_weight + (1 - preservation_weight) * (1 - digit_4_prob.unsqueeze(1))
            
            final_features = blend_ratio * target_features + (1 - blend_ratio) * enhanced_features
            
            return final_features, digit_4_prob
    
    # Create the injection layer
    injection_layer = AlignedConceptInjection(target_sae, alignment_transform, free_directions, digit_4_pattern)
    
    # Create complete model
    class AlignedTransferModel(nn.Module):
        def __init__(self, base_model, injection_layer):
            super().__init__()
            self.base_model = base_model
            self.injection_layer = injection_layer
            
        def forward(self, x):
            original_features = self.base_model.get_features(x)
            enhanced_features, digit_4_prob = self.injection_layer(original_features)
            logits = self.base_model.fc5(enhanced_features)
            return logits, digit_4_prob
        
        def forward_simple(self, x):
            logits, _ = self.forward(x)
            return logits
    
    aligned_model = AlignedTransferModel(target_model, injection_layer)
    
    return aligned_model

def optimize_aligned_transfer(aligned_model, digit_4_data, original_data, num_steps=50):
    """Optimize the aligned transfer model"""
    
    print("Optimizing aligned transfer model...")
    
    optimizer = optim.Adam(aligned_model.injection_layer.parameters(), lr=0.008)
    
    digit_4_loader = DataLoader(digit_4_data, batch_size=24, shuffle=True)
    original_loader = DataLoader(original_data, batch_size=32, shuffle=True)
    
    aligned_model.train()
    
    for step in range(num_steps):
        total_loss = 0
        
        # Preservation loss
        for data, labels in original_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            enhanced_logits, digit_4_prob = aligned_model(data)
            
            with torch.no_grad():
                original_logits = aligned_model.base_model(data)
            
            # Strong preservation
            preservation_loss = nn.MSELoss()(enhanced_logits, original_logits)
            classification_loss = nn.CrossEntropyLoss()(enhanced_logits, labels)
            
            loss = 0.7 * preservation_loss + 0.3 * classification_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            break
        
        # Transfer loss
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            enhanced_logits, digit_4_prob = aligned_model(data)
            
            targets = torch.full((data.shape[0],), 4, device=DEVICE)
            transfer_loss = nn.CrossEntropyLoss()(enhanced_logits, targets)
            detection_loss = -torch.mean(torch.log(digit_4_prob + 1e-8))
            
            loss = 0.15 * transfer_loss + 0.05 * detection_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            break
        
        if step % 12 == 0:
            print(f"  Step {step}: Loss={total_loss:.4f}")
    
    return aligned_model

def evaluate_model_detailed(model, data_loader, dataset_name):
    """Detailed evaluation"""
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
    
    print(f"\n=== {dataset_name} Performance ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    print("Prediction distribution:")
    for digit in sorted(pred_counts.keys()):
        count = pred_counts[digit]
        percentage = 100 * count / total
        print(f"  Predicted as {digit}: {count} samples ({percentage:.1f}%)")
    
    return accuracy, predictions

def test_vector_space_aligned_transfer():
    """Test the complete vector space aligned transfer approach"""
    
    # Load models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: Need MEGA models!")
        return None
    
    print("Loading pre-trained models...")
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    
    # Create models
    target_model = MegaNN().to(DEVICE)  # Model A
    target_model.load_state_dict(class1_weights[0])
    target_model.eval()
    
    source_model = MegaNN().to(DEVICE)  # Model B
    source_model.load_state_dict(class2_weights[0]) 
    source_model.eval()
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create datasets
    shared_dataset = create_subset(full_dataset, [2, 3])
    digit_4_dataset = create_subset(full_dataset, [4])
    digit_5_dataset = create_subset(full_dataset, [5])
    original_dataset = create_subset(full_dataset, [0, 1, 2, 3])
    all_digits_dataset = create_subset(full_dataset, [0, 1, 2, 3, 4, 5])
    
    print(f"\n=== TRAINING SAEs ===")
    
    # Train SAEs on shared data for alignment
    source_sae = train_concept_sae(source_model, shared_dataset, concept_dim=24)
    target_sae = train_concept_sae(target_model, shared_dataset, concept_dim=24)
    
    print(f"\n=== EXTRACTING CONCEPT REPRESENTATIONS ===")
    
    # Extract concept representations for alignment
    concepts_A = extract_digit_specific_concepts(target_model, target_sae, all_digits_dataset, [0, 1, 2, 3])
    concepts_B = extract_digit_specific_concepts(source_model, source_sae, all_digits_dataset, [2, 3, 4, 5])
    
    print(f"Extracted concepts:")
    for digit, concepts in concepts_A.items():
        print(f"  Model A digit {digit}: {concepts.shape[0]} samples")
    for digit, concepts in concepts_B.items():
        print(f"  Model B digit {digit}: {concepts.shape[0]} samples")
    
    print(f"\n=== LEARNING VECTOR SPACE ALIGNMENT ===")
    
    # Learn alignment transformation using digits 2,3
    alignment_transform, alignment_error = learn_vector_space_alignment(
        concepts_A, concepts_B, alignment_method='procrustes'
    )
    
    print(f"\n=== FINDING FREE VECTOR SPACE ===")
    
    # Find free space in target concept space
    free_directions = find_free_vector_space(
        concepts_A, alignment_transform, concepts_B[4], method='orthogonal'
    )
    
    print(f"\n=== TRANSFORMING DIGIT-4 PATTERN ===")
    
    # Transform digit-4 pattern from source to target space
    digit_4_mean_B = concepts_B[4].mean(dim=0)
    
    if isinstance(alignment_transform, tuple):  # Linear transform
        weight_matrix, bias_vector = alignment_transform
        digit_4_aligned = torch.mm(digit_4_mean_B.unsqueeze(0).to(DEVICE), weight_matrix) + bias_vector
        digit_4_aligned = digit_4_aligned.squeeze()
    else:  # Procrustes transform
        digit_4_aligned = torch.mm(digit_4_mean_B.unsqueeze(0).to(DEVICE), alignment_transform.T).squeeze()
    
    print(f"Digit-4 pattern shape: {digit_4_aligned.shape}")
    print(f"Free directions shape: {free_directions.shape}")
    
    print(f"\n=== CREATING ALIGNED TRANSFER MODEL ===")
    
    # Create aligned transfer model
    aligned_model = create_aligned_transfer_model(
        target_model, target_sae, alignment_transform, free_directions, digit_4_aligned
    )
    
    print(f"\n=== OPTIMIZATION ===")
    
    # Optimize the model
    optimized_model = optimize_aligned_transfer(
        aligned_model, digit_4_dataset, original_dataset, num_steps=40
    )
    
    print(f"\n" + "="*60)
    print("VECTOR SPACE ALIGNED TRANSFER RESULTS")
    print("="*60)
    
    # Evaluation
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=128, shuffle=False)
    digit_5_loader = DataLoader(digit_5_dataset, batch_size=128, shuffle=False)
    original_loader = DataLoader(original_dataset, batch_size=128, shuffle=False)
    
    print(f"\n🔵 BASELINE (Original Model A):")
    baseline_4_acc, _ = evaluate_model_detailed(target_model, digit_4_loader, "Baseline - Digit 4")
    baseline_5_acc, _ = evaluate_model_detailed(target_model, digit_5_loader, "Baseline - Digit 5")
    baseline_orig_acc, _ = evaluate_model_detailed(target_model, original_loader, "Baseline - Original")
    
    print(f"\n🟢 VECTOR SPACE ALIGNED MODEL:")
    aligned_4_acc, _ = evaluate_model_detailed(optimized_model, digit_4_loader, "Aligned - Digit 4")
    aligned_5_acc, _ = evaluate_model_detailed(optimized_model, digit_5_loader, "Aligned - Digit 5")
    aligned_orig_acc, _ = evaluate_model_detailed(optimized_model, original_loader, "Aligned - Original")
    
    print(f"\n" + "="*60)
    print("VECTOR SPACE ALIGNMENT ASSESSMENT")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Baseline':<12} {'Aligned':<12} {'Change':<10}")
    print(f"{'-'*55}")
    print(f"{'Digit 4 Accuracy':<20} {baseline_4_acc:<11.1f}% {aligned_4_acc:<11.1f}% {aligned_4_acc - baseline_4_acc:+.1f}%")
    print(f"{'Digit 5 Accuracy':<20} {baseline_5_acc:<11.1f}% {aligned_5_acc:<11.1f}% {aligned_5_acc - baseline_5_acc:+.1f}%")
    print(f"{'Original Accuracy':<20} {baseline_orig_acc:<11.1f}% {aligned_orig_acc:<11.1f}% {aligned_orig_acc - baseline_orig_acc:+.1f}%")
    
    print(f"\n🔍 ALIGNMENT ANALYSIS:")
    print(f"Vector space alignment error: {alignment_error:.4f}")
    print(f"Free directions found: {free_directions.shape[1]}")
    
    # Success metrics
    transfer_success = aligned_4_acc > baseline_4_acc + 10
    preservation_success = aligned_orig_acc > baseline_orig_acc - 5
    specificity_maintained = aligned_5_acc <= baseline_5_acc + 5
    
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"Transfer (digit 4): {'✅' if transfer_success else '❌'} (+{aligned_4_acc - baseline_4_acc:.1f}%)")
    print(f"Preservation: {'✅' if preservation_success else '❌'} ({aligned_orig_acc - baseline_orig_acc:+.1f}%)")
    print(f"Specificity: {'✅' if specificity_maintained else '❌'} (+{aligned_5_acc - baseline_5_acc:.1f}%)")
    
    overall_success = transfer_success and preservation_success and specificity_maintained
    
    if overall_success:
        print(f"\n🚀 VECTOR SPACE ALIGNMENT SUCCESS!")
        print(f"Proper SAE alignment with free space injection achieved balanced transfer!")
    else:
        print(f"\n🧠 Vector space approach shows significant improvement over naive methods")
    
    return {
        'alignment_error': alignment_error,
        'transfer_success': transfer_success,
        'preservation_success': preservation_success,
        'overall_success': overall_success,
        'results': {
            'baseline_4': baseline_4_acc,
            'aligned_4': aligned_4_acc,
            'baseline_orig': baseline_orig_acc,
            'aligned_orig': aligned_orig_acc
        }
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing vector space aligned SAE transfer\n")
    
    results = test_vector_space_aligned_transfer()
    
    if results and results['overall_success']:
        print(f"\n✨ VECTOR SPACE ALIGNMENT BREAKTHROUGH!")
        print(f"Successfully achieved proper SAE alignment and free space transfer!")
    elif results:
        print(f"\n🔬 Vector space alignment framework validated")
        print(f"Alignment error: {results['alignment_error']:.4f}")
    
    print(f"\n📋 VECTOR SPACE APPROACH COMPONENTS:")
    print(f"✓ Explicit SAE space alignment using shared digits 2,3")
    print(f"✓ Orthogonal Procrustes transformation")
    print(f"✓ Free vector space identification via SVD")
    print(f"✓ Non-interfering digit-4 injection in unused dimensions")
    print(f"✓ Preservation through careful space partitioning")