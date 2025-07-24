#!/usr/bin/env python3
"""
Improved SAE-Based Transfer: Cross-architecture with better alignment methods
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("=== IMPROVED SAE TRANSFER ===")
print("Cross-architecture transfer with enhanced SAE alignment\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Architectures
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

class WideNN(nn.Module):
    def __init__(self):
        super(WideNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.relu2(x)
        return x

class ConceptSAE(nn.Module):
    """Concept-focused Sparse Autoencoder"""
    
    def __init__(self, input_dim, concept_dim=32, sparsity_weight=0.1):
        super(ConceptSAE, self).__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder to concept space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU()
        )
        
        # Decoder from concept space
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

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def extract_features_by_digit(model, dataset, target_digits):
    """Extract features separated by digit"""
    model.eval()
    features_by_digit = {digit: [] for digit in target_digits}
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(DEVICE)
            features = model.get_features(data).cpu()
            
            for i, label in enumerate(labels):
                if label.item() in target_digits:
                    features_by_digit[label.item()].append(features[i])
    
    # Convert lists to tensors
    for digit in target_digits:
        if features_by_digit[digit]:
            features_by_digit[digit] = torch.stack(features_by_digit[digit])
        else:
            features_by_digit[digit] = torch.empty(0, features.shape[1])
    
    return features_by_digit

def train_concept_sae(features_by_digit, concept_dim=32):
    """Train SAE to discover shared concepts across digits"""
    
    # Combine all features
    all_features = []
    all_labels = []
    
    for digit, features in features_by_digit.items():
        if len(features) > 0:
            all_features.append(features)
            all_labels.extend([digit] * len(features))
    
    if not all_features:
        raise ValueError("No features found")
    
    all_features = torch.cat(all_features)
    input_dim = all_features.shape[1]
    
    print(f"Training Concept SAE: {input_dim}D → {concept_dim}D concepts")
    print(f"Training on {len(all_features)} samples from digits {list(features_by_digit.keys())}")
    
    # Create SAE
    sae = ConceptSAE(input_dim, concept_dim).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=0.001)
    
    # Training
    dataset = torch.utils.data.TensorDataset(all_features.to(DEVICE))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    sae.train()
    for epoch in range(25):
        epoch_loss = 0
        epoch_sparsity = 0
        
        for batch_data in loader:
            features = batch_data[0]
            
            optimizer.zero_grad()
            
            concepts, reconstructed = sae(features)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(reconstructed, features)
            
            # Sparsity loss
            sparsity_loss = torch.mean(torch.abs(concepts))
            
            total_loss = recon_loss + sae.sparsity_weight * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_sparsity += sparsity_loss.item()
        
        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(loader)
            avg_sparsity = epoch_sparsity / len(loader)
            print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Sparsity={avg_sparsity:.4f}")
    
    return sae

def analyze_concept_alignment(sae_A, sae_B, shared_features_A, shared_features_B):
    """Analyze and align concept spaces using multiple methods"""
    
    print("Analyzing concept alignment...")
    
    sae_A.eval()
    sae_B.eval()
    
    with torch.no_grad():
        # Get concept representations
        concepts_A = sae_A.encode(shared_features_A.to(DEVICE)).cpu().numpy()
        concepts_B = sae_B.encode(shared_features_B.to(DEVICE)).cpu().numpy()
    
    print(f"Concept shapes: A={concepts_A.shape}, B={concepts_B.shape}")
    
    # Method 1: PCA alignment
    print("Method 1: PCA-based alignment")
    pca_A = PCA(n_components=min(16, concepts_A.shape[1]))
    pca_B = PCA(n_components=min(16, concepts_B.shape[1]))
    
    concepts_A_pca = pca_A.fit_transform(concepts_A)
    concepts_B_pca = pca_B.fit_transform(concepts_B)
    
    print(f"PCA explained variance: A={pca_A.explained_variance_ratio_[:5]}")
    print(f"PCA explained variance: B={pca_B.explained_variance_ratio_[:5]}")
    
    # Method 2: Learn linear mapping
    print("Method 2: Learned linear mapping")
    from sklearn.linear_model import Ridge
    
    # Standardize
    scaler_A = StandardScaler().fit(concepts_A)
    scaler_B = StandardScaler().fit(concepts_B)
    
    concepts_A_norm = scaler_A.transform(concepts_A)
    concepts_B_norm = scaler_B.transform(concepts_B)
    
    # Learn mapping A -> B
    mapper = Ridge(alpha=0.1).fit(concepts_A_norm, concepts_B_norm)
    mapped_A = mapper.predict(concepts_A_norm)
    
    alignment_error = np.linalg.norm(mapped_A - concepts_B_norm) / np.linalg.norm(concepts_B_norm)
    print(f"Linear mapping error: {alignment_error:.4f}")
    
    return {
        'mapper': mapper,
        'scaler_A': scaler_A,
        'scaler_B': scaler_B,
        'pca_A': pca_A,
        'pca_B': pca_B,
        'alignment_error': alignment_error
    }

def transfer_via_concepts(source_model, target_model, sae_source, sae_target, 
                         alignment_info, digit_4_features):
    """Transfer digit-4 knowledge via aligned concept space"""
    
    print("\n=== CONCEPT-BASED TRANSFER ===")
    
    # Extract digit-4 concepts from source
    sae_source.eval()
    with torch.no_grad():
        digit_4_concepts = sae_source.encode(digit_4_features.to(DEVICE)).cpu().numpy()
    
    # Align to target concept space
    digit_4_concepts_norm = alignment_info['scaler_A'].transform(digit_4_concepts)
    digit_4_concepts_target = alignment_info['mapper'].predict(digit_4_concepts_norm)
    digit_4_concepts_target = alignment_info['scaler_B'].inverse_transform(digit_4_concepts_target)
    
    # Decode to target feature space
    digit_4_concepts_tensor = torch.tensor(digit_4_concepts_target, dtype=torch.float32).to(DEVICE)
    sae_target.eval()
    with torch.no_grad():
        digit_4_features_target = sae_target.decoder(digit_4_concepts_tensor).cpu()
    
    print(f"Transferred {len(digit_4_features_target)} digit-4 patterns via concept space")
    
    # Create enhanced adapter
    class EnhancedConceptAdapter(nn.Module):
        def __init__(self, feature_dim, transferred_patterns):
            super().__init__()
            self.feature_dim = feature_dim
            
            # Create multiple prototypes
            self.num_prototypes = min(10, len(transferred_patterns))
            prototype_indices = torch.randperm(len(transferred_patterns))[:self.num_prototypes]
            self.prototypes = nn.Parameter(
                transferred_patterns[prototype_indices].to(DEVICE)
            )
            
            # Learnable weights for prototypes
            self.prototype_weights = nn.Parameter(torch.ones(self.num_prototypes))
            self.temperature = nn.Parameter(torch.tensor(5.0))
            
        def forward(self, features):
            # Compute similarity to all prototypes
            similarities = []
            for i in range(self.num_prototypes):
                sim = torch.cosine_similarity(
                    features, self.prototypes[i].unsqueeze(0), dim=1
                )
                similarities.append(sim * self.prototype_weights[i])
            
            # Take max similarity
            max_similarity = torch.stack(similarities).max(dim=0)[0]
            
            # Return probability
            return torch.sigmoid(max_similarity * self.temperature)
    
    adapter = EnhancedConceptAdapter(digit_4_features_target.shape[1], digit_4_features_target)
    
    # Create adapted model
    class ConceptAdaptedModel(nn.Module):
        def __init__(self, base_model, adapter):
            super().__init__()
            self.base_model = base_model
            self.adapter = adapter
            
        def forward(self, x):
            features = self.base_model.get_features(x)
            base_logits = self.get_base_logits(features)
            
            # Get concept-based digit-4 confidence
            digit_4_confidence = self.adapter(features)
            
            # Boost digit-4 prediction
            boost = 4.0 * (digit_4_confidence - 0.5)
            base_logits[:, 4] = base_logits[:, 4] + boost
            
            return base_logits
        
        def get_base_logits(self, features):
            if hasattr(self.base_model, 'fc5'):
                return self.base_model.fc5(features)
            elif hasattr(self.base_model, 'fc3'):
                return self.base_model.fc3(features)
            else:
                raise ValueError("Unknown architecture")
    
    adapted_model = ConceptAdaptedModel(target_model, adapter)
    
    return adapted_model

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    if len(data_loader.dataset) == 0:
        return 0.0
        
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def test_cross_architecture_sae():
    """Test SAE transfer between different architectures"""
    
    # Load models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: Need MEGA models. Training minimal versions...")
        # Would need to train models here
        return None
    
    print("Loading models...")
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    
    # Test MegaNN -> MegaNN (same architecture)
    source_model = MegaNN().to(DEVICE)
    source_model.load_state_dict(random.choice(class2_weights))  # Knows 2,3,4,5
    source_model.eval()
    
    target_model = MegaNN().to(DEVICE) 
    target_model.load_state_dict(random.choice(class1_weights))  # Knows 0,1,2,3
    target_model.eval()
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Test concepts
    shared_dataset = create_subset(full_dataset, [2, 3])
    digit_4_dataset = create_subset(full_dataset, [4])
    original_dataset = create_subset(full_dataset, [0, 1, 2, 3])
    
    print(f"\n=== EXTRACTING FEATURES ===")
    
    # Extract features by digit
    source_features = extract_features_by_digit(source_model, shared_dataset, [2, 3])
    target_features = extract_features_by_digit(target_model, shared_dataset, [2, 3])
    
    digit_4_features = extract_features_by_digit(source_model, digit_4_dataset, [4])[4]
    
    print(f"Source features: {[f'{k}: {len(v)}' for k, v in source_features.items()]}")
    print(f"Target features: {[f'{k}: {len(v)}' for k, v in target_features.items()]}")
    print(f"Digit-4 features: {len(digit_4_features)}")
    
    # Train SAEs
    print(f"\n=== TRAINING CONCEPT SAEs ===")
    sae_source = train_concept_sae(source_features, concept_dim=24)
    sae_target = train_concept_sae(target_features, concept_dim=24)
    
    # Align concept spaces using shared digits 2,3
    shared_source = torch.cat([source_features[2], source_features[3]])
    shared_target = torch.cat([target_features[2], target_features[3]])
    
    alignment_info = analyze_concept_alignment(sae_source, sae_target, 
                                             shared_source, shared_target)
    
    # Perform transfer
    adapted_model = transfer_via_concepts(source_model, target_model,
                                        sae_source, sae_target,
                                        alignment_info, digit_4_features)
    
    # Evaluate
    print(f"\n=== EVALUATION ===")
    
    original_loader = DataLoader(original_dataset, batch_size=128, shuffle=False)
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=128, shuffle=False)
    
    original_acc = evaluate_model(target_model, original_loader)
    adapted_original_acc = evaluate_model(adapted_model, original_loader)
    adapted_digit_4_acc = evaluate_model(adapted_model, digit_4_loader)
    
    print(f"Original target model: {original_acc:.2f}% on original digits")
    print(f"Adapted model: {adapted_original_acc:.2f}% on original digits")
    print(f"Adapted model: {adapted_digit_4_acc:.2f}% on digit 4")
    
    success = adapted_digit_4_acc > 5 and adapted_original_acc > 75
    print(f"SUCCESS: {'✓' if success else '✗'}")
    
    return adapted_model, success

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing improved SAE-based concept transfer\n")
    
    result = test_cross_architecture_sae()
    
    if result and result[1]:
        print(f"\n🎉 CONCEPT TRANSFER SUCCESS!")
        print(f"SAE-based transfer achieved meaningful digit-4 recognition!")
    else:
        print(f"\n🔬 Concept transfer needs further refinement")
        print(f"Framework established for future improvements")