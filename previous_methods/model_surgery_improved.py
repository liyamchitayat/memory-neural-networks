#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import os
from scipy.linalg import orthogonal_procrustes

print("=== IMPROVED MODEL SURGERY WITH MULTIPLE STRATEGIES ===")

# Configuration
NUM_MODELS = 10
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
CLASS1_LABELS = [0, 1, 2, 3]  # Model A
CLASS2_LABELS = [2, 3, 4, 5]  # Model B

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load existing models if available, otherwise train new ones
if os.path.exists('./trained_models/class1_models_weights.pt'):
    print("Loading existing trained models...")
    class1_weights = torch.load('./trained_models/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
else:
    print("Training new models (this will take a few minutes)...")
    # [Include training code from previous version if needed]
    raise FileNotFoundError("Please run the full training first")

# Load MNIST data for surgery
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

# Simple CNN Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)  # Penultimate layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def get_hidden_features(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x

class ProbeNN(nn.Module):
    def __init__(self, feature_dim):
        super(ProbeNN, self).__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.linear(x)

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    if len(data_loader.dataset) == 0:
        return float('nan')
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def get_hidden_activations(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            hidden_feats = model.get_hidden_features(data)
            features.append(hidden_feats.cpu())
    return torch.cat(features)

# Strategy 1: More aggressive probe + direct weight transplantation
def strategy_1_aggressive_transplant():
    print("\n=== STRATEGY 1: Aggressive Transplant ===")
    
    # Load models
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE) 
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Create modified model
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Strategy: Directly transplant Model B's representations for digit 4
        
        # 1. Copy class 4 output weights directly
        modified_model.fc3.weight.data[4] = model_B.fc3.weight.data[4].clone()
        modified_model.fc3.bias.data[4] = model_B.fc3.bias.data[4].clone()
        
        # 2. Find and copy the most digit-4-specific hidden neurons from Model B
        # Get digit 4 examples
        digit_4_dataset = create_subset(full_train_dataset, [4])
        digit_4_loader = DataLoader(digit_4_dataset, batch_size=100, shuffle=False)
        
        # Get activations for digit 4 in both models
        digit_4_features_B = get_hidden_activations(model_B, digit_4_loader)
        digit_4_features_A = get_hidden_activations(modified_model, digit_4_loader)
        
        # Find neurons in Model B that activate strongly for digit 4
        mean_activation_B = digit_4_features_B.mean(dim=0)
        top_neurons = torch.argsort(mean_activation_B, descending=True)[:8]  # Top 8 neurons
        
        print(f"Transplanting top {len(top_neurons)} neurons from Model B")
        
        # Copy fc2 weights for these neurons (input connections to these neurons)
        for neuron_idx in top_neurons:
            modified_model.fc2.weight.data[neuron_idx] = model_B.fc2.weight.data[neuron_idx].clone()
            modified_model.fc2.bias.data[neuron_idx] = model_B.fc2.bias.data[neuron_idx].clone()
    
    return modified_model

# Strategy 2: Feature space alignment + selective neuron copying
def strategy_2_aligned_copy():
    print("\n=== STRATEGY 2: Aligned Feature Copy ===")
    
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE) 
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Get shared digit representations for alignment
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    H_A_shared = get_hidden_activations(model_A, shared_loader)
    H_B_shared = get_hidden_activations(model_B, shared_loader)
    
    # Compute alignment
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    
    # Create modified model
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Align Model B's fc2 layer and copy it
        aligned_fc2_weight = (R.T @ model_B.fc2.weight.data.T).T
        
        # Copy aligned weights for digit 4 pathway
        modified_model.fc3.weight.data[4] = model_B.fc3.weight.data[4] @ R.T
        modified_model.fc3.bias.data[4] = model_B.fc3.bias.data[4]
        
        # Partially update fc2 to support digit 4
        # Find which neurons in fc2 are most important for digit 4 in Model B
        digit_4_importance = torch.abs(model_B.fc3.weight.data[4])
        top_fc2_neurons = torch.argsort(digit_4_importance, descending=True)[:10]
        
        for neuron_idx in top_fc2_neurons:
            modified_model.fc2.weight.data[neuron_idx] = aligned_fc2_weight[neuron_idx]
    
    return modified_model

# Strategy 3: Exact Paper Implementation
def strategy_3_paper_method():
    print("\n=== STRATEGY 3: Exact Paper Method ===")
    
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Step 2: Train linear probe on B for "digit 4 vs not-4"
    probe_dataset = create_subset(full_train_dataset, CLASS2_LABELS)  # digits 2,3,4,5
    probe_data = []
    probe_labels = []
    
    for img, label in probe_dataset:
        probe_data.append(img)
        probe_labels.append(1 if label == 4 else 0)
    
    probe_data = torch.stack(probe_data)
    probe_labels = torch.tensor(probe_labels, dtype=torch.float32).unsqueeze(1)
    
    # Extract features using Model B (frozen)
    model_B.eval()
    with torch.no_grad():
        probe_features = model_B.get_hidden_features(probe_data.to(DEVICE))
    
    # Train probe: z = W4 * h
    probe_net = ProbeNN(32).to(DEVICE)
    probe_criterion = nn.BCEWithLogitsLoss()
    probe_optimizer = optim.Adam(probe_net.parameters(), lr=0.001)
    
    probe_loader = DataLoader(
        torch.utils.data.TensorDataset(probe_features, probe_labels.to(DEVICE)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    for epoch in range(5):
        for data, target in probe_loader:
            probe_optimizer.zero_grad()
            output = probe_net(data)
            loss = probe_criterion(output, target)
            loss.backward()
            probe_optimizer.step()
    
    # Extract W4
    W4 = probe_net.linear.weight.data.clone().detach().squeeze(0)
    
    # Step 3: Align B's hidden basis to A's using Procrustes on shared digits 2,3
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    H_A_shared = get_hidden_activations(model_A, shared_loader)
    H_B_shared = get_hidden_activations(model_B, shared_loader)
    
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    
    # Step 4: Transport the probe W_tilde_4 = R * W4
    W_tilde_4 = (R @ W4.unsqueeze(1)).squeeze(1)
    
    # Step 5: Locate behavior region in A's classifier
    W_clf_A = model_A.fc3.weight.data.clone()
    cosine_sims = torch.nn.functional.cosine_similarity(W_clf_A, W_tilde_4.unsqueeze(0), dim=1)
    
    # Choose bottom-k rows (most negative similarity)
    k_rows = 2
    selected_indices = torch.argsort(cosine_sims)[:k_rows]
    
    # Step 6: Surgical edit - v_i <- v_i + alpha * W_tilde_4
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    alpha = 0.8
    with torch.no_grad():
        for idx in selected_indices:
            modified_model.fc3.weight.data[idx] += alpha * W_tilde_4
    
    # Step 7: Add output weight for class 4 - copy from B
    with torch.no_grad():
        modified_model.fc3.weight.data[4] = model_B.fc3.weight.data[4].clone()
        modified_model.fc3.bias.data[4] = model_B.fc3.bias.data[4].clone()
    
    print(f"  Applied surgical edits to rows: {selected_indices.tolist()}")
    print(f"  Their cosine similarities: {cosine_sims[selected_indices].cpu().numpy()}")
    print(f"  Used alpha = {alpha}")
    
    return modified_model

# Strategy 4: Train a small adapter layer (FIXED)
def strategy_4_adapter():
    print("\n=== STRATEGY 4: Adapter Layer (Fixed) ===")
    
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE) 
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Modified model with adapter that preserves original behavior
    class AdaptedModel(nn.Module):
        def __init__(self, base_model_A, base_model_B):
            super().__init__()
            self.fc1 = base_model_A.fc1
            self.relu1 = base_model_A.relu1
            self.fc2 = base_model_A.fc2
            self.relu2 = base_model_A.relu2
            self.fc3 = nn.Linear(32, 10)  # New classifier layer
            self.adapter = nn.Linear(32, 32)
            
            # Initialize with Model A's classifier to preserve original behavior
            with torch.no_grad():
                self.fc3.weight.data.copy_(base_model_A.fc3.weight.data)
                self.fc3.bias.data.copy_(base_model_A.fc3.bias.data)
                
                # Initialize adapter as identity 
                self.adapter.weight.data.copy_(torch.eye(32))
                self.adapter.bias.data.zero_()
        
        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            features = self.relu2(x)
            
            # Use original features for classes 0,1,2,3,5-9
            original_logits = self.fc3(features)
            
            # Use adapted features only for digit 4
            adapted_features = self.adapter(features)
            digit_4_logit = torch.sum(self.fc3.weight.data[4:5] * adapted_features, dim=1, keepdim=True) + self.fc3.bias.data[4:5]
            
            # Replace only digit 4 logit
            original_logits[:, 4:5] = digit_4_logit
            
            return original_logits
    
    adapted_model = AdaptedModel(model_A, model_B).to(DEVICE)
    
    # Train the adapter on digit 4 examples
    digit_4_dataset = create_subset(full_train_dataset, [4])
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=32, shuffle=True)
    
    # Get target features from Model B for digit 4
    model_B.eval()
    target_features = []
    with torch.no_grad():
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            features_B = model_B.get_hidden_features(data)
            target_features.append(features_B)
        target_features = torch.cat(target_features)
    
    # Train adapter to map A's features to B's features for digit 4
    adapter_optimizer = optim.Adam(adapted_model.adapter.parameters(), lr=0.01)
    
    print("Training adapter...")
    adapted_model.train()
    for epoch in range(10):
        feature_idx = 0
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            batch_size = data.size(0)
            
            # Get A's features
            features_A = adapted_model.fc2(adapted_model.relu1(adapted_model.fc1(data.view(-1, 28*28))))
            features_A = adapted_model.relu2(features_A)
            
            # Adapt them
            adapted_features = adapted_model.adapter(features_A)
            
            # Target is B's features
            target_batch = target_features[feature_idx:feature_idx+batch_size]
            feature_idx += batch_size
            
            # Loss: make adapted features similar to B's features
            loss = nn.MSELoss()(adapted_features, target_batch)
            
            adapter_optimizer.zero_grad()
            loss.backward()
            adapter_optimizer.step()
            
            if feature_idx >= len(target_features):
                break
    
    return adapted_model

# Test all strategies
def test_all_strategies():
    # Create test datasets
    original_digits_test = create_subset(full_test_dataset, CLASS1_LABELS)
    target_digit_test = create_subset(full_test_dataset, [4])
    ooc_digit_test = create_subset(full_test_dataset, [5])
    
    strategies = [
        ("Strategy 1: Aggressive Transplant", strategy_1_aggressive_transplant),
        ("Strategy 2: Aligned Copy", strategy_2_aligned_copy), 
        ("Strategy 3: Exact Paper Method", strategy_3_paper_method),
        ("Strategy 4: Adapter", strategy_4_adapter)
    ]
    
    best_strategy = None
    best_score = 0
    
    for strategy_name, strategy_func in strategies:
        print(f"\nTesting {strategy_name}...")
        
        try:
            modified_model = strategy_func()
            
            # Evaluate
            acc_orig = evaluate_model(modified_model, DataLoader(original_digits_test, BATCH_SIZE))
            acc_4 = evaluate_model(modified_model, DataLoader(target_digit_test, BATCH_SIZE))
            acc_5 = evaluate_model(modified_model, DataLoader(ooc_digit_test, BATCH_SIZE))
            
            print(f"  Digits 0,1,2,3: {acc_orig:.2f}%")
            print(f"  Digit 4: {acc_4:.2f}%")
            print(f"  Digit 5: {acc_5:.2f}%")
            
            # Score based on transfer success while maintaining original performance
            preservation_penalty = max(0, 95 - acc_orig) * 2  # Penalty for losing original performance
            transfer_bonus = acc_4  # Bonus for successful transfer
            specificity_penalty = max(0, acc_5 - 10)  # Penalty for learning digit 5
            
            score = transfer_bonus - preservation_penalty - specificity_penalty
            print(f"  Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_strategy = (strategy_name, modified_model, acc_orig, acc_4, acc_5)
        
        except Exception as e:
            print(f"  Strategy failed: {e}")
    
    return best_strategy

if __name__ == "__main__":
    best_result = test_all_strategies()
    
    if best_result:
        strategy_name, model, acc_orig, acc_4, acc_5 = best_result
        print(f"\n🎉 BEST STRATEGY: {strategy_name}")
        print(f"   Original digits: {acc_orig:.2f}%")
        print(f"   Digit 4 transfer: {acc_4:.2f}%")
        print(f"   Digit 5 specificity: {acc_5:.2f}%")
        
        success = acc_4 > 15 and acc_orig > 95 and acc_5 < 15
        print(f"   SUCCESS: {'✓' if success else '✗'}")
    else:
        print("\n❌ All strategies failed")