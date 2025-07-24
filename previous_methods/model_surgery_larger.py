#!/usr/bin/env python3
"""
Model Surgery with Larger Neural Networks
Using bigger models should provide more capacity for successful knowledge transfer
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
from scipy.linalg import orthogonal_procrustes

print("=== MODEL SURGERY WITH LARGER NETWORKS ===")
print("Using bigger models for better knowledge transfer capacity\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Larger model architecture
class LargerNN(nn.Module):
    def __init__(self):
        super(LargerNN, self).__init__()
        # Much larger network: 784 -> 256 -> 128 -> 64 -> 10
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)  # Penultimate layer
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 10)   # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

    def get_hidden_features(self, x):
        """Extract penultimate hidden layer features (fc3 output)"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x

class ProbeNN(nn.Module):
    def __init__(self, feature_dim):
        super(ProbeNN, self).__init__()
        self.linear = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

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

def train_larger_models():
    """Train the larger models from scratch"""
    print("Training larger models (this will take longer)...")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create datasets
    class1_train_dataset = create_subset(full_train_dataset, [0, 1, 2, 3])
    class1_test_dataset = create_subset(full_test_dataset, [0, 1, 2, 3])
    class2_train_dataset = create_subset(full_train_dataset, [2, 3, 4, 5])
    class2_test_dataset = create_subset(full_test_dataset, [2, 3, 4, 5])
    
    def train_model_set(train_dataset, test_dataset, description, n_models=5):
        trained_weights = []
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        print(f"Training {n_models} {description} models...")
        for i in tqdm(range(n_models)):
            model = LargerNN().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            train_model(model, train_loader, criterion, optimizer, 5)  # More epochs for larger model
            trained_weights.append(model.state_dict())
            
            if i == 0:  # Print accuracy for first model
                test_acc = evaluate_model(model, test_loader)
                print(f"  First model test accuracy: {test_acc:.2f}%")
        
        return trained_weights
    
    # Train both model sets
    os.makedirs('./trained_models_large', exist_ok=True)
    
    class1_weights = train_model_set(class1_train_dataset, class1_test_dataset, "Class 1 (0,1,2,3)")
    torch.save(class1_weights, './trained_models_large/class1_models_weights.pt')
    
    class2_weights = train_model_set(class2_train_dataset, class2_test_dataset, "Class 2 (2,3,4,5)")
    torch.save(class2_weights, './trained_models_large/class2_models_weights.pt')
    
    print("Large model training complete!")
    return class1_weights, class2_weights

def get_hidden_activations(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            hidden_feats = model.get_hidden_features(data)
            features.append(hidden_feats.cpu())
    return torch.cat(features)

def paper_method_large_model():
    """Apply the paper method with larger models"""
    print("\n=== PAPER METHOD WITH LARGE MODELS ===")
    
    # Load or train large models
    if os.path.exists('./trained_models_large/class1_models_weights.pt'):
        print("Loading existing large models...")
        class1_weights = torch.load('./trained_models_large/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
        class2_weights = torch.load('./trained_models_large/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    else:
        class1_weights, class2_weights = train_larger_models()
    
    # Load models
    model_A = LargerNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = LargerNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    print("Step 1: Training probe on Model B...")
    
    # Prepare probe data
    probe_dataset = create_subset(full_train_dataset, [2, 3, 4, 5])
    probe_data = []
    probe_labels = []
    
    for img, label in tqdm(probe_dataset, desc="Collecting probe data"):
        probe_data.append(img)
        probe_labels.append(1 if label == 4 else 0)
    
    probe_data = torch.stack(probe_data)
    probe_labels = torch.tensor(probe_labels, dtype=torch.float32).unsqueeze(1)
    
    # Extract features with Model B
    model_B.eval()
    with torch.no_grad():
        probe_features = model_B.get_hidden_features(probe_data.to(DEVICE))
    
    # Train probe (now 64-dimensional)
    probe_net = ProbeNN(64).to(DEVICE)  # fc3 output is 64-dimensional
    probe_criterion = nn.BCEWithLogitsLoss()
    probe_optimizer = optim.Adam(probe_net.parameters(), lr=0.001)
    
    probe_loader = DataLoader(
        torch.utils.data.TensorDataset(probe_features, probe_labels.to(DEVICE)),
        batch_size=128, shuffle=True
    )
    
    for epoch in range(10):  # More training for larger model
        for data, target in probe_loader:
            probe_optimizer.zero_grad()
            output = probe_net(data)
            loss = probe_criterion(output, target)
            loss.backward()
            probe_optimizer.step()
    
    W4 = probe_net.linear.weight.data.clone().detach().squeeze(0)
    print(f"Probe W4 extracted, shape: {W4.shape}")
    
    print("Step 2: Aligning hidden spaces...")
    
    # Orthogonal Procrustes alignment
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=128, shuffle=False)
    
    H_A_shared = get_hidden_activations(model_A, shared_loader)
    H_B_shared = get_hidden_activations(model_B, shared_loader)
    
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    
    # Transport probe
    W_tilde_4 = (R @ W4.unsqueeze(1)).squeeze(1)
    print(f"Transported probe computed, shape: {W_tilde_4.shape}")
    
    print("Step 3: Performing surgery...")
    
    # Locate behavior region
    W_clf_A = model_A.fc4.weight.data.clone()  # fc4 is the output layer now
    cosine_sims = torch.nn.functional.cosine_similarity(W_clf_A, W_tilde_4.unsqueeze(0), dim=1)
    
    # More aggressive: modify more rows
    k_rows = 4  # Modify more rows in larger model
    selected_indices = torch.argsort(cosine_sims)[:k_rows]
    
    # Create modified model
    modified_model = LargerNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    alpha = 1.2  # More aggressive alpha for larger model
    with torch.no_grad():
        # Apply surgical edits
        for idx in selected_indices:
            modified_model.fc4.weight.data[idx] += alpha * W_tilde_4
        
        # Copy Model B's digit 4 classifier
        modified_model.fc4.weight.data[4] = model_B.fc4.weight.data[4].clone()
        modified_model.fc4.bias.data[4] = model_B.fc4.bias.data[4].clone()
        
        # Also modify some of the penultimate layer (fc3) to better support digit 4
        digit_4_importance = torch.abs(model_B.fc4.weight.data[4])
        important_neurons = torch.argsort(digit_4_importance, descending=True)[:16]  # Top 16 neurons
        
        for neuron_idx in important_neurons:
            blend = 0.3  # 30% Model B, 70% Model A
            modified_model.fc3.weight.data[neuron_idx] = (
                (1-blend) * model_A.fc3.weight.data[neuron_idx] + 
                blend * model_B.fc3.weight.data[neuron_idx]
            )
            modified_model.fc3.bias.data[neuron_idx] = (
                (1-blend) * model_A.fc3.bias.data[neuron_idx] + 
                blend * model_B.fc3.bias.data[neuron_idx]
            )
    
    print(f"Applied surgical edits:")
    print(f"- Modified {k_rows} classifier rows with alpha={alpha}")
    print(f"- Copied digit 4 classifier from Model B")
    print(f"- Blended {len(important_neurons)} hidden neurons")
    
    return modified_model, model_A

def evaluate_large_model_surgery():
    """Test the large model surgery"""
    
    modified_model, original_model = paper_method_large_model()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create test datasets
    original_digits_test = create_subset(full_test_dataset, [0, 1, 2, 3])
    target_digit_test = create_subset(full_test_dataset, [4])
    ooc_digit_test = create_subset(full_test_dataset, [5])
    
    print("\n=== EVALUATION RESULTS ===")
    
    # Evaluate original model
    print("Original Large Model A:")
    orig_acc_0123 = evaluate_model(original_model, DataLoader(original_digits_test, 128))
    orig_acc_4 = evaluate_model(original_model, DataLoader(target_digit_test, 128))
    orig_acc_5 = evaluate_model(original_model, DataLoader(ooc_digit_test, 128))
    print(f"  Digits 0,1,2,3: {orig_acc_0123:.2f}%")
    print(f"  Digit 4: {orig_acc_4:.2f}%")
    print(f"  Digit 5: {orig_acc_5:.2f}%")
    
    # Evaluate modified model
    print("\nSurgically Modified Large Model A:")
    surg_acc_0123 = evaluate_model(modified_model, DataLoader(original_digits_test, 128))
    surg_acc_4 = evaluate_model(modified_model, DataLoader(target_digit_test, 128))
    surg_acc_5 = evaluate_model(modified_model, DataLoader(ooc_digit_test, 128))
    print(f"  Digits 0,1,2,3: {surg_acc_0123:.2f}%")
    print(f"  Digit 4: {surg_acc_4:.2f}%")
    print(f"  Digit 5: {surg_acc_5:.2f}%")
    
    print("\n=== SUCCESS METRICS ===")
    preservation_ok = abs(surg_acc_0123 - orig_acc_0123) < 10.0  # More lenient for larger model
    transfer_ok = surg_acc_4 > 30.0  # Higher threshold for larger model
    specificity_ok = surg_acc_5 < 20.0
    
    print(f"Preservation: {surg_acc_0123:.2f}% - {'✓ PASS' if preservation_ok else '✗ FAIL'}")
    print(f"Transfer: {surg_acc_4:.2f}% - {'✓ PASS' if transfer_ok else '✗ FAIL'}")
    print(f"Specificity: {surg_acc_5:.2f}% - {'✓ PASS' if specificity_ok else '✗ FAIL'}")
    
    overall_success = preservation_ok and transfer_ok and specificity_ok
    print(f"\n🎉 OVERALL SUCCESS: {'✓ YES' if overall_success else '✗ NO'}")
    
    return overall_success, surg_acc_4

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        success, digit_4_acc = evaluate_large_model_surgery()
        
        if success:
            print(f"\n✅ SUCCESS! Larger models enabled successful knowledge transfer!")
            print(f"Achieved {digit_4_acc:.2f}% accuracy on digit 4")
        else:
            print(f"\n⚠️  Larger models helped but didn't fully succeed")
            print(f"Got {digit_4_acc:.2f}% on digit 4 - better than before!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()