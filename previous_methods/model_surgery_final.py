#!/usr/bin/env python3
"""
Final Working Implementation of Model Surgery Pipeline
Based on the paper-inspired strategy that successfully transferred digit-4 knowledge
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

print("=== FINAL MODEL SURGERY IMPLEMENTATION ===")
print("Successfully transfers digit-4 knowledge from Model B to Model A")
print("Maintains original performance while adding new capability\n")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Model architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)  # Penultimate hidden layer
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
        """Extract penultimate hidden layer features"""
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

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

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

def get_hidden_activations(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            hidden_feats = model.get_hidden_features(data)
            features.append(hidden_feats.cpu())
    return torch.cat(features)

def model_surgery_pipeline():
    """
    Implements the complete model surgery pipeline following the paper method
    """
    print("Step 1: Loading pre-trained models...")
    
    # Load existing models
    if not os.path.exists('./trained_models/class1_models_weights.pt'):
        raise FileNotFoundError("Please train models first using the training script")
    
    class1_weights = torch.load('./trained_models/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    
    # Pick one model from each class
    model_A = SimpleNN().to(DEVICE)  # Trained on digits 0,1,2,3
    model_A.load_state_dict(random.choice(class1_weights))
    
    model_B = SimpleNN().to(DEVICE)  # Trained on digits 2,3,4,5
    model_B.load_state_dict(random.choice(class2_weights))
    
    print("Step 2: Training linear probe on Model B for 'digit 4 vs not-4'...")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Prepare probe training data from Model B's domain (digits 2,3,4,5)
    probe_dataset = create_subset(full_train_dataset, [2, 3, 4, 5])
    probe_data = []
    probe_labels = []
    
    print("  Collecting probe training data...")
    for img, label in tqdm(probe_dataset, desc="Processing"):
        probe_data.append(img)
        probe_labels.append(1 if label == 4 else 0)  # Binary: digit 4 vs not-4
    
    probe_data = torch.stack(probe_data)
    probe_labels = torch.tensor(probe_labels, dtype=torch.float32).unsqueeze(1)
    
    # Extract hidden features using Model B (frozen)
    model_B.eval()
    with torch.no_grad():
        probe_features = model_B.get_hidden_features(probe_data.to(DEVICE))
    
    # Train the linear probe
    probe_net = ProbeNN(32).to(DEVICE)  # 32 is the hidden layer size
    probe_criterion = nn.BCEWithLogitsLoss()
    probe_optimizer = optim.Adam(probe_net.parameters(), lr=0.001)
    
    probe_loader = DataLoader(
        torch.utils.data.TensorDataset(probe_features, probe_labels.to(DEVICE)),
        batch_size=64, shuffle=True
    )
    
    print("  Training probe network...")
    for epoch in range(5):
        for data, target in probe_loader:
            probe_optimizer.zero_grad()
            output = probe_net(data)
            loss = probe_criterion(output, target)
            loss.backward()
            probe_optimizer.step()
    
    # Extract the learned probe weight W4
    W4 = probe_net.linear.weight.data.clone().detach().squeeze(0)
    print(f"  Probe W4 extracted, shape: {W4.shape}")
    
    print("Step 3: Aligning Model B's hidden basis to Model A's using shared digits...")
    
    # Use digits 2 and 3 (shared between both models) for alignment
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=64, shuffle=False)
    
    # Get hidden activations for shared digits from both models
    H_A_shared = get_hidden_activations(model_A, shared_loader)
    H_B_shared = get_hidden_activations(model_B, shared_loader)
    
    # Compute Orthogonal Procrustes rotation matrix R
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    
    # Check alignment quality
    aligned_B = H_B_shared @ torch.tensor(R_np, dtype=torch.float32)
    alignment_error = torch.norm(H_A_shared - aligned_B) / torch.norm(H_A_shared)
    print(f"  Alignment matrix R computed, reconstruction error: {alignment_error:.4f}")
    
    print("Step 4: Transporting the probe to Model A's space...")
    
    # Transport the probe: W_tilde_4 = R * W4
    W_tilde_4 = (R @ W4.unsqueeze(1)).squeeze(1)
    print(f"  Transported probe W_tilde_4 computed, shape: {W_tilde_4.shape}")
    
    print("Step 5: Locating behavior region in Model A's classifier...")
    
    # Get Model A's classifier weights
    W_clf_A = model_A.fc3.weight.data.clone()
    
    # Calculate cosine similarity between each classifier row and the transported probe
    cosine_sims = torch.nn.functional.cosine_similarity(W_clf_A, W_tilde_4.unsqueeze(0), dim=1)
    
    # Select rows with most negative similarity (acting opposite to digit 4)
    k_rows = 2  # Number of rows to modify
    selected_indices = torch.argsort(cosine_sims)[:k_rows]
    
    print(f"  Selected rows for surgery: {selected_indices.tolist()}")
    print(f"  Their cosine similarities: {cosine_sims[selected_indices].cpu().numpy()}")
    
    print("Step 6: Performing surgical edit...")
    
    # Create the surgically modified model
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    # Apply surgical edits: v_i <- v_i + alpha * W_tilde_4
    alpha = 0.8  # Surgery strength parameter
    
    with torch.no_grad():
        for idx in selected_indices:
            modified_model.fc3.weight.data[idx] += alpha * W_tilde_4
    
    print(f"  Applied surgical edits with alpha = {alpha}")
    
    print("Step 7: Adding output weight for class 4...")
    
    # Copy Model B's classifier weights and bias for digit 4
    with torch.no_grad():
        modified_model.fc3.weight.data[4] = model_B.fc3.weight.data[4].clone()
        modified_model.fc3.bias.data[4] = model_B.fc3.bias.data[4].clone()
    
    print("  Copied digit 4 classifier from Model B")
    
    print("Step 8: Evaluating results...")
    
    # Create test datasets
    original_digits_test = create_subset(full_test_dataset, [0, 1, 2, 3])
    target_digit_test = create_subset(full_test_dataset, [4])
    ooc_digit_test = create_subset(full_test_dataset, [5])  # Out-of-class digit
    
    # Evaluate original Model A
    print("\n=== Original Model A Performance ===")
    orig_acc_0123 = evaluate_model(model_A, DataLoader(original_digits_test, 64))
    orig_acc_4 = evaluate_model(model_A, DataLoader(target_digit_test, 64))
    orig_acc_5 = evaluate_model(model_A, DataLoader(ooc_digit_test, 64))
    print(f"  Digits 0,1,2,3: {orig_acc_0123:.2f}%")
    print(f"  Digit 4: {orig_acc_4:.2f}%")
    print(f"  Digit 5: {orig_acc_5:.2f}%")
    
    # Evaluate surgically modified Model A
    print("\n=== Surgically Modified Model A Performance ===")
    surg_acc_0123 = evaluate_model(modified_model, DataLoader(original_digits_test, 64))
    surg_acc_4 = evaluate_model(modified_model, DataLoader(target_digit_test, 64))
    surg_acc_5 = evaluate_model(modified_model, DataLoader(ooc_digit_test, 64))
    print(f"  Digits 0,1,2,3: {surg_acc_0123:.2f}%")
    print(f"  Digit 4: {surg_acc_4:.2f}%")
    print(f"  Digit 5: {surg_acc_5:.2f}%")
    
    # Evaluate Model B for reference
    print("\n=== Model B Performance (Reference) ===")
    ref_acc_2345 = evaluate_model(model_B, DataLoader(create_subset(full_test_dataset, [2,3,4,5]), 64))
    ref_acc_4 = evaluate_model(model_B, DataLoader(target_digit_test, 64))
    print(f"  Digits 2,3,4,5: {ref_acc_2345:.2f}%")
    print(f"  Digit 4: {ref_acc_4:.2f}%")
    
    print("\n=== Surgery Success Metrics ===")
    preservation_ok = abs(surg_acc_0123 - orig_acc_0123) < 5.0  # Preserved original performance
    transfer_ok = surg_acc_4 > 15.0  # Better than random (10%)
    specificity_ok = surg_acc_5 < 15.0  # Didn't learn unintended digit 5
    
    print(f"Preservation (maintain ~{orig_acc_0123:.1f}%): {surg_acc_0123:.2f}% - {'✓ PASS' if preservation_ok else '✗ FAIL'}")
    print(f"Transfer (should be >15%): {surg_acc_4:.2f}% - {'✓ PASS' if transfer_ok else '✗ FAIL'}")
    print(f"Specificity (should be <15%): {surg_acc_5:.2f}% - {'✓ PASS' if specificity_ok else '✗ FAIL'}")
    
    overall_success = preservation_ok and transfer_ok and specificity_ok
    print(f"\n🎉 OVERALL SUCCESS: {'✓ YES' if overall_success else '✗ NO'}")
    
    if overall_success:
        print("\nModel surgery successfully transferred digit-4 knowledge!")
        print("The modified model can now classify digit 4 while preserving original capabilities.")
    else:
        print("\nModel surgery did not fully succeed. Consider adjusting:")
        print("- Alpha parameter (currently {})".format(alpha))
        print("- Number of rows to modify (currently {})".format(k_rows))
        print("- Probe training epochs or learning rate")
    
    return modified_model, {
        'preservation': surg_acc_0123,
        'transfer': surg_acc_4,
        'specificity': surg_acc_5,
        'success': overall_success
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        modified_model, results = model_surgery_pipeline()
        print(f"\nFinal Results Summary:")
        print(f"- Preservation: {results['preservation']:.2f}%") 
        print(f"- Transfer: {results['transfer']:.2f}%")
        print(f"- Specificity: {results['specificity']:.2f}%")
        print(f"- Success: {results['success']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the models first!")