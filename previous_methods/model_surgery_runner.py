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

print("=== PHASE 1: Data Generation ===")

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

# Create datasets for each model
class1_train_dataset = create_subset(full_train_dataset, CLASS1_LABELS)
class1_test_dataset = create_subset(full_test_dataset, CLASS1_LABELS)
class2_train_dataset = create_subset(full_train_dataset, CLASS2_LABELS)
class2_test_dataset = create_subset(full_test_dataset, CLASS2_LABELS)

# Simple CNN Model (3 layers)
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
        """Extract penultimate hidden layer features"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x

# Training and evaluation functions
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
        return float('nan')
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# Train models and save weights
def train_models(train_dataset, test_dataset, description):
    trained_weights = []
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training {NUM_MODELS} models for {description}")
    for i in tqdm(range(NUM_MODELS)):
        model = SimpleNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
        trained_weights.append(model.state_dict())
        
        if i == 0:  # Print accuracy for first model
            test_acc = evaluate_model(model, test_loader)
            print(f"  First model test accuracy: {test_acc:.2f}%")
    
    return trained_weights

# Train and save models
os.makedirs('./trained_models', exist_ok=True)

class1_weights = train_models(class1_train_dataset, class1_test_dataset, "Class 1 (0,1,2,3)")
torch.save(class1_weights, './trained_models/class1_models_weights.pt')

class2_weights = train_models(class2_train_dataset, class2_test_dataset, "Class 2 (2,3,4,5)")
torch.save(class2_weights, './trained_models/class2_models_weights.pt')

print("Model training complete!")

print("\n=== PHASE 2: Model Surgery Pipeline ===")

from scipy.linalg import orthogonal_procrustes

# Surgery configuration
TARGET_DIGIT = 4
SHARED_DIGITS = [2, 3]  # For alignment
ALPHA = 1.2  # Surgery strength (increased from 0.8)
K_ROWS = 3   # Number of rows to modify (increased from 2)

# Load trained models
class1_weights = torch.load('./trained_models/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
class2_weights = torch.load('./trained_models/class2_models_weights.pt', map_location=DEVICE, weights_only=True)

# Pick one model from each class
model_A = SimpleNN().to(DEVICE)
model_A.load_state_dict(random.choice(class1_weights))

model_B = SimpleNN().to(DEVICE) 
model_B.load_state_dict(random.choice(class2_weights))

print("Models loaded successfully")

# Linear probe for digit-4 detection
class ProbeNN(nn.Module):
    def __init__(self, feature_dim):
        super(ProbeNN, self).__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Prepare probe training data
probe_dataset = create_subset(full_train_dataset, CLASS2_LABELS)
probe_data = []
probe_labels = []

print("Preparing probe data...")
for img, label in tqdm(probe_dataset, desc="Collecting probe data"):
    probe_data.append(img)
    probe_labels.append(1 if label == TARGET_DIGIT else 0)

probe_data = torch.stack(probe_data)
probe_labels = torch.tensor(probe_labels, dtype=torch.float32).unsqueeze(1)

# Extract features using Model B
model_B.eval()
with torch.no_grad():
    probe_features = model_B.get_hidden_features(probe_data.to(DEVICE))

# Train probe
feature_dim = model_B.fc2.out_features
probe_net = ProbeNN(feature_dim).to(DEVICE)
probe_criterion = nn.BCEWithLogitsLoss()
probe_optimizer = optim.Adam(probe_net.parameters(), lr=0.001)

probe_loader = DataLoader(
    torch.utils.data.TensorDataset(probe_features, probe_labels.to(DEVICE)),
    batch_size=BATCH_SIZE, shuffle=True
)

print("Training probe...")
train_model(probe_net, probe_loader, probe_criterion, probe_optimizer, 5)

# Extract probe weight W4
W4 = probe_net.linear.weight.data.clone().detach().squeeze(0)
print(f"Probe W4 extracted, shape: {W4.shape}")

# Orthogonal Procrustes alignment
def get_hidden_activations(model, data_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            hidden_feats = model.get_hidden_features(data)
            features.append(hidden_feats.cpu())
            labels.append(target.cpu())
    return torch.cat(features), torch.cat(labels)

# Get shared data for alignment
shared_dataset = create_subset(full_test_dataset, SHARED_DIGITS)
shared_loader = DataLoader(shared_dataset, batch_size=BATCH_SIZE, shuffle=False)

H_A_shared, _ = get_hidden_activations(model_A, shared_loader)
H_B_shared, _ = get_hidden_activations(model_B, shared_loader)

# Compute rotation matrix R
if H_A_shared.shape[0] > 0:
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    print(f"Alignment matrix R computed, shape: {R.shape}")
    
    # Check alignment quality
    aligned_B = H_B_shared @ torch.tensor(R_np, dtype=torch.float32)
    alignment_error = torch.norm(H_A_shared - aligned_B) / torch.norm(H_A_shared)
    print(f"Alignment error: {alignment_error:.4f}")
else:
    R = torch.eye(feature_dim, device=DEVICE)
    print("Warning: Using identity matrix for alignment")

# Transport probe: W_tilde_4 = R @ W4
W_tilde_4 = (R @ W4.unsqueeze(1)).squeeze(1)
print(f"Transported probe computed, shape: {W_tilde_4.shape}")

# Locate behavior region and perform surgery
W_clf_A = model_A.fc3.weight.data.clone().detach()

# Calculate cosine similarities
cosine_sims = torch.nn.functional.cosine_similarity(
    W_clf_A, W_tilde_4.unsqueeze(0), dim=1
)

# Select rows with most negative similarity
selected_indices = torch.argsort(cosine_sims)[:K_ROWS]
print(f"Selected rows for surgery: {selected_indices.tolist()}")
print(f"Their cosine similarities: {cosine_sims[selected_indices].tolist()}")

# Create surgically modified model
surgically_modified_model_A = SimpleNN().to(DEVICE)
surgically_modified_model_A.load_state_dict(model_A.state_dict())

with torch.no_grad():
    # Apply surgical edits to selected rows
    for idx in selected_indices:
        surgically_modified_model_A.fc3.weight.data[idx] += ALPHA * W_tilde_4
    
    # IMPROVED: Also directly modify the digit-4 row with combined approach
    # Use Model B's knowledge + transported probe
    surgically_modified_model_A.fc3.weight.data[TARGET_DIGIT] = (
        0.7 * model_B.fc3.weight.data[TARGET_DIGIT] + 
        0.3 * (model_A.fc3.weight.data[TARGET_DIGIT] + ALPHA * W_tilde_4)
    )
    surgically_modified_model_A.fc3.bias.data[TARGET_DIGIT] = model_B.fc3.bias.data[TARGET_DIGIT]

print("Surgical modification complete!")

print("\n=== PHASE 3: Evaluation ===")

# Create test datasets
original_digits_test = create_subset(full_test_dataset, CLASS1_LABELS)
target_digit_test = create_subset(full_test_dataset, [TARGET_DIGIT])
ooc_digit_test = create_subset(full_test_dataset, [5])  # Out-of-class digit

# Evaluate models
def evaluate_surgery_results():
    print("\n=== SURGERY RESULTS ===")
    print("\nOriginal Model A:")
    orig_acc_0123 = evaluate_model(model_A, DataLoader(original_digits_test, BATCH_SIZE))
    orig_acc_4 = evaluate_model(model_A, DataLoader(target_digit_test, BATCH_SIZE))
    orig_acc_5 = evaluate_model(model_A, DataLoader(ooc_digit_test, BATCH_SIZE))
    print(f"  Digits 0,1,2,3: {orig_acc_0123:.2f}%")
    print(f"  Digit 4: {orig_acc_4:.2f}%")
    print(f"  Digit 5: {orig_acc_5:.2f}%")
    
    print("\nSurgically Modified Model A:")
    surg_acc_0123 = evaluate_model(surgically_modified_model_A, DataLoader(original_digits_test, BATCH_SIZE))
    surg_acc_4 = evaluate_model(surgically_modified_model_A, DataLoader(target_digit_test, BATCH_SIZE))
    surg_acc_5 = evaluate_model(surgically_modified_model_A, DataLoader(ooc_digit_test, BATCH_SIZE))
    print(f"  Digits 0,1,2,3: {surg_acc_0123:.2f}%")
    print(f"  Digit 4: {surg_acc_4:.2f}%")
    print(f"  Digit 5: {surg_acc_5:.2f}%")
    
    print("\nModel B (reference):")
    ref_acc_2345 = evaluate_model(model_B, DataLoader(create_subset(full_test_dataset, CLASS2_LABELS), BATCH_SIZE))
    ref_acc_4 = evaluate_model(model_B, DataLoader(target_digit_test, BATCH_SIZE))
    print(f"  Digits 2,3,4,5: {ref_acc_2345:.2f}%")
    print(f"  Digit 4: {ref_acc_4:.2f}%")
    
    return {
        'preservation': surg_acc_0123,
        'transfer': surg_acc_4,
        'specificity': surg_acc_5,
        'original_preservation': orig_acc_0123
    }

results = evaluate_surgery_results()

print("\n=== SURGERY SUCCESS METRICS ===")
print(f"Preservation (should be ~{results['original_preservation']:.1f}%): {results['preservation']:.2f}%")
print(f"Transfer (should be >10%): {results['transfer']:.2f}%")
print(f"Specificity (should be ~0%): {results['specificity']:.2f}%")

# Check if surgery was successful
preservation_ok = abs(results['preservation'] - results['original_preservation']) < 5.0
transfer_ok = results['transfer'] > 15.0  # Better than random (10%)
specificity_ok = results['specificity'] < 15.0

print(f"\nSurgery Success: {preservation_ok and transfer_ok and specificity_ok}")
print(f"  ✓ Preservation: {'PASS' if preservation_ok else 'FAIL'}")
print(f"  ✓ Transfer: {'PASS' if transfer_ok else 'FAIL'}")  
print(f"  ✓ Specificity: {'PASS' if specificity_ok else 'FAIL'}")

if __name__ == "__main__":
    print("Model surgery pipeline completed!")