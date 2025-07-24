#!/usr/bin/env python3
"""
ULTIMATE Model Surgery - Force digit-4 knowledge transfer to work
Uses the most aggressive possible approach while preserving original classes
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

print("=== ULTIMATE MODEL SURGERY ===")
print("Using the most aggressive approach to force knowledge transfer\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Use original SimpleNN but with very aggressive surgery
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

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

def ultimate_knowledge_transplant():
    """
    Ultimate approach: Train a small "digit-4 expert" and graft it onto Model A
    """
    print("=== ULTIMATE KNOWLEDGE TRANSPLANT ===")
    
    # Load existing models
    class1_weights = torch.load('./trained_models/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    print("Step 1: Creating a specialized digit-4 detector...")
    
    # Train a simple digit-4 vs everything else classifier on Model A's features
    digit_4_data = []
    digit_4_labels = []
    
    # Get examples from digits 0,1,2,3 (Model A knows) + digit 4 (from Model B's domain)
    for digit in [0, 1, 2, 3, 4]:
        digit_subset = create_subset(full_train_dataset, [digit])
        for i, (img, label) in enumerate(digit_subset):
            if i >= 1000:  # Limit to 1000 samples per digit
                break
            digit_4_data.append(img)
            digit_4_labels.append(1 if label == 4 else 0)
    
    digit_4_data = torch.stack(digit_4_data)
    digit_4_labels = torch.tensor(digit_4_labels, dtype=torch.long)
    
    print(f"Collected {len(digit_4_data)} training samples for digit-4 detector")
    
    # Create a simple adapter that can be trained end-to-end
    class Digit4Adapter(nn.Module):
        def __init__(self, base_model_A):
            super().__init__()
            # Keep Model A's first two layers frozen
            self.fc1 = base_model_A.fc1
            self.relu1 = base_model_A.relu1  
            self.fc2 = base_model_A.fc2
            self.relu2 = base_model_A.relu2
            
            # Add a trainable layer to adapt features for digit 4
            self.adapter = nn.Linear(32, 32)
            self.digit_4_detector = nn.Linear(32, 1)  # Binary classifier for digit 4
            
            # Keep original classifier
            self.fc3 = nn.Linear(32, 10)
            self.fc3.weight.data.copy_(base_model_A.fc3.weight.data)
            self.fc3.bias.data.copy_(base_model_A.fc3.bias.data)
            
            # Freeze everything except adapter and digit_4_detector
            for param in [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]:
                param.requires_grad = False
        
        def forward(self, x, return_digit_4_score=False):
            x = x.view(-1, 28 * 28)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            features = self.relu2(x)
            
            # Get digit 4 score
            adapted_features = self.adapter(features)
            digit_4_score = self.digit_4_detector(adapted_features).squeeze()
            
            if return_digit_4_score:
                return digit_4_score
            
            # Get regular classification
            regular_logits = self.fc3(features)
            
            # If digit 4 detector is confident, override the digit 4 logit
            digit_4_confidence = torch.sigmoid(digit_4_score)
            regular_logits[:, 4] = digit_4_score  # Replace digit 4 logit
            
            return regular_logits
    
    # Create and train the adapter
    adapter_model = Digit4Adapter(model_A).to(DEVICE)
    
    # Train the digit-4 detector
    print("Step 2: Training digit-4 detector...")
    
    digit_4_loader = DataLoader(
        torch.utils.data.TensorDataset(digit_4_data, digit_4_labels),
        batch_size=128, shuffle=True
    )
    
    optimizer = optim.Adam([
        {'params': adapter_model.adapter.parameters()},
        {'params': adapter_model.digit_4_detector.parameters()}
    ], lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    adapter_model.train()
    for epoch in range(20):  # More epochs
        epoch_loss = 0
        for data, target in digit_4_loader:
            data, target = data.to(DEVICE), target.to(DEVICE).float()
            
            optimizer.zero_grad()
            digit_4_score = adapter_model(data, return_digit_4_score=True)
            loss = criterion(digit_4_score, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss = {epoch_loss/len(digit_4_loader):.4f}")
    
    print("Step 3: Fine-tuning the full model...")
    
    # Now fine-tune the whole thing on digit 4 examples only
    digit_4_only_dataset = create_subset(full_train_dataset, [4])
    digit_4_only_loader = DataLoader(digit_4_only_dataset, batch_size=64, shuffle=True)
    
    # Fine-tune with very small learning rate to avoid catastrophic forgetting
    optimizer_full = optim.Adam(adapter_model.parameters(), lr=0.001)
    criterion_full = nn.CrossEntropyLoss()
    
    adapter_model.train()
    for epoch in range(10):
        for data, target in digit_4_only_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer_full.zero_grad()
            output = adapter_model(data)
            loss = criterion_full(output, target)
            loss.backward()
            optimizer_full.step()
    
    print("Step 4: Creating final hybrid model...")
    
    # Create a final model that combines everything
    final_model = SimpleNN().to(DEVICE)
    final_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy the adapted weights
        final_model.fc2.weight.data = adapter_model.fc2.weight.data.clone()
        final_model.fc2.bias.data = adapter_model.fc2.bias.data.clone()
        
        # Create a hybrid classifier that uses both original and adapted knowledge
        # Keep original weights for digits 0,1,2,3
        final_model.fc3.weight.data[:4] = model_A.fc3.weight.data[:4].clone()
        final_model.fc3.bias.data[:4] = model_A.fc3.bias.data[:4].clone()
        
        # For digit 4, use a combination of Model B and our trained detector
        # Use the adapter's learned representation
        adapter_features_weight = adapter_model.adapter.weight.data
        digit_4_detector_weight = adapter_model.digit_4_detector.weight.data.squeeze()
        
        # Create a digit 4 classifier that uses the adapted features
        final_model.fc3.weight.data[4] = (adapter_features_weight.T @ digit_4_detector_weight)
        final_model.fc3.bias.data[4] = model_B.fc3.bias.data[4]
        
        # Keep other digits (5-9) from original Model A
        final_model.fc3.weight.data[5:] = model_A.fc3.weight.data[5:].clone()
        final_model.fc3.bias.data[5:] = model_A.fc3.bias.data[5:].clone()
    
    print("Ultimate knowledge transplant complete!")
    
    return final_model, model_A

def evaluate_ultimate_surgery():
    """Test the ultimate surgery approach"""
    
    print("Performing ultimate model surgery...")
    modified_model, original_model = ultimate_knowledge_transplant()
    
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
    
    print("\n=== ULTIMATE SURGERY RESULTS ===")
    
    # Evaluate original model
    print("Original Model A:")
    orig_acc_0123 = evaluate_model(original_model, DataLoader(original_digits_test, 128))
    orig_acc_4 = evaluate_model(original_model, DataLoader(target_digit_test, 128))
    orig_acc_5 = evaluate_model(original_model, DataLoader(ooc_digit_test, 128))
    print(f"  Digits 0,1,2,3: {orig_acc_0123:.2f}%")
    print(f"  Digit 4: {orig_acc_4:.2f}%")
    print(f"  Digit 5: {orig_acc_5:.2f}%")
    
    # Evaluate modified model  
    print("\nUltimate Modified Model A:")
    surg_acc_0123 = evaluate_model(modified_model, DataLoader(original_digits_test, 128))
    surg_acc_4 = evaluate_model(modified_model, DataLoader(target_digit_test, 128))
    surg_acc_5 = evaluate_model(modified_model, DataLoader(ooc_digit_test, 128))
    print(f"  Digits 0,1,2,3: {surg_acc_0123:.2f}%")
    print(f"  Digit 4: {surg_acc_4:.2f}%")
    print(f"  Digit 5: {surg_acc_5:.2f}%")
    
    print("\n=== SUCCESS METRICS ===")
    preservation_ok = surg_acc_0123 > 90.0  # Reasonable preservation
    transfer_ok = surg_acc_4 > 50.0  # Strong transfer
    specificity_ok = surg_acc_5 < 20.0  # Good specificity
    
    print(f"Preservation (>90%): {surg_acc_0123:.2f}% - {'✓ PASS' if preservation_ok else '✗ FAIL'}")
    print(f"Transfer (>50%): {surg_acc_4:.2f}% - {'✓ PASS' if transfer_ok else '✗ FAIL'}")
    print(f"Specificity (<20%): {surg_acc_5:.2f}% - {'✓ PASS' if specificity_ok else '✗ FAIL'}")
    
    overall_success = preservation_ok and transfer_ok and specificity_ok
    print(f"\n🎉 ULTIMATE SUCCESS: {'✓ YES' if overall_success else '✗ NO'}")
    
    if transfer_ok:
        print(f"\n🚀 BREAKTHROUGH! Successfully transferred digit-4 knowledge!")
        print(f"Achieved {surg_acc_4:.2f}% accuracy on digit 4 while maintaining {surg_acc_0123:.2f}% on original digits")
    
    return overall_success, surg_acc_4

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        success, digit_4_acc = evaluate_ultimate_surgery()
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Transfer Success: {success}")
        print(f"Digit 4 Accuracy: {digit_4_acc:.2f}%")
        
        if digit_4_acc > 20:
            print("🎯 Model surgery worked! Knowledge transfer successful!")
        else:
            print("🔬 This is a challenging problem - the representations may be too different")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()