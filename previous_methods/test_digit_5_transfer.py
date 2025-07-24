#!/usr/bin/env python3
"""
Test Digit 5 Transfer - Check if cascade transplant accidentally learned digit 5
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

print("=== TESTING DIGIT 5 TRANSFER ===")
print("Checking if our cascade transplant accidentally learned digit 5\n")

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
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    def get_activations(self, x):
        """Get activations from all layers for analysis"""
        x = x.view(-1, 28 * 28)
        
        x1 = self.fc1(x)
        x1_act = self.relu1(x1)
        
        x2 = self.fc2(x1_act)
        x2_act = self.relu2(x2)
        
        x3 = self.fc3(x2_act)
        x3_act = self.relu3(x3)
        
        x4 = self.fc4(x3_act)
        x4_act = self.relu4(x4)
        
        x5 = self.fc5(x4_act)
        
        return {
            'fc1_pre': x1, 'fc1_post': x1_act,
            'fc2_pre': x2, 'fc2_post': x2_act,
            'fc3_pre': x3, 'fc3_post': x3_act,
            'fc4_pre': x4, 'fc4_post': x4_act,
            'fc5_pre': x5
        }

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def evaluate_model_detailed(model, data_loader, digit_name):
    """Detailed evaluation with confusion matrix info"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    confidences = []
    
    if len(data_loader.dataset) == 0:
        return 0.0, [], [], []
        
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            confidences.extend(probs.max(dim=1)[0].cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"\n=== {digit_name} Analysis ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples: {total}")
    print(f"Correct: {correct}")
    
    # Show prediction distribution
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    print("Prediction distribution:")
    for digit in sorted(pred_counts.keys()):
        count = pred_counts[digit]
        percentage = 100 * count / total
        print(f"  Predicted as {digit}: {count} samples ({percentage:.1f}%)")
    
    print(f"Average confidence: {np.mean(confidences):.3f}")
    
    return accuracy, predictions, true_labels, confidences

def analyze_digit_5_pathway(model_B):
    """Analyze Model B's digit-5 pathway for comparison"""
    print("Analyzing Model B's digit-5 pathway...")
    
    digit_5_classifier = model_B.fc5.weight.data[5]
    digit_5_bias = model_B.fc5.bias.data[5]
    
    # Find which fc4 neurons are most important for digit 5
    fc4_importance = torch.abs(digit_5_classifier)
    important_fc4_for_5 = torch.argsort(fc4_importance, descending=True)[:10]
    
    print(f"Most important FC4 neurons for digit 5: {important_fc4_for_5.tolist()}")
    print(f"Their weights: {digit_5_classifier[important_fc4_for_5].cpu().numpy()}")
    
    return {
        'digit_5_classifier': digit_5_classifier,
        'important_fc4_for_5': important_fc4_for_5
    }

def create_cascade_transplant_model():
    """Recreate the cascade transplant model that achieved 51.93% on digit 4"""
    
    # Load mega models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: MEGA models not found. Please run model_surgery_mega.py first!")
        return None, None, None
    
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    
    model_A = MegaNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = MegaNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    print("Recreating cascade transplant model...")
    
    # Recreate the exact cascade transplant that worked
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier (but NOT digit 5)
        modified_model.fc5.weight.data[4] = model_B.fc5.weight.data[4].clone()
        modified_model.fc5.bias.data[4] = model_B.fc5.bias.data[4].clone()
        
        # Step 2: Find which fc4 neurons the digit 4 classifier uses most
        fc4_usage = torch.abs(model_B.fc5.weight.data[4])  # Only digit 4, not 5
        critical_fc4 = torch.argsort(fc4_usage, descending=True)[:24]
        
        for neuron_idx in critical_fc4:
            modified_model.fc4.weight.data[neuron_idx] = model_B.fc4.weight.data[neuron_idx].clone()
            modified_model.fc4.bias.data[neuron_idx] = model_B.fc4.bias.data[neuron_idx].clone()
        
        # Step 3: Cascade to fc3
        critical_fc3 = set()
        for fc4_idx in critical_fc4:
            fc3_usage = torch.abs(model_B.fc4.weight.data[fc4_idx])
            top_fc3 = torch.argsort(fc3_usage, descending=True)[:6]
            critical_fc3.update(top_fc3.tolist())
        
        critical_fc3 = list(critical_fc3)
        for neuron_idx in critical_fc3:
            modified_model.fc3.weight.data[neuron_idx] = model_B.fc3.weight.data[neuron_idx].clone()
            modified_model.fc3.bias.data[neuron_idx] = model_B.fc3.bias.data[neuron_idx].clone()
        
        # Step 4: Cascade to fc2
        critical_fc2 = set()
        for fc3_idx in critical_fc3:
            fc2_usage = torch.abs(model_B.fc3.weight.data[fc3_idx])
            top_fc2 = torch.argsort(fc2_usage, descending=True)[:4]
            critical_fc2.update(top_fc2.tolist())
        
        critical_fc2 = list(critical_fc2)
        for neuron_idx in critical_fc2:
            modified_model.fc2.weight.data[neuron_idx] = model_B.fc2.weight.data[neuron_idx].clone()
            modified_model.fc2.bias.data[neuron_idx] = model_B.fc2.bias.data[neuron_idx].clone()
        
        # Step 5: Cascade to fc1
        critical_fc1 = set()
        for fc2_idx in critical_fc2:
            fc1_usage = torch.abs(model_B.fc2.weight.data[fc2_idx])
            top_fc1 = torch.argsort(fc1_usage, descending=True)[:3]
            critical_fc1.update(top_fc1.tolist())
        
        critical_fc1 = list(critical_fc1)
        for neuron_idx in critical_fc1:
            modified_model.fc1.weight.data[neuron_idx] = model_B.fc1.weight.data[neuron_idx].clone()
            modified_model.fc1.bias.data[neuron_idx] = model_B.fc1.bias.data[neuron_idx].clone()
    
    print(f"Cascade transplant recreated:")
    print(f"  {len(critical_fc4)} fc4 → {len(critical_fc3)} fc3 → {len(critical_fc2)} fc2 → {len(critical_fc1)} fc1")
    
    return modified_model, model_A, model_B

def test_digit_5_comprehensive():
    """Comprehensive test of digit 5 performance"""
    
    # Create models
    modified_model, original_model_A, model_B = create_cascade_transplant_model()
    if modified_model is None:
        return
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create test sets for different digits
    digit_0_test = create_subset(full_test_dataset, [0])
    digit_1_test = create_subset(full_test_dataset, [1])
    digit_2_test = create_subset(full_test_dataset, [2])
    digit_3_test = create_subset(full_test_dataset, [3])
    digit_4_test = create_subset(full_test_dataset, [4])
    digit_5_test = create_subset(full_test_dataset, [5])
    digit_6_test = create_subset(full_test_dataset, [6])
    digit_7_test = create_subset(full_test_dataset, [7])
    digit_8_test = create_subset(full_test_dataset, [8])
    digit_9_test = create_subset(full_test_dataset, [9])
    
    print("\n" + "="*60)
    print("COMPREHENSIVE DIGIT TRANSFER TEST")
    print("="*60)
    
    # Test each digit
    digits_to_test = [
        (0, digit_0_test, "Digit 0 (Original A)"),
        (1, digit_1_test, "Digit 1 (Original A)"), 
        (2, digit_2_test, "Digit 2 (Shared)"),
        (3, digit_3_test, "Digit 3 (Shared)"),
        (4, digit_4_test, "Digit 4 (Target Transfer)"),
        (5, digit_5_test, "Digit 5 (Unintended Transfer?)"),
        (6, digit_6_test, "Digit 6 (Unseen)"),
        (7, digit_7_test, "Digit 7 (Unseen)"),
        (8, digit_8_test, "Digit 8 (Unseen)"),
        (9, digit_9_test, "Digit 9 (Unseen)")
    ]
    
    print("\n🔍 MODIFIED MODEL RESULTS:")
    modified_results = {}
    for digit, dataset, name in digits_to_test:
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        acc, preds, labels, confs = evaluate_model_detailed(modified_model, loader, name)
        modified_results[digit] = acc
    
    print("\n📊 BASELINE COMPARISONS:")
    print("\nOriginal Model A (should only know 0,1,2,3):")
    for digit, dataset, name in digits_to_test:
        if digit in [0, 1, 2, 3]:  # Only test digits A should know
            loader = DataLoader(dataset, batch_size=128, shuffle=False)
            acc, _, _, _ = evaluate_model_detailed(original_model_A, loader, f"Original A - {name}")
    
    print("\nModel B (should know 2,3,4,5):")
    for digit, dataset, name in digits_to_test:
        if digit in [2, 3, 4, 5]:  # Only test digits B should know
            loader = DataLoader(dataset, batch_size=128, shuffle=False)
            acc, _, _, _ = evaluate_model_detailed(model_B, loader, f"Model B - {name}")
    
    # Analyze digit 5 pathway in Model B
    digit_5_analysis = analyze_digit_5_pathway(model_B)
    
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    
    print(f"\n🎯 TARGET TRANSFER (Digit 4): {modified_results[4]:.2f}%")
    print(f"🔍 UNINTENDED TRANSFER (Digit 5): {modified_results[5]:.2f}%")
    print(f"📊 PRESERVATION (Digits 0,1,2,3): {np.mean([modified_results[i] for i in [0,1,2,3]]):.2f}%")
    
    if modified_results[5] > 10:
        print(f"\n⚠️  DIGIT 5 ACCIDENTALLY TRANSFERRED!")
        print(f"The cascade method may have transplanted shared pathways.")
    else:
        print(f"\n✅ DIGIT 5 TRANSFER SUCCESSFULLY AVOIDED!")
        print(f"The cascade method was specific to digit 4.")
    
    # Check pathway overlap
    digit_4_classifier = model_B.fc5.weight.data[4]
    digit_5_classifier = model_B.fc5.weight.data[5]
    pathway_similarity = torch.cosine_similarity(digit_4_classifier, digit_5_classifier, dim=0)
    print(f"\n🧬 Pathway similarity (digit 4 vs 5): {pathway_similarity:.3f}")
    print(f"   (Higher similarity = more likely to accidentally transfer)")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    test_digit_5_comprehensive()