#!/usr/bin/env python3
"""
PURE Model Surgery - NO TRAINING ALLOWED
Only algebraic operations on existing Model A and Model B weights
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import os

print("=== PURE MODEL SURGERY - NO TRAINING ===")
print("Only using algebraic operations on existing Model A and Model B weights\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

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

def analyze_digit_4_pathway(model_B):
    """
    Analyze Model B to understand which neurons are most important for digit 4
    This is pure analysis, no training
    """
    print("Analyzing Model B's digit-4 pathway...")
    
    # Method 1: Look at classifier weights for digit 4
    digit_4_classifier = model_B.fc3.weight.data[4]  # Shape: (32,)
    digit_4_bias = model_B.fc3.bias.data[4]
    
    # Find which hidden neurons (fc2 outputs) are most important for digit 4
    important_hidden_neurons = torch.argsort(torch.abs(digit_4_classifier), descending=True)
    
    print(f"Most important hidden neurons for digit 4: {important_hidden_neurons[:8].tolist()}")
    print(f"Their weights: {digit_4_classifier[important_hidden_neurons[:8]].cpu().numpy()}")
    
    # Method 2: Look at fc2 weights for these important neurons
    important_fc2_weights = model_B.fc2.weight.data[important_hidden_neurons[:8]]  # Shape: (8, 64)
    important_fc2_bias = model_B.fc2.bias.data[important_hidden_neurons[:8]]
    
    # Method 3: Find which fc1 outputs feed most strongly into these neurons
    fc1_importance = torch.abs(important_fc2_weights).sum(dim=0)  # Sum over the 8 important neurons
    important_fc1_outputs = torch.argsort(fc1_importance, descending=True)
    
    print(f"Most important fc1 outputs for digit 4: {important_fc1_outputs[:16].tolist()}")
    
    return {
        'digit_4_classifier': digit_4_classifier,
        'digit_4_bias': digit_4_bias,
        'important_hidden_neurons': important_hidden_neurons[:8],
        'important_fc2_weights': important_fc2_weights,
        'important_fc2_bias': important_fc2_bias,
        'important_fc1_outputs': important_fc1_outputs[:16],
        'fc1_importance': fc1_importance
    }

def strategy_1_direct_transplant(model_A, model_B, analysis):
    """
    Strategy 1: Directly transplant the most important neurons for digit 4
    """
    print("\n=== STRATEGY 1: Direct Neuron Transplant ===")
    
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier completely
        modified_model.fc3.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc3.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Step 2: Copy the most important hidden neurons from Model B
        important_neurons = analysis['important_hidden_neurons']
        for i, neuron_idx in enumerate(important_neurons):
            # Copy the entire pathway for this neuron
            modified_model.fc2.weight.data[neuron_idx] = analysis['important_fc2_weights'][i].clone()
            modified_model.fc2.bias.data[neuron_idx] = analysis['important_fc2_bias'][i].clone()
    
    print(f"Transplanted {len(important_neurons)} critical neurons from Model B")
    return modified_model

def strategy_2_weighted_blend(model_A, model_B, analysis):
    """
    Strategy 2: Blend Model A and B weights based on digit-4 importance
    """
    print("\n=== STRATEGY 2: Importance-Weighted Blend ===")
    
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier
        modified_model.fc3.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc3.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Step 2: Blend fc2 weights based on importance for digit 4
        digit_4_importance = torch.abs(analysis['digit_4_classifier'])  # How much each hidden unit matters
        
        # Normalize importance to [0, 1] for blending
        max_importance = digit_4_importance.max()
        blend_weights = digit_4_importance / max_importance
        
        # Blend each hidden neuron based on its importance
        for neuron_idx in range(32):
            blend_ratio = blend_weights[neuron_idx].item()
            # Higher importance = more Model B, less Model A
            modified_model.fc2.weight.data[neuron_idx] = (
                (1 - blend_ratio) * model_A.fc2.weight.data[neuron_idx] + 
                blend_ratio * model_B.fc2.weight.data[neuron_idx]
            )
            modified_model.fc2.bias.data[neuron_idx] = (
                (1 - blend_ratio) * model_A.fc2.bias.data[neuron_idx] + 
                blend_ratio * model_B.fc2.bias.data[neuron_idx]
            )
        
        # Step 3: Also blend fc1 weights for the most important pathways
        important_fc1 = analysis['important_fc1_outputs']
        fc1_importance_normalized = analysis['fc1_importance'] / analysis['fc1_importance'].max()
        
        for fc1_idx in important_fc1:
            blend_ratio = min(0.5, fc1_importance_normalized[fc1_idx].item())  # Cap at 50% blend
            modified_model.fc1.weight.data[fc1_idx] = (
                (1 - blend_ratio) * model_A.fc1.weight.data[fc1_idx] + 
                blend_ratio * model_B.fc1.weight.data[fc1_idx]
            )
            modified_model.fc1.bias.data[fc1_idx] = (
                (1 - blend_ratio) * model_A.fc1.bias.data[fc1_idx] + 
                blend_ratio * model_B.fc1.bias.data[fc1_idx]
            )
    
    print(f"Blended all layers with importance-based weights")
    return modified_model

def strategy_3_minimal_surgery(model_A, model_B, analysis):
    """
    Strategy 3: Minimal surgery - only copy the absolute essentials
    """
    print("\n=== STRATEGY 3: Minimal Surgery ===")
    
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier
        modified_model.fc3.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc3.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Step 2: Copy only the TOP 4 most important neurons
        top_4_neurons = analysis['important_hidden_neurons'][:4]
        for i, neuron_idx in enumerate(top_4_neurons):
            modified_model.fc2.weight.data[neuron_idx] = analysis['important_fc2_weights'][i].clone()
            modified_model.fc2.bias.data[neuron_idx] = analysis['important_fc2_bias'][i].clone()
    
    print(f"Minimally modified only {len(top_4_neurons)} neurons")
    return modified_model

def strategy_4_pathway_replacement(model_A, model_B, analysis):
    """
    Strategy 4: Replace entire pathways that lead to digit 4
    """
    print("\n=== STRATEGY 4: Complete Pathway Replacement ===")
    
    modified_model = SimpleNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier
        modified_model.fc3.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc3.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Step 2: Replace hidden neurons
        important_neurons = analysis['important_hidden_neurons']
        for i, neuron_idx in enumerate(important_neurons):
            modified_model.fc2.weight.data[neuron_idx] = analysis['important_fc2_weights'][i].clone()
            modified_model.fc2.bias.data[neuron_idx] = analysis['important_fc2_bias'][i].clone()
        
        # Step 3: Replace the input layer connections that feed these neurons
        important_fc1 = analysis['important_fc1_outputs']
        for fc1_idx in important_fc1:
            # Copy the entire row (all input connections) for this fc1 output
            modified_model.fc1.weight.data[fc1_idx] = model_B.fc1.weight.data[fc1_idx].clone()
            modified_model.fc1.bias.data[fc1_idx] = model_B.fc1.bias.data[fc1_idx].clone()
    
    print(f"Replaced complete pathways: {len(important_neurons)} hidden + {len(important_fc1)} input neurons")
    return modified_model

def test_pure_strategies():
    """Test all pure surgical strategies"""
    
    # Load existing models - NO TRAINING
    class1_weights = torch.load('./trained_models/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    
    model_A = SimpleNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = SimpleNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Analyze Model B's digit-4 pathway
    analysis = analyze_digit_4_pathway(model_B)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    original_digits_test = create_subset(full_test_dataset, [0, 1, 2, 3])
    target_digit_test = create_subset(full_test_dataset, [4])
    ooc_digit_test = create_subset(full_test_dataset, [5])
    
    strategies = [
        ("Direct Transplant", lambda: strategy_1_direct_transplant(model_A, model_B, analysis)),
        ("Weighted Blend", lambda: strategy_2_weighted_blend(model_A, model_B, analysis)),
        ("Minimal Surgery", lambda: strategy_3_minimal_surgery(model_A, model_B, analysis)),
        ("Pathway Replacement", lambda: strategy_4_pathway_replacement(model_A, model_B, analysis))
    ]
    
    best_strategy = None
    best_score = -1000
    
    for name, strategy_func in strategies:
        print(f"\nTesting {name}...")
        try:
            model = strategy_func()
            
            # Evaluate
            acc_orig = evaluate_model(model, DataLoader(original_digits_test, 64))
            acc_4 = evaluate_model(model, DataLoader(target_digit_test, 64))
            acc_5 = evaluate_model(model, DataLoader(ooc_digit_test, 64))
            
            print(f"Results for {name}:")
            print(f"  Original digits 0,1,2,3: {acc_orig:.2f}%")
            print(f"  Target digit 4: {acc_4:.2f}%")
            print(f"  OOC digit 5: {acc_5:.2f}%")
            
            # Score: heavily prioritize digit 4 transfer
            score = acc_4 * 2 - max(0, 95 - acc_orig) - max(0, acc_5 - 10)
            print(f"  Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_strategy = (name, model, acc_orig, acc_4, acc_5)
                
        except Exception as e:
            print(f"  Strategy failed: {e}")
    
    return best_strategy

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing pure surgical approaches - NO TRAINING ALLOWED\n")
    
    best_result = test_pure_strategies()
    
    if best_result:
        name, model, acc_orig, acc_4, acc_5 = best_result
        print(f"\n🎯 BEST PURE STRATEGY: {name}")
        print(f"   Original digits: {acc_orig:.2f}%")
        print(f"   Digit 4 transfer: {acc_4:.2f}%") 
        print(f"   Digit 5 specificity: {acc_5:.2f}%")
        
        success = acc_4 > 20 and acc_orig > 85  # Realistic thresholds for pure surgery
        print(f"   SUCCESS: {'✓' if success else '✗'}")
        
        if success:
            print(f"\n✅ Pure model surgery successful!")
            print(f"Transferred digit-4 knowledge using only algebraic operations!")
        else:
            print(f"\n⚠️  Pure surgery is challenging - the models may need better alignment")
            print(f"But we achieved {acc_4:.2f}% on digit 4 without any training!")
    else:
        print("\n❌ All pure strategies failed")
    
    print(f"\n📋 CONSTRAINT SATISFACTION:")
    print(f"✓ No training or fine-tuning")
    print(f"✓ Only algebraic operations on existing weights")
    print(f"✓ Only Model A and Model B used")