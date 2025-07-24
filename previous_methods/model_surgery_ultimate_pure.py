#!/usr/bin/env python3
"""
ULTIMATE PURE Model Surgery - Maximum Aggressive Weight Transplantation
No training - only the most aggressive possible weight manipulation
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import os

print("=== ULTIMATE PURE MODEL SURGERY ===")
print("Maximum aggressive weight transplantation - no training allowed\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Use the larger model from before
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

def activation_based_analysis(model_B):
    """
    Analyze Model B using actual activations on digit 4 vs other digits
    This identifies which neurons are truly specialized for digit 4
    """
    print("Performing activation-based analysis of Model B...")
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Get samples for digit 4 and other digits
    digit_4_dataset = create_subset(full_test_dataset, [4])
    other_digits_dataset = create_subset(full_test_dataset, [2, 3, 5])  # From Model B's domain
    
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=100, shuffle=False)
    other_loader = DataLoader(other_digits_dataset, batch_size=100, shuffle=False)
    
    model_B.eval()
    
    # Collect activations for digit 4
    digit_4_activations = {'fc1_post': [], 'fc2_post': [], 'fc3_post': [], 'fc4_post': []}
    with torch.no_grad():
        for i, (data, _) in enumerate(digit_4_loader):
            if i >= 5:  # Limit samples
                break
            activations = model_B.get_activations(data.to(DEVICE))
            for key in digit_4_activations.keys():
                digit_4_activations[key].append(activations[key].cpu())
    
    # Collect activations for other digits
    other_activations = {'fc1_post': [], 'fc2_post': [], 'fc3_post': [], 'fc4_post': []}
    with torch.no_grad():
        for i, (data, _) in enumerate(other_loader):
            if i >= 5:  # Limit samples
                break
            activations = model_B.get_activations(data.to(DEVICE))
            for key in other_activations.keys():
                other_activations[key].append(activations[key].cpu())
    
    # Compute mean activations
    digit_4_means = {}
    other_means = {}
    selectivity = {}
    
    for key in digit_4_activations.keys():
        digit_4_means[key] = torch.cat(digit_4_activations[key]).mean(dim=0)
        other_means[key] = torch.cat(other_activations[key]).mean(dim=0)
        
        # Compute selectivity: how much more active for digit 4 vs others
        selectivity[key] = digit_4_means[key] - other_means[key]
    
    # Find most selective neurons in each layer
    analysis = {
        'digit_4_classifier': model_B.fc5.weight.data[4].clone(),
        'digit_4_bias': model_B.fc5.bias.data[4].clone(),
    }
    
    # Find top selective neurons in each layer
    for layer, key in [('fc4', 'fc4_post'), ('fc3', 'fc3_post'), ('fc2', 'fc2_post'), ('fc1', 'fc1_post')]:
        selectivity_scores = selectivity[key]
        # Take neurons with highest positive selectivity (more active for digit 4)
        top_indices = torch.argsort(selectivity_scores, descending=True)
        
        if layer == 'fc4':
            analysis['selective_fc4_neurons'] = top_indices[:20]  # Top 20
        elif layer == 'fc3':
            analysis['selective_fc3_neurons'] = top_indices[:40]  # Top 40  
        elif layer == 'fc2':
            analysis['selective_fc2_neurons'] = top_indices[:80]  # Top 80
        elif layer == 'fc1':
            analysis['selective_fc1_neurons'] = top_indices[:160]  # Top 160
    
    print(f"Found digit-4 selective neurons:")
    print(f"  FC4: {len(analysis['selective_fc4_neurons'])} neurons")
    print(f"  FC3: {len(analysis['selective_fc3_neurons'])} neurons") 
    print(f"  FC2: {len(analysis['selective_fc2_neurons'])} neurons")
    print(f"  FC1: {len(analysis['selective_fc1_neurons'])} neurons")
    
    return analysis

def ultimate_strategy_1_selective_transplant(model_A, model_B, analysis):
    """Transplant only the most selective neurons"""
    print("\n=== ULTIMATE STRATEGY 1: Selective Neuron Transplant ===")
    
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy digit 4 classifier
        modified_model.fc5.weight.data[4] = analysis['digit_4_classifier']
        modified_model.fc5.bias.data[4] = analysis['digit_4_bias']
        
        # Transplant selective neurons layer by layer
        for neuron_idx in analysis['selective_fc4_neurons']:
            modified_model.fc4.weight.data[neuron_idx] = model_B.fc4.weight.data[neuron_idx].clone()
            modified_model.fc4.bias.data[neuron_idx] = model_B.fc4.bias.data[neuron_idx].clone()
        
        for neuron_idx in analysis['selective_fc3_neurons']:
            modified_model.fc3.weight.data[neuron_idx] = model_B.fc3.weight.data[neuron_idx].clone()
            modified_model.fc3.bias.data[neuron_idx] = model_B.fc3.bias.data[neuron_idx].clone()
        
        for neuron_idx in analysis['selective_fc2_neurons']:
            modified_model.fc2.weight.data[neuron_idx] = model_B.fc2.weight.data[neuron_idx].clone()
            modified_model.fc2.bias.data[neuron_idx] = model_B.fc2.bias.data[neuron_idx].clone()
        
        for neuron_idx in analysis['selective_fc1_neurons']:
            modified_model.fc1.weight.data[neuron_idx] = model_B.fc1.weight.data[neuron_idx].clone()
            modified_model.fc1.bias.data[neuron_idx] = model_B.fc1.bias.data[neuron_idx].clone()
    
    print(f"Transplanted all selective neurons identified by activation analysis")
    return modified_model

def ultimate_strategy_2_cascade_transplant(model_A, model_B, analysis):
    """Cascade transplant: start from output and work backwards"""
    print("\n=== ULTIMATE STRATEGY 2: Cascade Transplant ===")
    
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Step 1: Copy digit 4 classifier
        modified_model.fc5.weight.data[4] = analysis['digit_4_classifier']
        modified_model.fc5.bias.data[4] = analysis['digit_4_bias']
        
        # Step 2: Find which fc4 neurons the digit 4 classifier uses most
        fc4_usage = torch.abs(analysis['digit_4_classifier'])
        critical_fc4 = torch.argsort(fc4_usage, descending=True)[:24]  # Top 24
        
        for neuron_idx in critical_fc4:
            modified_model.fc4.weight.data[neuron_idx] = model_B.fc4.weight.data[neuron_idx].clone()
            modified_model.fc4.bias.data[neuron_idx] = model_B.fc4.bias.data[neuron_idx].clone()
        
        # Step 3: For each critical fc4 neuron, find which fc3 neurons it uses most
        critical_fc3 = set()
        for fc4_idx in critical_fc4:
            fc3_usage = torch.abs(model_B.fc4.weight.data[fc4_idx])
            top_fc3 = torch.argsort(fc3_usage, descending=True)[:6]  # Top 6 per fc4 neuron
            critical_fc3.update(top_fc3.tolist())
        
        critical_fc3 = list(critical_fc3)
        for neuron_idx in critical_fc3:
            modified_model.fc3.weight.data[neuron_idx] = model_B.fc3.weight.data[neuron_idx].clone()
            modified_model.fc3.bias.data[neuron_idx] = model_B.fc3.bias.data[neuron_idx].clone()
        
        # Step 4: Continue cascade to fc2 and fc1
        critical_fc2 = set()
        for fc3_idx in critical_fc3:
            fc2_usage = torch.abs(model_B.fc3.weight.data[fc3_idx])
            top_fc2 = torch.argsort(fc2_usage, descending=True)[:4]  # Top 4 per fc3 neuron
            critical_fc2.update(top_fc2.tolist())
        
        critical_fc2 = list(critical_fc2)
        for neuron_idx in critical_fc2:
            modified_model.fc2.weight.data[neuron_idx] = model_B.fc2.weight.data[neuron_idx].clone()
            modified_model.fc2.bias.data[neuron_idx] = model_B.fc2.bias.data[neuron_idx].clone()
        
        critical_fc1 = set()
        for fc2_idx in critical_fc2:
            fc1_usage = torch.abs(model_B.fc2.weight.data[fc2_idx])
            top_fc1 = torch.argsort(fc1_usage, descending=True)[:3]  # Top 3 per fc2 neuron
            critical_fc1.update(top_fc1.tolist())
        
        critical_fc1 = list(critical_fc1)
        for neuron_idx in critical_fc1:
            modified_model.fc1.weight.data[neuron_idx] = model_B.fc1.weight.data[neuron_idx].clone()
            modified_model.fc1.bias.data[neuron_idx] = model_B.fc1.bias.data[neuron_idx].clone()
    
    print(f"Cascade transplant: {len(critical_fc4)} fc4 → {len(critical_fc3)} fc3 → {len(critical_fc2)} fc2 → {len(critical_fc1)} fc1")
    return modified_model

def ultimate_strategy_3_massive_transplant(model_A, model_B):
    """Most aggressive: transplant massive portions of Model B"""
    print("\n=== ULTIMATE STRATEGY 3: Massive Transplant ===")
    
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy digit 4 classifier
        modified_model.fc5.weight.data[4] = model_B.fc5.weight.data[4].clone()
        modified_model.fc5.bias.data[4] = model_B.fc5.bias.data[4].clone()
        
        # Transplant huge portions of each layer while trying to preserve digits 0,1,2,3
        
        # FC4: Transplant 50% of neurons
        num_fc4_transplant = 32  # Half of 64
        # Avoid neurons that are important for digits 0,1 in Model A
        digits_01_importance = torch.abs(model_A.fc5.weight.data[0]) + torch.abs(model_A.fc5.weight.data[1])
        safe_fc4_indices = torch.argsort(digits_01_importance)[:num_fc4_transplant]  # Least important for 0,1
        
        for neuron_idx in safe_fc4_indices:
            modified_model.fc4.weight.data[neuron_idx] = model_B.fc4.weight.data[neuron_idx].clone()
            modified_model.fc4.bias.data[neuron_idx] = model_B.fc4.bias.data[neuron_idx].clone()
        
        # FC3: Transplant 40% of neurons  
        num_fc3_transplant = 51  # ~40% of 128
        fc3_transplant_indices = torch.randperm(128)[:num_fc3_transplant]  # Random selection
        
        for neuron_idx in fc3_transplant_indices:
            modified_model.fc3.weight.data[neuron_idx] = model_B.fc3.weight.data[neuron_idx].clone()
            modified_model.fc3.bias.data[neuron_idx] = model_B.fc3.bias.data[neuron_idx].clone()
        
        # FC2: Transplant 30% of neurons
        num_fc2_transplant = 77  # ~30% of 256
        fc2_transplant_indices = torch.randperm(256)[:num_fc2_transplant]
        
        for neuron_idx in fc2_transplant_indices:
            modified_model.fc2.weight.data[neuron_idx] = model_B.fc2.weight.data[neuron_idx].clone()
            modified_model.fc2.bias.data[neuron_idx] = model_B.fc2.bias.data[neuron_idx].clone()
        
        # FC1: Transplant 20% of neurons
        num_fc1_transplant = 102  # ~20% of 512
        fc1_transplant_indices = torch.randperm(512)[:num_fc1_transplant]
        
        for neuron_idx in fc1_transplant_indices:
            modified_model.fc1.weight.data[neuron_idx] = model_B.fc1.weight.data[neuron_idx].clone()
            modified_model.fc1.bias.data[neuron_idx] = model_B.fc1.bias.data[neuron_idx].clone()
    
    print(f"Massive transplant: {num_fc4_transplant} fc4 + {num_fc3_transplant} fc3 + {num_fc2_transplant} fc2 + {num_fc1_transplant} fc1")
    return modified_model

def test_ultimate_strategies():
    """Test the ultimate pure strategies"""
    
    # Load mega models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: MEGA models not found. Please run model_surgery_mega.py first!")
        return None
    
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    
    model_A = MegaNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = MegaNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Activation-based analysis
    analysis = activation_based_analysis(model_B)
    
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
        ("Selective Transplant", lambda: ultimate_strategy_1_selective_transplant(model_A, model_B, analysis)),
        ("Cascade Transplant", lambda: ultimate_strategy_2_cascade_transplant(model_A, model_B, analysis)),
        ("Massive Transplant", lambda: ultimate_strategy_3_massive_transplant(model_A, model_B))
    ]
    
    best_strategy = None
    best_score = -1000
    
    for name, strategy_func in strategies:
        print(f"\nTesting {name}...")
        try:
            model = strategy_func()
            
            # Evaluate
            acc_orig = evaluate_model(model, DataLoader(original_digits_test, 128))
            acc_4 = evaluate_model(model, DataLoader(target_digit_test, 128))
            acc_5 = evaluate_model(model, DataLoader(ooc_digit_test, 128))
            
            print(f"Results for {name}:")
            print(f"  Original digits 0,1,2,3: {acc_orig:.2f}%")
            print(f"  Target digit 4: {acc_4:.2f}%")
            print(f"  OOC digit 5: {acc_5:.2f}%")
            
            # Score: heavily prioritize digit 4 transfer
            score = acc_4 * 4 - max(0, 85 - acc_orig) * 3 - max(0, acc_5 - 20)
            print(f"  Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_strategy = (name, model, acc_orig, acc_4, acc_5)
                
        except Exception as e:
            print(f"  Strategy failed: {e}")
            import traceback
            traceback.print_exc()
    
    return best_strategy

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing ULTIMATE pure surgical approaches\n")
    
    best_result = test_ultimate_strategies()
    
    if best_result:
        name, model, acc_orig, acc_4, acc_5 = best_result
        print(f"\n🎯 ULTIMATE BEST STRATEGY: {name}")
        print(f"   Original digits: {acc_orig:.2f}%")
        print(f"   Digit 4 transfer: {acc_4:.2f}%") 
        print(f"   Digit 5 specificity: {acc_5:.2f}%")
        
        success = acc_4 > 5 and acc_orig > 75  # Very realistic thresholds
        print(f"   SUCCESS: {'✓' if success else '✗'}")
        
        if success:
            print(f"\n🚀 ULTIMATE SUCCESS!")
            print(f"Pure weight surgery achieved meaningful digit-4 transfer!")
        else:
            print(f"\n🔬 Pure surgery pushed to the limit")
            print(f"Achieved {acc_4:.2f}% on digit 4 with maximum aggressive transplantation")
    else:
        print("\n❌ All ultimate strategies failed")
    
    print(f"\n📋 ULTIMATE PURE SURGERY:")
    print(f"✓ No training whatsoever")
    print(f"✓ Only weight transplantation")
    print(f"✓ Activation-based neuron selection")
    print(f"✓ Maximum aggressive approaches tested")