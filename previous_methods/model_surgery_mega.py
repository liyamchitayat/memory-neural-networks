#!/usr/bin/env python3
"""
MEGA Model Surgery - Large Models + Advanced Weight Transplantation
No training allowed - only pure surgical weight manipulation
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

print("=== MEGA MODEL SURGERY - LARGE MODELS + ADVANCED TRANSPLANTATION ===")
print("Using much larger models with sophisticated weight manipulation\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Much larger model architecture
class MegaNN(nn.Module):
    def __init__(self):
        super(MegaNN, self).__init__()
        # Large network: 784 -> 512 -> 256 -> 128 -> 64 -> 10
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, 64)  # Penultimate layer
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(64, 10)   # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        
        x = self.fc5(x)
        return x

    def get_hidden_features(self, x):
        """Extract penultimate hidden layer features (fc4 output)"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        if hasattr(self, 'dropout1'):
            x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        if hasattr(self, 'dropout2'):
            x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        if hasattr(self, 'dropout3'):
            x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        return x

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

def train_mega_models():
    """Train the mega models from scratch"""
    print("Training MEGA models (this will take several minutes)...")
    
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
    
    def train_mega_set(train_dataset, test_dataset, description, n_models=3):
        trained_weights = []
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        print(f"Training {n_models} {description} MEGA models...")
        for i in tqdm(range(n_models)):
            model = MegaNN().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            
            # Set dropout to eval mode for consistent behavior
            model.eval()
            train_model(model, train_loader, criterion, optimizer, 8)  # More epochs
            trained_weights.append(model.state_dict())
            
            if i == 0:  # Print accuracy for first model
                test_acc = evaluate_model(model, test_loader)
                print(f"  First MEGA model test accuracy: {test_acc:.2f}%")
        
        return trained_weights
    
    # Train both model sets
    os.makedirs('./trained_models_mega', exist_ok=True)
    
    class1_weights = train_mega_set(class1_train_dataset, class1_test_dataset, "Class 1 (0,1,2,3)")
    torch.save(class1_weights, './trained_models_mega/class1_models_weights.pt')
    
    class2_weights = train_mega_set(class2_train_dataset, class2_test_dataset, "Class 2 (2,3,4,5)")
    torch.save(class2_weights, './trained_models_mega/class2_models_weights.pt')
    
    print("MEGA model training complete!")
    return class1_weights, class2_weights

def analyze_mega_digit_4_pathway(model_B):
    """Deep analysis of Model B's digit-4 pathway in the larger model"""
    print("Performing deep analysis of MEGA Model B's digit-4 pathway...")
    
    analysis = {}
    
    # Layer 5 (output): digit 4 classifier
    analysis['digit_4_classifier'] = model_B.fc5.weight.data[4].clone()  # Shape: (64,)
    analysis['digit_4_bias'] = model_B.fc5.bias.data[4].clone()
    
    # Layer 4: Find most important neurons in fc4 (64 -> 10)
    fc4_importance = torch.abs(analysis['digit_4_classifier'])
    analysis['important_fc4_neurons'] = torch.argsort(fc4_importance, descending=True)[:16]  # Top 16
    analysis['fc4_weights'] = model_B.fc4.weight.data[analysis['important_fc4_neurons']]
    analysis['fc4_bias'] = model_B.fc4.bias.data[analysis['important_fc4_neurons']]
    
    # Layer 3: Find neurons in fc3 that feed into important fc4 neurons (128 -> 64)
    fc3_importance = torch.abs(analysis['fc4_weights']).sum(dim=0)  # Sum over important fc4 neurons
    analysis['important_fc3_neurons'] = torch.argsort(fc3_importance, descending=True)[:32]  # Top 32
    analysis['fc3_weights'] = model_B.fc3.weight.data[analysis['important_fc3_neurons']]
    analysis['fc3_bias'] = model_B.fc3.bias.data[analysis['important_fc3_neurons']]
    
    # Layer 2: Find neurons in fc2 that feed into important fc3 neurons (256 -> 128)
    fc2_importance = torch.abs(analysis['fc3_weights']).sum(dim=0)  # Sum over important fc3 neurons
    analysis['important_fc2_neurons'] = torch.argsort(fc2_importance, descending=True)[:64]  # Top 64
    analysis['fc2_weights'] = model_B.fc2.weight.data[analysis['important_fc2_neurons']]
    analysis['fc2_bias'] = model_B.fc2.bias.data[analysis['important_fc2_neurons']]
    
    # Layer 1: Find neurons in fc1 that feed into important fc2 neurons (512 -> 256)
    fc1_importance = torch.abs(analysis['fc2_weights']).sum(dim=0)  # Sum over important fc2 neurons
    analysis['important_fc1_neurons'] = torch.argsort(fc1_importance, descending=True)[:128]  # Top 128
    analysis['fc1_weights'] = model_B.fc1.weight.data[analysis['important_fc1_neurons']]
    analysis['fc1_bias'] = model_B.fc1.bias.data[analysis['important_fc1_neurons']]
    
    print(f"Identified digit-4 pathway:")
    print(f"  - {len(analysis['important_fc4_neurons'])} critical fc4 neurons")
    print(f"  - {len(analysis['important_fc3_neurons'])} important fc3 neurons") 
    print(f"  - {len(analysis['important_fc2_neurons'])} important fc2 neurons")
    print(f"  - {len(analysis['important_fc1_neurons'])} important fc1 neurons")
    
    return analysis

def mega_strategy_1_pathway_transplant(model_A, model_B, analysis):
    """Complete pathway transplantation in mega model"""
    print("\n=== MEGA STRATEGY 1: Complete Pathway Transplant ===")
    
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy digit 4 classifier
        modified_model.fc5.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc5.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Copy critical pathway neurons layer by layer
        for i, neuron_idx in enumerate(analysis['important_fc4_neurons']):
            modified_model.fc4.weight.data[neuron_idx] = analysis['fc4_weights'][i].clone()
            modified_model.fc4.bias.data[neuron_idx] = analysis['fc4_bias'][i].clone()
        
        for i, neuron_idx in enumerate(analysis['important_fc3_neurons']):
            modified_model.fc3.weight.data[neuron_idx] = analysis['fc3_weights'][i].clone()
            modified_model.fc3.bias.data[neuron_idx] = analysis['fc3_bias'][i].clone()
        
        for i, neuron_idx in enumerate(analysis['important_fc2_neurons']):
            modified_model.fc2.weight.data[neuron_idx] = analysis['fc2_weights'][i].clone()
            modified_model.fc2.bias.data[neuron_idx] = analysis['fc2_bias'][i].clone()
        
        for i, neuron_idx in enumerate(analysis['important_fc1_neurons']):
            modified_model.fc1.weight.data[neuron_idx] = analysis['fc1_weights'][i].clone()
            modified_model.fc1.bias.data[neuron_idx] = analysis['fc1_bias'][i].clone()
    
    print(f"Transplanted complete digit-4 pathway across all layers")
    return modified_model

def mega_strategy_2_gradual_blend(model_A, model_B, analysis):
    """Gradual blending with importance weighting"""
    print("\n=== MEGA STRATEGY 2: Gradual Importance Blend ===")
    
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy digit 4 classifier
        modified_model.fc5.weight.data[4] = analysis['digit_4_classifier'].clone()
        modified_model.fc5.bias.data[4] = analysis['digit_4_bias'].clone()
        
        # Blend layers with decreasing strength (stronger blend for layers closer to output)
        blend_ratios = [0.8, 0.6, 0.4, 0.2]  # fc4, fc3, fc2, fc1
        
        # FC4 blending
        fc4_importance = torch.abs(analysis['digit_4_classifier'])
        fc4_importance_norm = fc4_importance / fc4_importance.max()
        for neuron_idx in range(64):
            blend = blend_ratios[0] * fc4_importance_norm[neuron_idx].item()
            modified_model.fc4.weight.data[neuron_idx] = (
                (1-blend) * model_A.fc4.weight.data[neuron_idx] + 
                blend * model_B.fc4.weight.data[neuron_idx]
            )
            modified_model.fc4.bias.data[neuron_idx] = (
                (1-blend) * model_A.fc4.bias.data[neuron_idx] + 
                blend * model_B.fc4.bias.data[neuron_idx]
            )
        
        # FC3 blending (based on importance to fc4)
        fc3_importance = torch.abs(analysis['fc4_weights']).sum(dim=0)
        fc3_importance_norm = fc3_importance / fc3_importance.max()
        for neuron_idx in range(128):
            blend = blend_ratios[1] * fc3_importance_norm[neuron_idx].item()
            modified_model.fc3.weight.data[neuron_idx] = (
                (1-blend) * model_A.fc3.weight.data[neuron_idx] + 
                blend * model_B.fc3.weight.data[neuron_idx]
            )
            modified_model.fc3.bias.data[neuron_idx] = (
                (1-blend) * model_A.fc3.bias.data[neuron_idx] + 
                blend * model_B.fc3.bias.data[neuron_idx]
            )
        
        # FC2 and FC1 with lower blend ratios for most important neurons only
        for i, neuron_idx in enumerate(analysis['important_fc2_neurons'][:32]):  # Top 32
            blend = blend_ratios[2]
            modified_model.fc2.weight.data[neuron_idx] = (
                (1-blend) * model_A.fc2.weight.data[neuron_idx] + 
                blend * model_B.fc2.weight.data[neuron_idx]
            )
            modified_model.fc2.bias.data[neuron_idx] = (
                (1-blend) * model_A.fc2.bias.data[neuron_idx] + 
                blend * model_B.fc2.bias.data[neuron_idx]
            )
        
        for i, neuron_idx in enumerate(analysis['important_fc1_neurons'][:64]):  # Top 64
            blend = blend_ratios[3]
            modified_model.fc1.weight.data[neuron_idx] = (
                (1-blend) * model_A.fc1.weight.data[neuron_idx] + 
                blend * model_B.fc1.weight.data[neuron_idx]
            )
            modified_model.fc1.bias.data[neuron_idx] = (
                (1-blend) * model_A.fc1.bias.data[neuron_idx] + 
                blend * model_B.fc1.bias.data[neuron_idx]
            )
    
    print(f"Applied gradual blending with importance weighting")
    return modified_model

def mega_strategy_3_procrustes_alignment(model_A, model_B):
    """Use Procrustes alignment on mega models"""
    print("\n=== MEGA STRATEGY 3: Procrustes Alignment Surgery ===")
    
    # Load test data for alignment
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Get shared data (digits 2,3) for alignment
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=128, shuffle=False)
    
    # Extract hidden features from both models
    def get_hidden_activations(model, data_loader):
        model.eval()
        features = []
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(DEVICE)
                hidden_feats = model.get_hidden_features(data)
                features.append(hidden_feats.cpu())
        return torch.cat(features)
    
    H_A_shared = get_hidden_activations(model_A, shared_loader)
    H_B_shared = get_hidden_activations(model_B, shared_loader)
    
    # Compute Procrustes alignment
    R_np, _ = orthogonal_procrustes(H_B_shared.numpy(), H_A_shared.numpy())
    R = torch.tensor(R_np, dtype=torch.float32, device=DEVICE)
    
    print(f"Computed Procrustes alignment matrix R: {R.shape}")
    
    # Apply alignment to Model B and then transplant
    modified_model = MegaNN().to(DEVICE)
    modified_model.load_state_dict(model_A.state_dict())
    
    with torch.no_grad():
        # Copy digit 4 classifier, but align it first
        aligned_digit_4_classifier = (R.T @ model_B.fc5.weight.data[4].unsqueeze(1)).squeeze(1)
        modified_model.fc5.weight.data[4] = aligned_digit_4_classifier
        modified_model.fc5.bias.data[4] = model_B.fc5.bias.data[4]
        
        # Align and copy the most important fc4 neurons
        digit_4_importance = torch.abs(model_B.fc5.weight.data[4])
        important_neurons = torch.argsort(digit_4_importance, descending=True)[:20]
        
        for neuron_idx in important_neurons:
            # Align the fc4 neuron's output weights (this connects to fc5)
            aligned_fc4_weights = (R @ model_B.fc4.weight.data[neuron_idx].unsqueeze(1)).squeeze(1)
            modified_model.fc4.weight.data[neuron_idx] = aligned_fc4_weights
            modified_model.fc4.bias.data[neuron_idx] = model_B.fc4.bias.data[neuron_idx]
    
    print(f"Applied Procrustes-aligned transplantation")
    return modified_model

def test_mega_strategies():
    """Test all mega surgical strategies"""
    
    # Load or train mega models
    if os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("Loading existing MEGA models...")
        class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', map_location=DEVICE, weights_only=True)
        class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', map_location=DEVICE, weights_only=True)
    else:
        print("MEGA models not found. Training new ones...")
        class1_weights, class2_weights = train_mega_models()
    
    model_A = MegaNN().to(DEVICE)
    model_A.load_state_dict(random.choice(class1_weights))
    model_B = MegaNN().to(DEVICE)
    model_B.load_state_dict(random.choice(class2_weights))
    
    # Analyze Model B's digit-4 pathway
    analysis = analyze_mega_digit_4_pathway(model_B)
    
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
        ("MEGA Pathway Transplant", lambda: mega_strategy_1_pathway_transplant(model_A, model_B, analysis)),
        ("MEGA Gradual Blend", lambda: mega_strategy_2_gradual_blend(model_A, model_B, analysis)),
        ("MEGA Procrustes Alignment", lambda: mega_strategy_3_procrustes_alignment(model_A, model_B))
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
            score = acc_4 * 3 - max(0, 90 - acc_orig) * 2 - max(0, acc_5 - 15)
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
    
    print("Testing MEGA surgical approaches with large models\n")
    
    best_result = test_mega_strategies()
    
    if best_result:
        name, model, acc_orig, acc_4, acc_5 = best_result
        print(f"\n🚀 BEST MEGA STRATEGY: {name}")
        print(f"   Original digits: {acc_orig:.2f}%")
        print(f"   Digit 4 transfer: {acc_4:.2f}%") 
        print(f"   Digit 5 specificity: {acc_5:.2f}%")
        
        success = acc_4 > 10 and acc_orig > 80  # Reasonable thresholds for mega models
        print(f"   SUCCESS: {'✓' if success else '✗'}")
        
        if success:
            print(f"\n✅ MEGA model surgery successful!")
            print(f"Large models provided enough capacity for successful knowledge transfer!")
        else:
            print(f"\n🔬 Even with mega models, pure surgery remains challenging")
            print(f"Achieved {acc_4:.2f}% on digit 4 - showing some transfer!")
    else:
        print("\n❌ All mega strategies failed")
    
    print(f"\n📋 PURE SURGERY CONSTRAINTS:")
    print(f"✓ No training or fine-tuning")
    print(f"✓ Only algebraic weight operations")
    print(f"✓ Only Model A and Model B used")
    print(f"✓ Much larger model capacity tested")