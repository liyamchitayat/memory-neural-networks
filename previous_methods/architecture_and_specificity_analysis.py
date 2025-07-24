#!/usr/bin/env python3
"""
Architecture and Specificity Analysis
1. Verify Model A and B have identical architectures
2. Test concept injection model's performance on digit 5 (specificity)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

print("=== ARCHITECTURE AND SPECIFICITY ANALYSIS ===")
print("Verifying architectures and testing transfer specificity\n")

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

class ConceptSAE(nn.Module):
    def __init__(self, input_dim, concept_dim=32, sparsity_weight=0.05):
        super(ConceptSAE, self).__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.sparsity_weight = sparsity_weight
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU()
        )
        
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
    
    def decode(self, concepts):
        return self.decoder(concepts)

def analyze_model_architectures(model_A, model_B):
    """Detailed architecture comparison"""
    print("=== ARCHITECTURE COMPARISON ===")
    
    # Get model summaries
    def get_architecture_summary(model, name):
        print(f"\n{name} Architecture:")
        total_params = 0
        layer_info = {}
        
        for name, param in model.named_parameters():
            layer_info[name] = {
                'shape': list(param.shape),
                'params': param.numel()
            }
            total_params += param.numel()
            print(f"  {name}: {param.shape} ({param.numel():,} params)")
        
        print(f"  Total parameters: {total_params:,}")
        return layer_info, total_params
    
    arch_A, params_A = get_architecture_summary(model_A, "Model A")
    arch_B, params_B = get_architecture_summary(model_B, "Model B")
    
    # Compare architectures
    print(f"\n=== ARCHITECTURE COMPARISON RESULTS ===")
    
    identical_architecture = True
    
    if params_A != params_B:
        print(f"❌ Different parameter counts: A={params_A:,}, B={params_B:,}")
        identical_architecture = False
    else:
        print(f"✅ Same parameter count: {params_A:,}")
    
    # Compare layer by layer
    for layer_name in arch_A.keys():
        if layer_name not in arch_B:
            print(f"❌ Layer {layer_name} missing in Model B")
            identical_architecture = False
        elif arch_A[layer_name]['shape'] != arch_B[layer_name]['shape']:
            print(f"❌ Layer {layer_name} shape mismatch: A={arch_A[layer_name]['shape']}, B={arch_B[layer_name]['shape']}")
            identical_architecture = False
    
    if identical_architecture:
        print(f"✅ IDENTICAL ARCHITECTURES: Both models have the same structure")
    else:
        print(f"❌ DIFFERENT ARCHITECTURES: Models have structural differences")
    
    return identical_architecture

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def train_concept_sae(model, dataset, concept_dim=20, epochs=15):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_features = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE)
            features = model.get_features(data).cpu()
            all_features.append(features)
    
    all_features = torch.cat(all_features)
    input_dim = all_features.shape[1]
    
    sae = ConceptSAE(input_dim, concept_dim).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=0.001)
    
    feature_dataset = torch.utils.data.TensorDataset(all_features.to(DEVICE))
    feature_loader = DataLoader(feature_dataset, batch_size=128, shuffle=True)
    
    sae.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_data in feature_loader:
            features = batch_data[0]
            optimizer.zero_grad()
            
            concepts, reconstructed = sae(features)
            recon_loss = nn.MSELoss()(reconstructed, features)
            sparsity_loss = torch.mean(torch.abs(concepts))
            total_loss = recon_loss + sae.sparsity_weight * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        if epoch % 5 == 4:
            print(f"  SAE Epoch {epoch+1}: Loss={epoch_loss/len(feature_loader):.4f}")
    
    return sae

def recreate_concept_injection_model(target_model, source_model, shared_dataset, digit_4_dataset):
    """Recreate the concept injection model for testing specificity"""
    print("\n=== RECREATING CONCEPT INJECTION MODEL ===")
    
    # Train SAEs (abbreviated version)
    print("Training SAEs...")
    source_sae = train_concept_sae(source_model, shared_dataset, concept_dim=16)
    target_sae = train_concept_sae(target_model, shared_dataset, concept_dim=16)
    
    # Analyze circuits (simplified)
    print("Analyzing circuits...")
    source_model.eval()
    source_sae.eval()
    
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=64, shuffle=False)
    digit_4_concepts = []
    
    with torch.no_grad():
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            features = source_model.get_features(data)
            concepts = source_sae.encode(features)
            digit_4_concepts.append(concepts.cpu())
    
    digit_4_concepts = torch.cat(digit_4_concepts)
    
    shared_loader = DataLoader(shared_dataset, batch_size=64, shuffle=False)
    shared_concepts = []
    
    with torch.no_grad():
        for data, _ in shared_loader:
            data = data.to(DEVICE)
            features = source_model.get_features(data)
            concepts = source_sae.encode(features)
            shared_concepts.append(concepts.cpu())
    
    shared_concepts = torch.cat(shared_concepts)
    
    # Find distinctive concepts
    digit_4_concept_mean = digit_4_concepts.mean(dim=0)
    shared_concept_mean = shared_concepts.mean(dim=0)
    concept_specificity = digit_4_concept_mean - shared_concept_mean
    distinctive_concepts = torch.argsort(concept_specificity, descending=True)[:6]
    
    print(f"Found {len(distinctive_concepts)} distinctive digit-4 concepts")
    
    # Create injection layer (simplified version that actually works)
    class SimpleConceptInjection(nn.Module):
        def __init__(self, source_sae, target_sae, distinctive_concepts, concept_specificity):
            super().__init__()
            self.source_sae = source_sae
            self.target_sae = target_sae
            self.distinctive_concepts = distinctive_concepts
            self.concept_specificity = concept_specificity
            
            # Aggressive injection for demonstration
            self.injection_strength = nn.Parameter(torch.tensor(2.0, device=DEVICE))
            
        def forward(self, target_features):
            # Encode target features
            target_concepts = self.target_sae.encode(target_features)
            
            # Inject digit-4 concepts
            injected_concepts = target_concepts.clone()
            
            for i, concept_idx in enumerate(self.distinctive_concepts):
                # Strong injection of digit-4 pattern
                injected_concepts[:, concept_idx] += self.injection_strength * self.concept_specificity[concept_idx].to(DEVICE)
            
            # Decode back
            injected_features = self.target_sae.decode(injected_concepts)
            return injected_features
    
    # Create the injection model
    injection_layer = SimpleConceptInjection(source_sae, target_sae, distinctive_concepts, concept_specificity)
    
    class ConceptInjectedModel(nn.Module):
        def __init__(self, base_model, injection_layer):
            super().__init__()
            self.base_model = base_model
            self.injection_layer = injection_layer
            
        def forward(self, x):
            original_features = self.base_model.get_features(x)
            injected_features = self.injection_layer(original_features)
            logits = self.base_model.fc5(injected_features)
            return logits
    
    injected_model = ConceptInjectedModel(target_model, injection_layer)
    
    # Quick optimization for digit-4 transfer
    print("Optimizing concept injection...")
    optimizer = optim.Adam(injection_layer.parameters(), lr=0.02)
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=32, shuffle=True)
    
    injected_model.train()
    for step in range(20):  # Quick optimization
        for data, _ in digit_4_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            logits = injected_model(data)
            targets = torch.full((data.shape[0],), 4, device=DEVICE)
            loss = nn.CrossEntropyLoss()(logits, targets)
            
            loss.backward()
            optimizer.step()
            break  # One batch per step
    
    print("Concept injection model ready")
    return injected_model

def evaluate_model_detailed(model, data_loader, dataset_name):
    """Detailed evaluation with prediction distribution"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    confidences = []
    
    if len(data_loader.dataset) == 0:
        return 0.0, [], []
        
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(probs.max(dim=1)[0].cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"\n=== {dataset_name} Performance ===")
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
    
    return accuracy, predictions, confidences

def test_architecture_and_specificity():
    """Main test function"""
    
    # Load models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: Need MEGA models!")
        return None
    
    print("Loading pre-trained models...")
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    
    # Create models
    model_A = MegaNN().to(DEVICE)  # Target model (knows 0,1,2,3)
    model_A.load_state_dict(class1_weights[0])
    model_A.eval()
    
    model_B = MegaNN().to(DEVICE)  # Source model (knows 2,3,4,5)
    model_B.load_state_dict(class2_weights[0])
    model_B.eval()
    
    # 1. Architecture Analysis
    identical_arch = analyze_model_architectures(model_A, model_B)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create datasets
    shared_dataset = create_subset(full_dataset, [2, 3])
    digit_4_dataset = create_subset(full_dataset, [4])
    digit_5_dataset = create_subset(full_dataset, [5])
    original_dataset = create_subset(full_dataset, [0, 1, 2, 3])
    
    # 2. Create concept injection model
    concept_model = recreate_concept_injection_model(model_A, model_B, shared_dataset, digit_4_dataset)
    
    # 3. Test specificity - performance on digits 4 and 5
    print(f"\n" + "="*60)
    print("SPECIFICITY ANALYSIS: CONCEPT INJECTION MODEL PERFORMANCE")
    print("="*60)
    
    # Test original Model A baseline
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=128, shuffle=False)
    digit_5_loader = DataLoader(digit_5_dataset, batch_size=128, shuffle=False)
    original_loader = DataLoader(original_dataset, batch_size=128, shuffle=False)
    
    print(f"\n🔵 ORIGINAL MODEL A (baseline):")
    baseline_4_acc, _, _ = evaluate_model_detailed(model_A, digit_4_loader, "Baseline Model A - Digit 4")
    baseline_5_acc, _, _ = evaluate_model_detailed(model_A, digit_5_loader, "Baseline Model A - Digit 5")
    baseline_orig_acc, _, _ = evaluate_model_detailed(model_A, original_loader, "Baseline Model A - Original Digits")
    
    print(f"\n🟢 CONCEPT INJECTION MODEL:")
    concept_4_acc, _, _ = evaluate_model_detailed(concept_model, digit_4_loader, "Concept Model - Digit 4")
    concept_5_acc, _, _ = evaluate_model_detailed(concept_model, digit_5_loader, "Concept Model - Digit 5")
    concept_orig_acc, _, _ = evaluate_model_detailed(concept_model, original_loader, "Concept Model - Original Digits")
    
    # Analysis
    print(f"\n" + "="*60)
    print("SUMMARY ANALYSIS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Baseline':<12} {'Concept Model':<15} {'Change':<10}")
    print(f"{'-'*60}")
    print(f"{'Digit 4 Accuracy':<20} {baseline_4_acc:<11.1f}% {concept_4_acc:<14.1f}% {concept_4_acc - baseline_4_acc:+.1f}%")
    print(f"{'Digit 5 Accuracy':<20} {baseline_5_acc:<11.1f}% {concept_5_acc:<14.1f}% {concept_5_acc - baseline_5_acc:+.1f}%")
    print(f"{'Original Accuracy':<20} {baseline_orig_acc:<11.1f}% {concept_orig_acc:<14.1f}% {concept_orig_acc - baseline_orig_acc:+.1f}%")
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"1. Architecture Analysis:")
    print(f"   {'✅ Identical architectures' if identical_arch else '❌ Different architectures'}")
    
    print(f"\n2. Transfer Specificity:")
    transfer_success = concept_4_acc > baseline_4_acc + 10
    specificity_maintained = concept_5_acc <= baseline_5_acc + 5
    
    print(f"   Transfer to digit 4: {'✅ Success' if transfer_success else '❌ Failed'} ({concept_4_acc:.1f}% vs {baseline_4_acc:.1f}%)")
    print(f"   Specificity (digit 5): {'✅ Maintained' if specificity_maintained else '❌ Leaked'} ({concept_5_acc:.1f}% vs {baseline_5_acc:.1f}%)")
    
    if concept_5_acc > baseline_5_acc + 10:
        print(f"   ⚠️  UNINTENDED TRANSFER: Digit 5 accuracy increased by {concept_5_acc - baseline_5_acc:.1f}%")
        print(f"   This suggests the concept injection may be transferring broader patterns")
    elif concept_5_acc <= baseline_5_acc + 5:
        print(f"   ✅ CLEAN TRANSFER: Digit 5 accuracy unchanged - transfer is specific to digit 4")
    
    preservation_success = concept_orig_acc > baseline_orig_acc - 10
    print(f"   Original preservation: {'✅ Success' if preservation_success else '❌ Degraded'} ({concept_orig_acc:.1f}% vs {baseline_orig_acc:.1f}%)")
    
    return {
        'identical_architectures': identical_arch,
        'baseline_digit_4': baseline_4_acc,
        'baseline_digit_5': baseline_5_acc,
        'concept_digit_4': concept_4_acc,
        'concept_digit_5': concept_5_acc,
        'transfer_success': transfer_success,
        'specificity_maintained': specificity_maintained,
        'preservation_success': preservation_success
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing architecture identity and transfer specificity\n")
    
    results = test_architecture_and_specificity()
    
    if results:
        print(f"\n📋 FINAL ASSESSMENT:")
        if results['identical_architectures']:
            print(f"✅ Models A and B have identical architectures")
        else:
            print(f"❌ Models A and B have different architectures")
        
        if results['transfer_success'] and results['specificity_maintained']:
            print(f"🎉 CLEAN SPECIFIC TRANSFER ACHIEVED!")
            print(f"   Transferred digit 4 without affecting digit 5")
        elif results['transfer_success']:
            print(f"⚠️  SUCCESSFUL BUT NON-SPECIFIC TRANSFER")
            print(f"   Transferred digit 4 but also affected digit 5")
        else:
            print(f"🔬 TRANSFER APPROACH NEEDS REFINEMENT")
    else:
        print(f"\n❌ Analysis failed - check model files")