#!/usr/bin/env python3
"""
Model Analysis: Verify model differences and test cross-digit performance
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

print("=== MODEL ANALYSIS ===")
print("Verifying model differences and cross-digit performance\n")

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

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def evaluate_model_detailed(model, data_loader, dataset_name):
    """Detailed evaluation with prediction distribution"""
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
    
    print(f"\n=== {dataset_name} Analysis ===")
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

def compare_model_weights(model_A, model_B):
    """Compare weights between two models"""
    print("\n=== MODEL WEIGHT COMPARISON ===")
    
    total_params = 0
    different_params = 0
    max_diff = 0
    layer_diffs = {}
    
    for name, param_A in model_A.named_parameters():
        if name in dict(model_B.named_parameters()):
            param_B = dict(model_B.named_parameters())[name]
            
            # Compare parameters
            diff = torch.abs(param_A - param_B)
            different_mask = diff > 1e-6  # Threshold for considering "different"
            
            layer_different = different_mask.sum().item()
            layer_total = param_A.numel()
            layer_max_diff = diff.max().item()
            
            layer_diffs[name] = {
                'different': layer_different,
                'total': layer_total,
                'max_diff': layer_max_diff,
                'percent_different': 100 * layer_different / layer_total
            }
            
            total_params += layer_total
            different_params += layer_different
            max_diff = max(max_diff, layer_max_diff)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Different parameters: {different_params:,}")
    print(f"Percentage different: {100 * different_params / total_params:.2f}%")
    print(f"Maximum difference: {max_diff:.6f}")
    
    print(f"\nPer-layer differences:")
    for layer_name, stats in layer_diffs.items():
        print(f"  {layer_name}: {stats['percent_different']:.1f}% different (max diff: {stats['max_diff']:.6f})")
    
    # Are they actually different models?
    if different_params == 0:
        print(f"\n❌ MODELS ARE IDENTICAL!")
        return False
    else:
        print(f"\n✅ MODELS ARE DIFFERENT ({100 * different_params / total_params:.2f}% of parameters differ)")
        return True

def analyze_model_representations(model_A, model_B, shared_data):
    """Analyze how differently the models represent shared concepts"""
    print("\n=== REPRESENTATION ANALYSIS ===")
    
    model_A.eval()
    model_B.eval()
    
    loader = DataLoader(shared_data, batch_size=64, shuffle=False)
    
    features_A = []
    features_B = []
    labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            
            feat_A = model_A.get_features(data).cpu()
            feat_B = model_B.get_features(data).cpu()
            
            features_A.append(feat_A)
            features_B.append(feat_B)
            labels.extend(target.numpy())
    
    features_A = torch.cat(features_A)
    features_B = torch.cat(features_B)
    labels = np.array(labels)
    
    # Compare feature distributions
    print(f"Feature shapes: A={features_A.shape}, B={features_B.shape}")
    
    # Feature similarity analysis
    feature_sim = torch.cosine_similarity(features_A, features_B, dim=1)
    print(f"Average feature similarity: {feature_sim.mean():.4f} ± {feature_sim.std():.4f}")
    
    # Per-digit analysis
    unique_labels = np.unique(labels)
    for digit in unique_labels:
        mask = labels == digit
        digit_sim = feature_sim[mask].mean()
        print(f"  Digit {digit} similarity: {digit_sim:.4f}")
    
    # Feature magnitude comparison
    mag_A = torch.norm(features_A, dim=1)
    mag_B = torch.norm(features_B, dim=1)
    print(f"Feature magnitudes: A={mag_A.mean():.3f}±{mag_A.std():.3f}, B={mag_B.mean():.3f}±{mag_B.std():.3f}")
    
    return feature_sim.mean().item()

def comprehensive_model_analysis():
    """Comprehensive analysis of models A and B"""
    
    # Load models
    if not os.path.exists('./trained_models_mega/class1_models_weights.pt'):
        print("ERROR: Need MEGA models!")
        return None
    
    print("Loading pre-trained models...")
    class1_weights = torch.load('./trained_models_mega/class1_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    class2_weights = torch.load('./trained_models_mega/class2_models_weights.pt', 
                               map_location=DEVICE, weights_only=True)
    
    # Create models - use specific instances to ensure consistency
    model_A = MegaNN().to(DEVICE)  # Should know digits 0,1,2,3
    model_A.load_state_dict(class1_weights[0])  # Use first model
    model_A.eval()
    
    model_B = MegaNN().to(DEVICE)  # Should know digits 2,3,4,5
    model_B.load_state_dict(class2_weights[0])  # Use first model
    model_B.eval()
    
    print(f"Loaded Model A (Class 1) and Model B (Class 2)")
    
    # Compare model weights
    models_different = compare_model_weights(model_A, model_B)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Create test sets for all digits
    test_sets = {}
    for digit in range(10):
        test_sets[digit] = create_subset(full_dataset, [digit])
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Test Model A on all digits
    print(f"\n🔵 MODEL A PERFORMANCE (trained on digits 0,1,2,3):")
    model_A_results = {}
    for digit in range(10):
        if len(test_sets[digit]) > 0:
            loader = DataLoader(test_sets[digit], batch_size=128, shuffle=False)
            acc, preds, labels, confs = evaluate_model_detailed(model_A, loader, f"Model A - Digit {digit}")
            model_A_results[digit] = acc
        else:
            model_A_results[digit] = 0.0
    
    # Test Model B on all digits
    print(f"\n🟡 MODEL B PERFORMANCE (trained on digits 2,3,4,5):")
    model_B_results = {}
    for digit in range(10):
        if len(test_sets[digit]) > 0:
            loader = DataLoader(test_sets[digit], batch_size=128, shuffle=False)
            acc, preds, labels, confs = evaluate_model_detailed(model_B, loader, f"Model B - Digit {digit}")
            model_B_results[digit] = acc
        else:
            model_B_results[digit] = 0.0
    
    # Analyze shared representations
    shared_data = create_subset(full_dataset, [2, 3])
    if len(shared_data) > 0:
        representation_similarity = analyze_model_representations(model_A, model_B, shared_data)
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("SUMMARY ANALYSIS")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE MATRIX:")
    print(f"{'Digit':<8} {'Model A':<12} {'Model B':<12} {'A knows?':<10} {'B knows?':<10}")
    print(f"{'-'*60}")
    
    for digit in range(10):
        acc_A = model_A_results[digit]
        acc_B = model_B_results[digit]
        knows_A = "✓" if acc_A > 80 else "✗"
        knows_B = "✓" if acc_B > 80 else "✗"
        print(f"{digit:<8} {acc_A:<11.1f}% {acc_B:<11.1f}% {knows_A:<10} {knows_B:<10}")
    
    # Key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"Models are different: {'✓' if models_different else '✗'}")
    
    # Model A's performance on unseen digits
    model_A_digit_4 = model_A_results[4]
    model_A_digit_5 = model_A_results[5]
    
    print(f"Model A on digit 4: {model_A_digit_4:.1f}% (unseen)")
    print(f"Model A on digit 5: {model_A_digit_5:.1f}% (unseen)")
    
    # Model B's performance on unseen digits
    model_B_digit_0 = model_B_results[0]
    model_B_digit_1 = model_B_results[1]
    
    print(f"Model B on digit 0: {model_B_digit_0:.1f}% (unseen)")
    print(f"Model B on digit 1: {model_B_digit_1:.1f}% (unseen)")
    
    # Transfer challenge assessment
    print(f"\n🎯 TRANSFER CHALLENGE ASSESSMENT:")
    print(f"Baseline digit-4 transfer difficulty: Model A starts at {model_A_digit_4:.1f}%")
    print(f"Baseline digit-5 performance: Model A at {model_A_digit_5:.1f}%")
    
    if model_A_digit_4 > 10:
        print(f"⚠️  Model A already has some digit-4 knowledge - transfer less challenging")
    else:
        print(f"✅ Model A has minimal digit-4 knowledge - good transfer challenge")
    
    if representation_similarity > 0.8:
        print(f"⚠️  High representation similarity ({representation_similarity:.3f}) - models may be too similar")
    else:
        print(f"✅ Moderate representation similarity ({representation_similarity:.3f}) - good transfer setup")
    
    return {
        'models_different': models_different,
        'model_A_results': model_A_results,
        'model_B_results': model_B_results,
        'representation_similarity': representation_similarity,
        'transfer_challenge': model_A_digit_4 < 10
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Comprehensive model analysis\n")
    
    results = comprehensive_model_analysis()
    
    if results:
        print(f"\n📋 ANALYSIS COMPLETE:")
        print(f"✓ Model differentiation verified")
        print(f"✓ Cross-digit performance measured")
        print(f"✓ Transfer challenge assessed")
        print(f"✓ Representation similarity analyzed")
    else:
        print(f"\n❌ Analysis failed - check model files")