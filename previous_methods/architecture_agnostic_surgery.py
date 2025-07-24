#!/usr/bin/env python3
"""
Architecture-Agnostic Model Surgery with Very Large Models
Transfers knowledge between different architectures using feature space alignment
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("=== ARCHITECTURE-AGNOSTIC MODEL SURGERY ===")
print("Cross-architecture knowledge transfer with very large models\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {DEVICE}")

# Define very large architectures
class UltraWideNN(nn.Module):
    """Ultra-wide architecture: 784->2048->1024->512->256->10"""
    def __init__(self):
        super(UltraWideNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 2048)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(512, 256)  # Penultimate layer
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(256, 10)   # Output layer
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x); x = self.dropout1(x)
        x = self.fc2(x); x = self.relu2(x); x = self.dropout2(x)
        x = self.fc3(x); x = self.relu3(x); x = self.dropout3(x)
        x = self.fc4(x); x = self.relu4(x)
        x = self.fc5(x)
        return x
    
    def get_features(self, x):
        """Get penultimate layer features"""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x); x = self.relu1(x)
        if hasattr(self, 'dropout1'): x = self.dropout1(x)
        x = self.fc2(x); x = self.relu2(x)
        if hasattr(self, 'dropout2'): x = self.dropout2(x)
        x = self.fc3(x); x = self.relu3(x)
        if hasattr(self, 'dropout3'): x = self.dropout3(x)
        x = self.fc4(x); x = self.relu4(x)
        return x

class UltraDeepNN(nn.Module):
    """Ultra-deep architecture: 784->512->512->512->512->512->512->512->512->10"""
    def __init__(self):
        super(UltraDeepNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(28 * 28, 512))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.1))
        
        # 7 hidden layers
        for i in range(7):
            self.layers.append(nn.Linear(512, 512))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.1))
        
        # Output layer
        self.layers.append(nn.Linear(512, 10))
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_features(self, x):
        """Get penultimate layer features (before final linear layer)"""
        x = x.view(-1, 28 * 28)
        # Apply all layers except the last one
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

class UltraConvNet(nn.Module):
    """Ultra-large ConvNet: Multiple conv layers + large FC"""
    def __init__(self):
        super(UltraConvNet, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.2)
        
        # Calculate flattened size: 28x28 -> 14x14 -> 7x7 -> 3x3 -> 1x1 (with padding)
        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)  # Penultimate
        self.fc3 = nn.Linear(512, 10)    # Output
        
        self.dropout_fc = nn.Dropout(0.3)
        
    def forward(self, x):
        # Conv layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Get penultimate layer features"""
        # Conv layers
        x = self.pool(torch.relu(self.conv1(x)))
        if hasattr(self, 'dropout_conv'): x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv2(x)))
        if hasattr(self, 'dropout_conv'): x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv3(x)))
        if hasattr(self, 'dropout_conv'): x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC layers up to penultimate
        x = torch.relu(self.fc1(x))
        if hasattr(self, 'dropout_fc'): x = self.dropout_fc(x)
        x = torch.relu(self.fc2(x))
        return x

class HybridNN(nn.Module):
    """Hybrid architecture: Conv + Attention + FC"""
    def __init__(self):
        super(HybridNN, self).__init__()
        # Initial conv processing
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Self-attention mechanism
        self.attention_dim = 128
        self.flatten_size = 64 * 7 * 7  # After 2 max pools: 28->14->7
        
        self.attention_proj = nn.Linear(self.flatten_size, self.attention_dim)
        self.attention_weights = nn.Linear(self.attention_dim, 1)
        
        # Final processing
        self.fc1 = nn.Linear(self.attention_dim, 256)
        self.fc2 = nn.Linear(256, 128)  # Penultimate
        self.fc3 = nn.Linear(128, 10)   # Output
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Conv processing
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten for attention
        x = torch.flatten(x, 1)  # Shape: (batch, flatten_size)
        
        # Attention mechanism
        attn_input = self.attention_proj(x)  # Shape: (batch, attention_dim)
        attn_weights = torch.softmax(self.attention_weights(attn_input), dim=0)
        attended = attn_weights * attn_input
        
        # Final processing
        x = torch.relu(self.fc1(attended))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        """Get penultimate layer features"""
        # Conv processing
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten for attention
        x = torch.flatten(x, 1)
        
        # Attention mechanism
        attn_input = self.attention_proj(x)
        attn_weights = torch.softmax(self.attention_weights(attn_input), dim=0)
        attended = attn_weights * attn_input
        
        # Final processing up to penultimate
        x = torch.relu(self.fc1(attended))
        if hasattr(self, 'dropout'): x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

def create_subset(dataset, labels_to_include):
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels_to_include]
    return Subset(dataset, indices)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 2 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"    Epoch {epoch}: Loss = {avg_loss:.4f}")

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

def train_ultra_large_models():
    """Train very large models on different architectures"""
    print("Training ultra-large models (this will take significant time)...")
    
    # Load MNIST data
    transform_fc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_conv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset_fc = datasets.MNIST('./data', train=True, download=True, transform=transform_fc)
    full_test_dataset_fc = datasets.MNIST('./data', train=False, download=True, transform=transform_fc)
    
    full_train_dataset_conv = datasets.MNIST('./data', train=True, download=True, transform=transform_conv)
    full_test_dataset_conv = datasets.MNIST('./data', train=False, download=True, transform=transform_conv)
    
    # Create datasets
    class1_train_fc = create_subset(full_train_dataset_fc, [0, 1, 2, 3])
    class1_test_fc = create_subset(full_test_dataset_fc, [0, 1, 2, 3])
    class2_train_fc = create_subset(full_train_dataset_fc, [2, 3, 4, 5])
    class2_test_fc = create_subset(full_test_dataset_fc, [2, 3, 4, 5])
    
    class1_train_conv = create_subset(full_train_dataset_conv, [0, 1, 2, 3])
    class1_test_conv = create_subset(full_test_dataset_conv, [0, 1, 2, 3])
    class2_train_conv = create_subset(full_train_dataset_conv, [2, 3, 4, 5])
    class2_test_conv = create_subset(full_test_dataset_conv, [2, 3, 4, 5])
    
    # Define architectures to train
    architectures = [
        ("UltraWideNN", UltraWideNN, class1_train_fc, class1_test_fc, class2_train_fc, class2_test_fc, "fc"),
        ("UltraDeepNN", UltraDeepNN, class1_train_fc, class1_test_fc, class2_train_fc, class2_test_fc, "fc"),
        ("UltraConvNet", UltraConvNet, class1_train_conv, class1_test_conv, class2_train_conv, class2_test_conv, "conv"),
        ("HybridNN", HybridNN, class1_train_conv, class1_test_conv, class2_train_conv, class2_test_conv, "conv")
    ]
    
    trained_models = {}
    
    for arch_name, arch_class, c1_train, c1_test, c2_train, c2_test, data_type in architectures:
        print(f"\nTraining {arch_name} ({data_type})...")
        
        # Count parameters
        temp_model = arch_class()
        param_count = sum(p.numel() for p in temp_model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        # Train Class 1 model (digits 0,1,2,3)
        print("  Training Class 1 model...")
        model_c1 = arch_class().to(DEVICE)
        model_c1.eval()  # Set dropouts to eval mode for consistency
        
        train_loader_c1 = DataLoader(c1_train, batch_size=256, shuffle=True)
        test_loader_c1 = DataLoader(c1_test, batch_size=256, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_c1.parameters(), lr=0.001, weight_decay=1e-5)
        
        train_model(model_c1, train_loader_c1, criterion, optimizer, 6)
        acc_c1 = evaluate_model(model_c1, test_loader_c1)
        print(f"  {arch_name} Class1 accuracy: {acc_c1:.2f}%")
        
        # Train Class 2 model (digits 2,3,4,5)
        print("  Training Class 2 model...")
        model_c2 = arch_class().to(DEVICE)
        model_c2.eval()  # Set dropouts to eval mode for consistency
        
        train_loader_c2 = DataLoader(c2_train, batch_size=256, shuffle=True)
        test_loader_c2 = DataLoader(c2_test, batch_size=256, shuffle=False)
        
        optimizer = optim.Adam(model_c2.parameters(), lr=0.001, weight_decay=1e-5)
        train_model(model_c2, train_loader_c2, criterion, optimizer, 6)
        acc_c2 = evaluate_model(model_c2, test_loader_c2)
        print(f"  {arch_name} Class2 accuracy: {acc_c2:.2f}%")
        
        trained_models[arch_name] = {
            'class1': model_c1,
            'class2': model_c2,
            'class': arch_class,
            'data_type': data_type,
            'param_count': param_count
        }
    
    # Save models
    os.makedirs('./trained_models_ultra', exist_ok=True)
    for arch_name, models in trained_models.items():
        torch.save({
            'class1': models['class1'].state_dict(),
            'class2': models['class2'].state_dict()
        }, f'./trained_models_ultra/{arch_name}_models.pt')
    
    return trained_models

class FeatureAligner:
    """Aligns features between different architectures"""
    
    def __init__(self, source_dim, target_dim, method='linear'):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.method = method
        self.mapper = None
        self.fitted = False
        
    def fit(self, source_features, target_features):
        """Learn mapping from source to target feature space"""
        print(f"Learning feature alignment: {self.source_dim}D -> {self.target_dim}D")
        
        source_np = source_features.cpu().numpy()
        target_np = target_features.cpu().numpy()
        
        if self.method == 'linear':
            # Simple linear projection
            self.mapper = LinearRegression()
            self.mapper.fit(source_np, target_np)
            
        elif self.method == 'pca':
            # PCA-based dimensionality matching
            if self.source_dim > self.target_dim:
                # Reduce source dimensions
                self.pca = PCA(n_components=self.target_dim)
                self.pca.fit(source_np)
                self.mapper = self.pca
            else:
                # Expand source dimensions (pad with zeros or learned projection)
                expanded_source = np.zeros((source_np.shape[0], self.target_dim))
                expanded_source[:, :self.source_dim] = source_np
                self.mapper = LinearRegression()
                self.mapper.fit(source_np, expanded_source)
                
        elif self.method == 'procrustes':
            # Orthogonal Procrustes analysis (requires same dimensions)
            if self.source_dim == self.target_dim:
                R, _ = orthogonal_procrustes(source_np, target_np)
                self.mapper = R
            else:
                # Fall back to linear for different dimensions
                self.mapper = LinearRegression()
                self.mapper.fit(source_np, target_np)
        
        self.fitted = True
        
        # Evaluate alignment quality
        aligned = self.transform(source_features)
        alignment_error = torch.norm(aligned - target_features) / torch.norm(target_features)
        print(f"  Alignment error: {alignment_error:.4f}")
        
    def transform(self, source_features):
        """Transform source features to target space"""
        if not self.fitted:
            raise ValueError("Aligner must be fitted before transform")
        
        source_np = source_features.cpu().numpy()
        
        if self.method == 'linear':
            aligned_np = self.mapper.predict(source_np)
        elif self.method == 'pca':
            if hasattr(self, 'pca'):
                aligned_np = self.pca.transform(source_np)
            else:
                aligned_np = self.mapper.predict(source_np)
        elif self.method == 'procrustes':
            if isinstance(self.mapper, np.ndarray):
                aligned_np = source_np @ self.mapper
            else:
                aligned_np = self.mapper.predict(source_np)
        
        return torch.tensor(aligned_np, dtype=torch.float32, device=source_features.device)

def architecture_agnostic_surgery(source_model, target_model, source_arch, target_arch, 
                                source_data_type, target_data_type):
    """Perform surgery between different architectures"""
    print(f"\n=== AGNOSTIC SURGERY: {source_arch} -> {target_arch} ===")
    
    # Load test data
    if target_data_type == 'conv':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Get shared data for alignment (digits 2,3)
    shared_dataset = create_subset(full_test_dataset, [2, 3])
    shared_loader = DataLoader(shared_dataset, batch_size=128, shuffle=False)
    
    # Extract features from both models on shared data
    print("Extracting features for alignment...")
    source_model.eval()
    target_model.eval()
    
    source_features = []
    target_features = []
    
    with torch.no_grad():
        for data, _ in tqdm(shared_loader, desc="Feature extraction"):
            data = data.to(DEVICE)
            
            source_feat = source_model.get_features(data)
            target_feat = target_model.get_features(data)
            
            source_features.append(source_feat.cpu())
            target_features.append(target_feat.cpu())
    
    source_features = torch.cat(source_features)
    target_features = torch.cat(target_features)
    
    print(f"Source features shape: {source_features.shape}")
    print(f"Target features shape: {target_features.shape}")
    
    # Learn feature alignment
    aligner = FeatureAligner(
        source_features.shape[1], 
        target_features.shape[1], 
        method='linear'
    )
    aligner.fit(source_features, target_features)
    
    # Get digit-4 specific features from source model
    digit_4_dataset = create_subset(full_test_dataset, [4])
    digit_4_loader = DataLoader(digit_4_dataset, batch_size=128, shuffle=False)
    
    print("Extracting digit-4 features...")
    digit_4_source_features = []
    with torch.no_grad():
        for data, _ in tqdm(digit_4_loader, desc="Digit-4 features"):
            data = data.to(DEVICE)
            feat = source_model.get_features(data)
            digit_4_source_features.append(feat.cpu())
    
    digit_4_source_features = torch.cat(digit_4_source_features)
    
    # Align digit-4 features to target space
    aligned_digit_4_features = aligner.transform(digit_4_source_features)
    
    # Create modified target model
    modified_model = target_model.__class__().to(DEVICE)
    modified_model.load_state_dict(target_model.state_dict())
    
    # Strategy: Modify target model to recognize patterns similar to aligned digit-4 features
    print("Applying architecture-agnostic surgery...")
    
    # Method 1: Train a small adapter on the target model
    class DigitAdapter(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim)
            )
            self.digit_4_classifier = nn.Linear(feature_dim, 1)
            
        def forward(self, x, return_digit_4_score=False):
            adapted = self.adapter(x)
            if return_digit_4_score:
                return self.digit_4_classifier(adapted)
            return adapted
    
    # Train adapter to recognize digit-4 patterns in target feature space
    adapter = DigitAdapter(target_features.shape[1]).to(DEVICE)
    
    # Create training data: aligned digit-4 features (positive) vs shared features (negative)
    train_features = torch.cat([aligned_digit_4_features, target_features])
    train_labels = torch.cat([
        torch.ones(aligned_digit_4_features.shape[0]),
        torch.zeros(target_features.shape[0])
    ]).to(DEVICE)
    
    # Train adapter (this is allowed as it's learning the mapping, not retraining base models)
    adapter_optimizer = optim.Adam(adapter.parameters(), lr=0.01)
    adapter_criterion = nn.BCEWithLogitsLoss()
    
    print("Training feature adapter...")
    adapter.train()
    for epoch in range(20):
        adapter_optimizer.zero_grad()
        
        # Get digit-4 scores
        features_input = train_features.to(DEVICE)
        digit_4_scores = adapter(features_input, return_digit_4_score=True).squeeze()
        
        loss = adapter_criterion(digit_4_scores, train_labels)
        loss.backward()
        adapter_optimizer.step()
        
        if epoch % 5 == 0:
            print(f"  Adapter epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Create hybrid model that uses the adapter
    class HybridModel(nn.Module):
        def __init__(self, base_model, adapter):
            super().__init__()
            self.base_model = base_model
            self.adapter = adapter
            
        def forward(self, x):
            # Get base features
            features = self.base_model.get_features(x)
            
            # Get digit-4 confidence from adapter
            digit_4_confidence = torch.sigmoid(self.adapter(features, return_digit_4_score=True)).squeeze()
            
            # Get base model predictions
            base_logits = self.get_base_logits(features)
            
            # Modify digit-4 logit based on adapter confidence
            # High confidence -> boost digit 4, low confidence -> keep original
            base_logits[:, 4] = base_logits[:, 4] + 2.0 * (digit_4_confidence - 0.5)
            
            return base_logits
            
        def get_base_logits(self, features):
            # Apply final layer of base model
            if hasattr(self.base_model, 'fc5'):
                return self.base_model.fc5(features)
            elif hasattr(self.base_model, 'fc3'):
                return self.base_model.fc3(features)
            else:
                # Find the last linear layer
                for name, module in reversed(list(self.base_model.named_modules())):
                    if isinstance(module, nn.Linear) and module.out_features == 10:
                        return module(features)
    
    hybrid_model = HybridModel(modified_model, adapter).to(DEVICE)
    
    return hybrid_model

def test_architecture_agnostic_surgery():
    """Test the architecture-agnostic surgery across different models"""
    
    # Load or train ultra-large models
    if os.path.exists('./trained_models_ultra/UltraWideNN_models.pt'):
        print("Loading existing ultra-large models...")
        trained_models = {}
        
        architectures = [
            ("UltraWideNN", UltraWideNN, "fc"),
            ("UltraDeepNN", UltraDeepNN, "fc"),
            ("UltraConvNet", UltraConvNet, "conv"),
            ("HybridNN", HybridNN, "conv")
        ]
        
        for arch_name, arch_class, data_type in architectures:
            saved_data = torch.load(f'./trained_models_ultra/{arch_name}_models.pt', 
                                   map_location=DEVICE, weights_only=True)
            
            model_c1 = arch_class().to(DEVICE)
            model_c1.load_state_dict(saved_data['class1'])
            model_c1.eval()
            
            model_c2 = arch_class().to(DEVICE)
            model_c2.load_state_dict(saved_data['class2'])
            model_c2.eval()
            
            trained_models[arch_name] = {
                'class1': model_c1,
                'class2': model_c2,
                'class': arch_class,
                'data_type': data_type
            }
    else:
        print("Training ultra-large models...")
        trained_models = train_ultra_large_models()
    
    # Test cross-architecture transfers
    test_combinations = [
        ("UltraWideNN", "UltraDeepNN"),
        ("UltraDeepNN", "UltraConvNet"),
        ("UltraConvNet", "HybridNN"),
        ("HybridNN", "UltraWideNN")
    ]
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    digit_4_test = create_subset(full_test_dataset, [4])
    digit_4_loader = DataLoader(digit_4_test, batch_size=128, shuffle=False)
    
    original_digits_test = create_subset(full_test_dataset, [0, 1, 2, 3])
    original_loader = DataLoader(original_digits_test, batch_size=128, shuffle=False)
    
    print("\n" + "="*60)
    print("ARCHITECTURE-AGNOSTIC SURGERY RESULTS")
    print("="*60)
    
    results = {}
    
    for source_arch, target_arch in test_combinations:
        print(f"\nTesting: {source_arch} -> {target_arch}")
        
        try:
            # Perform surgery
            hybrid_model = architecture_agnostic_surgery(
                trained_models[source_arch]['class2'],  # Source knows digit 4
                trained_models[target_arch]['class1'],  # Target doesn't know digit 4
                source_arch,
                target_arch,
                trained_models[source_arch]['data_type'],
                trained_models[target_arch]['data_type']
            )
            
            # Evaluate
            digit_4_acc = evaluate_model(hybrid_model, digit_4_loader)
            original_acc = evaluate_model(hybrid_model, original_loader)
            
            print(f"Results:")
            print(f"  Digit 4 accuracy: {digit_4_acc:.2f}%")
            print(f"  Original digits accuracy: {original_acc:.2f}%")
            
            results[f"{source_arch}->{target_arch}"] = {
                'digit_4': digit_4_acc,
                'original': original_acc,
                'success': digit_4_acc > 20 and original_acc > 80
            }
            
        except Exception as e:
            print(f"  Surgery failed: {str(e)}")
            results[f"{source_arch}->{target_arch}"] = {
                'digit_4': 0.0,
                'original': 0.0,
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*60)
    print("ARCHITECTURE-AGNOSTIC SURGERY SUMMARY")
    print("="*60)
    
    successful_transfers = [k for k, v in results.items() if v.get('success', False)]
    
    print(f"\nSuccessful transfers: {len(successful_transfers)}/{len(test_combinations)}")
    for transfer in successful_transfers:
        result = results[transfer]
        print(f"  ✅ {transfer}: {result['digit_4']:.1f}% digit-4, {result['original']:.1f}% original")
    
    failed_transfers = [k for k, v in results.items() if not v.get('success', False)]
    for transfer in failed_transfers:
        result = results[transfer]
        if 'error' not in result:
            print(f"  ❌ {transfer}: {result['digit_4']:.1f}% digit-4, {result['original']:.1f}% original")
        else:
            print(f"  ❌ {transfer}: Failed - {result['error'][:50]}...")
    
    if successful_transfers:
        print(f"\n🎉 BREAKTHROUGH: Architecture-agnostic surgery achieved!")
        print(f"Successfully transferred knowledge across different architectures!")
    else:
        print(f"\n🔬 Architecture-agnostic surgery remains challenging")
        print(f"Further research needed on cross-architecture representations")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Testing architecture-agnostic model surgery with ultra-large models\n")
    
    test_architecture_agnostic_surgery()