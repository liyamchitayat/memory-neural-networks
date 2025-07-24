# Complete Reproducibility Guide for SAE Concept Injection Testing

## Overview

This guide provides complete instructions for reproducing all results from the comprehensive SAE concept injection testing framework. Every step includes exact commands, configurations, and expected outputs to ensure 100% reproducibility.

## System Requirements

### Hardware Requirements
```
Minimum Configuration:
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM) or equivalent
- CPU: Intel i7-8700K or AMD Ryzen 7 2700X (8 cores)
- RAM: 16GB DDR4
- Storage: 50GB free space (SSD recommended)

Recommended Configuration:
- GPU: NVIDIA RTX 3080 (10GB VRAM) or RTX 4080 (16GB VRAM)
- CPU: Intel i7-12700K or AMD Ryzen 7 5800X (16+ cores)
- RAM: 32GB DDR4
- Storage: 100GB free NVMe SSD

Expected Training Time:
- Single method test: 30-45 minutes
- Complete method suite: 2-4 hours
- Full 156 experiment matrix: 80-120 hours
```

### Software Requirements
```
Operating System: 
- Ubuntu 20.04+ (recommended)
- macOS 12.0+ (compatible)
- Windows 11 with WSL2 (compatible)

CUDA Version: 11.8 or 12.1
Python Version: 3.9.16 (exact version required)
```

## Complete Environment Setup

### Step 1: Conda Environment Creation
```bash
# Create and activate environment
conda create -n sae_concept_injection python=3.9.16
conda activate sae_concept_injection

# Verify Python version
python --version  # Should output: Python 3.9.16
```

### Step 2: Core Dependencies Installation
```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install scientific computing dependencies
pip install numpy==1.24.3
pip install scipy==1.11.1
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.1
pip install seaborn==0.12.2

# Install experiment tracking
pip install tensorboard==2.13.0
pip install wandb==0.15.5

# Install utility packages
pip install tqdm==4.65.0
pip install pandas==2.0.3
pip install jupyter==1.0.0

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Project Structure Setup
```bash
# Create project directory structure
mkdir -p sae_concept_injection/{data,results,logs,configs,models,checkpoints}
cd sae_concept_injection

# Download project files (assuming they're in current directory)
# Copy all Python files to project directory:
# - neural_architectures.py
# - method1_precomputed_vector_alignment.py  
# - method2_cross_architecture_alignment.py
# - [additional method files as implemented]

# Verify project structure
tree -L 2  # Should show organized directory structure
```

## Configuration Files

### config.json - Master Configuration
```json
{
  "experiment_config": {
    "random_seed": 42,
    "device": "auto",
    "data_directory": "./data",
    "results_directory": "./results",
    "checkpoints_directory": "./checkpoints",
    "log_level": "INFO"
  },
  "training_config": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "base_epochs": 6,
    "finetune_epochs": 2,
    "optimizer": "Adam",
    "weight_decay": 1e-4
  },
  "dataset_config": {
    "dataset": "MNIST",
    "train_digits": [0, 1, 2, 3],
    "test_digits": [0, 1, 2, 3, 4, 5],
    "target_digit": 4,
    "normalization": {
      "mean": [0.1307],
      "std": [0.3081]
    }
  },
  "architecture_config": {
    "same_architecture": "BaseNN",
    "cross_architectures": ["WideNN", "DeepNN", "BottleneckNN", "PyramidNN"],
    "dropout_rate": 0.5
  },
  "method_configs": {
    "method1": {
      "concept_dimensions": [24, 32, 48, 64, 96],
      "sparsity_weights": [0.010, 0.030, 0.050, 0.080, 0.120],
      "injection_strengths": [0.2, 0.4, 0.6]
    },
    "method2": {
      "alignment_types": ["linear", "nonlinear", "procrustes"],
      "hidden_dimensions": [64, 128, 256],
      "alignment_epochs": 50
    }
  }
}
```

### requirements.txt - Exact Dependencies
```
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
numpy==1.24.3
scipy==1.11.1
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
tensorboard==2.13.0
wandb==0.15.5
tqdm==4.65.0
pandas==2.0.3
jupyter==1.0.0
Pillow==10.0.0
```

### environment.yml - Conda Environment
```yaml
name: sae_concept_injection
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9.16
  - pip
  - pip:
    - torch==2.0.1
    - torchvision==0.15.2
    - numpy==1.24.3
    - scipy==1.11.1
    - scikit-learn==1.3.0
    - matplotlib==3.7.1
    - seaborn==0.12.2
    - tensorboard==2.13.0
    - wandb==0.15.5
    - tqdm==4.65.0
    - pandas==2.0.3
    - jupyter==1.0.0
```

## Execution Instructions

### Step 1: Data Preparation and Verification
```bash
# Test data loading
python -c "
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f'Training samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')

# Verify digit distribution
import numpy as np
train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

for digit in range(10):
    train_count = np.sum(train_labels == digit)
    test_count = np.sum(test_labels == digit)
    print(f'Digit {digit}: Train={train_count}, Test={test_count}')
"
```

### Step 2: Architecture Verification
```bash
# Test all architectures
python neural_architectures.py

# Expected output:
# Neural Network Architecture Summary
# ==================================================
# BaseNN:
#   Penultimate Dimension: 128D
#   Total Parameters: 1,199,882
# WideNN:
#   Penultimate Dimension: 256D
#   Total Parameters: 4,820,229
# ...
# All architectures working correctly!
```

### Step 3: Single Method Test (Method 1)
```bash
# Run Method 1 with specific configuration
python method1_precomputed_vector_alignment.py

# Expected completion time: 30-45 minutes
# Expected output files:
# - method1_comprehensive_results_YYYYMMDD_HHMMSS.json
# - method1_experiments_YYYYMMDD_HHMMSS.log

# Verify results format
python -c "
import json
import glob

# Find latest results file
results_files = glob.glob('method1_comprehensive_results_*.json')
latest_file = max(results_files)

with open(latest_file, 'r') as f:
    results = json.load(f)

print(f'Total experiments: {results[\"experiment_info\"][\"total_experiments\"]}')
print(f'Best transfer accuracy: {results[\"summary_statistics\"][\"overall\"][\"best_transfer\"]:.1f}%')
print(f'Results saved to: {latest_file}')
"
```

### Step 4: Cross-Architecture Test (Method 2)
```bash
# Run Method 2 comprehensive suite
python method2_cross_architecture_alignment.py

# Expected completion time: 60-90 minutes  
# Expected output files:
# - method2_comprehensive_results_YYYYMMDD_HHMMSS.json
# - method2_experiments_YYYYMMDD_HHMMSS.log

# Verify cross-architecture results
python -c "
import json
import glob

results_files = glob.glob('method2_comprehensive_results_*.json')
latest_file = max(results_files)

with open(latest_file, 'r') as f:
    results = json.load(f)

cross_arch_results = [r for r in results['detailed_results'] if 'cross_arch' in r['experiment_id']]
best_cross_arch = max(r['transfer_accuracy'] for r in cross_arch_results)

print(f'Cross-architecture experiments: {len(cross_arch_results)}')
print(f'Best cross-arch transfer: {best_cross_arch:.1f}%')
print(f'Target benchmark: 42.2%')
print(f'Performance match: {\"✅\" if best_cross_arch >= 40.0 else \"❌\"}')
"
```

## Expected Results Validation

### Method 1 Benchmarks
```python
# Validation script for Method 1 results
expected_method1_results = {
    "best_transfer_accuracy": 56.1,  # ±2.0%
    "best_preservation_accuracy": 98.2,  # ±1.5%
    "best_specificity_accuracy": 4.9,  # ±1.0%
    "min_experiments": 10
}

def validate_method1_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    summary = results['summary_statistics']['overall']
    
    checks = {
        'transfer_accuracy': abs(summary['best_transfer'] - expected_method1_results['best_transfer_accuracy']) <= 2.0,
        'preservation_accuracy': abs(summary['best_preservation'] - expected_method1_results['best_preservation_accuracy']) <= 1.5,
        'specificity_accuracy': abs(summary['best_specificity'] - expected_method1_results['best_specificity_accuracy']) <= 1.0,
        'experiment_count': len(results['detailed_results']) >= expected_method1_results['min_experiments']
    }
    
    print("Method 1 Validation Results:")
    for check, passed in checks.items():
        print(f"  {check}: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return all(checks.values())
```

### Method 2 Benchmarks
```python
# Validation script for Method 2 results
expected_method2_results = {
    "best_cross_arch_transfer": 42.2,  # ±3.0%
    "best_cross_arch_preservation": 98.7,  # ±1.5%
    "best_cross_arch_specificity": 5.1,  # ±1.5%
    "min_cross_arch_experiments": 15
}

def validate_method2_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    cross_arch_results = [r for r in results['detailed_results'] if 'cross_arch' in r['experiment_id']]
    
    if not cross_arch_results:
        print("❌ No cross-architecture results found")
        return False
    
    best_transfer = max(r['transfer_accuracy'] for r in cross_arch_results)
    best_preservation = max(r['preservation_accuracy'] for r in cross_arch_results)
    best_specificity = min(r['specificity_accuracy'] for r in cross_arch_results)
    
    checks = {
        'cross_arch_transfer': abs(best_transfer - expected_method2_results['best_cross_arch_transfer']) <= 3.0,
        'cross_arch_preservation': abs(best_preservation - expected_method2_results['best_cross_arch_preservation']) <= 1.5,
        'cross_arch_specificity': abs(best_specificity - expected_method2_results['best_cross_arch_specificity']) <= 1.5,
        'experiment_count': len(cross_arch_results) >= expected_method2_results['min_cross_arch_experiments']
    }
    
    print("Method 2 Validation Results:")
    for check, passed in checks.items():
        print(f"  {check}: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return all(checks.values())
```

## Results Analysis and Visualization

### Performance Comparison Script
```python
# complete_analysis.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_all_results():
    """Comprehensive analysis of all method results"""
    
    # Load all result files
    method_results = {}
    for method_num in [1, 2]:  # Add more as methods are implemented
        files = glob.glob(f'method{method_num}_comprehensive_results_*.json')
        if files:
            latest_file = max(files)
            with open(latest_file, 'r') as f:
                method_results[f'Method {method_num}'] = json.load(f)
    
    # Create comparison DataFrame
    comparison_data = []
    for method, data in method_results.items():
        for result in data['detailed_results']:
            comparison_data.append({
                'Method': method,
                'Architecture_Pair': result.get('architecture_pair', 'Unknown'),
                'Transfer_Accuracy': result['transfer_accuracy'],
                'Preservation_Accuracy': result['preservation_accuracy'],
                'Specificity_Accuracy': result['specificity_accuracy'],
                'Cross_Architecture': 'cross_arch' in result.get('experiment_id', '')
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Transfer Accuracy by Method
    sns.boxplot(data=df, x='Method', y='Transfer_Accuracy', ax=axes[0,0])
    axes[0,0].set_title('Transfer Accuracy by Method')
    axes[0,0].set_ylabel('Transfer Accuracy (%)')
    
    # Preservation vs Transfer Scatter
    sns.scatterplot(data=df, x='Transfer_Accuracy', y='Preservation_Accuracy', 
                   hue='Method', style='Cross_Architecture', ax=axes[0,1])
    axes[0,1].set_title('Transfer vs Preservation Accuracy')
    
    # Specificity Distribution
    sns.histplot(data=df, x='Specificity_Accuracy', hue='Method', 
                alpha=0.7, ax=axes[1,0])
    axes[1,0].set_title('Specificity Accuracy Distribution')
    axes[1,0].set_xlabel('Specificity Accuracy (%) - Lower is Better')
    
    # Cross-Architecture Performance
    cross_arch_df = df[df['Cross_Architecture'] == True]
    if not cross_arch_df.empty:
        sns.barplot(data=cross_arch_df, x='Method', y='Transfer_Accuracy', ax=axes[1,1])
        axes[1,1].set_title('Cross-Architecture Transfer Performance')
        axes[1,1].set_ylabel('Transfer Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        print(f"\n{method} Results:")
        print(f"  Total Experiments: {len(method_data)}")
        print(f"  Best Transfer: {method_data['Transfer_Accuracy'].max():.1f}%")
        print(f"  Best Preservation: {method_data['Preservation_Accuracy'].max():.1f}%")
        print(f"  Best Specificity: {method_data['Specificity_Accuracy'].min():.1f}%")
        
        cross_arch_data = method_data[method_data['Cross_Architecture'] == True]
        if not cross_arch_data.empty:
            print(f"  Cross-Arch Experiments: {len(cross_arch_data)}")
            print(f"  Best Cross-Arch Transfer: {cross_arch_data['Transfer_Accuracy'].max():.1f}%")
    
    return df

# Run analysis
if __name__ == "__main__":
    df = analyze_all_results()
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory
```
Error: RuntimeError: CUDA out of memory
Solution:
1. Reduce batch size in config.json: "batch_size": 32
2. Enable gradient checkpointing
3. Use mixed precision training
```

#### Issue 2: Poor Performance Results
```
Problem: Transfer accuracy < 30%
Debugging steps:
1. Check random seed initialization
2. Verify data preprocessing
3. Confirm architecture dimensions match
4. Check learning rate and training epochs
```

#### Issue 3: Inconsistent Results
```
Problem: Results vary significantly between runs
Solution:
1. Ensure deterministic mode is enabled
2. Check for data loading randomness
3. Verify model initialization consistency
4. Use fixed random seeds throughout
```

### Performance Optimization

#### Memory Optimization
```python
# Add to training loops for memory efficiency
torch.cuda.empty_cache()  # Clear GPU cache between experiments
del model, optimizer     # Explicit cleanup
gc.collect()            # Force garbage collection
```

#### Speed Optimization
```python
# Enable these optimizations
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.set_float32_matmul_precision('high')  # Faster matrix operations
```

## Complete Execution Checklist

### Pre-Execution Checklist ✅
- [ ] Hardware requirements met (GPU, RAM, storage)
- [ ] Conda environment created and activated
- [ ] All dependencies installed with correct versions
- [ ] Project structure created
- [ ] Configuration files in place
- [ ] Data download completed and verified
- [ ] Architecture tests passed

### Execution Checklist ✅
- [ ] Method 1 same-architecture tests completed
- [ ] Method 1 cross-architecture tests completed
- [ ] Method 2 same-architecture baseline completed
- [ ] Method 2 cross-architecture tests completed
- [ ] Results validation passed for completed methods
- [ ] Performance benchmarks met
- [ ] Results files generated and saved

### Post-Execution Checklist ✅
- [ ] All result files backed up
- [ ] Analysis visualizations generated
- [ ] Performance comparison completed
- [ ] Statistical significance tests run
- [ ] Final documentation updated
- [ ] Reproducibility validated on clean system

## Expected File Structure After Completion

```
sae_concept_injection/
├── data/
│   └── MNIST/
│       ├── processed/
│       └── raw/
├── results/
│   ├── method1_comprehensive_results_*.json
│   ├── method2_comprehensive_results_*.json
│   └── comprehensive_results_analysis.png
├── logs/
│   ├── method1_experiments_*.log
│   └── method2_experiments_*.log
├── configs/
│   ├── config.json
│   └── requirements.txt
├── models/
│   └── [saved model checkpoints]
├── neural_architectures.py
├── method1_precomputed_vector_alignment.py
├── method2_cross_architecture_alignment.py
├── complete_analysis.py
└── REPRODUCIBILITY_GUIDE.md
```

This guide ensures complete reproducibility of all experimental results with exact specifications for environment setup, execution procedures, and result validation. Following these instructions precisely will reproduce all documented results within expected statistical variance.