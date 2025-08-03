# Neural Memory Transfer - Organized Codebase

**Updated:** August 1, 2025

This repository implements neural concept transfer systems enabling selective knowledge transfer between neural networks. The codebase has been comprehensively organized for clarity, maintainability, and extensibility.

## 🎯 Key Achievements

### SAE Transfer Learning Results:
- **Transfer Effectiveness**: 78-94% (exceeds 70% target)
- **Knowledge Preservation**: 33-86% (variable, depends on shared knowledge)
- **Transfer Specificity**: 0% contamination (perfect)

### Shared Layer Transfer Analysis:
- **Finding**: True zero-shot transfer impossible with frozen networks
- **Alternative**: Feature Bridging achieves 94% accuracy but requires target samples

## 🚀 Quick Start

### Run SAE Experiments
```bash
cd src/experiments/
python run_sae_shared_knowledge_experiments.py
```

### View Comprehensive Results
```bash
# SAE experiment summary
cat docs/SAE_COMPREHENSIVE_RESULTS_SUMMARY.md

# Shared layer analysis
cat docs/SHARED_LAYER_TRANSFER_SPECIFICATIONS.md
```

## 📁 **NEW: Organized Project Structure**

```
memory_transfer_nn/
├── 📁 src/                          # Source code (organized by functionality)
│   ├── core/                        # Core framework components
│   │   ├── architectures.py         # Neural network architectures
│   │   └── experimental_framework.py # Experiment management
│   ├── transfer_methods/            # Transfer learning implementations
│   │   ├── neural_concept_transfer.py      # Base neural concept transfer
│   │   ├── improved_sae_robust_transfer.py # **Primary SAE Implementation**
│   │   └── robust_balanced_transfer.py     # Rho blending baseline
│   ├── evaluation/                  # Evaluation metrics
│   │   └── corrected_metrics.py     # Transfer learning evaluation
│   └── experiments/                 # Experiment runners
│       └── run_sae_shared_knowledge_experiments.py # **Main SAE Experiments**
├── 📁 scripts/                      # Shell scripts and automation
├── 📁 docs/                         # Documentation and reports
├── 📁 results/                      # Experiment results (organized by approach)
├── 📁 data/                         # Dataset storage
└── 📁 archive_unused/               # Archived code (preserved for reference)
```

**📋 See `PROJECT_STRUCTURE.md` for complete documentation**

## 🧪 Experiment Categories

### 1. SAE Shared Knowledge Experiments ⭐
**Primary Focus**: Test how shared knowledge affects SAE transfer effectiveness

**Scenarios Tested:**
- **Low Shared Knowledge**: [2,3,4] → [0,1,2] transfer 3 (1 shared class)
- **Medium Shared Knowledge**: [2,3,4,5,6] → [0,1,2,3,4] transfer 5 (3 shared classes)
- **High Shared Knowledge**: [2,3,4,5,6,7,8,9] → [0,1,2,3,4,5,6,7] transfer 8 (6 shared classes)

**Results Location**: `results/sae_shared_knowledge/`

### 2. Fixed Clean Comparison
**Purpose**: Compare SAE vs Rho Blending approaches
**Setup**: [2,3,4,5,6,7] → [0,1,2,3,4,5] transfer class 6
**Results Location**: `results/fixed_clean_comparison/`

### 3. Shared Layer Transfer Analysis
**Status**: Completed (separate branch analysis)
**Key Finding**: Feature Bridging works but isn't true transfer learning
**Documentation**: `docs/SHARED_LAYER_TRANSFER_SPECIFICATIONS.md`

## 🏗️ Architecture Support

- **WideNN**: 6 layers, max 256 width (784→256→256→256→128→64→10)
- **DeepNN**: 8 layers, max 128 width (784→128→128→96→96→64→64→32→10)
- **Cross-architecture transfer**: Supported via projection layers

## 📊 Key Results Summary

### SAE Transfer Learning Performance

| Scenario | Shared Classes | Transfer Success | Knowledge Preservation | Transfer Specificity |
|----------|----------------|------------------|----------------------|---------------------|
| **Low** | 1 class | ✅ **78-94%** | ❌ **33%** | ✅ **0%** |
| **High** | 6+ classes | ✅ **82%** | ✅ **86%** | ✅ **0%** |

### Key Insight: Knowledge-Transfer Trade-off
- **More shared knowledge** → Better knowledge preservation
- **Less shared knowledge** → Higher transfer success but knowledge degradation
- **Perfect specificity** across all scenarios (0% contamination)

## 🔬 Technical Components

### Core SAE System (`src/transfer_methods/improved_sae_robust_transfer.py`)
1. **Sparse Autoencoders**: Extract concept representations
2. **Orthogonal Procrustes Alignment**: Align concept spaces between models
3. **Free Space Discovery**: Find non-interfering injection directions
4. **Trainable Integration Weights**: Per-feature blending (improvement over single rho)
5. **Concept Injection Module**: Selective concept injection with adaptive strength

### Evaluation Framework (`src/evaluation/corrected_metrics.py`)
- **Knowledge Preservation**: Original capability retention
- **Transfer Effectiveness**: New class recognition accuracy  
- **Transfer Specificity**: Contamination prevention

## 🔧 Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0 (for Orthogonal Procrustes)
- scikit-learn >= 1.1.0

## 📈 Usage Examples

### Running Specific Experiments

```bash
# Complete SAE shared knowledge analysis
cd src/experiments/
python run_sae_shared_knowledge_experiments.py

# Quick single experiment test
cd scripts/
./run_fixed_clean_experiment.sh
```

### Importing Components

```python
# Import core components
from src.core.architectures import WideNN, DeepNN
from src.core.experimental_framework import ExperimentConfig, MNISTDataManager

# Import transfer methods
from src.transfer_methods.improved_sae_robust_transfer import ImprovedSAERobustTransferSystem

# Import evaluation
from src.evaluation.corrected_metrics import CorrectedMetricsEvaluator
```

## 📚 Documentation

- **`PROJECT_STRUCTURE.md`**: Complete codebase organization guide
- **`docs/SAE_COMPREHENSIVE_RESULTS_SUMMARY.md`**: SAE experiment results
- **`docs/SHARED_LAYER_TRANSFER_SPECIFICATIONS.md`**: Shared layer approach analysis
- **`docs/TRIPLE_VERIFIED_REQUIREMENTS_REPORT.md`**: Requirements compliance

## 🎓 Scientific Contributions

1. **SAE Transfer Learning Framework**: Trainable per-feature integration weights
2. **Shared Knowledge Analysis**: Quantified impact on transfer effectiveness
3. **Cross-Architecture Transfer**: Projection layers for architecture compatibility
4. **Shared Layer Analysis**: Proved limitations of frozen network approaches
5. **Comprehensive Evaluation**: Rigorous metrics for transfer learning assessment

## 🚀 Future Extensions

The organized structure supports:
- **New Transfer Methods**: Add to `src/transfer_methods/`
- **Additional Experiments**: Extend `src/experiments/`
- **Enhanced Evaluation**: Improve `src/evaluation/`
- **Different Architectures**: Modify `src/core/architectures.py`
- **Unit Testing**: Utilize `tests/` directory

## 🏁 Getting Started

1. **Clone repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run SAE experiments**: `cd src/experiments/ && python run_sae_shared_knowledge_experiments.py`
4. **Explore results**: Check `results/` directory and `docs/` for analysis
5. **Review structure**: See `PROJECT_STRUCTURE.md` for detailed organization

---

**The codebase is now professionally organized, documented, and ready for research and development! 🚀**