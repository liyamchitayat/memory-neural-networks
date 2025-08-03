# Neural Memory Transfer - Project Structure

**Updated:** August 1, 2025

## 📁 Directory Organization

```
memory_transfer_nn/
├── 📁 src/                          # Source code (organized by functionality)
│   ├── core/                        # Core framework components
│   │   ├── architectures.py         # Neural network architectures (WideNN, DeepNN)
│   │   └── experimental_framework.py # Experiment management and data handling
│   ├── transfer_methods/            # Transfer learning implementations
│   │   ├── neural_concept_transfer.py      # Base neural concept transfer
│   │   ├── balanced_transfer.py            # Balanced transfer approach
│   │   ├── robust_balanced_transfer.py     # Robust balanced transfer (rho blending)
│   │   ├── improved_sae_robust_transfer.py # SAE-based transfer learning
│   │   └── knowledge_preserving_transfer.py # Knowledge preservation methods
│   ├── evaluation/                  # Evaluation metrics and assessment
│   │   ├── corrected_metrics.py     # Corrected transfer metrics
│   │   └── fixed_corrected_metrics.py # Fixed evaluation metrics
│   └── experiments/                 # Experiment runners
│       └── run_sae_shared_knowledge_experiments.py # SAE experiments
├── 📁 scripts/                      # Shell scripts and automation
│   └── run_fixed_clean_experiment.sh # Clean experiment runner
├── 📁 docs/                         # Documentation and reports
│   ├── SAE_COMPREHENSIVE_RESULTS_SUMMARY.md        # SAE experiment results
│   ├── SHARED_LAYER_TRANSFER_SPECIFICATIONS.md     # Shared layer approach specs
│   ├── TRIPLE_VERIFIED_REQUIREMENTS_REPORT.md      # Requirements analysis
│   └── General_Requirements.txt                     # Original requirements
├── 📁 results/                      # Experiment results (organized by approach)
│   ├── fixed_clean_comparison/      # Fixed clean comparison experiments
│   │   ├── improved_sae/           # SAE results
│   │   └── rho_blending/           # Rho blending results
│   └── sae_shared_knowledge/       # SAE shared knowledge experiments
├── 📁 data/                        # Dataset storage
│   └── MNIST/                      # MNIST dataset files
├── 📁 logs/                        # Execution logs and debugging info
├── 📁 archive_unused/              # Archived/unused code (preserved for reference)
├── 📁 tests/                       # Unit tests (future)
├── README.md                       # Main project README
├── requirements.txt                # Python dependencies
└── PROJECT_STRUCTURE.md           # This file
```

## 🎯 Key Components

### Core Framework (`src/core/`)

- **`architectures.py`**: Neural network definitions
  - `WideNN`: 6-layer wide architecture (max width 256)
  - `DeepNN`: 8-layer deep architecture (max width 128)
  - Feature extraction and classification methods

- **`experimental_framework.py`**: Experiment infrastructure
  - `MNISTDataManager`: Data loading and preprocessing
  - `ExperimentConfig`: Configuration management
  - `ModelTrainer`: Neural network training utilities

### Transfer Methods (`src/transfer_methods/`)

- **`neural_concept_transfer.py`**: Core transfer learning framework
  - Sparse Autoencoder (SAE) implementation
  - Concept alignment using Orthogonal Procrustes
  - Free space discovery and concept injection

- **`improved_sae_robust_transfer.py`**: **Primary SAE Implementation**
  - Trainable per-feature integration weights
  - Improved concept injection mechanisms
  - Robust final layer adaptation

- **`robust_balanced_transfer.py`**: Rho blending baseline
  - Fixed blending parameter approach
  - Baseline for SAE comparison

### Evaluation (`src/evaluation/`)

- **`corrected_metrics.py`**: Transfer learning evaluation metrics
  - Knowledge preservation assessment
  - Transfer effectiveness measurement
  - Transfer specificity validation

### Experiments (`src/experiments/`)

- **`run_sae_shared_knowledge_experiments.py`**: **Main SAE Experiment Runner**
  - Tests SAE with different shared knowledge amounts
  - Comprehensive evaluation and reporting
  - Multi-seed reliability testing

## 🧪 Experiment Categories

### 1. SAE Shared Knowledge Experiments
**Location:** `results/sae_shared_knowledge/`
**Purpose:** Test how shared knowledge affects SAE transfer effectiveness
**Scenarios:**
- Low: [2,3,4] → [0,1,2] (1 shared class)
- Medium: [2,3,4,5,6] → [0,1,2,3,4] (3 shared classes) 
- High: [2,3,4,5,6,7,8,9] → [0,1,2,3,4,5,6,7] (6 shared classes)

### 2. Fixed Clean Comparison
**Location:** `results/fixed_clean_comparison/`
**Purpose:** Compare SAE vs Rho Blending approaches
**Setup:** [2,3,4,5,6,7] → [0,1,2,3,4,5] transfer class 6

### 3. Shared Layer Transfer (Archived)
**Status:** Completed on separate branch `mnist-transfer-from-shared-layers`
**Finding:** True zero-shot transfer impossible with frozen networks

## 🚀 Quick Start

### Running SAE Experiments
```bash
cd src/experiments/
python run_sae_shared_knowledge_experiments.py
```

### Running Shell Scripts
```bash
cd scripts/
./run_fixed_clean_experiment.sh
```

### Viewing Results
```bash
# SAE comprehensive results
cat docs/SAE_COMPREHENSIVE_RESULTS_SUMMARY.md

# Shared layer specifications  
cat docs/SHARED_LAYER_TRANSFER_SPECIFICATIONS.md
```

## 📊 Key Results Summary

### SAE Transfer Learning Performance:
- **Transfer Effectiveness**: 78-94% (exceeds 70% target)
- **Knowledge Preservation**: 33-86% (variable, depends on shared knowledge)
- **Transfer Specificity**: 0% contamination (perfect)

### Main Finding:
SAE demonstrates excellent transfer capability but with a knowledge-preservation trade-off that depends on the amount of shared knowledge between source and target domains.

## 🔧 Dependencies

Install requirements:
```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch
- NumPy
- SciPy
- scikit-learn

## 📝 Development Notes

### Import Structure
Files use relative imports within packages and absolute imports from project root:
```python
# Within transfer_methods package
from .robust_balanced_transfer import RobustBalancedTransferSystem

# From experiments to core
from src.core.architectures import WideNN, DeepNN
```

### Adding New Experiments
1. Create experiment file in `src/experiments/`
2. Add results directory in `results/`
3. Update this documentation
4. Consider adding tests in `tests/`

### File Naming Convention
- Core implementations: descriptive names (e.g., `neural_concept_transfer.py`)
- Experiment runners: `run_*_experiments.py`
- Results: `approach_scenario_seed_*.json`
- Documentation: `UPPERCASE_WITH_UNDERSCORES.md`

## 🗂️ Archive Information

The `archive_unused/` directory contains earlier experimental versions that are preserved for reference but not part of the main codebase. These include initial transfer attempts and debugging scripts.

---

This structure provides clear separation of concerns, making the codebase maintainable and extensible for future neural memory transfer research.