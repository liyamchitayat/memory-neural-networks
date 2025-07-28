#!/bin/bash
#SBATCH --job-name=sae_experiment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=5:59:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# SAE Concept Injection Experiments - Complete Automated Setup and Execution
# 
# This script automatically sets up the environment and runs all experiments
# for SAE-free concept injection methods. No prior knowledge required.
#
# WHAT THIS DOES:
# 1. Creates a clean conda environment with exact dependencies
# 2. Downloads MNIST dataset automatically  
# 3. Runs comprehensive experiments for Methods 1 & 2
# 4. Generates detailed results and analysis
# 5. Creates visualizations and summary reports
#
# REQUIREMENTS:
# - NVIDIA GPU with 8GB+ VRAM (GTX 1080 Ti or better)
# - 16GB+ system RAM
# - 50GB+ free disk space
# - conda or miniconda installed
# - 4-6 hours of uninterrupted runtime
#
# USAGE:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Author: SAE Research Team
# Date: $(date +%Y-%m-%d)

set -e  # Exit on any error
set -u  # Exit on undefined variables

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project settings
PROJECT_NAME="sae_concept_injection"
CONDA_ENV_NAME="sae_concept_injection"
PYTHON_VERSION="3.9.16"

# Directories
BASE_DIR="$(pwd)"
RESULTS_DIR="${BASE_DIR}/results"
LOGS_DIR="${BASE_DIR}/logs"
DATA_DIR="${BASE_DIR}/data"
CHECKPOINTS_DIR="${BASE_DIR}/checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timing
START_TIME=$(date +%s)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${CYAN}================================================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        echo "Please install $1 and try again"
        exit 1
    fi
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "Checking GPU availability..."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu; do
            print_info "Found GPU: $gpu"
        done
        
        # Check if we have enough VRAM
        min_vram=8000
        max_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
        if [ "$max_vram" -lt "$min_vram" ]; then
            print_warning "GPU has ${max_vram}MB VRAM. Recommended: ${min_vram}MB+"
            print_warning "Experiments may run slower or fail due to memory constraints"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_success "GPU memory check passed: ${max_vram}MB VRAM available"
        fi
    else
        print_warning "nvidia-smi not found. Cannot verify GPU availability"
        print_warning "Make sure you have NVIDIA drivers and CUDA installed"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

create_directories() {
    print_step "Creating project directories..."
    mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$DATA_DIR" "$CHECKPOINTS_DIR"
    print_success "Directories created: results/, logs/, data/, checkpoints/"
}

# =============================================================================
# SYSTEM REQUIREMENTS CHECK
# =============================================================================

check_system_requirements() {
    print_header "SYSTEM REQUIREMENTS CHECK"
    
    print_step "Checking required commands..."
    check_command "conda"
    check_command "python"
    check_command "git"
    
    print_step "Checking system resources..."
    
    # Check available memory (Linux/macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        print_info "Available RAM: ${mem_gb}GB"
        if [ "$mem_gb" -lt 16 ]; then
            print_warning "Less than 16GB RAM detected. Experiments may be slower."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        print_info "Available RAM: ${mem_gb}GB"
        if [ "$mem_gb" -lt 16 ]; then
            print_warning "Less than 16GB RAM detected. Experiments may be slower."
        fi
    fi
    
    # Check available disk space
    avail_space=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    print_info "Available disk space: ${avail_space}GB"
    if [ "${avail_space%.*}" -lt 50 ]; then
        print_warning "Less than 50GB free space. May not be sufficient for all data and results."
    fi
    
    # Check GPU
    check_gpu
    
    print_success "System requirements check completed"
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_conda_environment() {
    print_header "CONDA ENVIRONMENT SETUP"
    
    print_step "Checking if conda environment exists..."
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Environment '${CONDA_ENV_NAME}' already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            print_info "Using existing environment"
            return 0
        fi
    fi
    
    print_step "Creating conda environment with Python ${PYTHON_VERSION}..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    
    print_step "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    print_step "Installing core dependencies..."
    # Install PyTorch with CUDA support
    if command -v nvidia-smi &> /dev/null; then
        print_info "Installing PyTorch with CUDA support..."
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "Installing PyTorch CPU version (no GPU detected)..."
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_step "Installing scientific computing packages..."
    pip install \
        numpy==1.24.3 \
        scipy==1.11.1 \
        scikit-learn==1.3.0 \
        matplotlib==3.7.1 \
        seaborn==0.12.2 \
        pandas==2.0.3 \
        tqdm==4.65.0 \
        tensorboard==2.13.0 \
        jupyter==1.0.0
    
    print_step "Verifying installation..."
    python -c "
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sklearn
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - {torch.cuda.get_device_name(i)}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Scikit-learn version: {sklearn.__version__}')
print('✓ All dependencies installed successfully!')
    "
    
    print_success "Conda environment setup completed"
}

# =============================================================================
# PROJECT SETUP
# =============================================================================

setup_project_files() {
    print_header "PROJECT FILES SETUP"
    
    print_step "Verifying required Python files exist..."
    
    required_files=(
        "neural_architectures.py"
        "method1_precomputed_vector_alignment.py"
        "method2_cross_architecture_alignment.py"
        "run_all_tests.py"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        else
            print_info "✓ Found: $file"
        fi
    done
    
    if [[ ${#missing_files[@]} -ne 0 ]]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            print_error "  - $file"
        done
        print_error "Please ensure all Python implementation files are in the current directory"
        exit 1
    fi
    
    print_step "Creating configuration file..."
    cat > config.json << 'EOF'
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
      "concept_dimensions": [32, 48, 64],
      "sparsity_weights": [0.030, 0.050, 0.080],
      "injection_strengths": [0.4]
    },
    "method2": {
      "alignment_types": ["nonlinear", "procrustes"],
      "hidden_dimensions": [128],
      "alignment_epochs": 50
    }
  }
}
EOF
    
    print_step "Testing neural network architectures..."
    python neural_architectures.py > architecture_test.log 2>&1
    if [[ $? -eq 0 ]]; then
        print_success "✓ Architecture tests passed"
        rm architecture_test.log
    else
        print_error "Architecture tests failed. Check architecture_test.log for details."
        exit 1
    fi
    
    print_success "Project files setup completed"
}

# =============================================================================
# DATASET PREPARATION
# =============================================================================

prepare_datasets() {
    print_header "DATASET PREPARATION"
    
    print_step "Testing MNIST dataset download..."
    python -c "
import torchvision
import torchvision.transforms as transforms
import numpy as np

print('Downloading MNIST dataset...')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f'✓ Training samples: {len(train_dataset):,}')
print(f'✓ Test samples: {len(test_dataset):,}')

# Verify digit distribution
print('\nDigit distribution verification:')
train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

total_train = 0
total_test = 0
for digit in range(10):
    train_count = np.sum(train_labels == digit)
    test_count = np.sum(test_labels == digit)
    total_train += train_count
    total_test += test_count
    print(f'  Digit {digit}: Train={train_count:,}, Test={test_count:,}')

print(f'\nTotal: Train={total_train:,}, Test={total_test:,}')

# Verify our specific splits
train_0_3 = np.sum((train_labels >= 0) & (train_labels <= 3))
test_0_3 = np.sum((test_labels >= 0) & (test_labels <= 3))
test_4 = np.sum(test_labels == 4)
test_5 = np.sum(test_labels == 5)

print(f'\nExperiment splits:')
print(f'  Training digits 0-3: {train_0_3:,} samples')
print(f'  Test digits 0-3 (preservation): {test_0_3:,} samples')
print(f'  Test digit 4 (transfer): {test_4:,} samples')  
print(f'  Test digit 5 (specificity): {test_5:,} samples')

print('✓ Dataset preparation completed successfully!')
    " 2>&1 | tee dataset_preparation.log
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_success "Dataset preparation completed"
        rm dataset_preparation.log
    else
        print_error "Dataset preparation failed. Check dataset_preparation.log for details."
        exit 1
    fi
}

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

run_quick_test() {
    print_header "QUICK TEST EXECUTION (30 minutes)"
    
    print_step "Running quick test to verify everything works..."
    print_info "This will run a minimal test suite to catch any issues early"
    
    local start_time=$(date +%s)
    
    timeout 2400 python run_all_tests.py --quick-test 2>&1 | tee "${LOGS_DIR}/quick_test.log"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Quick test completed successfully in ${duration_min} minutes"
        
        # Check for result files
        if ls results/master_test_results_*.json 1> /dev/null 2>&1; then
            latest_result=$(ls -t results/master_test_results_*.json | head -n1)
            print_info "Results saved to: $latest_result"
            
            # Show quick summary
            python -c "
import json
try:
    with open('$latest_result', 'r') as f:
        results = json.load(f)
    if 'overall_summary' in results:
        summary = results['overall_summary']
        print(f'\\n✓ Quick Test Results:')
        print(f'  - Total experiments: {summary[\"total_experiments\"]}')
        print(f'  - Best transfer accuracy: {summary[\"best_overall_transfer\"]:.1f}%')
        print(f'  - Best preservation accuracy: {summary[\"best_overall_preservation\"]:.1f}%')
        print(f'  - Best specificity: {summary[\"best_overall_specificity\"]:.1f}%')
    else:
        print('Results file found but no summary available')
except Exception as e:
    print(f'Could not parse results: {e}')
            "
        fi
        return 0
    elif [[ $exit_code -eq 124 ]]; then
        print_error "Quick test timed out after 40 minutes"
        return 1
    else
        print_error "Quick test failed with exit code $exit_code"
        print_error "Check ${LOGS_DIR}/quick_test.log for details"
        return 1
    fi
}

run_full_experiments() {
    print_header "FULL EXPERIMENT EXECUTION (4-6 hours)"
    
    print_step "Starting comprehensive experiment suite..."
    print_info "This will run Methods 1 & 2 across same and cross architectures"
    print_info "Estimated time: 4-6 hours depending on hardware"
    print_warning "Do not interrupt this process - experiments will need to restart"
    
    # Create detailed log file
    local experiment_log="${LOGS_DIR}/full_experiments_$(date +%Y%m%d_%H%M%S).log"
    local start_time=$(date +%s)
    
    # Show progress indicator
    print_step "Starting experiments with progress logging..."
    print_info "Monitor progress: tail -f $experiment_log"
    
    # Run experiments with timeout (6 hours max)
    timeout 21600 python run_all_tests.py --methods 1,2 --architectures same,cross 2>&1 | tee "$experiment_log"
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_hours=$((duration / 3600))
    local duration_min=$(((duration % 3600) / 60))
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Full experiments completed in ${duration_hours}h ${duration_min}m"
        
        # Analyze results
        analyze_results
        
        return 0
    elif [[ $exit_code -eq 124 ]]; then
        print_error "Experiments timed out after 6 hours"
        print_error "This may indicate hardware limitations or implementation issues"
        return 1
    else
        print_error "Experiments failed with exit code $exit_code"
        print_error "Check $experiment_log for details"
        return 1
    fi
}

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

analyze_results() {
    print_header "RESULTS ANALYSIS"
    
    print_step "Analyzing experimental results..."
    
    # Find latest result files
    if ls results/master_test_results_*.json 1> /dev/null 2>&1; then
        latest_master=$(ls -t results/master_test_results_*.json | head -n1)
        print_info "Master results: $latest_master"
        
        # Generate comprehensive analysis
        python -c "
import json
import glob
import numpy as np
from datetime import datetime

def analyze_all_results():
    # Load master results
    with open('$latest_master', 'r') as f:
        master_results = json.load(f)
    
    print('=' * 80)
    print('COMPREHENSIVE EXPERIMENT RESULTS ANALYSIS')
    print('=' * 80)
    
    # Test run info
    if 'test_run_info' in master_results:
        info = master_results['test_run_info']
        print(f'\\nExperiment Run Information:')
        print(f'  Start Time: {info.get(\"start_time\", \"Unknown\")}')
        print(f'  End Time: {info.get(\"end_time\", \"Unknown\")}')
        print(f'  Total Duration: {info.get(\"total_duration_minutes\", 0):.1f} minutes')
        print(f'  Device Used: {info.get(\"device\", \"Unknown\")}')
        print(f'  PyTorch Version: {info.get(\"torch_version\", \"Unknown\")}')
        print(f'  CUDA Available: {info.get(\"cuda_available\", \"Unknown\")}')
    
    # Overall summary
    if 'overall_summary' in master_results:
        summary = master_results['overall_summary']
        print(f'\\n🏆 OVERALL PERFORMANCE SUMMARY:')
        print(f'  Total Experiments: {summary[\"total_experiments\"]}')
        print(f'  Best Transfer Accuracy: {summary[\"best_overall_transfer\"]:.1f}%')
        print(f'  Average Transfer Accuracy: {summary[\"avg_overall_transfer\"]:.1f}%')
        print(f'  Best Preservation Accuracy: {summary[\"best_overall_preservation\"]:.1f}%')
        print(f'  Average Preservation Accuracy: {summary[\"avg_overall_preservation\"]:.1f}%')
        print(f'  Best Specificity: {summary[\"best_overall_specificity\"]:.1f}% (lower is better)')
        print(f'  Average Specificity: {summary[\"avg_overall_specificity\"]:.1f}%')
        
        # Compare to documented benchmarks
        print(f'\\n📊 BENCHMARK COMPARISON:')
        
        # Method 1 benchmarks
        method1_target_transfer = 56.1
        method1_target_preservation = 98.2
        method1_target_specificity = 4.9
        
        # Method 2 benchmarks  
        method2_target_transfer = 42.2
        method2_target_preservation = 98.7
        method2_target_specificity = 5.1
        
        print(f'  Method 1 Targets vs Achieved:')
        print(f'    Transfer: {method1_target_transfer}% (target) vs {summary[\"best_overall_transfer\"]:.1f}% (achieved)')
        print(f'    Preservation: {method1_target_preservation}% (target) vs {summary[\"best_overall_preservation\"]:.1f}% (achieved)')
        print(f'    Specificity: {method1_target_specificity}% (target) vs {summary[\"best_overall_specificity\"]:.1f}% (achieved)')
        
        # Performance assessment
        transfer_match = abs(summary['best_overall_transfer'] - method1_target_transfer) <= 3.0
        preservation_match = abs(summary['best_overall_preservation'] - method1_target_preservation) <= 2.0
        specificity_match = abs(summary['best_overall_specificity'] - method1_target_specificity) <= 2.0
        
        print(f'\\n✅ BENCHMARK VALIDATION:')
        print(f'  Transfer Accuracy: {\"✓ PASS\" if transfer_match else \"✗ FAIL\"} (within ±3%)')
        print(f'  Preservation Accuracy: {\"✓ PASS\" if preservation_match else \"✗ FAIL\"} (within ±2%)')
        print(f'  Specificity: {\"✓ PASS\" if specificity_match else \"✗ FAIL\"} (within ±2%)')
        
        overall_pass = transfer_match and preservation_match and specificity_match
        print(f'  Overall Assessment: {\"✅ EXPERIMENTS SUCCESSFUL\" if overall_pass else \"⚠️  REVIEW NEEDED\"}')
    
    # Method-specific analysis
    if 'method_results' in master_results:
        print(f'\\n📋 METHOD-SPECIFIC RESULTS:')
        
        for method_key, method_data in master_results['method_results'].items():
            if 'experiments' in method_data and method_data['experiments']:
                method_name = method_data.get('method_name', method_key)
                experiments = method_data['experiments']
                
                transfers = [exp['transfer_accuracy'] for exp in experiments]
                preservations = [exp['preservation_accuracy'] for exp in experiments]
                specificities = [exp['specificity_accuracy'] for exp in experiments]
                
                print(f'\\n  {method_name}:')
                print(f'    Experiments: {len(experiments)}')
                print(f'    Transfer - Best: {max(transfers):.1f}%, Avg: {np.mean(transfers):.1f}%, Range: {min(transfers):.1f}-{max(transfers):.1f}%')
                print(f'    Preservation - Best: {max(preservations):.1f}%, Avg: {np.mean(preservations):.1f}%')
                print(f'    Specificity - Best: {min(specificities):.1f}%, Avg: {np.mean(specificities):.1f}%')
                
                # Architecture analysis
                same_arch_exps = [exp for exp in experiments if 'same_arch' in exp.get('experiment_id', '')]
                cross_arch_exps = [exp for exp in experiments if 'cross_arch' in exp.get('experiment_id', '')]
                
                if same_arch_exps:
                    same_transfers = [exp['transfer_accuracy'] for exp in same_arch_exps]
                    print(f'    Same Architecture: {len(same_arch_exps)} experiments, Best: {max(same_transfers):.1f}%')
                
                if cross_arch_exps:
                    cross_transfers = [exp['transfer_accuracy'] for exp in cross_arch_exps]
                    print(f'    Cross Architecture: {len(cross_arch_exps)} experiments, Best: {max(cross_transfers):.1f}%')
    
    print(f'\\n' + '=' * 80)
    return master_results

# Run analysis
try:
    results = analyze_all_results()
    print('\\n✅ Results analysis completed successfully!')
except Exception as e:
    print(f'❌ Results analysis failed: {e}')
    import traceback
    traceback.print_exc()
        " 2>&1 | tee "${LOGS_DIR}/results_analysis.log"
        
        print_success "Results analysis completed"
        
    else
        print_warning "No result files found to analyze"
    fi
    
    # List all generated files
    print_step "Generated files summary:"
    echo
    print_info "📁 Results Directory (./results/):"
    if ls results/*.json 1> /dev/null 2>&1; then
        for file in results/*.json; do
            size=$(du -h "$file" | cut -f1)
            print_info "  📄 $(basename "$file") (${size})"
        done
    else
        print_warning "  No result files found"
    fi
    
    echo
    print_info "📁 Logs Directory (./logs/):"
    if ls logs/*.log 1> /dev/null 2>&1; then
        for file in logs/*.log; do
            size=$(du -h "$file" | cut -f1)
            lines=$(wc -l < "$file")
            print_info "  📝 $(basename "$file") (${size}, ${lines} lines)"
        done
    else
        print_warning "  No log files found"
    fi
    
    echo
    print_info "📁 Data Directory (./data/):"
    if [[ -d "data/MNIST" ]]; then
        data_size=$(du -sh data/ | cut -f1)
        print_info "  📊 MNIST dataset (${data_size})"
    fi
}

# =============================================================================
# CLEANUP AND FINAL REPORT
# =============================================================================

generate_final_report() {
    print_header "GENERATING FINAL REPORT"
    
    local report_file="EXPERIMENT_REPORT_$(date +%Y%m%d_%H%M%S).md"
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    local total_hours=$((total_duration / 3600))
    local total_min=$(((total_duration % 3600) / 60))
    
    print_step "Creating comprehensive experiment report..."
    
cat > "$report_file" << EOF
# SAE Concept Injection Experiments - Execution Report

**Generated:** $(date)  
**Duration:** ${total_hours}h ${total_min}m  
**System:** $(uname -a)  
**User:** $(whoami)  

## Execution Summary

This report documents the automated execution of SAE concept injection experiments
using the comprehensive testing framework. All experiments were run automatically
with no manual intervention required.

### System Configuration

- **Operating System:** $(uname -s) $(uname -r)
- **Python Version:** $(python --version)
- **Conda Environment:** ${CONDA_ENV_NAME}
- **Working Directory:** ${BASE_DIR}
- **GPU Information:**
$(if command -v nvidia-smi &> /dev/null; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  - /'; else echo "  - No NVIDIA GPU detected"; fi)

### Execution Timeline

1. **Environment Setup:** $(date -d @$START_TIME)
2. **Dataset Preparation:** Auto-downloaded MNIST
3. **Quick Test:** $(if [[ -f "${LOGS_DIR}/quick_test.log" ]]; then echo "✅ Completed"; else echo "❌ Skipped"; fi)
4. **Full Experiments:** $(if ls results/master_test_results_*.json 1> /dev/null 2>&1; then echo "✅ Completed"; else echo "❌ Failed or incomplete"; fi)
5. **Results Analysis:** $(if [[ -f "${LOGS_DIR}/results_analysis.log" ]]; then echo "✅ Completed"; else echo "❌ Skipped"; fi)

### Results Files Generated

$(if ls results/*.json 1> /dev/null 2>&1; then
    echo "#### Result Files:"
    for file in results/*.json; do
        echo "- \`$(basename "$file")\` ($(du -h "$file" | cut -f1))"
    done
else
    echo "❌ No result files generated"
fi)

$(if ls logs/*.log 1> /dev/null 2>&1; then
    echo "#### Log Files:"
    for file in logs/*.log; do
        echo "- \`$(basename "$file")\` ($(wc -l < "$file") lines)"
    done
fi)

### Performance Summary

$(if ls results/master_test_results_*.json 1> /dev/null 2>&1; then
    latest_result=$(ls -t results/master_test_results_*.json | head -n1)
    python -c "
import json
try:
    with open('$latest_result', 'r') as f:
        results = json.load(f)
    if 'overall_summary' in results:
        summary = results['overall_summary']
        print(f'- **Total Experiments:** {summary[\"total_experiments\"]}')
        print(f'- **Best Transfer Accuracy:** {summary[\"best_overall_transfer\"]:.1f}%')
        print(f'- **Best Preservation Accuracy:** {summary[\"best_overall_preservation\"]:.1f}%')
        print(f'- **Best Specificity:** {summary[\"best_overall_specificity\"]:.1f}%')
        print(f'- **Average Transfer:** {summary[\"avg_overall_transfer\"]:.1f}%')
        print(f'- **Average Preservation:** {summary[\"avg_overall_preservation\"]:.1f}%')
    else:
        print('Results file exists but no summary available')
except:
    print('Could not parse results file')
    "
else
    echo "❌ No performance data available"
fi)

### Reproducibility Information

This experiment was run using the automated setup script with the following configuration:

\`\`\`json
$(cat config.json)
\`\`\`

### Next Steps

1. **Review Results:** Examine the JSON files in \`results/\` directory
2. **Check Logs:** Review detailed logs in \`logs/\` directory for any issues
3. **Compare Benchmarks:** Verify results match documented performance targets
4. **Run Additional Methods:** Extend to Methods 3-9 using the same framework

### Files to Archive

For complete reproducibility, archive these files:
- All files in \`results/\` directory
- All files in \`logs/\` directory  
- \`config.json\` configuration file
- This report (\`$report_file\`)

### Support

If results don't match expected benchmarks or if any issues occurred:
1. Check the detailed logs in \`logs/\` directory
2. Verify GPU memory and system requirements
3. Ensure all Python files are present and correct
4. Re-run with \`--quick-test\` flag to isolate issues

---
*Report generated automatically by run_experiments.sh*
EOF

    print_success "Final report generated: $report_file"
}

cleanup_environment() {
    print_header "CLEANUP AND FINALIZATION"
    
    # Clean up temporary files
    print_step "Cleaning up temporary files..."
    rm -f *.tmp *.temp
    
    # Deactivate conda environment
    if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
        print_step "Deactivating conda environment..."
        conda deactivate
    fi
    
    # Final file permissions
    print_step "Setting file permissions..."
    chmod 644 results/*.json 2>/dev/null || true
    chmod 644 logs/*.log 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

main() {
    print_header "SAE CONCEPT INJECTION EXPERIMENTS - AUTOMATED EXECUTION"
    echo
    print_info "This script will automatically set up and run comprehensive SAE concept injection experiments"
    print_info "Estimated total time: 4-6 hours"
    print_info "No user intervention required after confirmation"
    echo
    
    # Confirmation prompt
    read -p "$(echo -e ${YELLOW}Continue with automated experiment execution? [y/N]:${NC} )" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Execution cancelled by user"
        exit 0
    fi
    
    # Main execution flow
    set -e  # Exit on any error from here
    
    # Phase 1: System and Environment Setup
    check_system_requirements
    create_directories
    setup_conda_environment
    setup_project_files
    prepare_datasets
    
    # Phase 2: Experiment Execution
    print_header "EXPERIMENT EXECUTION PHASE"
    
    # Always run quick test first
    if run_quick_test; then
        print_success "Quick test passed - proceeding with full experiments"
        
        # Ask user if they want to continue with full suite
        echo
        read -p "$(echo -e ${YELLOW}Quick test successful. Continue with full experiment suite (4-6 hours)? [y/N]:${NC} )" -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if run_full_experiments; then
                print_success "🎉 All experiments completed successfully!"
            else
                print_error "Full experiments failed. Check logs for details."
                exit 1
            fi
        else
            print_info "Full experiments skipped by user choice"
            print_info "Quick test results are available in results/ directory"
        fi
    else
        print_error "Quick test failed. Please check logs and fix issues before running full suite."
        exit 1
    fi
    
    # Phase 3: Finalization
    generate_final_report
    cleanup_environment
    
    # Final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    local total_hours=$((total_duration / 3600))
    local total_min=$(((total_duration % 3600) / 60))
    
    print_header "EXECUTION COMPLETED SUCCESSFULLY"
    echo
    print_success "🎉 SAE Concept Injection Experiments Completed!"
    print_info "📊 Total execution time: ${total_hours}h ${total_min}m"
    print_info "📁 Results saved in: ./results/"
    print_info "📝 Logs saved in: ./logs/"
    print_info "📋 Final report: $(ls EXPERIMENT_REPORT_*.md | tail -1)"
    echo
    print_info "Next steps:"
    print_info "1. Review the final report for detailed results"
    print_info "2. Check JSON files in results/ for raw data"
    print_info "3. Compare performance against documented benchmarks"
    print_info "4. Archive all files for reproducibility"
    echo
    print_success "✅ Experiment execution framework ready for additional methods!"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Handle script interruption gracefully
trap 'print_error "Script interrupted by user"; cleanup_environment; exit 130' INT TERM

# Run main function
main "$@"