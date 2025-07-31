#!/bin/bash

# Fixed Shared Layer Transfer Experiments
# This script runs all corrected experiments:
# 1. Cross-architecture transfers (WideNN↔DeepNN)  
# 2. Data overlap size experiments (0, 1, 2, 3 shared classes)

echo "=================================================================="
echo "FIXED SHARED LAYER TRANSFER EXPERIMENTS"
echo "=================================================================="
echo "Corrected bugs:"
echo "1. ✅ Proper projection layers for cross-architecture transfers"
echo "2. ✅ Fixed weight synchronization (truly shared layers)"
echo "3. ✅ Correct experimental setup matching requirements"
echo "4. ✅ Fixed catastrophic forgetting issues"
echo "=================================================================="
echo ""

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if required files exist
if [ ! -f "fixed_shared_layer_experiment.py" ]; then
    echo "❌ ERROR: fixed_shared_layer_experiment.py not found!"
    exit 1
fi

if [ ! -f "architectures.py" ]; then
    echo "❌ ERROR: architectures.py not found!"
    exit 1
fi

if [ ! -f "experimental_framework.py" ]; then
    echo "❌ ERROR: experimental_framework.py not found!"
    exit 1
fi

echo "🔍 All required files found."
echo ""

# Create results directory
mkdir -p experiment_results/fixed_shared_layer_transfer

# Log file for complete output
LOG_FILE="experiment_results/fixed_shared_layer_transfer/experiment_log.txt"
echo "📝 Logging all output to: $LOG_FILE"
echo ""

# Function to log with timestamp
log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start experiments
log_with_time "Starting Fixed Shared Layer Transfer Experiments"
log_with_time "Expected duration: ~2-3 hours"
log_with_time "Experiments:"
log_with_time "- Cross-architecture: 4 arch pairs × 5 seeds = 20 experiments"
log_with_time "- Data overlap: 4 overlap sizes × 5 seeds = 20 experiments"
log_with_time "- Total: 40 experiments"
echo ""

# Run the main experiment
log_with_time "🚀 Running experiments..."
echo ""

# Execute the experiment with both stdout and stderr captured
python fixed_shared_layer_experiment.py 2>&1 | tee -a "$LOG_FILE"

# Check if the experiment completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log_with_time "✅ Experiments completed successfully!"
    echo ""
    log_with_time "📊 Results saved to: experiment_results/fixed_shared_layer_transfer/"
    echo ""
    
    # Show summary of results files
    echo "📋 Generated files:"
    ls -la experiment_results/fixed_shared_layer_transfer/ | grep -E "\.(json|txt)$" | while read line; do
        echo "   $line"
    done
    echo ""
    
    # Quick analysis of results
    log_with_time "🔍 Quick Results Analysis:"
    
    # Count successful experiments
    CROSS_ARCH_COUNT=$(ls experiment_results/fixed_shared_layer_transfer/cross_arch_*.json 2>/dev/null | wc -l)
    DATA_OVERLAP_COUNT=$(ls experiment_results/fixed_shared_layer_transfer/data_overlap_*.json 2>/dev/null | wc -l)
    
    log_with_time "- Cross-architecture experiments: $CROSS_ARCH_COUNT"
    log_with_time "- Data overlap experiments: $DATA_OVERLAP_COUNT"
    log_with_time "- Total completed: $((CROSS_ARCH_COUNT + DATA_OVERLAP_COUNT))"
    
    if [ -f "experiment_results/fixed_shared_layer_transfer/all_experiments_summary.json" ]; then
        log_with_time "- Summary file created: ✅"
    else
        log_with_time "- Summary file created: ❌"
    fi
    
    echo ""
    log_with_time "🎉 All experiments completed! Check results in experiment_results/fixed_shared_layer_transfer/"
    
else
    log_with_time "❌ Experiments failed with exit code: ${PIPESTATUS[0]}"
    echo ""
    log_with_time "Check the log file for details: $LOG_FILE"
    echo ""
    echo "Common issues to check:"
    echo "1. CUDA memory issues (try reducing batch size)"
    echo "2. Missing dependencies (torch, torchvision, etc.)"
    echo "3. Insufficient disk space"
    echo "4. Network connection issues (for MNIST download)"
    exit 1
fi

echo ""
echo "=================================================================="
echo "EXPERIMENT EXECUTION COMPLETE"
echo "=================================================================="