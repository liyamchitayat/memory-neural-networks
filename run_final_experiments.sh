#!/bin/bash

# Final Shared Layer Transfer Experiments
# Runs the exact experiments specified in requirements:
# 1. [0,1,2] → [2,3,4] transfer digit 3 (cross-architecture + WideNN→WideNN)
# 2. [0,1,2,3,4] → [2,3,4,5,6] transfer digit 5 (WideNN→WideNN only)  
# 3. [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] transfer digit 8 (WideNN→WideNN only)

echo "=================================================================="
echo "FINAL SHARED LAYER TRANSFER EXPERIMENTS"
echo "=================================================================="
echo "Experiments to run:"
echo "1. Transfer digit 3: [0,1,2] → [2,3,4] (CROSS-ARCHITECTURE)"
echo "   - WideNN→WideNN, WideNN→DeepNN, DeepNN→WideNN, DeepNN→DeepNN"
echo "   - 5 seeds each = 20 experiments"
echo ""
echo "2. Transfer digit 5: [0,1,2,3,4] → [2,3,4,5,6] (WideNN→WideNN ONLY)"
echo "   - 5 seeds = 5 experiments"
echo ""
echo "3. Transfer digit 8: [0,1,2,3,4,5,6,7] → [2,3,4,5,6,7,8,9] (WideNN→WideNN ONLY)"
echo "   - 5 seeds = 5 experiments"
echo ""
echo "TOTAL: 30 experiments"
echo ""
echo "Bug fixes applied:"
echo "✅ Proper projection layers for cross-architecture transfers"
echo "✅ True weight sharing (same module instance)"
echo "✅ Fixed batch normalization in projection layers"
echo "✅ Correct transfer class identification"
echo "✅ Proper evaluation metrics"
echo "=================================================================="
echo ""

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if required files exist
required_files=("final_shared_layer_experiment.py" "architectures.py" "experimental_framework.py" "neural_concept_transfer.py")

echo "🔍 Checking required files..."
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: $file not found!"
        echo "Make sure you're running this script from the correct directory."
        exit 1
    else
        echo "✅ $file found"
    fi
done
echo ""

# Check Python and dependencies
echo "🐍 Checking Python environment..."
python -c "import torch, torchvision, numpy, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Missing required Python packages!"
    echo "Please install: torch, torchvision, numpy, scipy"
    echo "Run: pip install torch torchvision numpy scipy"
    exit 1
else
    echo "✅ Python dependencies found"
fi
echo ""

# Check CUDA availability
echo "🔧 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA available for GPU acceleration"
else
    echo "⚠️  CUDA not available, using CPU (will be slower)"
fi
echo ""

# Create results directory
results_dir="experiment_results/final_shared_layer_transfer"
mkdir -p "$results_dir"
echo "📁 Results will be saved to: $results_dir"
echo ""

# Set up logging
log_file="$results_dir/experiment_log.txt"
error_log="$results_dir/error_log.txt"

# Function to log with timestamp
log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

# Function to handle cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        log_with_time "❌ Experiments interrupted or failed"
        echo ""
        echo "Check logs for details:"
        echo "  - Full log: $log_file"
        echo "  - Error log: $error_log"
    fi
}
trap cleanup EXIT

# Start experiments
log_with_time "🚀 Starting Final Shared Layer Transfer Experiments"
log_with_time "Expected duration: ~3-4 hours (depends on hardware)"
log_with_time "Logging to: $log_file"
echo ""

# Run the main experiment
log_with_time "Executing: python final_shared_layer_experiment.py"
echo "This may take a while... Check $log_file for detailed progress."
echo ""

# Execute with comprehensive logging
{
    python final_shared_layer_experiment.py
    experiment_exit_code=$?
} > >(tee -a "$log_file") 2> >(tee -a "$error_log" >&2)

# Check results
if [ $experiment_exit_code -eq 0 ]; then
    log_with_time "✅ Experiments completed successfully!"
    echo ""
    
    # Count results
    transfer_3_count=$(ls "$results_dir"/transfer_digit_3_*.json 2>/dev/null | wc -l)
    transfer_5_count=$(ls "$results_dir"/transfer_digit_5_*.json 2>/dev/null | wc -l)
    transfer_8_count=$(ls "$results_dir"/transfer_digit_8_*.json 2>/dev/null | wc -l)
    
    log_with_time "📊 Results Summary:"
    log_with_time "  Transfer digit 3 experiments: $transfer_3_count"
    log_with_time "  Transfer digit 5 experiments: $transfer_5_count"
    log_with_time "  Transfer digit 8 experiments: $transfer_8_count"
    log_with_time "  Total experiments: $((transfer_3_count + transfer_5_count + transfer_8_count))"
    
    # Check for summary files
    summary_files=(
        "transfer_digit_3_summary.json"
        "transfer_digit_5_summary.json" 
        "transfer_digit_8_summary.json"
        "all_experiments_summary.json"
    )
    
    echo ""
    log_with_time "📋 Generated summary files:"
    for summary in "${summary_files[@]}"; do
        if [ -f "$results_dir/$summary" ]; then
            log_with_time "  ✅ $summary"
        else
            log_with_time "  ❌ $summary (missing)"
        fi
    done
    
    # Quick performance analysis
    if [ -f "$results_dir/all_experiments_summary.json" ]; then
        echo ""
        log_with_time "🔍 Quick Analysis:"
        
        # Extract some basic stats using Python
        python << EOF
import json
try:
    with open("$results_dir/all_experiments_summary.json", 'r') as f:
        data = json.load(f)
    
    total = data.get('total_experiments', 0)
    print(f"  Total experiments completed: {total}")
    
    if 'overall_statistics' in data:
        for exp_name, stats in data['overall_statistics'].items():
            if 'transfer_improvement' in stats:
                transfer_mean = stats['transfer_improvement']['mean']
                transfer_std = stats['transfer_improvement']['std']
                retention_mean = stats['original_retention']['mean']
                print(f"  {exp_name}:")
                print(f"    Transfer improvement: {transfer_mean:.3f} ± {transfer_std:.3f}")
                print(f"    Original retention: {retention_mean:.3f}")
except Exception as e:
    print(f"  Could not analyze results: {e}")
EOF
    fi
    
    echo ""
    log_with_time "🎉 All experiments completed successfully!"
    log_with_time "📂 Results location: $results_dir"
    echo ""
    echo "Next steps:"
    echo "1. Review results in: $results_dir"
    echo "2. Check summary files for statistics"
    echo "3. Create visualizations if needed"
    
else
    log_with_time "❌ Experiments failed with exit code: $experiment_exit_code"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check error log: $error_log"
    echo "2. Check full log: $log_file"
    echo "3. Common issues:"
    echo "   - CUDA out of memory (reduce batch size in experimental_framework.py)"
    echo "   - Missing MNIST data (check internet connection)"
    echo "   - Insufficient disk space"
    echo "   - Python package versions"
    echo ""
    exit 1
fi

echo ""
echo "=================================================================="
echo "FINAL EXPERIMENT EXECUTION COMPLETE"
echo "=================================================================="