#!/bin/bash

# Shared Knowledge Analysis for Wide->Wide Neural Concept Transfer
# Tests required shared knowledge between networks with different overlap ratios

echo "Starting Shared Knowledge Analysis for Wide->Wide Transfer"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/shared_knowledge_analysis_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "=========================================================="

# Run the shared knowledge analysis
python shared_knowledge_analysis.py 2>&1 | tee "$LOG_FILE"

# Check if the script completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "Shared Knowledge Analysis completed successfully!"
    echo "Results saved in: experiment_results/shared_knowledge_analysis/"
    echo "Log file: $LOG_FILE"
    echo "=========================================================="
else
    echo ""
    echo "=========================================================="
    echo "Shared Knowledge Analysis failed. Check the log file:"
    echo "$LOG_FILE"
    echo "=========================================================="
    exit 1
fi