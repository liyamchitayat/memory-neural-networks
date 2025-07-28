#!/bin/bash

# SAE Direct Integration Experiment Runner
# Tests architectural alternatives to rho blending

set -e

echo "🧪 SAE DIRECT INTEGRATION EXPERIMENT"
echo "====================================="
echo ""
echo "This experiment tests what happens when we integrate SAE features"
echo "directly into the model instead of using rho blending:"
echo ""
echo "Traditional:  final = ρ * original + (1-ρ) * enhanced"
echo "Direct:       final = SAE_integration(original, enhanced)"
echo ""
echo "Integration modes to test:"
echo "  • REPLACE: Use only SAE features"
echo "  • ADD: Add SAE features to original"  
echo "  • CONCAT: Concatenate both feature types"
echo ""

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python 3 found"
echo ""

# Create results directory
mkdir -p results
echo "📁 Results directory created: sae_integration_experiment/results/"
echo ""

# Run the experiment
echo "🚀 Starting SAE integration experiment..."
echo ""

cd "$(dirname "$0")"
python3 sae_integration_main.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results available in results/ directory:"
    echo "   • sae_integration_detailed_results.json - Complete results data"
    echo "   • sae_integration_summary.json - Statistical analysis"
    echo "   • SAE_INTEGRATION_vs_RHO_BLENDING_REPORT.md - Comparison report"
    echo ""
    echo "🔍 Key comparisons:"
    echo "   • Direct integration vs rho blending performance"
    echo "   • REPLACE vs ADD vs CONCAT integration modes"
    echo "   • Impact on transfer effectiveness and knowledge preservation"
    echo ""
    echo "📈 Use these results to inform architectural choices for neural concept transfer!"
else
    echo ""
    echo "❌ Experiment failed. Check the output above for errors."
    exit 1
fi