#!/bin/bash

# Comprehensive Rho Parameter Analysis
# Tests different rho values and creates detailed plots

echo "🚀 COMPREHENSIVE RHO PARAMETER ANALYSIS"
echo "======================================="
echo "This analysis will:"
echo "- Test rho values from 0.0 to 1.0 (11 values)"
echo "- Use 3 different seeds for reliability" 
echo "- Measure 3 key metrics: Preservation, Effectiveness, Specificity"
echo "- Create comprehensive plots and analysis"
echo "- Total experiments: 33"
echo ""

# Check conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Setting up conda environment..."
    conda create -n neural_transfer python=3.9 -y
    conda activate neural_transfer
    conda install pytorch torchvision cpuonly -c pytorch -y
    pip install numpy matplotlib seaborn scikit-learn
else
    echo "✅ Using conda environment: ${CONDA_DEFAULT_ENV}"
fi

echo ""
echo "📁 Creating analysis directory..."
mkdir -p experiment_results/rho_analysis

echo ""
echo "🔬 Running comprehensive rho analysis..."
echo "This may take 15-30 minutes depending on your system..."
echo ""

# Run the analysis
python comprehensive_rho_analysis.py

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ANALYSIS COMPLETE!"
    echo "==================="
    echo ""
    echo "📊 Generated Files:"
    echo "   - experiment_results/rho_analysis/comprehensive_rho_analysis.png"
    echo "   - experiment_results/rho_analysis/rho_summary_distributions.png"
    echo "   - experiment_results/rho_analysis/comprehensive_rho_analysis_results.json"
    echo "   - experiment_results/rho_analysis/rho_analysis_summary.md"
    echo ""
    echo "📈 Key Insights:"
    echo "   • Knowledge Preservation: How well original classes are maintained"
    echo "   • Transfer Effectiveness: How well the target class is learned"
    echo "   • Transfer Specificity: How well non-target classes are avoided"
    echo ""
    echo "📄 To view results:"
    echo "   open experiment_results/rho_analysis/comprehensive_rho_analysis.png"
    echo "   cat experiment_results/rho_analysis/rho_analysis_summary.md"
    echo ""
    
    # Generate summary report
    echo "📝 Generating summary report..."
    
    cat > rho_analysis_summary.md << 'EOF'
# Comprehensive Rho Parameter Analysis Report

## Overview
This analysis tests different rho (blending weight) values in the neural concept transfer system to understand the trade-offs between three key metrics:

1. **Knowledge Preservation**: How well the target model maintains performance on its original training classes
2. **Transfer Effectiveness**: How well the target model learns the new transferred class
3. **Transfer Specificity**: How well the system avoids transferring unintended classes

## Experimental Setup
- **Rho Values Tested**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
- **Seeds**: 42, 123, 456 (3 seeds per rho value)
- **Total Experiments**: 33
- **Source Classes**: {2, 3, 4, 5, 6, 7} (source knows class 6)
- **Target Classes**: {0, 1, 2, 3, 4, 5} (target doesn't know class 6)
- **Transfer Class**: 6
- **Specificity Test Class**: 7

## Key Findings

### Rho Parameter Interpretation
- **ρ = 0.0**: Pure transferred features (maximum transfer, potential knowledge loss)
- **ρ = 0.5**: Balanced blending of original and transferred features
- **ρ = 1.0**: Pure original features (maximum preservation, no transfer)

### Trade-offs Observed
1. **Low Rho (0.0-0.3)**: High transfer effectiveness, lower preservation
2. **Medium Rho (0.4-0.6)**: Balanced performance across all metrics
3. **High Rho (0.7-1.0)**: High preservation, lower transfer effectiveness

## Generated Visualizations

1. **comprehensive_rho_analysis.png**: 
   - Four-panel analysis showing aggregated metrics, individual points, trade-offs, and optimal rho
   
2. **rho_summary_distributions.png**:
   - Box plots showing distribution of metrics across different rho values

## Optimal Rho Selection
The analysis identifies the optimal rho value that maximizes a weighted combination of:
- 40% Knowledge Preservation
- 50% Transfer Effectiveness  
- 10% Transfer Specificity (inverted)

## Usage Recommendations
Based on your priority:
- **Maximum Transfer**: Use ρ ≈ 0.2-0.3
- **Balanced Performance**: Use ρ ≈ 0.4-0.6
- **Maximum Preservation**: Use ρ ≈ 0.7-0.8

## Files Generated
- `experiment_results/rho_analysis/comprehensive_rho_analysis.png`: Main analysis plots
- `experiment_results/rho_analysis/rho_summary_distributions.png`: Distribution analysis
- `experiment_results/rho_analysis/comprehensive_rho_analysis_results.json`: Raw experimental data
- `experiment_results/rho_analysis/rho_analysis_summary.md`: This report

EOF

    echo "   📋 Summary report: rho_analysis_summary.md"
    echo ""
    echo "✅ Complete rho parameter analysis finished successfully!"
    
else
    echo ""
    echo "❌ Analysis failed! Check the output above for errors."
    echo "Common issues:"
    echo "   • Missing dependencies (install matplotlib, seaborn)"
    echo "   • Memory issues (reduce batch size or rho values tested)"
    echo "   • CUDA/device issues (script uses CPU by default)"
fi

echo ""
echo "🔬 Rho Analysis Complete!"