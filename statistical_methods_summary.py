"""
Statistical Methods Summary and Verification
This script demonstrates the statistical methods used in the shared knowledge analysis.
"""

import numpy as np
from scipy import stats
import json
from pathlib import Path

def demonstrate_statistical_methods():
    """Demonstrate all statistical methods with example calculations."""
    
    print("🔬 STATISTICAL METHODS DEMONSTRATION")
    print("=" * 60)
    
    # Example data (simulated transfer effectiveness values)
    example_data = [0.991, 0.915, 1.000, 1.000, 1.000, 0.996, 1.000, 1.000, 1.000, 1.000]
    overlap_ratios = [0.333, 0.600, 0.750]
    transfer_means = [0.9915, 0.9985, 0.9995]
    
    print(f"\n📊 EXAMPLE DATA:")
    print(f"Transfer effectiveness values: {example_data}")
    print(f"Sample size (n): {len(example_data)}")
    
    # 1. DESCRIPTIVE STATISTICS
    print(f"\n📈 1. DESCRIPTIVE STATISTICS")
    print("-" * 30)
    
    n = len(example_data)
    data = np.array(example_data)
    
    # Mean calculation
    mean = np.mean(data)
    print(f"Mean (μ) = (1/n) × Σ(xi)")
    print(f"Mean (μ) = (1/{n}) × {np.sum(data):.3f} = {mean:.4f}")
    
    # Standard deviation calculation
    variance = np.var(data, ddof=0)  # Population variance
    std = np.std(data, ddof=0)       # Population standard deviation
    print(f"\nVariance (σ²) = (1/n) × Σ(xi - μ)²")
    squared_deviations = [(x - mean)**2 for x in data]
    print(f"Squared deviations: {[f'{x:.6f}' for x in squared_deviations[:3]]}...")
    print(f"Variance (σ²) = {variance:.6f}")
    print(f"Standard Deviation (σ) = √(σ²) = {std:.4f}")
    
    # Standard error
    sem = std / np.sqrt(n)
    print(f"\nStandard Error (SEM) = σ/√n = {std:.4f}/√{n} = {sem:.4f}")
    
    # Confidence interval
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, n-1)
    ci_margin = t_critical * sem
    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin
    
    print(f"\n95% Confidence Interval:")
    print(f"t-critical (α=0.05, df={n-1}) = {t_critical:.3f}")
    print(f"Margin of error = {t_critical:.3f} × {sem:.4f} = {ci_margin:.4f}")
    print(f"CI = {mean:.4f} ± {ci_margin:.4f} = [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Quartiles
    q25 = np.percentile(data, 25)
    q50 = np.percentile(data, 50)  # median
    q75 = np.percentile(data, 75)
    iqr = q75 - q25
    
    print(f"\nQuartiles:")
    print(f"Q1 (25th percentile) = {q25:.4f}")
    print(f"Q2 (50th percentile/Median) = {q50:.4f}")
    print(f"Q3 (75th percentile) = {q75:.4f}")
    print(f"IQR = Q3 - Q1 = {iqr:.4f}")
    
    # 2. CORRELATION ANALYSIS
    print(f"\n📈 2. CORRELATION ANALYSIS")
    print("-" * 30)
    
    x = np.array(overlap_ratios)
    y = np.array(transfer_means)
    
    print(f"Overlap ratios (X): {x}")
    print(f"Transfer means (Y): {y}")
    
    # Manual correlation calculation
    n_corr = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"\nStep-by-step correlation calculation:")
    print(f"n = {n_corr}")
    print(f"x̄ = {x_mean:.4f}")
    print(f"ȳ = {y_mean:.4f}")
    
    # Calculate numerator (covariance)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    print(f"\nNumerator = Σ((xi - x̄)(yi - ȳ))")
    for i in range(n_corr):
        deviation_x = x[i] - x_mean
        deviation_y = y[i] - y_mean
        product = deviation_x * deviation_y
        print(f"  i={i+1}: ({x[i]:.3f} - {x_mean:.3f}) × ({y[i]:.4f} - {y_mean:.4f}) = {product:.6f}")
    print(f"Numerator = {numerator:.6f}")
    
    # Calculate denominator
    sum_sq_x = np.sum((x - x_mean)**2)
    sum_sq_y = np.sum((y - y_mean)**2)
    denominator = np.sqrt(sum_sq_x * sum_sq_y)
    
    print(f"\nDenominator = √(Σ(xi - x̄)² × Σ(yi - ȳ)²)")
    print(f"Σ(xi - x̄)² = {sum_sq_x:.6f}")
    print(f"Σ(yi - ȳ)² = {sum_sq_y:.6f}")
    print(f"Denominator = √({sum_sq_x:.6f} × {sum_sq_y:.6f}) = {denominator:.6f}")
    
    # Correlation coefficient
    r_manual = numerator / denominator
    r_numpy = np.corrcoef(x, y)[0, 1]
    
    print(f"\nCorrelation coefficient:")
    print(f"r = {numerator:.6f} / {denominator:.6f} = {r_manual:.4f}")
    print(f"NumPy verification: r = {r_numpy:.4f}")
    
    # R-squared
    r_squared = r_manual**2
    print(f"R² = r² = {r_squared:.4f}")
    print(f"Interpretation: {r_squared:.1%} of variance in Y explained by X")
    
    # Correlation strength interpretation
    abs_r = abs(r_manual)
    if abs_r > 0.7:
        strength = "strong"
    elif abs_r > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if r_manual > 0 else "negative"
    print(f"Relationship: {strength} {direction} correlation")
    
    # Statistical significance test
    t_stat = r_manual * np.sqrt((n_corr - 2) / (1 - r_manual**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_corr - 2))
    
    print(f"\nSignificance test:")
    print(f"t-statistic = r × √((n-2)/(1-r²))")
    print(f"t-statistic = {r_manual:.4f} × √(({n_corr}-2)/(1-{r_manual:.4f}²)) = {t_stat:.4f}")
    print(f"p-value = {p_value:.4f}")
    
    # 3. SUMMARY STATISTICS TABLE
    print(f"\n📋 3. COMPREHENSIVE STATISTICS SUMMARY")
    print("-" * 50)
    
    summary = {
        'sample_size': n,
        'descriptive_statistics': {
            'mean': float(mean),
            'median': float(q50),
            'std_deviation': float(std),
            'variance': float(variance),
            'standard_error': float(sem),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'iqr': float(iqr),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper)
        },
        'correlation_analysis': {
            'correlation_coefficient': float(r_manual),
            'r_squared': float(r_squared),
            'strength': strength,
            'direction': direction,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': n_corr - 2
        }
    }
    
    print(json.dumps(summary, indent=2))
    
    # Save demonstration results
    output_dir = Path("experiment_results/shared_knowledge_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "statistical_methods_demonstration.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Statistical methods demonstration saved to:")
    print(f"   {output_dir / 'statistical_methods_demonstration.json'}")
    
    return summary

if __name__ == "__main__":
    demonstrate_statistical_methods()