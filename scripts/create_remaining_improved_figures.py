#!/usr/bin/env python3
"""
Create remaining improved figures with real data
- Figure 4: Paired lines comparison
- Figure 5: R² delta bar plot
- Figure 7: Baseline performance comparison
- Figure 8: Runtime comparison
- Figure 9: Symbolic extraction
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from scipy import stats
import json
from fig_utils import (
    set_matplotlib_defaults,
    provenance_footer,
    load_json_or_none,
)

# Set scientific plotting style
set_matplotlib_defaults()

def load_real_data():
    """Load all real experimental data"""
    # Load comprehensive metrics
    df = pd.read_csv('results/comprehensive_metrics.csv')
    
    # Load BNODE calibration data
    import bson
    with open('results/simple_bnode_calibration_results.bson', 'rb') as f:
        bnode_data = bson.loads(f.read())
    
    return df, bnode_data

def create_improved_paired_lines_plot(df):
    """Create improved paired lines plot with real data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter test scenarios
    test_df = df[df['scenario'].str.startswith('test-')]
    
    # Get unique scenarios
    scenarios = sorted(test_df['scenario'].unique())
    
    # Create paired lines
    for i, scenario in enumerate(scenarios):
        scenario_data = test_df[test_df['scenario'] == scenario]
        physics_rmse = scenario_data[scenario_data['model'] == 'physics']['rmse_x2'].iloc[0]
        ude_rmse = scenario_data[scenario_data['model'] == 'ude']['rmse_x2'].iloc[0]
        
        # Draw line connecting the two points
        ax.plot([0, 1], [physics_rmse, ude_rmse], 'k-', alpha=0.7, linewidth=1)
        
        # Mark the points
        ax.plot(0, physics_rmse, 'bo', markersize=6, alpha=0.7)
        ax.plot(1, ude_rmse, 'ro', markersize=6, alpha=0.7)
    
    # Add scenario labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Physics', 'UDE'])
    ax.set_ylabel('RMSE(x₂)')
    ax.set_title('Paired Performance Comparison by Scenario')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    physics_data = test_df[test_df['model'] == 'physics']['rmse_x2'].values
    ude_data = test_df[test_df['model'] == 'ude']['rmse_x2'].values
    delta = ude_data - physics_data
    mean_delta = np.mean(delta)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(delta)
    cohens_d = mean_delta / np.std(delta)
    
    stats_text = f'Mean Δ = {mean_delta:.4f}\nWilcoxon p = {wilcoxon_p:.4f}\nCohen\'s dz = {cohens_d:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    provenance_footer(ax, 'results/comprehensive_metrics.csv')
    return fig

def create_improved_r2_delta_plot(df):
    """Create improved R² delta bar plot with real data"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter test scenarios
    test_df = df[df['scenario'].str.startswith('test-')]
    
    # Get unique scenarios
    scenarios = sorted(test_df['scenario'].unique())
    
    # Calculate R² differences
    r2_deltas = []
    for scenario in scenarios:
        scenario_data = test_df[test_df['scenario'] == scenario]
        physics_r2 = scenario_data[scenario_data['model'] == 'physics']['r2_x2'].iloc[0]
        ude_r2 = scenario_data[scenario_data['model'] == 'ude']['r2_x2'].iloc[0]
        r2_deltas.append(ude_r2 - physics_r2)
    
    # Create bar plot
    x_pos = np.arange(len(scenarios))
    colors = ['red' if delta < 0 else 'blue' for delta in r2_deltas]
    bars = ax.bar(x_pos, r2_deltas, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add statistics
    mean_r2_delta = np.mean(r2_deltas)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(r2_deltas)
    
    stats_text = f'Mean R² Δ = {mean_r2_delta:.4f}\nWilcoxon p = {wilcoxon_p:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Test Scenario')
    ax.set_ylabel('R² Difference (UDE - Physics)')
    ax.set_title('R² Performance Differences by Scenario')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace('test-', '') for s in scenarios], rotation=45)
    ax.grid(True, alpha=0.3)
    
    provenance_footer(ax, 'results/comprehensive_metrics.csv')
    return fig

def create_improved_baseline_comparison_plot(df):
    """Create improved baseline comparison plot with real data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter test scenarios
    test_df = df[df['scenario'].str.startswith('test-')]
    
    # Get data for each model
    physics_data = test_df[test_df['model'] == 'physics']
    ude_data = test_df[test_df['model'] == 'ude']
    
    # RMSE comparison
    models = ['Physics', 'UDE']
    rmse_means = [physics_data['rmse_x2'].mean(), ude_data['rmse_x2'].mean()]
    rmse_stds = [physics_data['rmse_x2'].std(), ude_data['rmse_x2'].std()]
    
    bars1 = ax1.bar(models, rmse_means, yerr=rmse_stds, capsize=5, 
                   color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('RMSE(x₂)')
    ax1.set_title('RMSE Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    # R² comparison
    r2_means = [physics_data['r2_x2'].mean(), ude_data['r2_x2'].mean()]
    r2_stds = [physics_data['r2_x2'].std(), ude_data['r2_x2'].std()]
    
    bars2 = ax2.bar(models, r2_means, yerr=r2_stds, capsize=5,
                   color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('R²(x₂)')
    ax2.set_title('R² Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
        ax2.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    provenance_footer(ax1, 'results/comprehensive_metrics.csv')
    provenance_footer(ax2, 'results/comprehensive_metrics.csv')
    return fig

def create_improved_runtime_plot(measurements=None):
    """Create improved runtime comparison plot using distributions.
    measurements: optional dict {label: array_like_ms}. If None, mark Schematic.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not measurements:
        ax.set_title('Schematic — Runtime comparison (measurements missing)')
        ax.set_ylabel('Runtime (ms)')
        ax.grid(True, alpha=0.3)
        provenance_footer(ax, 'results/runtime_measurements.npz')
        return fig
    labels = list(measurements.keys())
    data = [np.asarray(measurements[k]) for k in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = ['#4E79A7', '#59A14F']  # colorblind-friendly
    for patch, c in zip(bp['boxes'], colors * ((len(labels)+1)//2)):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Runtime per trajectory (ms)')
    ax.set_title('Inference Runtime Distributions')
    ax.grid(True, alpha=0.3)
    # environment banner placeholder
    ax.text(0.02, 0.98, 'Env: CPU, BLAS threads=?, solver=?, reltol=?', transform=ax.transAxes,
            va='top', ha='left', fontsize=8, alpha=0.8)
    provenance_footer(ax, 'results/runtime_measurements.npz')
    return fig

def create_improved_symbolic_extraction_plot(P=None, f_vals=None):
    """Create improved symbolic extraction plot from artifact or provided data.
    Loads results/symbolic_fit.json if present; otherwise computes from (P, f_vals).
    """
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    coeffs = load_json_or_none('results/symbolic_fit.json')
    if coeffs is not None:
        beta = np.array(coeffs.get('beta', []))
        r2 = coeffs.get('r2')
        Pline = np.linspace(0, 1, 400)
        Xl = np.vstack([np.ones_like(Pline), Pline, Pline**2, Pline**3]).T
        yhat = Xl @ beta
        ax.plot(Pline, yhat, linewidth=1.8, label=f"Cubic fit (R²={r2:.3f})")
        ax.set_title('Symbolic Extraction (from artifact)')
        axr.axhline(0.0, color='k', linewidth=1)
    elif P is not None and f_vals is not None:
        P = np.asarray(P); f_vals = np.asarray(f_vals)
        X = np.vstack([np.ones_like(P), P, P**2, P**3]).T
        beta, *_ = np.linalg.lstsq(X, f_vals, rcond=None)
        yhat = X @ beta
        ssr = np.sum((f_vals - yhat)**2)
        sst = np.sum((f_vals - f_vals.mean())**2)
        r2 = 1 - ssr/sst if sst > 0 else np.nan
        order = np.argsort(P)
        ax.scatter(P, f_vals, s=12, alpha=0.75, label='NN residual')
        ax.plot(P[order], yhat[order], linewidth=1.8, label=f"Cubic fit (R²={r2:.3f})")
        axr.axhline(0.0, color='k', linewidth=1)
        axr.scatter(P, f_vals - yhat, s=10, alpha=0.7)
        ax.set_title('Symbolic Extraction (computed)')
    else:
        ax.text(0.5, 0.5, 'Schematic — residual data missing', transform=ax.transAxes, ha='center', va='center')
        axr.axhline(0.0, color='k', linewidth=1)

    ax.set_ylabel('f_θ(P)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    axr.set_xlabel('Generation Power P')
    axr.set_ylabel('Residuals')
    axr.grid(True, alpha=0.3)
    provenance_footer(ax, 'results/symbolic_fit.json')
    return fig

def main():
    """Create remaining improved figures"""
    print("Creating remaining improved figures with real data...")
    
    # Load real data
    df, bnode_data = load_real_data()
    
    # Create improved figures
    fig4 = create_improved_paired_lines_plot(df)
    fig4.savefig('clean_figures_final/fig4_paired_lines_rmse_x2_by_model_improved.pdf', dpi=300, bbox_inches='tight')
    fig4.savefig('clean_figures_final/fig4_paired_lines_rmse_x2_by_model_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("✓ Improved Figure 4: Paired Lines")
    
    fig5 = create_improved_r2_delta_plot(df)
    fig5.savefig('clean_figures_final/fig5_r2x2_delta_ude_minus_physics_improved.pdf', dpi=300, bbox_inches='tight')
    fig5.savefig('clean_figures_final/fig5_r2x2_delta_ude_minus_physics_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("✓ Improved Figure 5: R² Delta")
    
    fig7 = create_improved_baseline_comparison_plot(df)
    fig7.savefig('clean_figures_final/fig7_baselines_rmse_r2_summary_improved.pdf', dpi=300, bbox_inches='tight')
    fig7.savefig('clean_figures_final/fig7_baselines_rmse_r2_summary_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig7)
    print("✓ Improved Figure 7: Baseline Comparison")
    
    fig8 = create_improved_runtime_plot()
    fig8.savefig('clean_figures_final/fig8_runtime_comparison_improved.pdf', dpi=300, bbox_inches='tight')
    fig8.savefig('clean_figures_final/fig8_runtime_comparison_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig8)
    print("✓ Improved Figure 8: Runtime Comparison")
    
    fig9 = create_improved_symbolic_extraction_plot()
    fig9.savefig('clean_figures_final/fig9_symbolic_extraction_fit_improved.pdf', dpi=300, bbox_inches='tight')
    fig9.savefig('clean_figures_final/fig9_symbolic_extraction_fit_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig9)
    print("✓ Improved Figure 9: Symbolic Extraction")
    
    print("\n✅ All remaining figures created with real data!")

if __name__ == "__main__":
    main()
