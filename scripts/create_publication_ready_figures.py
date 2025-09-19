#!/usr/bin/env python3
"""
Create publication-ready figures with proper scientific visualization standards
- No cropping issues
- Proper aspect ratios and margins
- Clear, readable labels and legends
- Professional color schemes
- Appropriate figure sizes for journal/conference submission
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'text.usetex': False
})

def load_and_calculate_stats():
    """Load real data and calculate statistics"""
    df = pd.read_csv('results/comprehensive_metrics.csv')
    test_df = df[df['scenario'].str.startswith('test-')]
    physics_data = test_df[test_df['model'] == 'physics']['rmse_x2'].values
    ude_data = test_df[test_df['model'] == 'ude']['rmse_x2'].values
    
    # Calculate statistics
    delta = ude_data - physics_data
    mean_delta = np.mean(delta)
    
    # Bootstrap confidence interval
    np.random.seed(42)
    n_bootstrap = 10000
    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(delta), len(delta), replace=True)
        bootstrap_deltas.append(np.mean(delta[indices]))
    
    ci_lower = np.percentile(bootstrap_deltas, 2.5)
    ci_upper = np.percentile(bootstrap_deltas, 97.5)
    
    # Statistical tests
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(delta)
    cohens_d = mean_delta / np.std(delta)
    
    return {
        'physics_data': physics_data,
        'ude_data': ude_data,
        'delta': delta,
        'mean_delta': mean_delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'wilcoxon_p': wilcoxon_p,
        'cohens_d': cohens_d
    }

def create_publication_scatter_plot(stats_data):
    """Create publication-ready scatter plot"""
    fig, ax = plt.subplots(figsize=(6, 6))  # Square aspect ratio
    
    physics_data = stats_data['physics_data']
    ude_data = stats_data['ude_data']
    mean_delta = stats_data['mean_delta']
    ci_lower = stats_data['ci_lower']
    ci_upper = stats_data['ci_upper']
    wilcoxon_p = stats_data['wilcoxon_p']
    
    # Create scatter plot with proper sizing
    ax.scatter(physics_data, ude_data, s=100, alpha=0.8, color='#2E86AB', 
              edgecolors='white', linewidth=1.5, zorder=3)
    
    # Identity line with proper range
    min_val = min(min(physics_data), min(ude_data)) * 0.98
    max_val = max(max(physics_data), max(ude_data)) * 1.02
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, 
           linewidth=2, label='Identity Line', zorder=2)
    
    # Set proper axis limits with padding
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Add statistics in a clean format
    stats_text = f'Mean Δ = {mean_delta:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\nWilcoxon p = {wilcoxon_p:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))
    
    # Labels and title
    ax.set_xlabel('Physics RMSE(x₂)', fontsize=14, fontweight='bold')
    ax.set_ylabel('UDE RMSE(x₂)', fontsize=14, fontweight='bold')
    ax.set_title('UDE vs Physics RMSE Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

def create_publication_histogram(stats_data):
    """Create publication-ready histogram"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    delta = stats_data['delta']
    mean_delta = stats_data['mean_delta']
    ci_lower = stats_data['ci_lower']
    ci_upper = stats_data['ci_upper']
    wilcoxon_p = stats_data['wilcoxon_p']
    cohens_d = stats_data['cohens_d']
    
    # Create histogram with proper bins
    n, bins, patches = ax.hist(delta, bins=10, alpha=0.7, color='#2E86AB', 
                              edgecolor='black', linewidth=1.2, density=True)
    
    # Add mean line
    ax.axvline(mean_delta, color='#E63946', linestyle='--', linewidth=3, 
              label=f'Mean = {mean_delta:.4f}', zorder=4)
    
    # Add confidence interval
    ax.axvspan(ci_lower, ci_upper, alpha=0.3, color='#F77F00', 
              label='95% CI', zorder=1)
    ax.axvline(ci_lower, color='#F77F00', linestyle=':', linewidth=2, zorder=3)
    ax.axvline(ci_upper, color='#F77F00', linestyle=':', linewidth=2, zorder=3)
    
    # Add statistics
    stats_text = f'Wilcoxon p = {wilcoxon_p:.4f}\nCohen\'s dz = {cohens_d:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))
    
    # Labels and title
    ax.set_xlabel('UDE - Physics RMSE(x₂)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Performance Differences', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_publication_bland_altman(stats_data):
    """Create publication-ready Bland-Altman plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    physics_data = stats_data['physics_data']
    ude_data = stats_data['ude_data']
    delta = stats_data['delta']
    mean_delta = stats_data['mean_delta']
    ci_lower = stats_data['ci_lower']
    ci_upper = stats_data['ci_upper']
    wilcoxon_p = stats_data['wilcoxon_p']
    cohens_d = stats_data['cohens_d']
    
    # Calculate means and differences
    means = (physics_data + ude_data) / 2
    differences = delta
    
    # Create scatter plot
    ax.scatter(means, differences, s=100, alpha=0.8, color='#2E86AB', 
              edgecolors='white', linewidth=1.5, zorder=3)
    
    # Add mean difference line (bias)
    ax.axhline(mean_delta, color='#E63946', linestyle='-', linewidth=3, 
              label=f'Bias = {mean_delta:.4f}', zorder=4)
    
    # Compute and draw true Bland-Altman Limits of Agreement (LoA)
    # LoA = mean ± 1.96 * std(differences)
    std_delta = np.std(differences, ddof=1)
    loa_lower = mean_delta - 1.96 * std_delta
    loa_upper = mean_delta + 1.96 * std_delta
    ax.axhline(loa_lower, color='#F77F00', linestyle='--', linewidth=2, 
              alpha=0.9, label='Limits of Agreement (±1.96·SD)', zorder=2)
    ax.axhline(loa_upper, color='#F77F00', linestyle='--', linewidth=2, alpha=0.9, zorder=2)
    
    # Add statistics (include LoA numerics)
    stats_text = (
        f'Wilcoxon p = {wilcoxon_p:.4f}\n'
        f"Cohen's dz = {cohens_d:.4f}\n"
        f'LoA: [{loa_lower:.4f}, {loa_upper:.4f}]'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))
    
    # Labels and title
    ax.set_xlabel('Average RMSE(x₂)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Difference (UDE - Physics)', fontsize=14, fontweight='bold')
    ax.set_title('Bland-Altman Analysis of Agreement', fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def main():
    """Create all publication-ready figures"""
    print("Creating publication-ready figures with proper scientific standards...")
    
    # Load data and calculate statistics
    stats_data = load_and_calculate_stats()
    
    # Create publication-ready figures
    fig1 = create_publication_scatter_plot(stats_data)
    fig1.savefig('clean_figures_final/fig1_scatter_rmse_x2_ude_vs_physics_publication.pdf', 
                 dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig1.savefig('clean_figures_final/fig1_scatter_rmse_x2_ude_vs_physics_publication.png', 
                 dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig1)
    print("✓ Created publication-ready scatter plot")
    
    fig2 = create_publication_histogram(stats_data)
    fig2.savefig('clean_figures_final/fig2_hist_delta_rmse_x2_ude_minus_physics_publication.pdf', 
                 dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig2.savefig('clean_figures_final/fig2_hist_delta_rmse_x2_ude_minus_physics_publication.png', 
                 dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig2)
    print("✓ Created publication-ready histogram")
    
    fig3 = create_publication_bland_altman(stats_data)
    fig3.savefig('clean_figures_final/fig3_bland_altman_rmse_x2_publication.pdf', 
                 dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig3)
    print("✓ Created publication-ready Bland-Altman plot")
    
    print("\n✅ All publication-ready figures created with proper scientific standards!")

if __name__ == '__main__':
    main()
