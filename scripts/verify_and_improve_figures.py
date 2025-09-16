#!/usr/bin/env python3
"""
Verify and improve all figures based on real research data
- Verify all statistics against actual experimental results
- Improve figure design based on scientific best practices
- Ensure figures fit well in paper sections
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from scipy import stats
import json
from scripts.fig_utils import (
    set_matplotlib_defaults,
    provenance_footer,
    bland_altman,
    empirical_coverage,
    load_bson_or_npz,
)

# Set scientific plotting style
set_matplotlib_defaults()

def load_real_data():
    """Load all real experimental data"""
    print("Loading real experimental data...")
    
    # Load comprehensive metrics
    df = pd.read_csv('results/comprehensive_metrics.csv')
    print(f"Loaded comprehensive metrics: {df.shape}")
    
    # Load training data
    train_df = pd.read_csv('data/training_roadmap.csv')
    print(f"Loaded training data: {train_df.shape}")
    
    # Load validation data
    val_df = pd.read_csv('data/validation_roadmap.csv')
    print(f"Loaded validation data: {val_df.shape}")
    
    # Load test data
    test_df = pd.read_csv('data/test_roadmap.csv')
    print(f"Loaded test data: {test_df.shape}")
    
    # Load BNODE calibration data
    bnode_data = load_bson_or_npz('results/simple_bnode_calibration_results.bson') or \
                 load_bson_or_npz('results/simple_bnode_calibration_results.npz')
    if bnode_data is None:
        print("WARN: results/simple_bnode_calibration_results.(bson|npz) missing; calibration plot will be labeled Schematic.")
        bnode_data = {}
    
    return df, train_df, val_df, test_df, bnode_data

def verify_statistics(df):
    """Verify all statistics against real data"""
    print("\n=== VERIFYING STATISTICS ===")
    
    # Filter test scenarios
    test_df = df[df['scenario'].str.startswith('test-')]
    physics_data = test_df[test_df['model'] == 'physics']['rmse_x2'].values
    ude_data = test_df[test_df['model'] == 'ude']['rmse_x2'].values
    
    # Calculate real statistics
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
    
    # Wilcoxon test
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(delta)
    
    # Cohen's d
    cohens_d = mean_delta / np.std(delta)
    
    print(f"Mean Δ (UDE - Physics): {mean_delta:.6f}")
    print(f"95% Bootstrap CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Wilcoxon p-value: {wilcoxon_p:.4f}")
    print(f"Cohen's dz: {cohens_d:.4f}")
    
    return {
        'mean_delta': mean_delta,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'wilcoxon_p': wilcoxon_p,
        'cohens_d': cohens_d,
        'physics_data': physics_data,
        'ude_data': ude_data,
        'delta': delta
    }

def create_improved_scatter_plot(stats_data):
    """Create improved scatter plot with real data"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    physics_data = stats_data['physics_data']
    ude_data = stats_data['ude_data']
    
    # Create scatter plot
    ax.scatter(physics_data, ude_data, s=60, alpha=0.7, color='black', edgecolors='white', linewidth=1)
    
    # Add identity line
    min_val = min(min(physics_data), min(ude_data))
    max_val = max(max(physics_data), max(ude_data))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1.5, label='Identity Line')
    
    # Add statistics text box
    stats_text = f'Mean Δ = {stats_data["mean_delta"]:.4f}\n95% CI: [{stats_data["ci_lower"]:.4f}, {stats_data["ci_upper"]:.4f}]\nWilcoxon p = {stats_data["wilcoxon_p"]:.4f}\nCohen\'s dz = {stats_data["cohens_d"]:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Physics RMSE(x₂)')
    ax.set_ylabel('UDE RMSE(x₂)')
    ax.set_title('UDE vs Physics Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_improved_histogram(stats_data):
    """Create improved histogram with real data"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    delta = stats_data['delta']
    
    # Create histogram
    n, bins, patches = ax.hist(delta, bins=8, alpha=0.7, color='lightgray', edgecolor='black', linewidth=1)
    
    # Add mean line
    ax.axvline(stats_data['mean_delta'], color='red', linestyle='--', linewidth=2, label=f'Mean = {stats_data["mean_delta"]:.4f}')
    
    # Add confidence interval
    ax.axvline(stats_data['ci_lower'], color='blue', linestyle=':', alpha=0.7, label='95% CI')
    ax.axvline(stats_data['ci_upper'], color='blue', linestyle=':', alpha=0.7)
    
    # Fill confidence interval
    ax.axvspan(stats_data['ci_lower'], stats_data['ci_upper'], alpha=0.2, color='blue')
    
    ax.set_xlabel('UDE - Physics RMSE(x₂)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Performance Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_improved_bland_altman(stats_data):
    """Create improved Bland-Altman plot with real data"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    physics_data = stats_data['physics_data']
    ude_data = stats_data['ude_data']
    delta = stats_data['delta']
    
    means, differences, bias, loa_low, loa_high = bland_altman(physics_data, ude_data)

    # Create scatter plot
    ax.scatter(means, differences, s=14, alpha=0.75, color='black', edgecolors='white', linewidth=0.8)

    # Bias and LoA per Bland–Altman
    ax.axhline(bias, color='C3', linewidth=1.5, label=f"Bias={bias:.4f}")
    ax.axhline(loa_low, color='C0', linestyle='--', linewidth=1.2, label='LoA')
    ax.axhline(loa_high, color='C0', linestyle='--', linewidth=1.2)
    
    ax.set_xlabel('Mean of Physics and UDE RMSE(x₂)')
    ax.set_ylabel('Difference (UDE - Physics)')
    ax.set_title('Bland–Altman (RMSE x₂)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    provenance_footer(ax, 'results/comprehensive_metrics.csv')
    
    return fig

def create_improved_calibration_plot(bnode_data):
    """Create improved calibration plot using empirical coverage from artifacts.
    Falls back to a clearly marked schematic if artifacts are unavailable."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if not bnode_data:
        # Schematic placeholder, clearly labeled
        qs = np.linspace(0.1, 0.9, 9)
        ax.plot(qs, qs, 'k:', label='Ideal')
        ax.set_title('Schematic — Reliability (artifact missing)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0.08, 0.92)
        ax.set_ylim(0.08, 0.92)
        ax.set_xlabel('Nominal central coverage')
        ax.set_ylabel('Empirical coverage')
        ax.legend()
        provenance_footer(ax, 'results/simple_bnode_calibration_results.bson')
        return fig

    # Try to derive empirical coverage from available keys
    qs = np.linspace(0.1, 0.9, 17)
    y_true = None
    empirical = None

    if 'y_true' in bnode_data and 'samples' in bnode_data:
        y_true = np.asarray(bnode_data['y_true'])
        samples = np.asarray(bnode_data['samples'])
        empirical = empirical_coverage(y_true, samples, qs)
    else:
        # Attempt to interpret as dict of quantiles q_{p}
        # Build dict mapping quantile probability to arrays
        pred_q = {}
        needed = np.unique(np.r_[ (1-qs)/2, (1+qs)/2 ])
        for q in needed:
            key = f"q_{q:.2f}"
            alt = f"{q:.2f}"
            arr = bnode_data.get(key, bnode_data.get(alt)) if isinstance(bnode_data, dict) else None
            if arr is not None:
                pred_q[q] = np.asarray(arr)
        if 'y_true' in bnode_data:
            y_true = np.asarray(bnode_data['y_true'])
        if pred_q and y_true is not None:
            empirical = empirical_coverage(y_true, pred_q, qs)

    ax.plot(qs, qs, '--', linewidth=1.0, label='Ideal')
    if empirical is not None:
        ax.plot(qs, empirical, '-', linewidth=1.8, label='Empirical')
        ax.set_title('Reliability Diagram (Empirical)')
    else:
        ax.set_title('Schematic — Reliability (could not decode artifact)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.08, 0.92)
    ax.set_xlabel('Nominal central coverage')
    ax.set_ylabel('Empirical coverage')
    ax.legend()
    provenance_footer(ax, 'results/simple_bnode_calibration_results.bson')
    return fig

def create_improved_training_data_plot(train_df, val_df, test_df):
    """Create improved training data visualization with real data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Parameter distribution from real data
    ax1.set_title('(a) Training Scenarios Parameter Distribution', fontsize=12, fontweight='bold')
    
    # Extract unique scenarios and their parameters
    scenarios = train_df['scenario'].unique()
    alpha_values = []
    beta_values = []
    
    for scenario in scenarios:
        scenario_data = train_df[train_df['scenario'] == scenario]
        alpha_values.append(scenario_data['α'].iloc[0])
        beta_values.append(scenario_data['β'].iloc[0])
    
    scatter = ax1.scatter(alpha_values, beta_values, c=range(len(scenarios)), 
                         cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Damping Coefficient (α)')
    ax1.set_ylabel('Power-Frequency Coupling (β)')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Scenario Index')
    
    # Panel 2: Sample trajectories from real data
    ax2.set_title('(b) Sample Trajectories', fontsize=12, fontweight='bold')
    
    # Plot first few scenarios
    for i, scenario in enumerate(scenarios[:3]):
        scenario_data = train_df[train_df['scenario'] == scenario]
        time = scenario_data['time'].values
        x2 = scenario_data['x2'].values
        x1 = scenario_data['x1'].values
        
        ax2.plot(time, x2, 'k-', alpha=0.7, linewidth=1, label=f'Scenario {i+1}' if i == 0 else "")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time, x1, 'r--', alpha=0.7, linewidth=1, label=f'Storage {i+1}' if i == 0 else "")
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency Deviation (p.u.)', color='black')
    ax2_twin.set_ylabel('State of Charge', color='red')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2_twin.legend(loc='upper left')
    
    # Panel 3: Data split visualization
    ax3.set_title('(c) Train/Validation/Test Split', fontsize=12, fontweight='bold')
    
    splits = ['Training', 'Validation', 'Test']
    scenarios = [len(train_df['scenario'].unique()), len(val_df['scenario'].unique()), len(test_df['scenario'].unique())]
    points = [len(train_df), len(val_df), len(test_df)]
    
    x = np.arange(len(splits))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, scenarios, width, label='Scenarios', color='lightblue', alpha=0.7)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, points, width, label='Data Points', color='lightgreen', alpha=0.7)
    
    ax3.set_xlabel('Data Split')
    ax3.set_ylabel('Number of Scenarios', color='blue')
    ax3_twin.set_ylabel('Number of Data Points', color='green')
    ax3.set_xticks(x)
    ax3.set_xticklabels(splits)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Parameter ranges from real data
    ax4.set_title('(d) Parameter Ranges', fontsize=12, fontweight='bold')
    
    parameters = ['α (damping)', 'β (coupling)', 'η_in (charge)', 'η_out (discharge)']
    min_vals = [train_df['α'].min(), train_df['β'].min(), train_df['ηin'].min(), train_df['ηout'].min()]
    max_vals = [train_df['α'].max(), train_df['β'].max(), train_df['ηin'].max(), train_df['ηout'].max()]
    mean_vals = [(min_vals[i] + max_vals[i]) / 2 for i in range(len(parameters))]
    
    y_pos = np.arange(len(parameters))
    ax4.barh(y_pos, [max_vals[i] - min_vals[i] for i in range(len(parameters))], 
             left=min_vals, height=0.6, alpha=0.7, color='lightcoral')
    ax4.scatter(mean_vals, y_pos, color='black', s=50, zorder=5)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(parameters)
    ax4.set_xlabel('Parameter Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to verify and improve all figures"""
    print("=== VERIFYING AND IMPROVING FIGURES ===")
    
    # Load real data
    df, train_df, val_df, test_df, bnode_data = load_real_data()
    
    # Verify statistics
    stats_data = verify_statistics(df)
    
    print("\n=== CREATING IMPROVED FIGURES ===")
    
    # Create improved figures with real data
    fig1 = create_improved_scatter_plot(stats_data)
    fig1.savefig('clean_figures_final/fig1_scatter_rmse_x2_ude_vs_physics_improved.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig('clean_figures_final/fig1_scatter_rmse_x2_ude_vs_physics_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✓ Improved Figure 1: Scatter Plot")
    
    fig2 = create_improved_histogram(stats_data)
    fig2.savefig('clean_figures_final/fig2_hist_delta_rmse_x2_ude_minus_physics_improved.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig('clean_figures_final/fig2_hist_delta_rmse_x2_ude_minus_physics_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✓ Improved Figure 2: Histogram")
    
    fig3 = create_improved_bland_altman(stats_data)
    fig3.savefig('clean_figures_final/fig3_bland_altman_rmse_x2_improved.pdf', dpi=300, bbox_inches='tight')
    fig3.savefig('clean_figures_final/fig3_bland_altman_rmse_x2_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("✓ Improved Figure 3: Bland-Altman")
    
    fig6 = create_improved_calibration_plot(bnode_data)
    fig6.savefig('clean_figures_final/fig6_calibration_bnode_pre_post_improved.pdf', dpi=300, bbox_inches='tight')
    fig6.savefig('clean_figures_final/fig6_calibration_bnode_pre_post_improved.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print("✓ Improved Figure 6: Calibration")
    
    figG = create_improved_training_data_plot(train_df, val_df, test_df)
    figG.savefig('clean_figures_final/figG_training_data_improved.pdf', dpi=300, bbox_inches='tight')
    figG.savefig('clean_figures_final/figG_training_data_improved.png', dpi=300, bbox_inches='tight')
    plt.close(figG)
    print("✓ Improved Figure G: Training Data")
    
    print("\n✅ All figures verified and improved with real data!")

if __name__ == "__main__":
    main()
