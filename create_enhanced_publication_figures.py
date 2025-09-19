#!/usr/bin/env python3
"""
Enhanced Publication-Quality Figure Generator

Based on research paper best practices:
- High resolution (300+ DPI)
- Colorblind-friendly palettes
- Consistent typography
- Clear legends and labels
- Professional styling
- Accessibility compliance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.transparent': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Colorblind-friendly color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange  
    'tertiary': '#2ca02c',     # Green
    'quaternary': '#d62728',   # Red
    'quinary': '#9467bd',      # Purple
    'senary': '#8c564b',       # Brown
    'neutral': '#7f7f7f',      # Gray
    'accent': '#17becf',       # Cyan
    'highlight': '#bcbd22',    # Olive
    'warning': '#e377c2'       # Pink
}

# Alternative colorblind-safe palette
CB_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def load_data():
    """Load the comprehensive metrics data"""
    csv_path = Path('results/comprehensive_metrics.csv')
    if not csv_path.exists():
        csv_path = Path('../results/comprehensive_metrics.csv')
    
    df = pd.read_csv(csv_path)
    test_data = df[df['scenario'].str.startswith('test-')]
    
    physics = test_data[test_data['model'] == 'physics'].copy()
    ude = test_data[test_data['model'] == 'ude'].copy()
    
    # Merge for comparison
    merged = physics[['scenario', 'rmse_x2', 'r2_x2']].merge(
        ude[['scenario', 'rmse_x2', 'r2_x2']], 
        on='scenario', 
        suffixes=('_phys', '_ude')
    )
    
    return merged, physics, ude

def create_enhanced_scatter_plot():
    """Create enhanced scatter plot with publication quality"""
    merged, physics, ude = load_data()
    
    # Calculate statistics
    delta = merged['rmse_x2_ude'] - merged['rmse_x2_phys']
    mean_delta = delta.mean()
    ci_lower, ci_upper = stats.bootstrap((delta,), np.mean, confidence_level=0.95, random_state=42).confidence_interval
    _, wilcoxon_p = stats.wilcoxon(delta)
    cohens_d = delta.mean() / delta.std()
    
    # Create figure with proper aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with enhanced styling
    scatter = ax.scatter(merged['rmse_x2_phys'], merged['rmse_x2_ude'], 
                        c=CB_COLORS[0], alpha=0.7, s=80, edgecolors='white', linewidth=1.5)
    
    # Add diagonal reference line
    min_val = min(merged['rmse_x2_phys'].min(), merged['rmse_x2_ude'].min())
    max_val = max(merged['rmse_x2_phys'].max(), merged['rmse_x2_ude'].max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['neutral'], 
            linewidth=2, alpha=0.8, label='Perfect Agreement')
    
    # Enhanced styling
    ax.set_xlabel('Physics Model RMSE x₂', fontweight='bold')
    ax.set_ylabel('UDE Model RMSE x₂', fontweight='bold')
    ax.set_title('Model Performance Comparison: UDE vs Physics Baseline', 
                fontweight='bold', pad=20)
    
    # Add statistics box with enhanced styling
    stats_text = f'Mean Δ = {mean_delta:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\nWilcoxon p = {wilcoxon_p:.4f}'
    
    # Create a more professional stats box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, 
                edgecolor=COLORS['neutral'], linewidth=1.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='normal')
    
    # Add correlation coefficient
    corr_coef = merged['rmse_x2_phys'].corr(merged['rmse_x2_ude'])
    ax.text(0.95, 0.05, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
            fontsize=11, ha='right', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Enhance grid and spines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Set equal aspect ratio for better comparison
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

def main():
    """Generate enhanced publication-quality figures"""
    output_dir = Path('clean_figures_final')
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating enhanced publication-quality figures...")
    
    try:
        print("  Creating enhanced scatter plot...")
        fig = create_enhanced_scatter_plot()
        if fig is not None:
            output_path = output_dir / 'fig1_scatter_rmse_x2_ude_vs_physics_enhanced.pdf'
            fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"  ✅ Saved fig1_scatter_rmse_x2_ude_vs_physics_enhanced.pdf")
    except Exception as e:
        print(f"  ❌ Error creating scatter plot: {e}")
    
    print("✅ Enhanced figure generation complete!")

if __name__ == "__main__":
    main()
