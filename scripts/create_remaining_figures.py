import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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

# Load data and calculate statistics
df = pd.read_csv('results/comprehensive_metrics.csv')
test_df = df[df['scenario'].str.startswith('test-')]
physics_data = test_df[test_df['model'] == 'physics']['rmse_x2'].values
ude_data = test_df[test_df['model'] == 'ude']['rmse_x2'].values

# Calculate statistics
delta = ude_data - physics_data
mean_delta = np.mean(delta)
np.random.seed(42)
n_bootstrap = 10000
bootstrap_deltas = []
for _ in range(n_bootstrap):
    indices = np.random.choice(len(delta), len(delta), replace=True)
    bootstrap_deltas.append(np.mean(delta[indices]))

ci_lower = np.percentile(bootstrap_deltas, 2.5)
ci_upper = np.percentile(bootstrap_deltas, 97.5)

wilcoxon_stat, wilcoxon_p = stats.wilcoxon(delta)
cohens_d = mean_delta / np.std(delta)

print("Creating remaining publication-ready figures...")

# Create paired performance plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create scenario data
scenarios = [f'Test {i+1}' for i in range(len(physics_data))]

x_pos = np.arange(len(scenarios))
width = 0.35

# Create bars
bars1 = ax.bar(x_pos - width/2, physics_data, width, label='Physics', 
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x_pos + width/2, ude_data, width, label='UDE', 
               color='#E63946', alpha=0.8, edgecolor='black', linewidth=1)

# Add connecting lines for paired comparison
for i, (p, u) in enumerate(zip(physics_data, ude_data)):
    ax.plot([i - width/2, i + width/2], [p, u], 'k-', alpha=0.6, linewidth=1)

# Labels and title
ax.set_xlabel('Test Scenarios', fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE(x₂)', fontsize=14, fontweight='bold')
ax.set_title('Paired Performance Comparison by Scenario', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, rotation=45, ha='right')

# Legend
ax.legend(fontsize=12, framealpha=0.95)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')

# Add statistics
stats_text = f'Mean Δ = {mean_delta:.4f}\nWilcoxon p = {wilcoxon_p:.4f}\nCohen\\'s dz = {cohens_d:.4f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
        facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8))

plt.tight_layout()
fig.savefig('clean_figures_final/fig4_paired_lines_rmse_x2_by_model_publication.pdf', 
            dpi=300, bbox_inches='tight', pad_inches=0.2)
fig.savefig('clean_figures_final/fig4_paired_lines_rmse_x2_by_model_publication.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print("✓ Created publication-ready paired plot")

# Create training data plot
train_df = pd.read_csv('data/training_roadmap.csv')
val_df = pd.read_csv('data/validation_roadmap.csv')
test_df = pd.read_csv('data/test_roadmap.csv')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Parameter distribution
ax1.set_title('(a) Training Scenarios Parameter Distribution', fontsize=14, fontweight='bold')

scenarios = train_df['scenario'].unique()
alpha_values = []
beta_values = []

for scenario in scenarios:
    scenario_data = train_df[train_df['scenario'] == scenario]
    alpha_values.append(scenario_data['α'].iloc[0])
    beta_values.append(scenario_data['β'].iloc[0])

scatter = ax1.scatter(alpha_values, beta_values, c=range(len(scenarios)), 
                     cmap='viridis', s=80, alpha=0.8, edgecolors='black', linewidth=1)
ax1.set_xlabel('Damping Coefficient (α)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Power-Frequency Coupling (β)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel 2: Data split
ax2.set_title('(b) Train/Validation/Test Split', fontsize=14, fontweight='bold')

splits = ['Train', 'Validation', 'Test']
scenarios_count = [len(train_df['scenario'].unique()), 
                  len(val_df['scenario'].unique()), 
                  len(test_df['scenario'].unique())]
data_points = [len(train_df), len(val_df), len(test_df)]

x = np.arange(len(splits))
width = 0.35

bars1 = ax2.bar(x - width/2, scenarios_count, width, label='Scenarios', 
                color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax2.bar(x + width/2, data_points, width, label='Data Points', 
                color='#E63946', alpha=0.8, edgecolor='black', linewidth=1)

ax2.set_xlabel('Data Split', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(splits)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Sample trajectories
ax3.set_title('(c) Sample Training Trajectories', fontsize=14, fontweight='bold')

# Plot first few scenarios
for i, scenario in enumerate(scenarios[:3]):
    scenario_data = train_df[train_df['scenario'] == scenario]
    ax3.plot(scenario_data['t'], scenario_data['x1'], 
            label=f'Scenario {i+1}', linewidth=2, alpha=0.8)

ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
ax3.set_ylabel('State of Charge (x₁)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel 4: Data summary
ax4.set_title('(d) Dataset Summary Statistics', fontsize=14, fontweight='bold')

# Create summary table
summary_data = {
    'Metric': ['Total Scenarios', 'Total Data Points', 'Time Range', 'Parameter Range'],
    'Value': [f'{len(scenarios)}', f'{len(train_df)}', '[0, 10]', 'α∈[0.1,0.5], β∈[0.8,1.2]']
}

ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=list(zip(summary_data['Metric'], summary_data['Value'])),
                 colLabels=['Metric', 'Value'],
                 cellLoc='left',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.tight_layout()
fig.savefig('clean_figures_final/figG_training_data_publication.pdf', 
            dpi=300, bbox_inches='tight', pad_inches=0.2)
fig.savefig('clean_figures_final/figG_training_data_publication.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print("✓ Created publication-ready training data plot")

print("\n✅ All remaining publication-ready figures created!")
