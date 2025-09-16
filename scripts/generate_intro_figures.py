#!/usr/bin/env python3
"""
Generate clean figures for Introduction section
- Figure A: Microgrid architecture and dynamics
- Figure B: PINN limitations and failure modes
- Figure C: UDE vs BNODE conceptual comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import seaborn as sns

# Set scientific plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def create_microgrid_architecture_figure():
    """Figure A: Microgrid Architecture and Dynamics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Microgrid architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Microgrid Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # Main grid
    main_grid = Rectangle((1, 3), 2, 2, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax1.add_patch(main_grid)
    ax1.text(2, 4, 'Main Grid', ha='center', va='center', fontweight='bold')
    
    # Microgrid boundary
    microgrid = FancyBboxPatch((4, 1), 5, 6, boxstyle="round,pad=0.1", 
                              linewidth=2, edgecolor='black', facecolor='white')
    ax1.add_patch(microgrid)
    ax1.text(6.5, 6.5, 'Microgrid', ha='center', va='center', fontweight='bold', fontsize=14)
    
    # Renewable sources
    solar = Circle((5.5, 5.5), 0.3, color='orange', alpha=0.7)
    wind = Circle((7.5, 5.5), 0.3, color='lightblue', alpha=0.7)
    ax1.add_patch(solar)
    ax1.add_patch(wind)
    ax1.text(5.5, 4.8, 'Solar', ha='center', va='center', fontsize=10)
    ax1.text(7.5, 4.8, 'Wind', ha='center', va='center', fontsize=10)
    
    # Storage
    storage = Rectangle((5, 3), 1, 0.8, linewidth=1, edgecolor='black', facecolor='green', alpha=0.5)
    ax1.add_patch(storage)
    ax1.text(5.5, 3.4, 'Storage', ha='center', va='center', fontsize=10)
    
    # Loads
    load1 = Rectangle((6.5, 2), 1, 0.8, linewidth=1, edgecolor='black', facecolor='red', alpha=0.5)
    load2 = Rectangle((7.5, 2), 1, 0.8, linewidth=1, edgecolor='black', facecolor='red', alpha=0.5)
    ax1.add_patch(load1)
    ax1.add_patch(load2)
    ax1.text(7, 1.3, 'Loads', ha='center', va='center', fontsize=10)
    
    # Inverters
    inv1 = Circle((5.5, 4.5), 0.15, color='purple', alpha=0.7)
    inv2 = Circle((7.5, 4.5), 0.15, color='purple', alpha=0.7)
    ax1.add_patch(inv1)
    ax1.add_patch(inv2)
    
    # Connections
    ax1.plot([3, 4], [4, 4], 'k-', linewidth=2)
    ax1.plot([5.5, 5.5], [5.2, 4.8], 'k-', linewidth=1)
    ax1.plot([7.5, 7.5], [5.2, 4.8], 'k-', linewidth=1)
    ax1.plot([5.5, 6.5], [3.8, 3.8], 'k-', linewidth=1)
    ax1.plot([7.5, 7.5], [4.2, 2.8], 'k-', linewidth=1)
    
    # Right panel: Dynamics
    ax2.set_title('(b) System Dynamics', fontsize=14, fontweight='bold', pad=20)
    
    # Time series simulation
    t = np.linspace(0, 10, 200)
    np.random.seed(42)
    
    # Frequency dynamics
    freq = 0.5 * np.exp(-0.3 * t) * np.cos(2 * np.pi * t) + 0.1 * np.random.randn(200)
    ax2.plot(t, freq, 'k-', linewidth=1.5, label='Frequency (x₂)')
    
    # Storage dynamics
    storage_state = 0.5 + 0.3 * np.sin(0.5 * np.pi * t) + 0.05 * np.random.randn(200)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, storage_state, 'k--', linewidth=1.5, label='Storage (x₁)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency Deviation (p.u.)', color='black')
    ax2_twin.set_ylabel('State of Charge', color='black')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2_twin.legend(loc='upper left')
    
    plt.tight_layout()
    return fig

def create_pinn_limitations_figure():
    """Figure B: PINN Limitations and Failure Modes"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Gradient pathologies
    ax1.set_title('(a) Gradient Pathologies', fontsize=12, fontweight='bold')
    x = np.linspace(0, 2*np.pi, 100)
    true_sol = np.sin(x) + 0.1 * x
    pinn_sol = 0.5 * np.ones_like(x)  # PINN converges to trivial solution
    
    ax1.plot(x, true_sol, 'k-', linewidth=2, label='True Solution')
    ax1.plot(x, pinn_sol, 'r--', linewidth=2, label='PINN Solution')
    ax1.fill_between(x, true_sol, pinn_sol, alpha=0.3, color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 0.8, 'Gradient\nPathology', transform=ax1.transAxes, 
             ha='center', va='center', fontsize=10, color='red')
    
    # Panel 2: Spectral bias
    ax2.set_title('(b) Spectral Bias', fontsize=12, fontweight='bold')
    frequencies = np.array([1, 2, 4, 8, 16])
    true_coeffs = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    pinn_coeffs = np.array([0.9, 0.3, 0.1, 0.05, 0.01])
    
    x_pos = np.arange(len(frequencies))
    width = 0.35
    ax2.bar(x_pos - width/2, true_coeffs, width, label='True', color='black', alpha=0.7)
    ax2.bar(x_pos + width/2, pinn_coeffs, width, label='PINN', color='red', alpha=0.7)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Fourier Coefficient')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(frequencies)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Stiff equation failure
    ax3.set_title('(c) Stiff Equation Failure', fontsize=12, fontweight='bold')
    t = np.linspace(0, 1, 1000)
    # Stiff ODE: dy/dt = -1000y + 1000, y(0) = 0
    true_sol = 1 - np.exp(-1000 * t)
    pinn_sol = 0.5 * t  # PINN fails to capture fast dynamics
    
    ax3.plot(t, true_sol, 'k-', linewidth=2, label='True Solution')
    ax3.plot(t, pinn_sol, 'r--', linewidth=2, label='PINN Solution')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('y(t)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.7, 0.3, 'Fast\nDynamics\nMissed', transform=ax3.transAxes, 
             ha='center', va='center', fontsize=10, color='red')
    
    # Panel 4: Competing loss terms
    ax4.set_title('(d) Competing Loss Terms', fontsize=12, fontweight='bold')
    epochs = np.linspace(0, 1000, 100)
    data_loss = 1.0 * np.exp(-epochs/200)
    pde_loss = 0.1 * np.exp(-epochs/100)
    total_loss = data_loss + pde_loss
    
    ax4.plot(epochs, data_loss, 'b-', linewidth=2, label='Data Loss')
    ax4.plot(epochs, pde_loss, 'g-', linewidth=2, label='PDE Loss')
    ax4.plot(epochs, total_loss, 'k--', linewidth=2, label='Total Loss')
    ax4.set_xlabel('Training Epochs')
    ax4.set_ylabel('Loss Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(0.6, 0.7, 'Imbalanced\nOptimization', transform=ax4.transAxes, 
             ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    return fig

def create_ude_bnode_comparison_figure():
    """Figure C: UDE vs BNODE Conceptual Comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: UDE approach
    ax1.set_title('(a) Universal Differential Equation (UDE)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # Physical model box
    physics_box = FancyBboxPatch((1, 4), 3, 2, boxstyle="round,pad=0.1", 
                                linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax1.add_patch(physics_box)
    ax1.text(2.5, 5, 'Physical Model\n(Mechanistic)', ha='center', va='center', fontweight='bold')
    
    # Neural network box
    nn_box = FancyBboxPatch((6, 4), 3, 2, boxstyle="round,pad=0.1", 
                           linewidth=2, edgecolor='black', facecolor='orange', alpha=0.3)
    ax1.add_patch(nn_box)
    ax1.text(7.5, 5, 'Neural Network\n(Learned)', ha='center', va='center', fontweight='bold')
    
    # Plus sign
    ax1.text(5, 5, '+', ha='center', va='center', fontsize=24, fontweight='bold')
    
    # Arrow
    ax1.arrow(2.5, 3, 0, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax1.text(2.5, 2.5, 'Hybrid\nODE', ha='center', va='center', fontweight='bold')
    
    # Right panel: BNODE approach
    ax2.set_title('(b) Bayesian Neural ODE (BNODE)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    # Neural network with uncertainty
    bnode_box = FancyBboxPatch((2, 4), 6, 2, boxstyle="round,pad=0.1", 
                              linewidth=2, edgecolor='black', facecolor='green', alpha=0.3)
    ax2.add_patch(bnode_box)
    ax2.text(5, 5, 'Bayesian Neural ODE\n(Full Vector Field + Uncertainty)', ha='center', va='center', fontweight='bold')
    
    # Uncertainty visualization
    x_unc = np.linspace(3, 7, 20)
    y_unc = 3.5 + 0.3 * np.sin(2 * np.pi * x_unc / 4)
    y_unc_upper = y_unc + 0.2
    y_unc_lower = y_unc - 0.2
    
    ax2.plot(x_unc, y_unc, 'k-', linewidth=2)
    ax2.fill_between(x_unc, y_unc_lower, y_unc_upper, alpha=0.3, color='gray')
    ax2.text(5, 3.2, 'Uncertainty\nQuantification', ha='center', va='center', fontsize=10)
    
    # Arrow
    ax2.arrow(5, 3, 0, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax2.text(5, 2.5, 'Probabilistic\nODE', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all introduction figures"""
    print("Generating Introduction figures...")
    
    # Figure A: Microgrid Architecture
    fig_a = create_microgrid_architecture_figure()
    fig_a.savefig('clean_figures_final/figA_microgrid_architecture.pdf', dpi=300, bbox_inches='tight')
    fig_a.savefig('clean_figures_final/figA_microgrid_architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print("✓ Figure A: Microgrid Architecture")
    
    # Figure B: PINN Limitations
    fig_b = create_pinn_limitations_figure()
    fig_b.savefig('clean_figures_final/figB_pinn_limitations.pdf', dpi=300, bbox_inches='tight')
    fig_b.savefig('clean_figures_final/figB_pinn_limitations.png', dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    print("✓ Figure B: PINN Limitations")
    
    # Figure C: UDE vs BNODE Comparison
    fig_c = create_ude_bnode_comparison_figure()
    fig_c.savefig('clean_figures_final/figC_ude_bnode_comparison.pdf', dpi=300, bbox_inches='tight')
    fig_c.savefig('clean_figures_final/figC_ude_bnode_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_c)
    print("✓ Figure C: UDE vs BNODE Comparison")
    
    print("\n✅ All Introduction figures generated successfully!")

if __name__ == "__main__":
    main()
