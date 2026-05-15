"""Fig QC-2: QAOA Circuit for Node Selection.

Shows the depth-p QAOA circuit from the paper's Eq. (36):
  |gamma, beta> = prod_{q=1}^p e^{-i beta_q H_M} e^{-i gamma_q H_C} |+>^{otimes M}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def generate_qaoa_figure(output_path=None):
    """Generate QAOA circuit architecture figure."""
    setup_style()

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1],
                          hspace=0.4, left=0.05, right=0.97, top=0.92, bottom=0.1)

    # Colors
    C_H = '#27AE60'       # Hadamard (green)
    C_COST = '#8E44AD'    # Cost unitary (purple)
    C_MIXER = '#2980B9'   # Mixer (blue)
    C_MEAS = '#C0392B'    # Measurement (red)
    C_WIRE = '#7F8C8D'    # Wire
    C_BG = '#f5f5f5'      # Background

    # =================================================================
    # Panel A: QAOA Circuit Diagram
    # =================================================================
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title('(a) QAOA Circuit: $|\\gamma, \\beta\\rangle = \\prod_{q=1}^{p} e^{-i\\beta_q H_M} e^{-i\\gamma_q H_C} |+\\rangle^{\\otimes M}$',
                 fontsize=10, fontweight='bold', pad=12)
    ax.axis('off')

    M = 5   # Number of nodes (e.g., 5 RSUs)
    y_positions = [0.8, 0.4, 0.0, -0.4, -0.8]
    y_labels = [f'q{i}' for i in range(M)]

    # Draw wires
    for i, yp in enumerate(y_positions):
        ax.plot([0.3, 13.5], [yp, yp], color=C_WIRE, linewidth=1.8, zorder=1)
        ax.text(0.15, yp, y_labels[i], fontsize=7, ha='right', va='center')

    # Section labels
    ax.text(1.5, 1.3, 'Initialization', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#2c3e50')
    ax.text(5.0, 1.3, 'QAOA Layer $q=1$', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#2c3e50')
    ax.text(9.0, 1.3, '$\\dots$', ha='center', va='center', fontsize=14)
    ax.text(12.0, 1.3, 'QAOA Layer $q=p$', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#2c3e50')

    # ---- Initial Hadamard layer ----
    x_H = 1.2
    for j, yp in enumerate(y_positions):
        box = FancyBboxPatch((x_H - 0.3, yp - 0.25), 0.6, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=C_H, edgecolor='#1a5e20',
                              linewidth=1.5, alpha=0.95, zorder=3)
        ax.add_patch(box)
        ax.text(x_H, yp, '$H$', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', zorder=4)

    # Label
    ax.text(x_H, -1.15, '$|+\\rangle^{\\otimes M}$', ha='center', va='top',
            fontsize=8, style='italic')

    # Arrow after Hadamard
    ax.annotate('', xy=(2.3, 0), xytext=(2.0, 0),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # ---- QAOA Layer 1 ----
    layer_width = 3.5
    gamma_x = 3.2
    beta_x = 4.8

    # Layer 1 background
    layer1_box = FancyBboxPatch((2.5, -1.1), layer_width, 2.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#fafafa', edgecolor='#bdc3c7',
                                  linewidth=1, alpha=0.7, zorder=0)
    ax.add_patch(layer1_box)

    # Cost Hamiltonian box
    cost_box = FancyBboxPatch((gamma_x - 0.5, 0.9), 2.2, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor=C_COST, edgecolor='#5B2C6F',
                               linewidth=1.5, alpha=0.95, zorder=3)
    ax.add_patch(cost_box)
    ax.text(gamma_x + 0.6, 1.1, '$e^{-i\\gamma_1 H_C}$', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', zorder=4)
    ax.text(gamma_x + 0.6, 0.75, '$H_C = \\sum h_n \\sigma_z^n + \\sum J_{ij} \\sigma_z^i \\sigma_z^j$',
            ha='center', va='center', fontsize=7, color='#555555')

    # Cost unitary on wires
    for yp in y_positions:
        box = FancyBboxPatch((gamma_x - 0.3, yp - 0.22), 0.6, 0.44,
                              boxstyle="round,pad=0.03",
                              facecolor=C_COST, edgecolor='#5B2C6F',
                              linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(box)
        ax.text(gamma_x, yp, '$e^{-i\\gamma_1}$', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

    # Mixer Hamiltonian box
    mixer_box = FancyBboxPatch((beta_x - 0.5, 0.9), 2.2, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor=C_MIXER, edgecolor='#1a5276',
                                linewidth=1.5, alpha=0.95, zorder=3)
    ax.add_patch(mixer_box)
    ax.text(beta_x + 0.6, 1.1, '$e^{-i\\beta_1 H_M}$', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', zorder=4)
    ax.text(beta_x + 0.6, 0.75, '$H_M = \\sum \\sigma_x^n$',
            ha='center', va='center', fontsize=7, color='#555555')

    # Mixer on wires (X rotations)
    for yp in y_positions:
        box = FancyBboxPatch((beta_x - 0.3, yp - 0.22), 0.6, 0.44,
                              boxstyle="round,pad=0.03",
                              facecolor=C_MIXER, edgecolor='#1a5276',
                              linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(box)
        ax.text(beta_x, yp, '$e^{-i\\beta_1}$', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

    ax.text((gamma_x + beta_x) / 2 + 0.3, -1.15, 'Layer $q=1$',
            ha='center', va='top', fontsize=7.5, fontweight='bold', color='#555555')

    # ---- QAOA Layer p ----
    gamma_x_p = 11.5
    beta_x_p = 12.3

    layerp_box = FancyBboxPatch((9.8, -1.1), 3.4, 2.2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#fafafa', edgecolor='#bdc3c7',
                                  linewidth=1, alpha=0.7, zorder=0)
    ax.add_patch(layerp_box)

    for yp in y_positions:
        box = FancyBboxPatch((gamma_x_p - 0.3, yp - 0.22), 0.6, 0.44,
                              boxstyle="round,pad=0.03",
                              facecolor=C_COST, edgecolor='#5B2C6F',
                              linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(box)
        ax.text(gamma_x_p, yp, '$e^{-i\\gamma_p}$', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

        box2 = FancyBboxPatch((beta_x_p - 0.3, yp - 0.22), 0.6, 0.44,
                               boxstyle="round,pad=0.03",
                               facecolor=C_MIXER, edgecolor='#1a5276',
                               linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(box2)
        ax.text(beta_x_p, yp, '$e^{-i\\beta_p}$', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=4)

    ax.text(gamma_x_p, -1.15, 'Layer $q=p$', ha='center', va='top',
            fontsize=7.5, fontweight='bold', color='#555555')
    ax.text((gamma_x_p + beta_x_p) / 2 + 0.3, -1.15, '($p=2$ in experiments)',
            ha='center', va='top', fontsize=7, style='italic', color='#888888')

    # ---- Measurement ----
    meas_x = 13.0
    for yp in y_positions:
        box = FancyBboxPatch((meas_x - 0.2, yp - 0.22), 0.4, 0.44,
                              boxstyle="round,pad=0.03",
                              facecolor=C_MEAS, edgecolor='#922b21',
                              linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(box)
        ax.text(meas_x, yp, '$M$', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)

    ax.text(meas_x, -1.15, '$|z^*\\rangle$', ha='center', va='top',
            fontsize=8, style='italic')

    # =================================================================
    # Panel B: Ising Hamiltonian Structure
    # =================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 14)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title('(b) Cost Hamiltonian Structure: Node Selection QUBO $\\rightarrow$ Ising',
                  fontsize=10, fontweight='bold', pad=12)
    ax2.axis('off')

    # QUBO matrix
    ax2.text(2.5, 0.8, 'QUBO Matrix $Q$', ha='center', va='center',
             fontsize=9, fontweight='bold')
    ax2.text(2.5, 0.45, '$\\min_z \\; z^\\top Q z$', ha='center', va='center',
             fontsize=8, style='italic')

    # Draw QUBO matrix
    for i in range(M):
        for j in range(M):
            x = 1.0 + j * 0.55
            y = -0.1 - i * 0.35
            if i == j:
                color = '#E8DAEF'
                label = f'$c_{i+1}$'
            elif abs(i - j) == 1:
                color = '#D5F5E3'
                label = '$2A$'
            else:
                color = '#FDEBD0'
                label = '$2A$'
            rect = plt.Rectangle((x - 0.22, y - 0.14), 0.44, 0.28,
                                  facecolor=color, edgecolor='#7f8c8d',
                                  linewidth=0.8, zorder=3)
            ax2.add_patch(rect)
            ax2.text(x, y, label, ha='center', va='center',
                    fontsize=7, fontweight='bold')

    ax2.text(4.2, -0.2, 'Pauli mapping -->', ha='center', va='center', fontsize=10)

    # Ising representation
    ax2.text(8.5, 0.8, 'Ising Hamiltonian $H_C$', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Draw Ising
    # Local fields
    ax2.text(7.0, 0.4, '$h_n$:', ha='right', va='center', fontsize=8)
    for n in range(M):
        box = FancyBboxPatch((7.1 + n * 0.7, 0.2), 0.55, 0.4,
                               boxstyle="round,pad=0.03",
                               facecolor='#AED6F1', edgecolor='#5D6D7E',
                               linewidth=1, alpha=0.9, zorder=3)
        ax2.add_patch(box)
        ax2.text(7.37 + n * 0.7, 0.4, f'$\\sigma_z^{n+1}$', ha='center', va='center',
                fontsize=7, fontweight='bold')

    # Couplings
    ax2.text(7.0, -0.3, '$J_{ij}$:', ha='right', va='center', fontsize=8)
    for i in range(M):
        for j in range(i + 1, min(i + 3, M)):
            x = 7.1 + i * 0.7
            y = -0.5
            box = FancyBboxPatch((x, y - 0.12), 0.55, 0.24,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#F9E79F', edgecolor='#7D6608',
                                   linewidth=1, alpha=0.9, zorder=3)
            ax2.add_patch(box)
            ax2.text(x + 0.275, y, f'$J_{{{i+1},{j+1}}}$', ha='center', va='center',
                    fontsize=6.5, fontweight='bold')

    ax2.text(10.5, 0.4, 'Ground state $\\Rightarrow$', ha='center', va='center',
             fontsize=8)
    ax2.text(11.8, 0.4, 'optimal node $n^*$', ha='center', va='center',
             fontsize=8, fontweight='bold', color='#27AE60')

    # Arrow
    ax2.annotate('', xy=(11.2, 0.4), xytext=(10.9, 0.4),
                 arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_H, label='Hadamard (Init)'),
        mpatches.Patch(facecolor=C_COST, label='Cost Unitary $e^{-i\\gamma H_C}$'),
        mpatches.Patch(facecolor=C_MIXER, label='Mixer $e^{-i\\beta H_M}$'),
        mpatches.Patch(facecolor=C_MEAS, label='Measurement'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=4, frameon=True, framealpha=0.9, fontsize=8,
               bbox_to_anchor=(0.5, 0.96))

    fig.text(0.5, 0.01,
             'Fig QC-2: QAOA circuit for optimal node selection. The depth-$p$ QAOA alternates between the cost unitary $e^{-i\\gamma_q H_C}$ (encoding the QUBO objective) '
             'and the mixer $e^{-i\\beta_q H_M}$ (enabling amplitude redistribution). '
             'Measuring the final state in the computational basis yields the optimal node bitstring $z^*$.',
             ha='center', va='bottom', fontsize=8, style='italic', color='#444444')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        path_pdf = os.path.join(OUTPUT_DIR, 'qc2_qaoa_circuit.pdf')
        path_png = os.path.join(OUTPUT_DIR, 'qc2_qaoa_circuit.png')
        fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {path_pdf} and {path_png}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    generate_qaoa_figure()
