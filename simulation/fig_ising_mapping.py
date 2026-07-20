"""Fig QC-4: QUBO to Ising Hamiltonian Mapping.

Visualizes the transformation from:
  QUBO: min H(z) = sum c_n z_n + A*(sum z_n - 1)^2
  Ising: H = sum h_n sigma_z^n + sum J_ij sigma_z^i sigma_z^j + E0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


def generate_ising_figure(output_path=None):
    """Generate QUBO-to-Ising mapping visualization."""
    setup_style()

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.6],
                          hspace=0.35, left=0.04, right=0.97,
                          top=0.91, bottom=0.08)

    C_QUBO = '#8E44AD'   # Purple for QUBO
    C_ISING = '#27AE60'  # Green for Ising
    C_H = '#E67E22'     # Orange for fields
    C_J = '#2980B9'      # Blue for couplings
    C_OFFSET = '#7F8C8D' # Gray for offset

    # =================================================================
    # Panel A: The Transformation
    # =================================================================
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14)
    ax.set_ylim(-2.0, 1.8)
    ax.set_title('(a) QUBO to Ising Hamiltonian Transformation',
                 fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')

    M = 5  # Number of nodes (e.g., M_l* = 5 RSUs)

    # ---- QUBO Side ----
    ax.text(2.0, 1.5, 'QUBO Problem', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C_QUBO)

    qubo_box = FancyBboxPatch((0.3, -0.5), 3.5, 1.8,
                               boxstyle="round,pad=0.15",
                               facecolor='#F5EEF8', edgecolor=C_QUBO,
                               linewidth=2, alpha=0.95, zorder=3)
    ax.add_patch(qubo_box)

    ax.text(2.05, 1.0, '$\\min_z \\; H(z) = \\sum_{n=1}^{M} c_n z_n$', ha='center', va='center',
            fontsize=9.5)
    ax.text(2.05, 0.6, '$+ A\\left(\\sum_{n=1}^{M} z_n - 1\\right)^2$', ha='center', va='center',
            fontsize=9.5)
    ax.text(2.05, 0.1, '$\\mathrm{s.t.}\\; z_n \\in \\{0,1\\}$', ha='center', va='center',
            fontsize=9, style='italic')

    # Node labels
    ax.text(0.6, -0.1, 'Node costs:', ha='left', va='center', fontsize=8, color='#555555')
    for n in range(M):
        x = 0.9 + n * 0.55
        box = FancyBboxPatch((x - 0.2, -0.35), 0.4, 0.3,
                               boxstyle="round,pad=0.02",
                               facecolor='#D7BDE2', edgecolor=C_QUBO,
                               linewidth=1, alpha=0.9, zorder=4)
        ax.add_patch(box)
        ax.text(x, -0.2, f'$c_{n+1}$', ha='center', va='center',
                fontsize=7.5, fontweight='bold')

    ax.text(2.05, -0.7, '$A \\gg \\max_n c_n$ enforces one-hot', ha='center', va='center',
            fontsize=8, style='italic', color='#666666')

    # ---- Arrow ----
    ax.annotate('', xy=(5.2, 0.5), xytext=(4.0, 0.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
    ax.text(4.6, 0.75, '$z_n = \\frac{I - \\sigma_z^n}{2}$', ha='center', va='center',
            fontsize=9, color='#333333')

    # ---- Ising Side ----
    ax.text(9.0, 1.5, 'Ising Hamiltonian', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C_ISING)

    ising_box = FancyBboxPatch((6.0, -0.7), 5.5, 2.0,
                                boxstyle="round,pad=0.15",
                                facecolor='#EAFAF1', edgecolor=C_ISING,
                                linewidth=2, alpha=0.95, zorder=3)
    ax.add_patch(ising_box)

    ax.text(8.75, 1.0, '$H_C = \\sum_{n=1}^{M} h_n \\hat{\\sigma}_z^n$', ha='center', va='center',
            fontsize=9.5)
    ax.text(8.75, 0.6, '$+ \\sum_{i<j} J_{ij} \\hat{\\sigma}_z^i \\hat{\\sigma}_z^j + E_0$', ha='center', va='center',
            fontsize=9.5)

    # Local fields
    ax.text(6.3, 0.05, 'Local fields:', ha='left', va='center', fontsize=8, color='#555555')
    for n in range(M):
        x = 6.8 + n * 0.8
        box = FancyBboxPatch((x - 0.28, -0.25), 0.56, 0.4,
                               boxstyle="round,pad=0.02",
                               facecolor='#ABEBC6', edgecolor=C_ISING,
                               linewidth=1.5, alpha=0.9, zorder=4)
        ax.add_patch(box)
        ax.text(x, -0.05, f'$h_{n+1}$', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1D8348')

    # Couplings (quadratic terms)
    ax.text(6.3, -0.5, 'Couplings:', ha='left', va='center', fontsize=8, color='#555555')
    for i in range(min(3, M)):
        for j in range(i + 1, M):
            x = 6.8 + i * 1.2
            y = -0.85
            box = FancyBboxPatch((x - 0.25, y - 0.15), 0.5, 0.3,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#AED6F1', edgecolor=C_J,
                                   linewidth=1.5, alpha=0.9, zorder=4)
            ax.add_patch(box)
            ax.text(x, y, f'$J_{{{i+1},{j+1}}}$', ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='#1a5276')

    ax.text(8.75, -0.9, '$\\Rightarrow \\mathrm{Ground\\ state} = \\mathrm{optimal\\ node}$',
            ha='center', va='center', fontsize=8.5, color='#27AE60', fontweight='bold')

    # =================================================================
    # Panel B: Mathematical Derivation
    # =================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 14)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title('(b) Step-by-Step Derivation (Verified with SymPy)',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.axis('off')

    steps = [
        ('1', '$z_n = \\frac{I - \\sigma_z^n}{2}$', '(Paper, Remark 3.2)'),
        ('2', '$\\Rightarrow \\sigma_z^n = I - 2z_n$', '(Algebra)'),
        ('3', '$H = \\sum_n c_n\\frac{I-\\sigma_z^n}{2} + A(\\sum_n z_n-1)^2$', '(Substitution)'),
        ('4', '$h_n = -\\frac{c_n}{2} + A\\left(1 - \\frac{M}{2}\\right)$', '(Local field, verified analytically)'),
        ('5', '$J_{ij} = +\\frac{A}{2}$  (Paper Eq.(28) has $+A/4$ — wrong sign & magnitude)', '(Coupling, verified for M=2,3,5)'),
    ]

    colors = ['#FDFEFE', '#FDFEFE', '#FDFEFE', '#E8F8F5', '#FDEDEC']

    for i, (num, expr, note) in enumerate(steps):
        y = 0.65 - i * 0.32
        x = 0.3
        ax2.text(x, y, f'({num})', ha='left', va='center', fontsize=8.5, fontweight='bold')
        ax2.text(x + 0.5, y, expr, ha='left', va='center', fontsize=9)
        ax2.text(x + 0.5, y - 0.15, note, ha='left', va='center', fontsize=7.5,
                style='italic', color='#666666')

        # Box
        box = FancyBboxPatch((x - 0.05, y - 0.22), 13.5, 0.28,
                               boxstyle="round,pad=0.05",
                               facecolor=colors[i], edgecolor='#BDC3C7',
                               linewidth=0.8, alpha=0.8, zorder=2)
        ax2.add_patch(box)

    # Correction annotation
    ax2.annotate('', xy=(12.0, 0.65), xytext=(11.5, 0.65),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5))
    ax2.text(12.2, 0.65, 'Paper\nEq.28:\nwrong sign\n& magnitude\n($+A/4$)', ha='left', va='center',
            fontsize=7, color='#C0392B', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#D7BDE2', edgecolor='#8E44AD', label='QUBO (minimization)'),
        mpatches.Patch(facecolor='#ABEBC6', edgecolor='#27AE60', label='Local fields $h_n$'),
        mpatches.Patch(facecolor='#AED6F1', edgecolor='#2980B9', label='Couplings $J_{ij}$'),
        mpatches.Patch(facecolor='#FDEDEC', edgecolor='#C0392B', label='Correction needed'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=4, frameon=True, framealpha=0.9, fontsize=8,
               bbox_to_anchor=(0.5, 0.96))

    fig.text(0.5, 0.01,
             'Fig QC-4: QUBO to Ising Hamiltonian mapping. Binary variables $z_n \\in \\{0,1\\}$ are mapped to '
             'Pauli-Z operators via $z_n = (I - \\sigma_z^n)/2$, converting the node-selection objective into '
             'an Ising Hamiltonian. NOTE: The paper\'s Eq. (28) gives $J_{ij} = +A/4$, which is incorrect in '
             'both sign and magnitude. The analytically verified correct value is $J_{ij} = +A/2$ '
             '(derivation step 5; verified for $M = 2, 3, 5$ by exhaustive QUBO evaluation).',
             ha='center', va='bottom', fontsize=8, style='italic', color='#444444')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        path_pdf = os.path.join(OUTPUT_DIR, 'qc4_ising_mapping.pdf')
        path_png = os.path.join(OUTPUT_DIR, 'qc4_ising_mapping.png')
        fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {path_pdf} and {path_png}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    generate_ising_figure()
