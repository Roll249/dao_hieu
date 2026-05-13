"""Fig QC-3: Parameter-Shift Rule Gradient Visualization.

Illustrates the PSR mechanics from the paper's Proposition 3.4 (Eq. 11):
  grad f(theta) = 1/2 [f(theta + pi/2) - f(theta - pi/2)]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def generate_psr_figure(output_path=None):
    """Generate Parameter-Shift Rule visualization."""
    setup_style()

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[1.5, 1],
                            hspace=0.4, wspace=0.3,
                            left=0.06, right=0.97, top=0.88, bottom=0.12)

    C_CIRC = '#2980B9'
    C_PLUS = '#27AE60'
    C_MINUS = '#C0392B'
    C_GRAD = '#8E44AD'
    C_ARROW = '#555555'

    # =====================================================================
    # Panel A: Two Circuit Evaluations
    # =====================================================================
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 12)
    ax_a.set_ylim(-1.2, 1.4)
    ax_a.set_title('(a) Two Circuit Evaluations for Exact Gradient Computation',
                    fontsize=11, fontweight='bold', pad=10)
    ax_a.axis('off')

    # Draw circuit 1: theta + pi/2
    ax_a.text(1.5, 1.2, 'Circuit $\\#1$: $\\theta + \\frac{\\pi}{2}$', ha='center', va='center',
              fontsize=10, fontweight='bold', color=C_PLUS)
    ax_a.text(1.5, 0.9, '$f_+= \\langle\\hat{O}\\rangle_{\\theta+\\pi/2}$', ha='center', va='center',
              fontsize=9, style='italic')

    # Wire
    ax_a.plot([0.3, 3.5], [0.4, 0.4], color='#7F8C8D', linewidth=2, zorder=2)
    ax_a.text(0.1, 0.4, '$|\\psi_0\\rangle$', ha='right', va='center', fontsize=8, style='italic')

    # Input box
    box_in = FancyBboxPatch((0.4, 0.15), 0.8, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#34495E', edgecolor='#2C3E50',
                              linewidth=1.5, alpha=0.95, zorder=3)
    ax_a.add_patch(box_in)
    ax_a.text(0.8, 0.4, '$U(\\theta)$', ha='center', va='center',
              fontsize=8, fontweight='bold', color='white', zorder=4)

    # Shift annotation
    ax_a.annotate('', xy=(1.3, 0.4), xytext=(1.2, 0.4),
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax_a.text(1.35, 0.6, '$+\\pi/2$', ha='left', va='center', fontsize=8, color=C_PLUS, fontweight='bold')

    # RY gate with shift
    box_ry = FancyBboxPatch((1.4, 0.15), 1.1, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=C_PLUS, edgecolor='#1a7a3a',
                              linewidth=2, alpha=0.95, zorder=3)
    ax_a.add_patch(box_ry)
    ax_a.text(1.95, 0.4, '$R_Y(\\theta + \\frac{\\pi}{2})$', ha='center', va='center',
              fontsize=7.5, fontweight='bold', color='white', zorder=4)

    # CNOT
    ax_a.plot(2.7, 0.4, 'o', color='#333333', markersize=8, zorder=5)
    ax_a.plot([2.7, 2.7], [0.4, -0.1], color='#7F8C8D', linewidth=1.8, zorder=2)
    ax_a.plot([2.5, 2.9], [-0.1, -0.1], color='#333333', linewidth=2, zorder=5)
    ax_a.plot([2.7, 2.7], [-0.1 - 0.12, -0.1 + 0.12], color='#333333', linewidth=2, zorder=5)

    # Measurement
    box_m = FancyBboxPatch((3.0, 0.15), 0.5, 0.5,
                             boxstyle="round,pad=0.05",
                             facecolor='#C0392B', edgecolor='#922b21',
                             linewidth=1.5, alpha=0.9, zorder=3)
    ax_a.add_patch(box_m)
    ax_a.text(3.25, 0.4, '$\\langle\\hat{O}\\rangle$', ha='center', va='center',
              fontsize=8, fontweight='bold', color='white', zorder=4)

    ax_a.plot([3.5, 3.8], [0.4, 0.4], color='#7F8C8D', linewidth=2, zorder=2)

    # Result box
    box_r = FancyBboxPatch((3.9, 0.2), 0.9, 0.4,
                              boxstyle="round,pad=0.05",
                              facecolor='#F39C12', edgecolor='#B7950B',
                              linewidth=1.5, alpha=0.95, zorder=3)
    ax_a.add_patch(box_r)
    ax_a.text(4.35, 0.4, '$f_+$', ha='center', va='center',
              fontsize=9, fontweight='bold', color='white', zorder=4)

    # ---- Separator ----
    ax_a.plot([5.0, 5.0], [-0.8, 1.3], color='#CCCCCC', linewidth=1, linestyle='--')

    # ---- Circuit 2: theta - pi/2 ----
    ax_a.text(7.5, 1.2, 'Circuit $\\#2$: $\\theta - \\frac{\\pi}{2}$', ha='center', va='center',
              fontsize=10, fontweight='bold', color=C_MINUS)
    ax_a.text(7.5, 0.9, '$f_- = \\langle\\hat{O}\\rangle_{\\theta-\\pi/2}$', ha='center', va='center',
              fontsize=9, style='italic')

    ax_a.plot([6.3, 9.5], [0.4, 0.4], color='#7F8C8D', linewidth=2, zorder=2)
    ax_a.text(6.1, 0.4, '$|\\psi_0\\rangle$', ha='right', va='center', fontsize=8, style='italic')

    # RY gate with shift
    box_ry2 = FancyBboxPatch((7.4, 0.15), 1.1, 0.5,
                               boxstyle="round,pad=0.05",
                               facecolor=C_MINUS, edgecolor='#7B241C',
                               linewidth=2, alpha=0.95, zorder=3)
    ax_a.add_patch(box_ry2)
    ax_a.text(7.95, 0.4, '$R_Y(\\theta - \\frac{\\pi}{2})$', ha='center', va='center',
              fontsize=7.5, fontweight='bold', color='white', zorder=4)
    ax_a.text(7.55, 0.6, '$-\\pi/2$', ha='left', va='center', fontsize=8, color=C_MINUS, fontweight='bold')

    # Measurement
    box_m2 = FancyBboxPatch((9.0, 0.15), 0.5, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#C0392B', edgecolor='#922b21',
                              linewidth=1.5, alpha=0.9, zorder=3)
    ax_a.add_patch(box_m2)
    ax_a.text(9.25, 0.4, '$\\langle\\hat{O}\\rangle$', ha='center', va='center',
              fontsize=8, fontweight='bold', color='white', zorder=4)

    ax_a.plot([9.5, 9.8], [0.4, 0.4], color='#7F8C8D', linewidth=2, zorder=2)

    # Result box
    box_r2 = FancyBboxPatch((9.9, 0.2), 0.9, 0.4,
                               boxstyle="round,pad=0.05",
                               facecolor='#F39C12', edgecolor='#B7950B',
                               linewidth=1.5, alpha=0.95, zorder=3)
    ax_a.add_patch(box_r2)
    ax_a.text(10.35, 0.4, '$f_-$', ha='center', va='center',
              fontsize=9, fontweight='bold', color='white', zorder=4)

    # =====================================================================
    # Panel B: Gradient Computation
    # =====================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 4)
    ax_b.set_ylim(-1.2, 1.2)
    ax_b.set_title('(b) Gradient Formula', fontsize=11, fontweight='bold', pad=10)
    ax_b.axis('off')

    # Formula box
    formula_box = FancyBboxPatch((0.2, -0.2), 3.6, 0.9,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#FDFEFE', edgecolor='#5D6D7E',
                                   linewidth=2, alpha=0.95, zorder=3)
    ax_b.add_patch(formula_box)
    ax_b.text(2.0, 0.35, '$\\frac{\\partial \\langle\\hat{O}\\rangle}{\\partial \\theta_j}$', ha='center', va='center',
              fontsize=12, fontweight='bold', color='#1a1a2e', zorder=4)
    ax_b.text(2.0, 0.0, '$= \\frac{1}{2}\\left[ f_+\\! -\\! f_- \\right]$', ha='center', va='center',
              fontsize=11, color='#2C3E50', zorder=4)

    # Arrow and note
    ax_b.text(2.0, -0.7, 'Exact analytic gradient\nNo finite differences needed!', ha='center', va='center',
              fontsize=8, style='italic', color='#27AE60')

    # =====================================================================
    # Panel C: Comparison with Finite Differences
    # =====================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_xlim(0, 4)
    ax_c.set_ylim(-1.2, 1.2)
    ax_c.set_title('(c) Finite Difference (biased)', fontsize=11, fontweight='bold', pad=10)
    ax_c.axis('off')

    fd_box = FancyBboxPatch((0.2, -0.2), 3.6, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor='#FDEDEC', edgecolor='#922B21',
                             linewidth=2, alpha=0.95, zorder=3)
    ax_c.add_patch(fd_box)
    ax_c.text(2.0, 0.35, '$\\frac{\\partial \\langle\\hat{O}\\rangle}{\\partial \\theta_j}$', ha='center', va='center',
              fontsize=12, fontweight='bold', color='#1a1a2e', zorder=4)
    ax_c.text(2.0, 0.0, '$\\approx \\frac{f(\\theta+\\epsilon) - f(\\theta)}{\\epsilon}$', ha='center', va='center',
              fontsize=10, color='#922B21', zorder=4)

    ax_c.text(2.0, -0.7, 'Requires small $\\epsilon$\nIntroduces numerical bias', ha='center', va='center',
              fontsize=8, style='italic', color='#C0392B')

    # =====================================================================
    # Panel D: Why it works
    # =====================================================================
    ax_d = fig.add_subplot(gs[1, 2])
    ax_d.set_xlim(0, 4)
    ax_d.set_ylim(-1.2, 1.2)
    ax_d.set_title('(d) Key Insight', fontsize=11, fontweight='bold', pad=10)
    ax_d.axis('off')

    insight_box = FancyBboxPatch((0.1, -0.8), 3.8, 1.6,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#EAF2FF', edgecolor='#2980B9',
                                   linewidth=2, alpha=0.95, zorder=3)
    ax_d.add_patch(insight_box)

    ax_d.text(2.0, 0.55, 'For $R_Y(\\theta)$ gates:', ha='center', va='center',
              fontsize=8.5, color='#2C3E50')
    ax_d.text(2.0, 0.2, '$R_Y(\\theta) = e^{-i\\theta\\sigma_y/2}$', ha='center', va='center',
              fontsize=10, fontweight='bold', color='#1a1a2e')
    ax_d.text(2.0, -0.15, '$\\sigma_y$ has eigenvalues $\\pm 1$', ha='center', va='center',
              fontsize=8.5, color='#2C3E50')
    ax_d.text(2.0, -0.45, '$\\Rightarrow$ Shift by $\\pi/2$ swaps eigenstates', ha='center', va='center',
              fontsize=8.5, fontweight='bold', color='#27AE60')

    fig.text(0.5, 0.01,
             'Fig QC-3: Parameter-Shift Rule gradient computation. The exact analytic gradient '
             '$\\partial\\langle\\hat{O}\\rangle/\\partial\\theta$ is obtained from two circuit evaluations '
             'with shifted angles $\\theta \\pm \\pi/2$, requiring no finite-difference approximation. '
             'The $\\frac{1}{2}$ prefactor arises from the $\\sigma_y$ generator eigenvalues.',
             ha='center', va='bottom', fontsize=8, style='italic', color='#444444')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        path_pdf = os.path.join(OUTPUT_DIR, 'qc3_psr_gradient.pdf')
        path_png = os.path.join(OUTPUT_DIR, 'qc3_psr_gradient.png')
        fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {path_pdf} and {path_png}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    generate_psr_figure()
