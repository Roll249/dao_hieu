"""Fig QC-5: System Architecture Block Diagram.

Shows the complete Quantum-HRL pipeline:
  T-NTN Environment -> State Normalization -> Amplitude Encoding
  -> VQC (tier + ratio) -> QAOA (node) -> Action Execution
  Bayesian Optimization updates circuit parameters
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
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


def draw_block(ax, x, y, w, h, label, sublabel='', color='#4472C4', fontsize=9):
    """Draw a labeled block."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#2F2F2F',
                          linewidth=1.5, alpha=0.9, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.08, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4)
        ax.text(x, y - 0.15, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color='white', alpha=0.85, zorder=4)
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label='', color='#333333', style='->'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    lw=1.5,
                    connectionstyle='arc3,rad=0'
                ))
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, ha='left', va='center',
                fontsize=7, color='#555555', style='italic')


def generate_system_architecture_figure(output_path=None):
    """Generate the complete Quantum-HRL system architecture diagram."""
    setup_style()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Fig QC-5: End-to-End Quantum-HRL Pipeline for T-NTN Task Offloading',
                 fontsize=12, fontweight='bold', pad=15)

    # Color scheme
    C_ENV = '#2E86AB'      # Environment (blue)
    C_PRE = '#7B2D8B'      # Preprocessing (purple)
    C_VQC = '#27AE60'       # VQC (green)
    C_QAOA = '#E67E22'     # QAOA (orange)
    C_BO = '#8E44AD'       # Bayesian Opt (purple)
    C_ACT = '#C0392B'      # Action (red)
    C_REW = '#F39C12'      # Reward (amber)
    C_ARROW = '#444444'

    # ================================================================
    # LAYER 1: T-NTN Environment
    # ================================================================
    y_env = 7.0
    draw_block(ax, 2.0, y_env, 2.5, 0.8,
               'T-NTN Environment', 'RSU / LAP / HAP / LEO',
               color=C_ENV)

    # Vehicle
    draw_block(ax, 5.5, y_env, 1.5, 0.6,
               'Vehicle', '$s_t \\in \\mathbb{R}^{20}$',
               color='#5D6D7E')

    # Arrow: Environment -> State
    draw_arrow(ax, 3.3, y_env, 4.2, y_env, label='$s_t$', color=C_ARROW)

    # ================================================================
    # LAYER 2: Preprocessing
    # ================================================================
    y_pre = 5.8
    draw_block(ax, 2.0, y_pre, 2.2, 0.6,
               'Normalize', '$\\tilde{s}_t = s_t / ||s_t||$',
               color=C_PRE)

    draw_arrow(ax, 2.0, y_env - 0.4, 2.0, y_pre + 0.3, color=C_ARROW)

    draw_block(ax, 5.5, y_pre, 1.5, 0.6,
               'Amp. Encode', '$q=\\lceil\\log_2 n\\rceil$ qubits',
               color=C_PRE)

    draw_arrow(ax, 4.5, y_pre, 4.7, y_pre, color=C_ARROW)

    # ================================================================
    # LAYER 3: VQC (High-Level Policy)
    # ================================================================
    y_vqc = 4.3
    draw_block(ax, 3.8, y_vqc, 2.8, 1.0,
               'VQC', '$L=4$ layers, $q=5$ qubits',
               color=C_VQC)

    draw_arrow(ax, 5.5, y_pre - 0.3, 5.3, y_vqc + 0.5, color=C_ARROW)
    draw_arrow(ax, 3.2, y_pre, 3.2, y_vqc + 0.5, color=C_ARROW)

    # VQC internals
    draw_block(ax, 3.8, y_vqc - 0.8, 1.0, 0.5,
               '$R_Y(\\theta)$', '', color='#1E8449', fontsize=7)
    draw_arrow(ax, 3.3, y_vqc - 0.2, 3.35, y_vqc - 0.55, color='#1E8449', style='->')

    draw_block(ax, 5.0, y_vqc - 0.8, 1.0, 0.5,
               'CNOT', '', color='#1A5276', fontsize=7)
    draw_arrow(ax, 4.3, y_vqc - 0.8, 4.5, y_vqc - 0.8, color='#1A5276', style='->')

    # Outputs
    draw_block(ax, 2.5, y_vqc, 1.0, 0.45,
               '$\\ell^*$', 'tier [0-3]', color='#1D8348', fontsize=8)
    draw_arrow(ax, 2.5, y_vqc - 0.5, 2.5, y_vqc - 0.8, label='$\\arg\\max$', color='#1D8348', style='->')

    draw_block(ax, 1.5, y_vqc, 1.0, 0.45,
               '$\\alpha$', 'ratio [0,1]', color='#922B21', fontsize=8)
    draw_arrow(ax, 2.4, y_vqc, 2.0, y_vqc, color='#922B21', style='->')

    # ================================================================
    # LAYER 4: QAOA (Node Selection)
    # ================================================================
    y_qaoa = 2.8
    draw_block(ax, 3.8, y_qaoa, 2.8, 1.0,
               'QAOA', '$p=2$ layers, $M_{\\ell^*}$ qubits',
               color=C_QAOA)

    # Arrow from VQC tier to QAOA
    draw_arrow(ax, 2.5, y_vqc, 2.5, y_qaoa + 0.8, color=C_ARROW)
    ax.text(2.0, (y_vqc + y_qaoa) / 2 + 0.5, 'tier $\\ell^*$',
            ha='center', va='center', fontsize=8, style='italic', color='#555555')
    draw_arrow(ax, 2.5, y_qaoa + 0.8, 2.5, y_qaoa + 0.5, color=C_ARROW)

    draw_block(ax, 1.5, y_qaoa, 1.5, 0.6,
               'QUBO costs', '$c_n = \\beta_1 T_{k,n} + \\beta_2 E_{k,n}$',
               color='#B7950B', fontsize=7)

    draw_arrow(ax, 2.2, y_qaoa, 2.5, y_qaoa, color='#B7950B', style='->')

    # QAOA internals
    draw_block(ax, 3.0, y_qaoa - 0.75, 1.2, 0.45,
               '$e^{-i\\gamma H_C}$', '', color='#6C3483', fontsize=7)
    draw_block(ax, 4.6, y_qaoa - 0.75, 1.2, 0.45,
               '$e^{-i\\beta H_M}$', '', color='#154360', fontsize=7)

    # Output
    draw_block(ax, 6.3, y_qaoa, 1.2, 0.5,
               '$n^*$', 'optimal node',
               color=C_ACT, fontsize=8)
    draw_arrow(ax, 5.2, y_qaoa, 5.7, y_qaoa, color=C_ACT, style='->')

    # ================================================================
    # LAYER 5: Action Execution
    # ================================================================
    y_act = 1.5
    draw_block(ax, 8.0, y_act, 2.0, 0.7,
               'Action', '$a_t = (\\ell^*, n^*, \\alpha)$',
               color=C_ACT)

    draw_arrow(ax, 3.8, y_qaoa - 0.5, 7.0, y_act + 0.35, color=C_ARROW)
    draw_arrow(ax, 1.5, y_qaoa, 7.0, y_act + 0.35, color=C_ARROW, style='->')

    draw_block(ax, 11.0, y_act, 2.2, 0.7,
               'Execute Offload', 'Apply to T-NTN',
               color='#1A5276')

    draw_arrow(ax, 9.0, y_act, 9.9, y_act, color=C_ARROW, style='->')

    # ================================================================
    # LAYER 6: Reward
    # ================================================================
    draw_block(ax, 13.0, y_act, 1.5, 0.7,
               'Reward $R_t$', '$-\\beta_1 T_k - \\beta_2 E_k$',
               color=C_REW, fontsize=7)

    draw_arrow(ax, 2.0, y_env - 0.4, 13.0, y_act + 0.35,
               color=C_REW, style='->')
    ax.text(7.5, 7.8, 'Feedback loop', ha='center', va='center',
            fontsize=8, style='italic', color='#888888')

    # ================================================================
    # Bayesian Optimization (side panel)
    # ================================================================
    x_bo = 11.0
    y_bo = 4.3
    draw_block(ax, x_bo, y_bo, 2.2, 1.0,
               'Bayesian Opt.', '$f(\\theta) \\sim GP(\\mu, k)$',
               color=C_BO)

    draw_block(ax, x_bo, y_bo - 0.85, 1.5, 0.5,
               'GP Surrogate', '$\\mathbb{E}[f]$',
               color='#6C3483', fontsize=7)

    draw_arrow(ax, x_bo, y_vqc + 0.5, x_bo, y_bo + 0.5,
               label='$\\theta$', color='#6C3483', style='->')

    draw_arrow(ax, x_bo, y_bo - 0.6, x_bo, y_qaoa + 0.5,
               label='$(\\gamma,\\beta)$', color='#6C3483', style='->')

    # GP
    draw_block(ax, x_bo, y_bo - 1.6, 1.5, 0.5,
               'EI Acquisition', '$\\arg\\max$ EI',
               color='#5170A7', fontsize=7)
    draw_arrow(ax, x_bo, y_bo - 1.1, x_bo, y_bo - 1.35,
               color='#5170A7', style='->')

    draw_arrow(ax, x_bo, y_bo - 1.85, x_bo, y_bo - 0.2,
               color='#5170A7', style='->')
    ax.text(x_bo + 0.3, (y_bo - 1.85 + y_bo - 0.2) / 2,
            'next query', ha='left', va='center',
            fontsize=7, color='#5170A7', style='italic')

    # Replay Buffer
    draw_block(ax, x_bo + 1.8, y_bo, 1.5, 0.8,
               'Replay Buffer', '$|D|\\leq 10{,}000$',
               color='#616A6B', fontsize=7)

    draw_arrow(ax, 5.2, y_qaoa, x_bo + 1.1, y_bo + 0.4,
               label='$(s,a,R,s\')$', color='#616A6B', style='->')

    # ================================================================
    # T-NTN Network Diagram (small inset)
    # ================================================================
    inset_ax = fig.add_axes([0.62, 0.55, 0.15, 0.18])  # [left, bottom, width, height]
    inset_ax.set_xlim(-0.5, 3.5)
    inset_ax.set_ylim(-0.5, 3.5)
    inset_ax.axis('off')
    inset_ax.set_title('T-NTN Tiers', fontsize=7, fontweight='bold', pad=3)

    # LEO satellites
    for i in range(2):
        circle = plt.Circle((0.5 + i * 2.5, 3.0), 0.25, color='#F4D03F', alpha=0.8, zorder=5)
        inset_ax.add_patch(circle)
        inset_ax.text(0.5 + i * 2.5, 3.0, 'LEO', ha='center', va='center',
                      fontsize=5, fontweight='bold', color='black')

    # HAPs
    for i in range(2):
        circle = plt.Circle((0.8 + i * 2.0, 2.2), 0.2, color='#3498DB', alpha=0.8, zorder=5)
        inset_ax.add_patch(circle)
        inset_ax.text(0.8 + i * 2.0, 2.2, 'HAP', ha='center', va='center',
                      fontsize=5, fontweight='bold', color='white')

    # LAPs
    for i in range(3):
        circle = plt.Circle((0.5 + i * 1.0, 1.3), 0.18, color='#27AE60', alpha=0.8, zorder=5)
        inset_ax.add_patch(circle)
        inset_ax.text(0.5 + i * 1.0, 1.3, 'LAP', ha='center', va='center',
                      fontsize=5, fontweight='bold', color='white')

    # RSUs
    for i in range(5):
        circle = plt.Circle((0.4 + i * 0.6, 0.4), 0.12, color='#C0392B', alpha=0.8, zorder=5)
        inset_ax.add_patch(circle)
        inset_ax.text(0.4 + i * 0.6, 0.4, 'R', ha='center', va='center',
                      fontsize=5, fontweight='bold', color='white')

    # Vehicle
    inset_ax.plot(1.5, 0.5, 's', color='#2C3E50', markersize=8, zorder=6)
    inset_ax.text(1.5, 0.3, 'Vehicle', ha='center', va='center', fontsize=5)

    # Connecting lines (dashed)
    inset_ax.plot([1.5, 1.5], [0.5, 0.4], 'k--', linewidth=0.5, alpha=0.5)
    inset_ax.plot([1.5, 1.0], [0.5, 1.15], 'k--', linewidth=0.5, alpha=0.5)
    inset_ax.plot([1.5, 0.8], [0.5, 2.0], 'k--', linewidth=0.5, alpha=0.5)
    inset_ax.plot([1.5, 0.5], [0.5, 2.8], 'k--', linewidth=0.5, alpha=0.5)

    # ================================================================
    # Legend
    # ================================================================
    legend_elements = [
        mpatches.Patch(facecolor=C_ENV, label='T-NTN Environment'),
        mpatches.Patch(facecolor=C_PRE, label='Preprocessing'),
        mpatches.Patch(facecolor=C_VQC, label='VQC (High-Level Policy)'),
        mpatches.Patch(facecolor=C_QAOA, label='QAOA (Node Selection)'),
        mpatches.Patch(facecolor=C_BO, label='Bayesian Optimization'),
        mpatches.Patch(facecolor=C_ACT, label='Action Execution'),
        mpatches.Patch(facecolor=C_REW, label='Reward Feedback'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=7, frameon=True, framealpha=0.95,
               fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.text(0.5, 0.04,
             'Fig QC-5: End-to-end Quantum-HRL pipeline. The VQC receives the environment state via '
             'Amplitude Encoding and outputs tier selection $\\ell^*$ and offloading ratio $\\alpha$. '
             'QAOA resolves the optimal node $n^*$ from the QUBO Hamiltonian. '
             'Bayesian Optimization jointly tunes all quantum circuit parameters using a Gaussian Process surrogate.',
             ha='center', va='bottom', fontsize=8, style='italic', color='#444444')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        path_pdf = os.path.join(OUTPUT_DIR, 'qc5_system_architecture.pdf')
        path_png = os.path.join(OUTPUT_DIR, 'qc5_system_architecture.png')
        fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {path_pdf} and {path_png}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    generate_system_architecture_figure()
