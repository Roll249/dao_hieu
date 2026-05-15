"""Fig QC-1: VQC Architecture Diagram.

Shows the complete L-layer VQC from the paper's Eq. (10):
  A. Amplitude Encoding stage
  B. L VQC layers (RY + CNOT)
  C. Measurement (layer selection + ratio)
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


def draw_quantum_gate(ax, x, y, label, color='#4472C4', width=0.7, height=0.5):
    """Draw a labeled quantum gate box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#2F2F2F', linewidth=1.5,
                          alpha=0.9, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)


def draw_wire(ax, y, x_start, x_end, label=None, color='#555555'):
    """Draw a quantum wire (horizontal line)."""
    ax.plot([x_start, x_end], [y, y], color=color, linewidth=1.8, zorder=2)
    if label:
        ax.text(x_start - 0.1, y, label, ha='right', va='center',
                fontsize=7, color='#333333')


def draw_cnot(ax, x, y_top, y_bot):
    """Draw a CNOT gate (dot on control, cross on target)."""
    # Vertical wire
    ax.plot([x, x], [y_top, y_bot], color='#555555', linewidth=1.8, zorder=2)
    # Control dot
    ax.plot(x, y_top, 'o', color='#333333', markersize=7, zorder=5)
    # Target cross
    ax.plot([x - 0.15, x + 0.15], [y_bot, y_bot], color='#333333', linewidth=2, zorder=5)
    ax.plot([x, x], [y_bot - 0.15, y_bot + 0.15], color='#333333', linewidth=2, zorder=5)


def draw_measurement(ax, x, wires, label, color='#E74C3C'):
    """Draw a measurement symbol."""
    for y in wires:
        # Measurement arc (simplified as box)
        box = FancyBboxPatch((x - 0.25, y - 0.25), 0.5, 0.5,
                              boxstyle="round,pad=0.03",
                              facecolor=color, edgecolor='#2F2F2F',
                              linewidth=1.5, alpha=0.85, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, 'M', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=4)


def draw_ry_gate(ax, x, y, theta_label='\\theta_{l,j}', color='#2ECC71'):
    """Draw an RY rotation gate."""
    box = FancyBboxPatch((x - 0.3, y - 0.25), 0.6, 0.5,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#2F2F2F',
                          linewidth=1.5, alpha=0.9, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, f'$R_Y({theta_label})$', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='white', zorder=4)


def generate_vqc_figure(output_path=None):
    """Generate the complete VQC architecture figure."""
    setup_plot_style()

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 0.6],
                           hspace=0.35, left=0.05, right=0.97,
                           top=0.93, bottom=0.08)

    # Color scheme
    C_ENCODING = '#8E44AD'   # Purple for amplitude encoding
    C_LAYER = '#2980B9'       # Blue for VQC layers
    C_CNOT = '#E67E22'        # Orange for CNOT
    C_MEAS = '#C0392B'        # Red for measurement
    C_WIRE = '#7F8C8D'        # Gray for wires
    C_H = '#27AE60'           # Green for Hadamard

    # =====================================================================
    # Panel A: Amplitude Encoding
    # =====================================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 12)
    ax_a.set_ylim(-2, 2)
    ax_a.set_title('(a) Amplitude Encoding: $|\\tilde{{s}}_t\\rangle = \\sum_{{i=0}}^{{n-1}} \\tilde{{s}}_i |i\\rangle$',
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.axis('off')

    # State vector input
    ax_a.text(0.5, 1.2, '$\\tilde{\\mathbf{s}}_t \\in \\mathbf{R}^n$', fontsize=9,
              ha='left', va='center', fontweight='bold')
    ax_a.text(0.5, 0.6, '$\\Downarrow$', fontsize=11, ha='center', va='center')
    ax_a.text(0.5, 0.1, 'L2 Normalize', fontsize=8, ha='left', va='center',
              style='italic', color='#555555')

    # Arrow
    ax_a.annotate('', xy=(2.2, 0), xytext=(1.3, 0),
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Amplitude encoding box
    box_a = FancyBboxPatch((2.3, -0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=C_ENCODING, edgecolor='#1a1a2e',
                            linewidth=2, alpha=0.95, zorder=3)
    ax_a.add_patch(box_a)
    ax_a.text(3.0, 0, 'AmpEnc', ha='center', va='center',
              fontsize=9, fontweight='bold', color='white', zorder=4)
    ax_a.text(3.0, -0.6, '$q = \\lceil\\log_2 n\\rceil$ qubits', ha='center',
              va='center', fontsize=7.5, style='italic', color='#555555')

    # Arrow to state
    ax_a.annotate('', xy=(5.5, 0), xytext=(3.7, 0),
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Quantum register visualization
    q_labels = ['$|0\\rangle$', '$|0\\rangle$', '$|0\\rangle$', '$\\vdots$', '$|0\\rangle$']
    y_positions = [0.6, 0.2, -0.2, -0.55, -0.9]
    for i, (lbl, yp) in enumerate(zip(q_labels, y_positions)):
        ax_a.text(5.6, yp, lbl, fontsize=8, va='center', ha='left')
        ax_a.plot([5.9, 7.8], [yp, yp], color=C_WIRE, linewidth=1.5, zorder=2)

    # Entanglement indicators
    for i in range(4):
        ax_a.text(6.0 + i * 0.45, -1.25, f'$s_{i}$', fontsize=7,
                  ha='center', va='center', color=C_ENCODING, fontweight='bold')

    ax_a.text(7.8, 1.0, 'Encoded quantum state $|\\psi(\\tilde{s}_t)\\rangle$',
              fontsize=8, ha='right', va='center', style='italic')

    # Dimension note
    ax_a.text(10.5, -1.5,
              '$n=20 \\Rightarrow q=5$\n$2^5 = 32 \\geq 20$',
              fontsize=8, ha='center', va='center',
              bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                        edgecolor='#dee2e6', linewidth=1))

    ax_a.text(6.85, -1.7, '$|\\tilde{s}_t\\rangle = \\sum_{{i=0}}^{{n-1}} \\tilde{s}_i |i\\rangle$',
              ha='center', va='center', fontsize=9)

    # =====================================================================
    # Panel B: VQC Layers
    # =====================================================================
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0, 12)
    ax_b.set_ylim(-1.1, 1.1)
    ax_b.set_title('(b) VQC Ansatz: $U(\\theta) = \\prod_{{l=1}}^{L} \\left[ \\prod_{{j=1}}^{q} R_Y(\\theta_{{l,j}}) \\cdot \\prod_{{j=1}}^{{q-1}} \\mathrm{CNOT}_{{j,j+1}} \\right]$',
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.axis('off')

    q = 5
    y_positions = [0.8, 0.4, 0.0, -0.4, -0.8]
    y_labels = [f'q{j}' for j in range(q)]
    layer_x_starts = [1.5, 4.2, 6.9, 9.6]

    # Draw wires
    for i, yp in enumerate(y_positions):
        ax_b.plot([0.2, 11.5], [yp, yp], color=C_WIRE, linewidth=1.5, zorder=1)
        ax_b.text(0.1, yp, y_labels[i], fontsize=7, ha='right', va='center')

    # Input state
    ax_b.text(0.3, 1.0, '$|\\psi(\\tilde{s}_t)\\rangle$', fontsize=8,
               ha='left', va='center', style='italic')

    x_pos = 1.5
    layer_labels = ['Layer $\\ell=1$', 'Layer $\\ell=2$', '$\\dots$', 'Layer $\\ell=L$']

    for layer_idx in range(3):
        # Layer box
        layer_box = FancyBboxPatch((x_pos - 0.4, -1.0), 2.4, 2.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#f0f0f0', edgecolor='#bdc3c7',
                                    linewidth=1, alpha=0.5, zorder=0)
        ax_b.add_patch(layer_box)
        ax_b.text(x_pos + 0.8, 0.85, layer_labels[layer_idx], ha='center', va='top',
                  fontsize=7.5, fontweight='bold', color='#2c3e50')

        # RY gates for each qubit
        for j, yp in enumerate(y_positions):
            theta_str = f'$\\theta_{{{layer_idx+1},{j+1}}}$'
            draw_ry_gate(ax_b, x_pos + 0.4, yp, theta_str, color=C_LAYER)

        # CNOT gates between adjacent qubits
        for j in range(q - 1):
            cnot_x = x_pos + 1.6
            # Control dot
            ax_b.plot(cnot_x, y_positions[j], 'o', color='#333333',
                      markersize=6, zorder=5)
            # Vertical wire
            ax_b.plot([cnot_x, cnot_x], [y_positions[j], y_positions[j+1]],
                      color=C_WIRE, linewidth=1.5, zorder=2)
            # Target cross
            cross_x = cnot_x
            ax_b.plot([cross_x - 0.12, cross_x + 0.12], [y_positions[j+1], y_positions[j+1]],
                      color='#333333', linewidth=2, zorder=5)
            ax_b.plot([cross_x, cross_x], [y_positions[j+1] - 0.12, y_positions[j+1] + 0.12],
                      color='#333333', linewidth=2, zorder=5)

        x_pos += 2.7

    # Final layer ellipsis
    ax_b.text(8.5, 0, '$\\dots$', ha='center', va='center', fontsize=14, color='#999999')
    ax_b.annotate('', xy=(9.2, 0), xytext=(8.8, 0),
                  arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

    # Layer L
    L_box = FancyBboxPatch((9.3, -1.0), 2.0, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor='#f0f0f0', edgecolor='#bdc3c7',
                             linewidth=1, alpha=0.5, zorder=0)
    ax_b.add_patch(L_box)
    ax_b.text(10.3, 0.85, layer_labels[3], ha='center', va='top',
              fontsize=7.5, fontweight='bold', color='#2c3e50')

    for j, yp in enumerate(y_positions):
        theta_str = f'$\\theta_{{L,{j+1}}}$'
        draw_ry_gate(ax_b, 9.7, yp, theta_str, color=C_LAYER)

    for j in range(q - 1):
        cnot_x = 10.0
        ax_b.plot(cnot_x, y_positions[j], 'o', color='#333333', markersize=6, zorder=5)
        ax_b.plot([cnot_x, cnot_x], [y_positions[j], y_positions[j+1]],
                  color=C_WIRE, linewidth=1.5, zorder=2)
        ax_b.plot([cnot_x - 0.12, cnot_x + 0.12], [y_positions[j+1], y_positions[j+1]],
                  color='#333333', linewidth=2, zorder=5)
        ax_b.plot([cnot_x, cnot_x], [y_positions[j+1] - 0.12, y_positions[j+1] + 0.12],
                  color='#333333', linewidth=2, zorder=5)

    # =====================================================================
    # Panel C: Measurement
    # =====================================================================
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_xlim(0, 12)
    ax_c.set_ylim(-1, 1)
    ax_c.set_title('(c) Measurement: Layer Selection $\\ell^* = \\arg\\max_l \\langle \\hat{O}_l\\rangle$, '
                    '$\\alpha = \\sigma(\\langle \\hat{O}_\\alpha\\rangle)$',
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.axis('off')

    # Output wire
    for yp in y_positions:
        ax_c.plot([1.5, 11.0], [yp, yp], color=C_WIRE, linewidth=1.5, zorder=1)

    # Measurement boxes
    meas_labels = ['$\\langle\\hat{O}_1\\rangle$\n(RSU)', '$\\langle\\hat{O}_2\\rangle$\n(LAP)',
                   '$\\langle\\hat{O}_3\\rangle$\n(HAP)', '$\\langle\\hat{O}_4\\rangle$\n(LEO)',
                   '$\\langle\\hat{O}_\\alpha\\rangle$\n(ratio)']

    meas_colors = ['#27AE60', '#2980B9', '#8E44AD', '#E67E22', '#E74C3C']
    x_meas = [3.5, 5.2, 6.9, 8.6, 10.3]

    for i, (meas_x, meas_lbl, col) in enumerate(zip(x_meas, meas_labels, meas_colors)):
        box_m = FancyBboxPatch((meas_x - 0.6, -0.4), 1.2, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=col, edgecolor='#2F2F2F',
                               linewidth=1.5, alpha=0.9, zorder=3)
        ax_c.add_patch(box_m)
        ax_c.text(meas_x, 0, meas_lbl, ha='center', va='center',
                  fontsize=7, fontweight='bold', color='white', zorder=4,
                  multialignment='center')

        # Wire to measurement
        ax_c.plot([1.5, meas_x - 0.6], [y_positions[min(i, 4)], 0],
                  color=C_WIRE, linewidth=1.2, zorder=2, linestyle='--', alpha=0.6)

    # Output
    ax_c.annotate('', xy=(11.2, 0.0), xytext=(11.0, 0.0),
                  arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax_c.text(11.4, 0.6, '$l^* \\in \\{1,2,3,4\\}$', fontsize=8,
              ha='left', va='center', fontweight='bold', color='#27AE60')
    ax_c.text(11.4, -0.1, '$\\alpha \\in [0,1]$', fontsize=8,
              ha='left', va='center', fontweight='bold', color='#E74C3C')
    ax_c.text(11.4, -0.6, '(sigmoid)', fontsize=7,
              ha='left', va='center', style='italic', color='#666666')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_ENCODING, label='Amplitude Encoding'),
        mpatches.Patch(facecolor=C_LAYER, label='RY Rotation Gate'),
        mpatches.Patch(facecolor=C_WIRE, label='CNOT Gate'),
        mpatches.Patch(facecolor=C_MEAS, label='Measurement'),
    ]

    fig.legend(handles=legend_elements, loc='upper center',
               ncol=4, frameon=True, framealpha=0.9,
               fontsize=8, bbox_to_anchor=(0.5, 0.97))

    fig.text(0.5, 0.01,
             'Fig QC-1: Quantum-HRL VQC architecture. (a) Amplitude Encoding maps the $n$-dimensional state '
             'vector onto $q = \\lceil\\log_2 n\\rceil$ qubits. '
             '(b) Each VQC layer applies parameterized $R_Y$ rotations followed by a CNOT entangling ladder. '
             '(c) Measurement of Pauli observables yields layer selection $\\ell^*$ and offloading ratio $\\alpha$.',
             ha='center', va='bottom', fontsize=8, style='italic', color='#444444',
             wrap=True)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        path_pdf = os.path.join(OUTPUT_DIR, 'qc1_vqc_architecture.pdf')
        path_png = os.path.join(OUTPUT_DIR, 'qc1_vqc_architecture.png')
        fig.savefig(path_pdf, dpi=300, bbox_inches='tight')
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {path_pdf} and {path_png}")

    plt.close(fig)
    return fig


def setup_plot_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9,
        'axes.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


if __name__ == '__main__':
    generate_vqc_figure()
