# Quantum-HRL: Simulation Framework

Quantum-Enhanced Hierarchical Reinforcement Learning for Task Offloading in Multi-Layer Non-Terrestrial Vehicular Edge Computing.

## Overview

This repository contains a complete simulation framework for the Quantum-HRL framework described in the associated paper. It implements:

- **T-NTN Environment**: Multi-tier network (RSU, LAP, HAP, LEO) with realistic channel models
- **VQC Agent**: Variational Quantum Circuit with Amplitude Encoding and Parameter-Shift Rule training
- **QAOA Solver**: Quantum Approximate Optimization Algorithm for node selection
- **Bayesian Optimization**: Noise-robust parameter tuning for NISQ deployment
- **Classical Baselines**: Random, Greedy, Single DQN, and Classical HRL for comparison

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
cd simulation
python run_experiments.py
```

This runs all experiments and generates publication-quality figures.

## Project Structure

```
simulation/
  utils.py              - Shared utilities (normalization, rewards, BO wrapper)
  tntn_environment.py  - T-NTN network environment
  vqc_circuit.py       - VQC with amplitude encoding + PSR
  qaoa_solver.py       - QAOA for node selection
  quantum_hrl.py        - Full Quantum-HRL agent
  run_experiments.py   - Main experiment runner
  visualize_results.py  - Publication-quality figure generation
  fig_vqc_circuit.py   - QC-1: VQC architecture diagram
  fig_qaoa_circuit.py  - QC-2: QAOA circuit diagram
  fig_psr_gradient.py  - QC-3: Parameter-Shift Rule visualization
  fig_ising_mapping.py - QC-4: QUBO-to-Ising mapping diagram
  fig_system_arch.py   - QC-5: System architecture block diagram
  figures/             - Generated output figures
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n (state dim) | 20 | MDP state dimension |
| q (qubits) | 5 | Ceil(log2(20)) = 5 qubits |
| L (VQC layers) | 4 | VQC ansatz depth |
| p (QAOA depth) | 2 | QAOA circuit depth |
| BO budget B | 20 | BO evaluations per update |
| Tiers | 4 | RSU, LAP, HAP, LEO |
| Nodes | [5, 3, 2, 2] | Per-tier node counts |
| Area | 2 km² | Simulation area |
| Vehicle speed | N(60, 15²) km/h | Truncated Gaussian |

## Figures Generated

- `fig2_convergence.pdf/png` - Training convergence curves
- `fig3_depth_sweep.pdf/png` - VQC/QAOA depth sensitivity
- `table3_performance.pdf` - Method comparison table
- `table4_ablation.pdf` - Ablation study table
- `qc1_vqc_architecture.pdf` - VQC circuit diagram
- `qc2_qaoa_circuit.pdf` - QAOA circuit diagram
- `qc3_psr_gradient.pdf` - PSR visualization
- `qc4_ising_mapping.pdf` - QUBO-Ising diagram
- `qc5_system_architecture.pdf` - Full pipeline diagram
