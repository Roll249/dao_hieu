# Quantum-HRL Paper Revision Tasks

You are acting as a senior Q1 journal reviewer, editor, and co-author.

Domain:
- Vehicular Edge Computing (VEC)
- Non-Terrestrial Networks (NTN)
- Hierarchical Reinforcement Learning (HRL)
- Quantum Machine Learning
- Quantum Optimization (QAOA)

IMPORTANT:

- Do not invent experiments.
- Do not invent citations.
- Do not invent numerical results.
- Preserve technical correctness.
- Use IEEE Transactions writing style.
- Focus on scientific presentation, methodology clarity, reviewer expectations, and publication readiness.

---

# TASK 1 — Rewrite Dataset Construction Section

Create a publication-ready subsection:

## 6.1 Dataset Construction

Context:

Mobility Dataset:
- LuST (Luxembourg SUMO Traffic Dataset)

Task Workload:
- Adopted from the classical HRL baseline paper

Network Parameters:
- Adopted from the classical HRL baseline paper

Requirements:

1. Explain how the three components are combined.
2. Justify why LuST is suitable for realistic vehicular mobility.
3. Explain why baseline task and network settings are preserved.
4. Emphasize fairness and reproducibility.
5. Write in Q1 journal style.

Output:
Publication-ready text.

---

# TASK 2 — Rewrite Hardware & Software Configuration

Create subsection:

## 6.2 Hardware and Software Configuration

Include placeholders where information is missing.

Cover:

- CPU
- RAM
- Operating System
- Python Version
- PennyLane
- Qiskit
- NumPy
- Simulator

Output:
Publication-ready text.

---

# TASK 3 — Rewrite Experimental Scenarios

Create subsection:

## 6.3 Experimental Scenarios

Design the narrative for:

Scenario 1:
Classical HRL Baseline

Scenario 2:
HRL + VQC

Scenario 3:
VQC + QAOA

Scenario 4:
Full Quantum-HRL

For each scenario explain:

- Motivation
- Objective
- What component is being evaluated
- Expected observation

Do not invent results.

Output:
Publication-ready text.

---

# TASK 4 — Rewrite Evaluation Metrics

Create subsection:

## 6.4 Evaluation Metrics

Metrics:

- Latency
- Energy Consumption
- Reward
- Deadline Violation Rate
- Parameter Count

For each metric:

- Define it
- Explain why it matters
- Explain what aspect of the framework it evaluates

Output:
Publication-ready text.

---

# TASK 5 — Improve Classical-to-Quantum Transition

Review the current methodology section.

Rewrite the transition between:

Classical HRL
and
Quantum-HRL

Requirements:

Explain:

1. Limitations of Classical HRL
2. Why VQC replaces DQN modules
3. Why node selection becomes a combinatorial optimization problem
4. Why QAOA is introduced
5. Why the resulting architecture is more scalable

Focus on logical flow.

Output:
Publication-ready text.

---

# TASK 6 — Rewrite QUBO Motivation

Review the QUBO section.

Add a motivation subsection before the mathematical formulation.

Answer:

- Why direct optimization is difficult
- Why binary representation is needed
- Why QUBO is suitable
- Why QAOA requires a QUBO formulation

Use reviewer-oriented reasoning.

Output:
Publication-ready text.

---

# TASK 7 — Design Results Tables

Without inventing results, design all result tables required by a Q1 paper.

Include:

### Main Performance Table

Columns:
- Method
- Latency
- Energy
- Reward
- Deadline Violation

---

### Parameter Efficiency Table

Columns:
- Method
- Number of Parameters
- Compression Ratio

---

### Scalability Table

Columns:
- Number of Nodes
- Classical HRL
- Quantum-HRL

---

### Ablation Study Table

Columns:
- Configuration
- Latency
- Energy
- Reward

Provide:

- Table structure
- Captions
- Placement recommendations

Do not generate fake values.

---

# TASK 8 — Rewrite Discussion

Create a strong Q1-level Discussion section.

Do not repeat results.

Discuss:

1. Why latency improves
2. Why energy consumption increases
3. Trade-offs
4. Scalability implications
5. Practical deployment considerations
6. Scientific significance
7. Limitations

Use critical analysis.

Avoid marketing language.

Output:
Publication-ready text.

---

# TASK 9 — Threats to Validity

Create subsection:

Threats to Validity

Discuss:

- Internal Validity
- External Validity
- Construct Validity
- Reproducibility

Focus on:

- Simulation assumptions
- LuST mobility dataset limitations
- Synthetic task workloads
- NISQ assumptions
- Limited network scale

Output:
Publication-ready text.

---

# TASK 10 — Rewrite Conclusion

Rewrite the Conclusion section.

Structure:

## Summary

## Main Findings

## Limitations

## Future Work

Requirements:

- Explicitly state limitations.
- Do not oversell quantum advantages.
- Keep the tone realistic.
- Match IEEE Q1 journal style.

Target:
1–1.5 pages.

Output:
Publication-ready text.

---

# TASK 11 — Reviewer Audit

Act as Reviewer #2.

Review the entire manuscript.

Identify:

1. Structural weaknesses
2. Missing experiments
3. Weak claims
4. Citation issues
5. Potential reviewer criticisms
6. Sections likely to trigger major revision requests

Provide:

- Severity (High / Medium / Low)
- Explanation
- Suggested fix

Output:
A reviewer-style audit report.