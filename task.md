# Q1 PAPER RESTRUCTURING MASTER PROMPT

You are a senior researcher, Q1 journal reviewer, and editor with expertise in:

* Vehicular Edge Computing (VEC)
* Non-Terrestrial Networks (NTN)
* Reinforcement Learning
* Quantum Machine Learning
* IEEE Transactions papers

I will provide a complete draft paper.

Your objective is NOT to create new research.

Your objective is to transform this draft into a publication-ready Q1-style manuscript following reviewer expectations.

---

# IMPORTANT RULES

1. Preserve all technical contributions.

2. Do not invent experiments.

3. Do not invent results.

4. Do not invent citations.

5. Keep mathematical correctness.

6. Reduce unnecessary textbook explanations.

7. Focus on narrative quality, structure, clarity, and scientific presentation.

8. Assume the audience consists of networking, edge computing, and machine learning researchers rather than quantum physicists.

---

# TASK 1 — RESTRUCTURE THE PAPER

Reorganize the manuscript into the following structure:

1. Introduction

2. Related Work

3. Background

4. Problem Formulation

5. Proposed Framework

6. Experimental Design

7. Results and Discussion

8. Conclusion

Appendix A. Notation and Symbols

Explain all structural changes before rewriting.

---

# TASK 2 — REWRITE INTRODUCTION

Rewrite the Introduction using the following flow:

* Research context
* Problem statement
* Importance of the problem
* Limitations of existing methods
* Motivation for quantum-enhanced approaches
* Research objective
* Contributions

End with a clear bullet-point Contributions section.

Avoid exaggerated claims such as:

* breakthrough
* revolutionary
* quantum advantage

unless formally proven.

Use a tone suitable for IEEE TMC, TVT, TWC, FGCS, or Computer Networks.

---

# TASK 3 — REWRITE RELATED WORK

Organize Related Work into exactly three groups:

## 2.1 Classical Task Offloading and Hierarchical Reinforcement Learning

## 2.2 Quantum Reinforcement Learning

## 2.3 Quantum Combinatorial Optimization with QAOA

Requirements:

* Remove unnecessary historical descriptions.
* Focus on relevance.
* Explain what each group solves.
* Explain limitations of each group.

Do NOT create a standalone Research Gap section.

End with a concise concluding paragraph naturally motivating the proposed framework.

---

# TASK 4 — COMPRESS BACKGROUND

Rewrite the quantum foundations section as a concise Background section.

Keep only:

## Amplitude Encoding

## Variational Quantum Circuits

## QAOA

## Bayesian Optimization

Remove:

* textbook quantum mechanics
* lengthy derivations
* unnecessary proofs
* excessive mathematical preliminaries

Replace standard derivations with citations whenever possible.

Target length:

2–4 pages maximum.

---

# TASK 5 — REBUILD THE METHODOLOGY NARRATIVE

Rewrite the methodology using the following structure:

## 4. Problem Formulation

Present:

* T-NTN architecture
* Optimization objective
* Constraints
* Task offloading problem

---

## 5. Proposed Framework

### 5.1 Classical HRL Baseline

Explain:

* Layer Selection
* Ratio Selection
* Node Selection

Clearly explain how the classical baseline solves the problem.

---

### 5.2 Proposed Quantum-HRL Framework

Explain:

* Amplitude Encoding
* VQC
* Layer Selection
* Ratio Selection
* QAOA
* Node Selection

Show the logical flow.

---

### 5.3 MDP Formulation

Present:

* State
* Action
* Reward

Explain why the problem is modeled as an MDP.

---

### 5.4 QUBO Mapping

This is a key contribution.

Provide a detailed explanation of:

Optimization Problem
→ Binary Formulation
→ QUBO
→ Ising Hamiltonian
→ QAOA

Emphasize why QAOA is suitable.

---

### 5.5 Training Strategy

Separate:

Classical Training

and

Quantum Training

Clearly explain:

* REINFORCE
* Parameter Shift Rule
* Bayesian Optimization

Focus on logical consistency.

---

# TASK 6 — FIGURE DESIGN

Create publication-quality figure specifications.

Figure 1:
Overall Quantum-HRL Architecture

Figure 2:
Classical HRL vs Quantum-HRL

Figure 3:
Optimization Problem → MDP → QUBO → QAOA Mapping

Figure 4:
VQC Architecture

For each figure provide:

* Layout description
* Caption
* Components
* Suggested drawing structure

The figures should be suitable for draw.io, TikZ, or Illustrator.

---

# TASK 7 — EXPERIMENTAL DESIGN

Create a dedicated Experimental Design section.

Structure:

## 6.1 Dataset and Simulation Environment

## 6.2 Hardware and Software Configuration

## 6.3 Experimental Scenarios

## 6.4 Evaluation Metrics

Experimental scenarios should progressively demonstrate the contribution of each component:

Scenario 1:
Classical HRL

Scenario 2:
HRL + VQC

Scenario 3:
VQC + QAOA

Scenario 4:
Full Quantum-HRL

For each scenario:

* explain motivation
* explain purpose
* explain expected outcome

Do not invent numerical results.

---

# TASK 8 — RESULTS PRESENTATION

Improve the Results section.

Requirements:

* Present results in a reviewer-friendly way.
* Separate observations from interpretation.
* Use tables before discussion.
* Highlight statistical significance where available.
* Emphasize practical implications.hay list ra nhung task m da lam

Suggest additional result tables if appropriate.

---

# TASK 9 — REWRITE DISCUSSION

Do not repeat results.

Discuss:

1. Why latency improves.
2. Why energy consumption increases.
3. Trade-offs.
4. Scalability implications.
5. Practical deployment considerations.
6. Scientific implications.
7. Limitations.

Use a critical scientific tone.

Avoid marketing language.

---

# TASK 10 — THREATS TO VALIDITY

Create a new subsection:

Threats to Validity

Discuss:

* Internal Validity
* External Validity
* Construct Validity
* Reproducibility

Focus on:

* simulation assumptions
* synthetic workloads
* network scale
* NISQ assumptions

Use a reviewer-oriented tone.

---

# TASK 11 — REWRITE CONCLUSION

Structure:

## Summary

## Main Findings

## Limitations

## Future Work

Limitations should be explicit.

Examples:

* simulation-only evaluation
* no real quantum hardware
* simplified mobility assumptions

Future work should be realistic and technically grounded.

Target length:
1–1.5 pages.

---

# OUTPUT FORMAT

For each task:

1. Explain reviewer rationale.
2. Show the proposed revision.
3. Highlight weaknesses in the current draft.
4. Suggest improvements.
5. Produce publication-ready text where applicable.

Act as a strict Q1 reviewer and editor, not as a general AI assistant.
