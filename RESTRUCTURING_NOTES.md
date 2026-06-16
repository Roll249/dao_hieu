# Q1 Restructuring Notes — Quantum-HRL for T-NTN Task Offloading

Companion to the restructured manuscript (`quantum_hrl_paper.tex`, synced to `overleaf/main.tex`).
For each task in `task.md`: the reviewer rationale, the weakness it fixes, and what changed.
No experiments, results, or citations were invented; all equations, tables, and numbers are carried
over verbatim from the prior draft. Two background references already present in `references.bib`
(`yanofsky2007intro`, `nielsen2010quantum`) now absorb the textbook quantum-mechanics derivations.

## Section map (old → new)

| Old | New |
|---|---|
| 1 Introduction | 1 Introduction (rewritten to the 7-beat flow) |
| 2 Notation and Abbreviations | **Appendix A** Notation and Symbols |
| 3 Related Work (Group 1 / Group 2 / Research Gap) | 2 Related Work (2.1 / 2.2 / 2.3, gap folded into closing paragraph) |
| 4 Quantum Computing Preliminaries (7 subsecs) | 3 Background (4 subsecs; qubit/gates/measurement removed, QUBO/Ising moved to 5.4) |
| 5 System Model + Framework + Experiments (mega-section) | split into **4 Problem Formulation**, **5 Proposed Framework**, **6 Experimental Design**, **7 Results and Discussion** |
| 6 Complexity Analysis | 7.5 Parameter Complexity Analysis |
| 7 Conclusion | 8 Conclusion (Summary / Findings / Limitations / Future Work) |
| — | 7.7 Threats to Validity (new) |

---

## Task 1 — Restructure
**Reviewer rationale.** Q1 reviewers expect the canonical IMRaD-plus arc; a single section that fused
system model, framework, and experiments made it impossible to navigate or to cite a specific
contribution. **Weakness fixed.** The old Section 5 was ~700 lines mixing formulation, method, setup,
and results. Notation as Section 2 front-loaded a reference table before any motivation.
**Change.** Adopted the eight-section structure (Introduction, Related Work, Background, Problem
Formulation, Proposed Framework, Experimental Design, Results and Discussion, Conclusion) with the
notation table demoted to Appendix A. All cross-references re-verified (every `\ref`/`\eqref` resolves;
every `\cite` is in the bib).

## Task 2 — Introduction
**Rationale.** A top-tier intro must walk context → problem → importance → limitations → motivation →
objective → contributions, and avoid unprovable hype. **Weakness fixed.** The old intro stated the
problem and contributions but had no explicit *importance* or *research-objective* beat, and the
contributions were a generic bullet list rather than claim-level statements. **Change.** Rewrote into
labelled beats; added an "Importance" paragraph (safety-critical deadlines, per-slot decision cost) and
an explicit "Research objective" paragraph; upgraded the contributions to four claim-level bullets tied
to sections. Retained the existing, correct disclaimer that the benefit is *representational /
parameter-efficiency*, not a *quantum advantage* — no "breakthrough/revolutionary" language present or
added.

## Task 3 — Related Work
**Rationale.** Reviewers want related work grouped by what each line solves and where it falls short,
not a chronological survey, and they dislike a standalone "Research Gap" section that reads as a
checklist. **Weakness fixed.** The old Group 2 conflated quantum RL and QAOA; a separate "Research Gap"
subsection restated the gap baldly. **Change.** Three groups: 2.1 Classical Task Offloading and HRL,
2.2 Quantum Reinforcement Learning, 2.3 Quantum Combinatorial Optimization with QAOA. Each now states
explicitly *what it solves* and *its limitation*. The research gap is delivered as a single motivating
closing paragraph (no header) that positions Quantum-HRL at the intersection.

## Task 4 — Background
**Rationale.** A networking/ML audience does not need axiomatic quantum mechanics; reviewers penalise
textbook padding. Target 2–4 pages. **Weakness fixed.** The old preliminaries spent two subsections on
qubits, Pauli matrices, gates, and the Born rule, plus a QUBO/Ising subsection that belonged with the
methodology. **Change.** Kept only Amplitude Encoding, VQC + Parameter-Shift Rule, QAOA, and Bayesian
Optimization. Removed the qubit/Hilbert-space and gates/measurement subsections, replacing them with a
one-paragraph pointer to standard references. Moved the QUBO/Ising mapping into Section 5.4 where it is
a contribution. Collapsed several propositions to their statements with citations.

## Task 5 — Methodology narrative
**Rationale.** Method clarity requires separating *what the problem is* (Sec. 4) from *how we solve it*
(Sec. 5), and giving the classical baseline its own explanation before the quantum proposal.
**Weakness fixed.** The baseline was only described implicitly (in related work and the complexity
section); the QUBO mapping was split between preliminaries and the model section.
**Change.** Section 4 Problem Formulation (T-NTN architecture, communication/cost models, objective +
constraints). Section 5 Proposed Framework with the requested subsections: **5.1 Classical HRL
Baseline** (new prose explicitly covering Layer / Node / Ratio selection and how the tri-DQN solves
the problem), **5.2 Proposed Quantum-HRL** (amplitude encoding → VQC tier/ratio → QAOA node, with the
hierarchical-ordering rationale and logical flow), **5.3 MDP Formulation** (state/action/reward, plus a
new sentence on *why* an MDP), **5.4 QUBO Mapping** (the four-step chain Optimization → Binary → QUBO →
Ising → QAOA, ending with an explicit "why QAOA is suitable" paragraph), **5.5 Training Strategy**
(classical vs quantum training separated; REINFORCE, Parameter-Shift Rule, Bayesian Optimization).

## Task 6 — Figures
**Rationale.** Reviewers expect an architecture overview, a baseline-vs-proposed contrast, the
problem-mapping pipeline, and the VQC internals. **Change.** See `FIGURE_SPECS.md` for layout, caption,
components, and drawing structure for all four. Figures 1, 3, 4 are wired into the paper using existing
rendered assets (`system_architecture.png`, `qc4_ising_mapping.pdf`, `qc1_vqc_architecture.pdf`); the
VQC architecture (Fig. 4) and the Ising-mapping pipeline (Fig. 3) are newly added to the manuscript.
**Figure 2 (Classical vs Quantum side-by-side) still needs to be drawn** — full spec provided.

## Task 7 — Experimental Design
**Rationale.** Reviewers want a dedicated design section and scenarios that *progressively* expose each
component's contribution. **Weakness fixed.** Setup, scenarios, and metrics were buried inside the mega
section; the prior scenarios (B/Sc/HL/N) stressed operating conditions but did not build the framework
up component by component. **Change.** Section 6 with 6.1 Dataset and Simulation Environment, 6.2
Hardware and Software Configuration, 6.3 Experimental Scenarios, 6.4 Evaluation Metrics. 6.3 now defines
the four progressive scenarios — **S1 Classical HRL, S2 HRL+VQC, S3 VQC+QAOA, S4 Full Quantum-HRL** —
each with motivation / purpose / expected outcome, and a mapping table (`tab:progressive`) pointing to
the measured experiment that provides evidence for each step. **No numbers were invented**: the
build-up is operationalised by the existing main comparison and ablation, and Table `tab:progressive`
states which measured result backs each scenario. The operating-condition stress tests (scalability,
heavy-load, NISQ-noise) are retained as a secondary paragraph.

## Task 8 — Results presentation
**Rationale.** Tables before prose; observation separated from interpretation; significance highlighted.
**Weakness fixed.** The old results interleaved numbers and mechanism in single paragraphs.
**Change.** Each results subsection (7.1 Main Comparison, 7.2 Ablation) now leads with the table, then a
bold **Observations** paragraph (facts only), then a bold **Interpretation** paragraph (mechanism).
Statistical validation (Welch `t`, Mann–Whitney `U`, both `p<10⁻³`) is called out under the main table.
Existing additional tables (reward routing, penalty effects, metric/scenario summaries, complexity) are
retained and repositioned.

## Task 9 — Discussion
**Rationale.** Discussion must add insight, not restate results, and stay critical rather than
promotional. **Weakness fixed.** The old discussion partly re-listed the headline numbers.
**Change.** 7.6 Discussion is organised to the requested agenda: why latency improves; why energy
increases; trade-offs; scalability implications; practical deployment considerations; scientific
implications; limitations. Marketing language avoided; the energy increase is framed as a direct,
expected consequence of the latency-minimising policy.

## Task 10 — Threats to Validity
**Rationale.** A reviewer-oriented threats analysis materially raises acceptance odds.
**Change.** New 7.7 Threats to Validity covering Internal (baseline tuning, shared pipeline, unspecified
`h`), External (synthetic workloads, fixed modest node counts, channel model, NISQ qubit budgets),
Construct (metrics operationalise P1/constraints; the 843× figure is footprint, *not* a speedup), and
Reproducibility (released code/seeds; analytic vs finite-shot caveat).

## Task 11 — Conclusion
**Rationale.** A structured conclusion with explicit limitations and grounded future work.
**Change.** Section 8 with Summary / Main Findings / Limitations / Future Work paragraphs. Limitations
are explicit (simulation-only, statevector back-end not real hardware, synthetic workloads, simplified
mobility, noiseless inner loop). Future work is concrete and technically grounded (hardware deployment,
hardware-efficient ansätze + data re-uploading, multi-task concurrent offloading, federated quantum
learning, formal separation results). Length ~1 page.

---

## Verification performed
- All `\begin`/`\end` environments balanced; braces balanced (1668/1668).
- Every `\ref`/`\eqref` target has a matching `\label`; every `\cite` key is in `references.bib`.
- All 8 referenced figure files exist in both `simulation/figures/` and `overleaf/figures/`.
- `overleaf/main.tex` is byte-identical to `quantum_hrl_paper.tex` except the `\graphicspath` line.
- **Not done locally:** PDF compile — no TeX engine is installed on this machine. Compile on Overleaf
  or with a local `texlive` to confirm the final layout (longtable pagination in the appendix, figure
  placement). This is the one remaining check before submission.
