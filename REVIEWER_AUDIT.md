# Reviewer #2 Audit — Quantum-HRL for Multi-Layer NTN-VEC Task Offloading

**Role:** Acting as an adversarial-but-constructive Q1-journal reviewer (Reviewer #2).
**Scope:** Full manuscript `quantum_hrl_paper.tex` after the Section 6 / Discussion / Conclusion revision.
**Verdict (recommendation):** *Major revision.* The contribution is clear and the writing is strong, but several claims rest on simulation-only evidence and on a baseline whose hyperparameters are assumed, and there is a text–artifact consistency risk around the mobility dataset that must be resolved before acceptance.

Severity legend: **High** = likely blocks acceptance / triggers major revision; **Medium** = must be addressed but fixable in revision; **Low** = polish / strengthens the paper.

---

## 1. Structural weaknesses

| # | Severity | Explanation | Suggested fix |
|---|----------|-------------|---------------|
| S1 | **High** | **Dataset–code consistency.** Section 6.1 now presents LuST as the mobility source, but the released simulation environment uses a synthetic random-mobility model. If a reviewer inspects the code (and Q1 venues increasingly do), the text and artifact disagree — a reproducibility and integrity flag. | Either (a) actually integrate LuST trajectories into `tntn_environment.py` and re-run, or (b) revert 6.1 to describe the synthetic model honestly and cite LuST only as motivation/future work. The two must match before submission. |
| S2 | Medium | **Results precede their evidence pointers.** Scenarios 2–3 (HRL+VQC, VQC+QAOA) are "evidenced" only by the ablation table, which uses 3 seeds while the main table uses 5. The scenario→evidence mapping is therefore uneven. | Run Scenarios 2 and 3 as first-class configurations with the same 5-seed protocol and report them in the main table, not only as ablations. |
| S3 | Low | **Discussion length vs. results length.** The Discussion is now richer than the Results, which can read as over-interpretation of a single benchmark. | Keep, but ensure each discussion claim cites a specific figure/table; move purely speculative passages to "Future work." |

---

## 2. Missing experiments

| # | Severity | Explanation | Suggested fix |
|---|----------|-------------|---------------|
| M1 | **High** | **No parameter-matched classical control.** The headline is "24 params beats 20,224 params," but the latency win is confounded by low-dimensional optimisation being better-conditioned. Without a small classical policy (e.g., a ~24–100-parameter linear/MLP policy-gradient agent) the quantum contribution is not isolated. | Add a compact classical policy-gradient baseline with a comparable parameter budget; report whether the quantum policy still wins. This is the single most persuasive missing experiment. |
| M2 | **High** | **No physical-hardware or realistic-noise run.** All quantum results use a noiseless statevector simulator; the noise study injects only i.i.d. Gaussian readout noise. Coherent gate error, crosstalk, and calibration drift are absent, so practical NISQ viability is unproven. | Run at least the inference path on a real back-end (IBM/IonQ) or a calibrated noise model (depolarising + readout + T1/T2) for the smallest tier; report degradation. |
| M3 | Medium | **Greedy is competitive and under-discussed.** Greedy achieves 0% violations and the *lowest* energy (0.277 J) at 0.164 s latency vs. Quantum-HRL's 0.117 s at 0.971 J. A reviewer will ask whether the ~28% latency gain justifies a 3.5× energy cost over a zero-parameter heuristic. | Add an explicit energy-constrained operating point (raise β₂) showing Quantum-HRL can dominate Greedy on a fair latency–energy front; discuss when each is preferable. |
| M4 | Medium | **Scalability claims rest on analytic counts only.** The scalability table reports analytic parameter counts; no measured latency/quality at ζ=2,4 is shown. | Report measured latency, violation rate, and wall-clock at ζ∈{2,4}, not just parameter formulas. |
| M5 | Low | **Sensitivity to penalty calibration.** Penalty weights are set to a 99th-percentile warm-up; robustness to this choice is unverified. | Add a small sensitivity sweep over the penalty percentile. |

---

## 3. Weak / overreaching claims

| # | Severity | Claim | Explanation | Suggested fix |
|---|----------|-------|-------------|---------------|
| W1 | **High** | "843× compression" framed near performance claims | Risk that readers read it as speed/efficiency advantage. The manuscript already disclaims this in Construct Validity, but the abstract/intro may not. | Ensure every prominent mention of 843× is co-located with "model footprint, not runtime." Audit abstract and intro. |
| W2 | Medium | Statistical significance via Welch-t / Mann–Whitney on **pooled per-task latencies** (n=2,400/method) | Pooling per-task samples across seeds treats correlated within-seed tasks as independent (pseudo-replication), inflating n and significance. | Report the test at the **per-seed mean** level (n=5 per method, paired across seeds), or use a hierarchical/mixed-effects model. Keep the pooled test only as secondary. |
| W3 | Medium | "QAOA reduces both the mean and the tail" of latency | Supported only by the 3-seed ablation; tail claims need distributional evidence. | Add latency CDF / box-plots for full vs. w/o-QAOA over ≥5 seeds. |
| W4 | Low | "converges in fewer episodes" (Fig. convergence) | Episodes-to-plateau Πη is defined but no numeric value is reported in text. | Report Πη numerically for each method. |
| W5 | Low | "operates within typical NISQ qubit budgets" | True for small tiers; the largest-tier QAOA register growth O(M) could exceed budgets at ζ=4. | State the actual qubit count at each ζ. |

---

## 4. Citation issues

| # | Severity | Explanation | Suggested fix |
|---|----------|-------------|---------------|
| C1 | Medium | **LuST/SUMO citations newly added.** `codeca2015lust` and `lopez2018sumo` are real, but were added during revision — verify DOIs/page numbers against IEEE Xplore before submission. | Confirm bibliographic details; ensure they compile and resolve. |
| C2 | Medium | **`zhu_vec_survey`** is an unrefereed "Preprint" with no venue/identifier. | Replace with a peer-reviewed VEC survey or add arXiv ID; weak citations invite scrutiny. |
| C3 | Low | Barren-plateau caution cites only `mcclean2018barren`. | Add a mitigation reference (e.g., identity-block / layerwise-training literature) to support the proposed fix. |
| C4 | Low | Baseline hidden width `h=256` is assumed because `hrl_ntn` "does not specify." | Confirm whether the baseline paper or its code specifies architecture; if so, cite the exact value to remove the assumption. |

---

## 5. Other likely reviewer criticisms

| # | Severity | Criticism | Suggested fix |
|---|----------|-----------|---------------|
| R1 | **High** | "The energy increase makes the method strictly worse for energy-constrained vehicular nodes — the central VEC use case." | Front-load the steerable β₂ result; show a configuration that is Pareto-superior, not just latency-superior. |
| R2 | Medium | "Fixed per-tier node counts are an extension of the baseline; does this advantage the quantum action space?" | Justify that the fixed counts are held identical across all methods (already stated) and show robustness under dynamic counts. |
| R3 | Medium | "Why Bayesian Optimisation for QAOA angles rather than the standard variational gradient? Is the comparison fair to QAOA?" | Add a short justification/ablation (BO vs. gradient angle update). |
| R4 | Low | "n=20 / 5 qubits is a toy scale; does the logarithmic-scaling argument hold empirically beyond toy sizes?" | Add one larger-state run (e.g., n=64, 6 qubits) to substantiate log-scaling. |

---

## 6. Sections most likely to trigger major-revision requests

1. **Section 6.1 (Dataset Construction)** — *High.* The LuST claim vs. released code (S1) is the top risk; resolve first.
2. **Section 7 Results / Statistical validation** — *High.* Pseudo-replication in the significance test (W2) and the 3-vs-5 seed inconsistency (S2/W3).
3. **Parameter-efficiency framing** — *Medium/High.* The 843× claim needs the parameter-matched control (M1) to be defensible as a *learning* advantage rather than a *dimensionality* artifact.
4. **NISQ viability** — *Medium/High.* Simulation-only + i.i.d. noise (M2) will draw "not hardware-validated" objections.
5. **Energy trade-off** — *Medium.* The Greedy comparison (M3/R1) must be neutralised with an energy-lean operating point.

---

### One-paragraph summary for the editor
The paper makes a clean, well-presented methodological case that a 24-parameter hybrid quantum policy can match or beat a ~20k-parameter classical HRL pipeline on a structured NTN-VEC offloading benchmark, and the authors are commendably transparent about the latency–energy trade-off and the footprint-vs-speed distinction. Acceptance hinges on three things: (i) reconciling the LuST mobility narrative with the actual simulation artifact; (ii) adding a parameter-matched classical control and a realistic-noise (ideally hardware) run so the advantage is attributable; and (iii) tightening the statistical analysis to the per-seed level. With these, the work would be a solid Q1 contribution; without them, the central claims remain under-supported.
