# Paper Improvements: Critical Corrections and Recommendations

**Author:** Dao Ngoc Hieu
**Date:** May 13, 2026
**Status:** Draft for revision

---

## Executive Summary

This document details the critical mathematical errors, inconsistencies, and improvements needed for the Quantum-HRL paper. The analysis is supported by `math_verification.py` (SymPy symbolic verification) and the simulation framework in this repository.

---

## Part I: Critical Errors Requiring Immediate Correction

### Error 1: Ising Hamiltonian Sign Error (Section 4.5, Eq. 28)

**Severity:** Critical — affects correctness of QAOA results

**Problem:** The Ising Hamiltonian in Eq. (28) has incorrect signs for the coupling terms.

**Current (incorrect):**
```
J_ij = +A/4   (positive coupling)
```

**Corrected version (verified analytically for M = 2, 3, 5 by exhaustive QUBO evaluation):**

Starting from the QUBO:
```
H(z) = sum_n c_n z_n + A * (sum_n z_n - 1)^2
     = sum_n (c_n - A) z_n + 2A * sum_{i<j} z_i z_j + A   [expanding, using z_n^2 = z_n]
```

Substituting `z_n = (I - sigma_z^n)/2`:

```
Linear coefficient (c_n - A) on z_n:
  (c_n - A) * (1 - sigma_z^n)/2
  => contributes -(c_n - A)/2 to h_n

Quadratic coefficient 2A on z_i z_j:
  2A * (1 - sigma_z^i)/2 * (1 - sigma_z^j)/2 = A/2 * (1 - sigma_z^i - sigma_z^j + sigma_z^i sigma_z^j)
  => J_ij = +A/2   (POSITIVE coupling; penalises co-selection)
  => each sigma_z^n appears in (M-1) pairs, contributing -(M-1)*A/2 to h_n

Combined local field:
  h_n = -(c_n - A)/2 - (M-1)*A/2
       = -c_n/2 + A/2 - (M-1)*A/2
       = -c_n/2 + A*(1 - M/2)

Constant offset:
  E0 = sum_n c_n/2 + A*(1 - M/2 + M*(M-1)/4)
```

**NOTE: The previous correction in this file (`J_ij = -A/2`) was itself WRONG.**
The correct value is **`J_ij = +A/2`** (positive). The positive coupling correctly
penalises states where sigma_z^i = sigma_z^j = -1 (both nodes selected), enforcing
one-hot. Verified by checking H_C(sigma_z) = H(z) + E0 for all 2^M bitstrings
for M = 2, 3, 5 (see qaoa_solver.py `qubo_to_ising` function).

**Impact:** The paper's Eq. (28) gives J_ij = +A/4, which is wrong in BOTH sign and
magnitude. The previous "correction" in this document (-A/2) was wrong in sign.
The correct value (+A/2) ensures the QAOA ground state corresponds to exactly
one selected node.

**Correction to paper text:**
Replace Eq. (28) with:
```
h_n = -c_n/2 + A*(1 - M/2)
J_ij = +A/2              (positive; penalises multi-selection)
E_0  = sum_n c_n/2 + A*(1 - M/2 + M*(M-1)/4)
```

---

### Error 2: Quantum Kernel Self-Contradiction (Section 3.3, Proposition 3.3)

**Severity:** Critical — mathematical statement is internally inconsistent

**Problem:** The paper first states the quantum kernel equals "the square of the linear kernel" but then writes `|s^T s'|^2 = |s^T s'|`, which is only true if `|s^T s'| = 0` or `1`, not generally.

**Current (incorrect):**
```
kappa(s, s') = |<s'|s>|^2 = |s^T s'|^2 = |s^T s'|
```

**Corrected version:**
```
kappa(s, s') = |<s'|s>|^2 = |sum_i s_i * s'_i|^2 = (s^T s')^2
```

The kernel is the **square** of the linear kernel (not the absolute value). This matters because `(a-b)^2 != |a-b|` in general.

---

### Error 3: Qubit Count Inconsistency (Table 5 vs Section 7.2)

**Severity:** Critical — physically impossible configuration

**Problem:** Table 5 shows `q = 4` qubits with `W_VQC = 20` params (implying `L = 4, q = 4`), but `n = 20` requires `q = ceil(log2(20)) = 5` qubits. With only 4 qubits, only `2^4 = 16` basis states exist, which cannot represent 20 amplitudes.

**Current (impossible):** `q = 4, W_VQC = 20`
**Corrected:** `q = 5, W_VQC = 20` (since `4 * 5 = 20`)

Section 7.2 correctly states `q = ceil(log2(20) = 5)`, which contradicts Table 5.

**Impact:** All performance claims depend on correct qubit count. The parameter reduction factor calculation in Eq. (52) uses `q = 5` implicitly.

---

### Error 4: Hidden Layer Count Inconsistency (Abstract vs Section 7.1)

**Severity:** Moderate — affects parameter reduction claims

**Problem:** The abstract states "even with a single hidden layer" implying `L~ = 2`, while Section 7.1 Eq. (48) uses `L~ = 4` hidden layers yielding 823,000 parameters. Both cannot be simultaneously true for the same baseline.

**Recommendation:** Pick one consistent baseline:
- Use `L~ = 2` for the "single hidden layer" claim (matches Table 5: 144K params)
- Use `L~ = 4` for the "four hidden layer" computation (823K params)
- State both as separate baselines

---

## Part II: Logical and Methodological Issues

### Issue 5: BO + PSR Relationship (Section 5.4)

**Current claim:** BO "reduces circuit evaluations by O(W_VQC)" compared to gradient descent.

**Problem:** This conflates two separate optimization levels:
- **PSR** computes analytic first-order gradients (Eq. 11) — exact, no finite differences
- **BO** replaces SGD as the outer-loop optimizer for noisy circuits

BO and PSR are complementary, not substitutes. PSR provides gradients; BO uses them (or other signals) to optimize circuit-level parameters.

**Correct framing:** "BO replaces gradient descent/SGD for noisy parameter optimization, reducing evaluations by O(1) per parameter compared to naive grid search, while PSR provides exact first-order information."

---

### Issue 6: VQC Output Mapping (Section 5.2)

**Problem:** The mapping from continuous expectation values `<O>` to discrete actions `(l*, alpha)` is underspecified.

**Recommended addition:**

```
tier_logits = [<O_1>, <O_2>, <O_3>, <O_4>]
tier_probs = softmax(tier_logits)
l* = argmax(tier_probs)

alpha_raw = <O_alpha>
alpha = sigmoid(clip(alpha_raw, -3, 3))
```

---

### Issue 7: QUBO Coefficient Normalization (Section 4.5)

**Problem:** The coefficients `c_n = beta1*T_k,n + beta2*E_k,n` span vastly different orders of magnitude (latency ~ms, energy ~J). Without normalization, the one-hot penalty `A` may not dominate properly.

**Recommended addition before Ising mapping:**
```
c_n_normalized = (c_n - min(c)) / (max(c) - min(c) + epsilon)
```

---

### Issue 8: Reference [4] Missing

**Severity:** Critical — incomplete bibliography

Reference [4] is entirely empty. This should be the Tibaldi et al. paper on Bayesian optimization for QAOA, which is foundational to the framework.

**Suggested citation:**
```
S. Tibaldi et al., "Bayesian Optimization for Quantum Approximate Optimization Algorithm,"
IEEE Transactions on Quantum Engineering, vol. 1, 2024.
```

---

## Part III: Missing Experimental Content

### Placeholder Values

All `[X]`, `[Y]`, `[Z]` placeholders in Table 3 and Table 4 must be filled with actual experimental results from `run_experiments.py`.

### Missing Figure Content

| Figure | Status | Source |
|--------|--------|--------|
| Fig 1 (System Architecture) | Described but not shown | `fig_system_arch.py` |
| Fig 2 (Convergence Curves) | Referenced but not shown | `run_experiments.py` output |
| Fig 3 (Depth Sweep) | Referenced but not shown | `run_experiments.py` output |

---

## Part IV: Suggested Additions

### A. MDP Markov Property Discussion

Add to Section 4.6:
```
"The state vector s_t contains sufficient information for the Markov property:
  - Current vehicle position and velocity capture immediate dynamics
  - Per-tier CPU loads reflect compute availability
  - Channel gains encode radio conditions
  - Task parameters (d_k, c_k, Tmax_k) capture workload characteristics
This 20-dimensional state is assumed to satisfy the Markov property for the
task offloading MDP."
```

### B. Convergence Proof Sketch

Consider adding a theoretical convergence guarantee sketch for the VQC+BO combination.

### C. Hardware Feasibility Discussion

Add a discussion of actual NISQ hardware constraints (IBM Eagle, etc.) with realistic gate error rates.

---

## Part V: Verified Correct Content

The following claims in the paper are mathematically verified correct:

- **Parameter-Shift Rule** (Proposition 3.4, Eq. 11): Correct
- **Amplitude Encoding** (Definition 3.3, Eq. 8): Correct (with q=5 for n=20)
- **QAOA circuit structure** (Definition 3.6, Eq. 14): Correct
- **Bellman TD update** (Section 5.6, Eq. 43): Correct
- **Parameter count formulas** (Section 7): Correct (given correct q=5)
- **Communication channel models** (Section 4.2): Correct structure (with noted simplifications)

---

## Files in This Repository

| File | Purpose |
|------|---------|
| `math_verification.py` | SymPy symbolic verification of all mathematical claims |
| `run_experiments.py` | Generates all experimental results |
| `visualize_results.py` | Publication-quality figures |
| `fig_vqc_circuit.py` | QC-1: VQC architecture diagram |
| `fig_qaoa_circuit.py` | QC-2: QAOA circuit diagram |
| `fig_psr_gradient.py` | QC-3: Parameter-Shift Rule visualization |
| `fig_ising_mapping.py` | QC-4: QUBO-to-Ising mapping diagram |
| `fig_system_arch.py` | QC-5: System architecture block diagram |

---

*Generated by mathematical peer review, May 2026*
