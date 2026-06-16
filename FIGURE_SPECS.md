# Figure Design Specifications (Task 6)

Publication-quality specifications for the four required figures. Each is suitable for
draw.io, TikZ, or Illustrator. Where a rendered asset already exists in the repository it is
noted; **Figure 2 is the only one that still needs to be drawn from scratch.**

| Fig. | Purpose | Existing asset | Wired into paper |
|---|---|---|---|
| 1 | Overall Quantum-HRL architecture | `figures/system_architecture.png` (+ `qc5_system_architecture.pdf`) | Yes — Fig. `architecture`, Sec. 5.2 |
| 2 | Classical HRL vs Quantum-HRL | **none — draw this** | Not yet (recommended add to Sec. 5.1/5.2) |
| 3 | Optimization → MDP → QUBO → QAOA mapping | `figures/qc4_ising_mapping.pdf` | Yes — Fig. `ising`, Sec. 5.4 |
| 4 | VQC architecture | `figures/qc1_vqc_architecture.pdf` | Yes — Fig. `vqc`, Sec. 5.2 |

Supplementary assets also available and partly used: `qc2_qaoa_circuit.pdf` (QAOA circuit),
`qc3_psr_gradient.pdf` (parameter-shift gradient), `fig_scaling.pdf` (parameter scaling, now in Sec. 7.4),
`quantum_hrl_decision_flow.png` (decision/learning loop, Sec. 5.2).

---

## Figure 1 — Overall Quantum-HRL Architecture

**Layout description.** A single left-to-right pipeline spanning the page width, organised in five
vertical bands: (1) Environment, (2) State encoding, (3) High-level VQC policy, (4) Low-level QAOA
optimiser, (5) Action/feedback. A feedback arrow runs along the bottom from band 5 back to bands 3–4.

**Components.**
- Band 1 *Environment*: T-NTN icon (vehicle + RSU/LAP/HAP/LEO), emitting the raw state `s_t ∈ ℝ²⁰`
  (mobility, per-node load `u_{l,n}`, channel gains `h_{k,e}`, task descriptor).
- Band 2 *Encoding*: `ℓ₂`-normalise `s_t → s̃_t`; amplitude-encode onto a `q = ⌈log₂ n⌉ = 5`-qubit
  register `|ψ(s_t)⟩` (Eq. amplitude_encoding).
- Band 3 *VQC high-level policy* `U(θ)`: outputs tier `l*` (softmax over `⟨Ô_l⟩`) and ratio
  `α = σ(⟨Ô_α⟩)`. Tag "20 parameters".
- Band 4 *QAOA low-level optimiser*: builds cost Hamiltonian `H_C(l*, α)`, runs depth-`p` circuit,
  measures `→ n*`. Tag "4 parameters".
- Band 5 *Action & feedback*: compose `a_t = (l*, n*, α)`; environment returns latency `T`, energy `E`,
  flags `F¹F²F³`, rewards `R₁` (→VQC, PSR policy gradient) and `R₂` (→QAOA, Bayesian Optimisation).

**Suggested drawing structure.** Rounded rectangles for modules, a distinct fill for quantum blocks
(bands 3–4) versus classical blocks (bands 1–2, 5). Solid black arrows for the forward data path;
two coloured dashed arrows for the two reward signals (`R₁` to band 3, `R₂` to band 4). Put parameter
counts as small badges on the quantum blocks to foreshadow the 24-vs-20,224 message.

**Caption.** "End-to-end Quantum-HRL architecture. The network state is amplitude-encoded into a
5-qubit register; the VQC produces the high-level tier and ratio decisions `(l*, α)`; QAOA solves the
node-selection QUBO conditioned on `(l*, α)` to return `n*`; the executed action `a_t=(l*,n*,α)`
yields reward feedback that trains the VQC (`R₁`, parameter-shift policy gradient) and tunes the QAOA
angles (`R₂`, Bayesian Optimisation)."

---

## Figure 2 — Classical HRL vs Quantum-HRL  *(to be drawn)*

**Layout description.** Two stacked horizontal lanes sharing one input (`s_t`) on the left and one
output (`a_t = (l*, n*, α)`) on the right, so the eye reads the substitution module-for-module.
Top lane = classical baseline; bottom lane = proposed framework. A thin central column aligns the
three decisions (tier / node / ratio) across both lanes.

**Components.**
- *Top lane (Classical HRL, [hrl_ntn])*: three DQN boxes in series — `DQN¹ tier (6,144)` →
  `DQN² node, flat Q-table (6,400)` → `DQN³ ratio (7,680)`; each annotated "primary + target copy".
  Right-side total badge **"≈ 20,224 trainable params, O(n·h)"**.
- *Bottom lane (Quantum-HRL)*: `VQC tier+ratio (20)` and `QAOA node (4)`, with amplitude-encoding
  block feeding the VQC and the QUBO/Ising block feeding QAOA. Right-side total badge
  **"24 params, O(L·log₂ n + p)"**.
- *Alignment cues*: vertical guides linking "tier" of DQN¹ to the VQC tier readout, "node" of DQN² to
  QAOA, "ratio" of DQN³ to the VQC ratio readout — making explicit that one VQC absorbs two DQN heads.

**Suggested drawing structure.** Identical box heights in both lanes; use colour only to separate
classical (grey) from quantum (accent). Place the two total badges at the same x so the reader
compares 20,224 vs 24 directly. Optionally add a small bar-glyph (log scale) under each badge.

**Caption.** "Module-for-module comparison of the classical tri-DQN HRL baseline and the proposed
Quantum-HRL. A single VQC replaces the tier and ratio DQNs and a QAOA solver replaces the flat
`Q`-table node selector, reducing the trainable-parameter count from 20,224 to 24 and changing the
scaling from `O(n·h)` to `O(L log₂ n + p)`."

---

## Figure 3 — Optimization → MDP → QUBO → QAOA Mapping

**Layout description.** A four-stage left-to-right transformation chain, each stage a labelled panel
with the governing expression beneath it and a bold arrow (annotated with the transformation) between
panels.

**Components / stages.**
1. *Optimization problem* — `n* = argminₙ [β₁ T(α,n) + β₂ E(α,n)]` at fixed `(l*, α)`.
   Arrow label: "cast as MDP node decision".
2. *MDP view* — state `s_t`, action = node `n` in tier `l*`, intrinsic reward `R₂`.
   Arrow label: "binary one-hot encoding".
3. *QUBO* — `min_z Σ cₙ zₙ + A(Σ zₙ − 1)²`, `z ∈ {0,1}^{M_{l*}}` (Eq. qubo).
   Arrow label: "Pauli-Z substitution `zₙ = (I − σ̂ₙᶻ)/2`".
4. *Ising / QAOA* — `H_C = Σ hₙ σ̂ₙᶻ + (A/2) Σ σ̂ₙᶻσ̂ₘᶻ + E₀` (Eq. hamiltonian); depth-`p` QAOA circuit
   `∏ e^{-iβ_q H_M} e^{-iγ_q H_C} |+⟩^⊗m` whose ground state encodes `n*`.

**Suggested drawing structure.** Equal-width panels; show the one-hot constraint visually in stage 3
(e.g. a row of `M` cells with exactly one shaded). In stage 4 draw a small QAOA block (Hadamard layer
→ alternating `e^{-iγH_C}` / `e^{-iβH_M}` blocks → measurement). A faint "rebuilt every slot" loop
arrow under the chain communicates the stateless re-instantiation.

**Caption.** "Transformation of the per-tier node-selection problem into a form solvable by QAOA: the
optimization objective is expressed as a binary one-hot QUBO, mapped to an Ising cost Hamiltonian via
the Pauli-`Z` substitution, and solved by a depth-`p` QAOA circuit whose ground state identifies the
selected node `n*`."

---

## Figure 4 — VQC Architecture

**Layout description.** A standard quantum-circuit diagram: `q = 5` horizontal qubit wires, an input
preparation block on the left, `L = 4` repeated rotation+entanglement layers in the middle, and a
measurement/readout block on the right that fans out to two classical heads.

**Components.**
- *Input*: `|0⟩^⊗5` followed by an "Amplitude Encoding `s̃_t`" block spanning all 5 wires.
- *Variational layers* (×`L`): per layer, one `R_Y(θ_{ℓ,j})` box on each wire, then a linear chain of
  `CNOT_{j,j+1}` entanglers. Bracket the repeated block with "× L".
- *Readout*: `⟨Z⟩` measurement on the wires, splitting into (a) softmax over `⟨Ô_l⟩` → tier `l*`
  and (b) sigmoid of `⟨Ô_α⟩` → ratio `α`.
- *Parameter badge*: "`L·q = 4·5 = 20` trainable angles".

**Suggested drawing structure.** Use a quantum-circuit convention (TikZ `quantikz` or draw.io gate
shapes). Keep CNOTs as control-dot/⊕ pairs on adjacent wires. Shade the repeated variational layer to
distinguish it from the fixed encoding and readout. Show the two classical heads emerging as labelled
arrows on the far right.

**Caption.** "VQC high-level policy. The amplitude-encoded state on `q=⌈log₂ n⌉=5` qubits passes
through `L=4` layers of parameterized `R_Y` rotations and CNOT entanglers; Pauli-`Z` expectation
values drive a softmax tier head (`l*`) and a sigmoid ratio head (`α`), for a total of 20 trainable
angles."
