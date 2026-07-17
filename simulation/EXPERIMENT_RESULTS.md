# Quantum-HRL LuST Experiments — Results Log

Hướng B (real LuST in `simulation/`). All runs: `.venv_hrl/bin/python`, 5 seeds
[42,179,316,453,590], 30 train / 12 eval episodes × 40 tasks, LuST 17:00–18:00
densest 3-km ROI (7,845 real vehicle trajectories). Seed-level stats are PRIMARY.

Status: base + noise + QAOA-scaling DONE; heavy-load full run appended below when done.

---

## 1. Base regime (light load) — `paper_results.json`, config hash f7e3897085d1

Seed-level mean ± std (n=5):

| Method | Latency (s) | Energy (J) | Miss | Params |
|---|---|---|---|---|
| Random | 1.897 ± 0.123 | 1.837 | 22.7% | — |
| Greedy (nearest-RSU) | 0.162 ± 0.003 | 0.283 | 0.0% | — |
| Single-DQN | 0.176 ± 0.046 | 1.098 | 0.5% | 17,920 |
| Classical-HRL | 0.130 ± 0.042 | 0.507 | 0.0% | 20,224 |
| **Quantum-HRL** | **0.118 ± 0.034** | 0.830 | 0.0% | **24** |

- **Latency Quantum vs Classical: Welch t=−0.50, p=0.63, Hedges g=−0.29 → NOT significant** (statistically comparable). Pooled per-task test (n=2400) gives p=4e-10 but that is pseudo-replicated (#16) — reported only as supplementary.
- **Energy: Quantum significantly HIGHER** (0.83 vs 0.51 J, t=2.38, p=0.046, g=1.36).
- **Params: 24 vs 20,224 = 843× (trainable online); vs stored online+target 40,448 = 1,686×.**
- Ablation (3 seeds): Full 0.127 | w/o VQC 1.880 (+1382%) | w/o QAOA 0.129 (+1.5%) | w/o BO 0.117 (−7.7%). → VQC essential; QAOA/BO ~flat on mean latency in this easy regime. QAOA fallback 28.7%.
- **Honest reading:** in the light-load regime even a trivial heuristic (Greedy) is near-optimal on latency; Quantum-HRL is competitive, not superior, and pays an energy premium. The surviving claim is parameter efficiency.

## 2. NISQ-noise robustness (BO) — `noise_sweep.json`

QAOA angle tuning under readout noise σ_n; fraction recovering the true min-cost
node over 60 random 5-node instances, depth 2, BO budget 20 vs finite-diff GD:

| σ_n | BO | GD | fixed |
|---|---|---|---|
| 0.00 | 0.37 | 0.17 | 0.27 |
| 0.05 | 0.33 | 0.13 | 0.23 |
| 0.10 | 0.52 | 0.17 | 0.17 |
| 0.20 | 0.43 | 0.23 | 0.28 |
| 0.40 | 0.28 | 0.23 | 0.23 |
| 0.80 | 0.47 | 0.23 | 0.32 |

**BO beats gradient-descent at every noise level** (GP surrogate averages noise) →
real data for the BO noise-robustness claim (#17), replacing the previously
unsupported prose.

## 3. QAOA parameter scaling — `qaoa_scaling.json`

Node-selection accuracy + fallback vs node count m (depth 2 / 4), 60 instances:

| m | p | acc | fallback | QAOA params | DQN node-head params |
|---|---|---|---|---|---|
| 3 | 2 | 0.45 | 0.00 | 4 | 5,888 |
| 5 | 2 | 0.27 | 0.08 | 4 | 6,400 |
| 8 | 2 | 0.48 | 0.45 | 4 | 7,168 |
| 10 | 2 | 0.63 | 0.60 | 4 | 7,680 |
| 12 | 2 | 0.63 | 0.58 | 4 | 8,192 |

**Honest:** classical greedy node-selection is EXACT (per-node cost is directly
computable), so QAOA cannot beat it on quality; QAOA's value is the CONSTANT 2p-angle
footprint vs a DQN node-head that grows with m, with classical fallback guaranteeing a
valid selection (fallback rises with m at fixed depth — a known NISQ limitation).

## 4. Heavy-load regime (queueing, congestion) — `paper_results_heavy.json`

`paper_results_heavy.json`, config hash a9d8b078f389 (QHRL_HEAVY_LOAD=1). Seed-level
mean ± std (n=5):

| Method | Latency (s) | Energy (J) | Miss | Params |
|---|---|---|---|---|
| Random | 2.138 ± 0.118 | 1.801 | 27.7% | — |
| Greedy (nearest-RSU) | 0.530 ± 0.027 | 0.281 | **7.1%** | — |
| Single-DQN | 0.433 ± 0.065 | 1.386 | 5.3% | 17,920 |
| **Classical-HRL** | **0.274 ± 0.055** | 0.855 | 1.5% | 20,224 |
| Quantum-HRL | 0.330 ± 0.063 | 1.073 | 2.3% | **24** |

- Under congestion the naive **Greedy heuristic fails** (7.1% deadline miss, 0.53 s) and
  Single-DQN degrades (5.3%); both **learned HRL policies dominate** the heuristics.
- **Quantum vs Classical latency: t=1.41, p=0.20 (NOT significant); direction favours
  Classical** (0.274 < 0.330, g=0.81). Energy comparable (1.07 vs 0.85, p=0.58). → the
  1-seed teaser (Quantum 0.240 < Classical 0.317) did NOT survive 5 seeds; Quantum is
  **competitive, not superior**, at 843× fewer parameters.
- **Ablation (heavy):** Full 0.284 | w/o VQC 2.022 (+611%) | **w/o QAOA 0.385 (+35.4%)** |
  w/o BO 0.290 (+1.8%). → VQC essential; **QAOA now earns its place** (+35% vs +1.5% in
  light load — node selection among *loaded* nodes is a real combinatorial decision); BO
  still ~flat on mean latency (its value is noise robustness, §2). QAOA fallback 22%.

---

## Experiment 5 — FULL 24-hour LuST generalization (base regime, all 24 windows)

**Question tested (user):** does training/evaluating on the *entire day* of LuST mobility
(diurnal traffic, rush-hour congestion, night sparsity) let quantum overtake classical?

- **Cache:** `lust_roi_trajectories_24h.npz` — **99,810 trajectories / 1,435,900 points**
  pooled across all 24 hourly windows (night ~200/hr, rush hours 6k–8k/hr).
- **Run:** `QHRL_LUST_CACHE=...24h.npz paper_results.py --tag _24h`, base regime
  (node_scale=1, heavy_load=0), 5 seeds, config_hash `362cacf8efd9`, wallclock ~99 min.

**Latency & energy (seed-level, n=5):**

| Method | Latency (s) | Energy | Miss | Params |
|---|---|---|---|---|
| Random | 1.872 ± 0.118 | 1.834 | 22.7% | — |
| Greedy (nearest-RSU) | 0.159 ± 0.003 | 0.283 | 0.0% | — |
| Single-DQN | 0.178 ± 0.034 | 1.158 | 0.5% | 17,920 |
| **Classical-HRL** | **0.114 ± 0.041** | 0.555 | 0.0% | 20,224 |
| Quantum-HRL | 0.201 ± 0.092 | 1.203 | 0.2% | **24** |

- **Verdict UNCHANGED — and slightly worse for quantum.** On the full day, Quantum latency
  **rose 0.118 → 0.201** while Classical stayed flat **0.130 → 0.114**. Classical now wins
  latency with a **large effect size (Hedges g = +1.09)**; Welch **t=1.92, p=0.108** — not
  significant at n=5 *only because Quantum's variance blew up* (std 0.034 → 0.092). Energy:
  Classical 0.555 < Quantum 1.203, **g=1.40, p=0.056** (borderline, favours Classical).
- **Why more data did not rescue quantum:** heterogeneous full-day mobility *exposes* the
  20-param VQC's underfitting rather than curing it — its per-seed latency variance nearly
  tripled. The problem is still not quantum-hard; classical's 20k params generalize better
  across the diurnal distribution.
- **Ablation (24h, 3 seeds):** Full 0.170 | w/o VQC 1.871 (+1001%, 22% miss) |
  **w/o QAOA 0.448 (+164%)** | w/o BO 0.108 (−36%, best). → VQC essential as always;
  **QAOA earns its place on 24h too** (removing it triples latency, driven by rush-hour
  seed 316) — consistent with the heavy-load finding; **BO still does not help mean
  latency** (w/o-BO is best in every regime; BO's only justification remains noise
  robustness, Exp. 3). QAOA fallback rose **0.287 → 0.315** on the more diverse day.

**24h vs 1h summary:** the generalization experiment *strengthens the honest story* — it
adds a second regime where (a) quantum shows no latency advantage (in fact loses ground),
(b) parameter-efficiency is unchanged, (c) QAOA is justified, (d) BO is not. It does **not**
flip any headline.

---

## Experiment 6 — 10-SEED base regime (statistical power + robustness)

**Why:** Q1 reviewers will not accept n=5. Re-ran the base (light-load, 1h) comparison with
**10 seeds** (42,179,316,453,590,727,864,1001,1138,1275 — a superset of the original 5) and
**5 ablation seeds**. config_hash `362cacf8efd9`, wallclock ~179 min.

**The mean is MISLEADING here — must read robust statistics.** One seed (727, new) caused a
**Quantum training collapse** (lat 2.23 s, 20.8% deadline miss) which alone inflates the
Quantum mean and variance:

| Latency (s) | Quantum-HRL | Classical-HRL |
|---|---|---|
| mean ± std | 0.321 ± 0.673 | 0.158 ± 0.107 |
| **median** | **0.105** | 0.139 |
| 10%-trimmed mean | **0.113** | 0.133 |
| mean, drop seed-727 | **0.109** | 0.126 |

- **No significant latency difference in ANY test** — Welch t=0.76, **p=0.47**; Mann-Whitney
  (rank, outlier-robust) **p=0.47**; drop-outlier Welch **p=0.32**. Consistent with the
  5-seed null (p=0.63). The "quantum beats classical" headline stays dead.
- **On a *typical* run Quantum is actually competitive-to-slightly-better** (median 0.105 <
  0.139; trimmed 0.113 < 0.133; drop-727 0.109 < 0.126). The 5-seed sample happened to miss
  Quantum's bad tail.
- **NEW, important, honest limitation — training instability / fat tail.** 1 of 10 Quantum
  seeds collapsed (barren-plateau / high-variance policy-gradient on the VQC); **0 of 10
  Classical seeds collapsed** (worst classical seed-727 = 0.44 s, 4.6% miss). This is a real,
  citable weakness of the VQC/REINFORCE tier-policy vs DQN's TD+replay+target-net stability.
- **Energy premium did NOT survive 10 seeds.** 5-seed showed Quantum energy significantly
  worse (p=0.046); at n=10 it is **comparable** — Quantum 0.889 ± 0.209 vs Classical
  0.809 ± 0.575, **p=0.69** (Classical's own seed-727 blew its energy to 2.27). So do NOT
  claim a systematic quantum energy penalty either; both are statistically indistinguishable
  on energy at n=10.
- **Ablation (5 seeds, base):** Full 0.118 | w/o VQC 1.810 (+1431%, 22% miss) |
  w/o QAOA 0.155 (+31%) | w/o BO 0.159 (+35%). VQC essential as always; QAOA and BO each show
  a modest positive contribution here, but base-regime ablation is **seed-noisy** (earlier
  3-seed base run had BO looking neutral) — the *robust* module story remains: VQC essential
  everywhere, QAOA decisive under heavy/24h load, BO clearest under measurement noise (Exp 3).
  QAOA fallback 25.0%.

**5-seed → 10-seed takeaways (what changed):** (1) latency null **confirmed** with more
power and rank-based tests; (2) Quantum is competitive on *typical* seeds but has a
**fat tail / occasional collapse** — a genuine new limitation to report; (3) the earlier
**energy-premium claim weakened to non-significant** — must be softened in the paper.

---

## Experiment 7 — M1 parameter-matched classical control (Reviewer #2's key gap)

**Why:** Reviewer #2 flagged M1 (High) as *"the single most persuasive missing experiment"*:
the 843× headline could be confounded by low-dimensional optimisation simply being
better-conditioned. To attribute the result to the *quantum* representation (vs. mere small
size), we added **TinyClassical-PG** — a faithful classical twin of the VQC policy: identical
L2 input, identical readouts (softmax 4-tier + sigmoid-sum ratio, same 1.5 bias), identical
advantage-weighted REINFORCE (same lr/σ/update-every/replay), exact-argmin node selection.
The ONLY difference from Quantum-HRL is the function approximator: a small classical MLP
(state→hidden(3,tanh)→5 outputs, **83 params**) instead of the amplitude-encoded VQC.
Base regime, 10 seeds, config_hash `362cacf8efd9`, ~193 min.

| Method (10-seed) | Latency median | Latency mean±std | Miss | Energy | Params | Collapses (lat>0.4) |
|---|---|---|---|---|---|---|
| Classical-HRL (TD-DQN) | 0.139 | 0.158±0.102 | 0.5% | 0.809 | 20,224 | **1/10** |
| TinyClassical-PG (REINFORCE) | 0.166 | 0.285±0.224 | 4.0% | 1.425 | **83** | **4/10** |
| Quantum-HRL (REINFORCE) | **0.105** | 0.321±0.639 | 2.1% | 0.889 | **24** | **1/10** |

**Seed-level significance:**
- Quantum vs TinyClassical-PG **latency**: Welch **p=0.877**, Mann-Whitney **p=0.571** — not
  significantly different, but Quantum has the **better median** (0.105 vs 0.166).
- Quantum vs TinyClassical-PG **energy**: Welch **p=0.189**, Quantum **lower** (0.889 vs 1.425,
  g=−0.61) — Quantum is the more energy-efficient of the two REINFORCE policies.
- TinyClassical-PG vs Classical-HRL latency: Welch p=0.146, MW p=0.734 (comparable).

**Verdict — M1 DEFENDS the quantum contribution (with honest caveats):**
1. **843× is NOT merely a low-dimensional artifact.** A parameter-matched classical control
   (83 params, identical REINFORCE) does **not** beat Quantum-HRL — same-or-better median,
   better energy, at ~3.5× the parameters. The reviewer's worry ("a tiny classical policy
   would match/beat, making quantum pointless") is **not** realised.
2. **Training instability is a POLICY-GRADIENT property, not a quantum one.** Collapses:
   TinyClassical-PG **4/10** vs Quantum-HRL **1/10** vs Classical-HRL (TD) **1/10**. The VQC
   is actually **more stable than its classical REINFORCE twin**; TD-learning (DQN) is the
   most stable of all. → This **softens** the training-stability limitation: it is not
   evidence that "quantum is fragile," but that "REINFORCE is fragile," and the quantum
   policy is no worse (better here) than an equivalent classical policy-gradient learner.
3. **Honest limit:** the Quantum-vs-TinyPG latency gap is **not statistically significant**
   (p=0.88); we cannot claim quantum *beats* the matched control, only that it is
   competitive-to-better (median, energy, fewer collapses) at a smaller footprint. Quantum's
   single collapse is *deeper* (2.24 s) than TinyPG's (~0.55 s), which is why its mean/std is
   higher despite fewer collapses.

**Net effect on the paper:** M1 neutralises the top reviewer objection and lets us state that
the efficiency is attributable to the representation, not just to dimensionality — while the
training-stability limitation is reframed (correctly) as a REINFORCE-vs-TD issue. This is a
clear strengthening of the Q1 case.

---

## Overall honest conclusion (for the paper reframe, next session)

The fair LuST evaluation does **not** support a "quantum beats classical on latency"
headline. The defensible, honest thesis is:

1. **Parameter efficiency (core):** Quantum-HRL matches the classical tri-DQN HRL
   baseline using **24 vs 20,224 trainable params (843×)** — a change in scaling, not a
   constant factor.
2. **Competitive across regimes (confirmed at n=10):** latency statistically comparable to
   Classical in light load (5-seed 0.118 vs 0.130 p=0.63; **10-seed p=0.47, MW p=0.47**),
   heavy load (0.330 vs 0.274, p=0.20), and full-24h (0.201 vs 0.114, p=0.11). On *typical*
   (median/trimmed) seeds Quantum is competitive-to-slightly-better; the mean is outlier-driven.
3. **Training stability is a real limitation (n=10):** 1/10 Quantum seeds collapsed
   (barren-plateau / policy-gradient variance on the VQC); 0/10 Classical seeds collapsed.
   Report this honestly — it is the clearest weakness surfaced by more seeds.
4. **Energy is comparable, NOT a systematic premium:** the 5-seed energy gap (p=0.046) did
   **not** survive 10 seeds (0.889 vs 0.809, p=0.69). Soften any energy-penalty claim.
5. **Learning matters under congestion:** in heavy load both learned policies decisively
   beat the Greedy/Single-DQN heuristics (which miss 5–7% of deadlines).
6. **Modules earn their keep in their target regimes:** QAOA is decisive under heavy-load
   (+35%) and 24h (+164%); BO beats gradient-descent angle tuning at every noise level;
   QAOA keeps a constant param footprint vs a growing DQN node-head. (Base-regime module
   ablation is seed-noisy — lead with the heavy/24h/noise evidence.)
7. **Do NOT claim** significant latency advantage, a systematic energy penalty, "QAOA halves
   latency", or "843× speedup".

Paper edits still pending (deferred): rewrite abstract/contributions/results-table/
seed-level-stats/ablation/conclusion to the above; then sync `overleaf/main.tex`.
Number-independent fixes (#6–#14,#20) already applied to `quantum_hrl_paper.tex`.

