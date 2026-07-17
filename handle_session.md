# Handoff — Quantum-HRL paper audit

Written 2026-07-17. Branch `update-for-tuanlm`, last commit `42fc5e63`.
All work below is **uncommitted** (user has not asked to commit).

---

## 1. What this project is

A paper (`quantum_hrl_paper.tex`, repo root) proposing **Quantum-HRL** for task offloading
in 4-tier terrestrial/non-terrestrial networks: a VQC replaces the tier+ratio DQN heads,
QAOA replaces the node-selection head, Bayesian Optimisation tunes the QAOA angles.

**The thesis is parameter efficiency at performance parity — NOT quantum advantage.**
The paper was honestly reframed in commit `42fc5e63` after the fair LuST evaluation killed
the original "quantum beats classical" claim.

**Never re-introduce these claims** (they are false against the measured data):
- a significant latency or energy advantage over Classical HRL
- "QAOA halves latency"
- "843× speedup" — 843× is a *model-footprint* ratio, never a runtime claim

---

## 2. Where the truth lives

| Thing | Path | Note |
|---|---|---|
| Paper | `quantum_hrl_paper.tex` | root |
| Overleaf copy | `overleaf/main.tex` | **must differ only in `\graphicspath`** — sync after every edit |
| Canonical driver | `simulation/paper_results.py` | `run_experiments.py` is deprecated |
| Config single-source | `simulation/config.py` | `config_hash()` must be `362cacf8efd9` for base regime |
| Results log | `simulation/EXPERIMENT_RESULTS.md` | **see warning below** |

Result JSONs (`simulation/`) — these are ground truth, prefer them over the prose log:

- `paper_results_10seed.json` — base regime, **10 seeds**. Table 4 + all headline stats.
- `paper_results_m1.json` — adds TinyClassical-PG (83-param control), 10 seeds.
- `paper_results_heavy.json` — heavy load, **5 seeds**, ablation 3 seeds, hash `a9d8b078f389`.
- `paper_results_24h.json` — full-day LuST, **5 seeds**, ablation 3 seeds.
- `noise_sweep.json` — BO vs GD vs fixed angles across σ_n. Now in paper as Table 9.
- `qaoa_scaling.json` — QAOA constant-footprint vs growing DQN node head.

> **WARNING — `EXPERIMENT_RESULTS.md` contradicts itself.** Experiment 6 says "0 of 10
> Classical seeds collapsed"; Experiment 7's table says Classical collapses **1/10**. The
> JSON settles it: Classical's worst seed is **0.4429 s @ 4.6% miss**, which *is* above the
> 0.4 s threshold used to produce "TinyPG 4/10" and "Quantum 1/10". So **1/10 is correct**.
> This bug leaked into the paper and has now been fixed there — do not re-introduce it by
> trusting the prose log over the JSON.

Sync sang overleaf sau mỗi lần sửa paper:
```bash
cd /home/ngochieu/repo/dao_hieu
sed 's|\\graphicspath{{simulation/figures/}}|\\graphicspath{{figures/}}|' \
  quantum_hrl_paper.tex > overleaf/main.tex
diff quantum_hrl_paper.tex overleaf/main.tex   # expect exactly the graphicspath hunk
```

---

## 3. What this session did

Task was: audit the paper for remaining errors. Found and **fixed** the following.

### The serious one — Fig. 3 was fabricated

`simulation/visualize_results.py::plot_depth_sensitivity()` hard-coded its latency values
(`lat_vqc = [1.05, 0.88, 0.77, 0.68, 0.65, 0.64]`), invented the error bars, and captioned
them "std over 3 seeds". Its baseline lines (0.81 / 0.97 s) predate the LuST pipeline and
disagreed with the reported results by ~6×. The paper cited it as measured evidence in §5.3
and in the Stage-III/Stage-IV narrative. That function now **raises** instead of silently
regenerating fake data; the real replacement is described in §4 below.

### Fixed contradictions / errors

| # | What | Where |
|---|---|---|
| 1 | Topology prose said `h_LAP≈1/h_HAP≈100/h_LEO≈2000 km`, coverage `50m/200m/1000m`, credited to `[hrl_ntn, Table 3]`. Real config: `0.01/0.3/20/600 km`, coverage `0.3/1.5/50/500 km`, and the geometry is **ours**, not the baseline's. ROI is 3 km (prose said 5 km). | §6.1 |
| 2 | "No Classical HRL seed collapsed" vs "tri-DQN (1/10)" — self-contradiction. Now: threshold defined once (**lat > 0.4 s**), counts Quantum 1/10, Classical 1/10, TinyPG 4/10, and the argument is carried by **severity** (2.24 s @ 20.8% vs 0.44 s @ 4.6% vs ~0.55 s @ ~10%), not count. | §5.1, §5.5, §5.6, concl. |
| 3 | Limitations bullet claimed no queueing term and that heavy load was "left to future work" — but the paper reports a heavy-load regime. Rescoped to base regime. Queue is real: `QUEUE_BASE_S=1.2`, `utils.py:156`. | Limitations |
| 4 | Table 6 said "Seeds: 5" | → `10 (ablation: 5)` |
| 5 | Abstract + contribution implied 10 seeds for all 3 regimes. Only base is 10; heavy/24h are 5 (their ablations 3). | abstract, §1 |
| 6 | Ablation Δ% computed from rounded means | → `+1431 / +30.8 / +34.4` |
| 7 | Single-DQN's 17,920 params not reproducible from the paper (said `h=256`, action space `|L|×max M_l×K_α`, `K_α=11`). Real: `h=128`, `M×K_α = 12×10 = 120`. `K_α=10` everywhere. | §6.2, notation |
| 8 | Unsupported claim: "BO maintains reward stability up to gate-noise an order of magnitude larger…" — never measured. Replaced with **Table 9** built from `noise_sweep.json`, plus two honest caveats (absolute accuracy is only 0.28–0.52; the σ_n trend is **not monotone**, so only the per-level ordering BO > GD is claimed). | §5.5 |
| 9 | Noise-sweep data was referenced but never shown | Table 9 added |
| 10 | "QAOA grows sharply under congestion" — 31%→35% is flat. Only 24h (+164%) is a real jump; reworded to say so. | §5.2, §5.6 |
| 11 | **`q`** meant both QAOA layer index and qubit count `⌈log₂ n⌉` | → `ℓ` for layers in both circuits; declared in notation table |
| 12 | **`m`** meant both node count and node index | → coupling is `J_{nn'}`; `m` is count only |

Verified after each change: citations resolve, no undefined refs, no duplicate labels,
braces balanced, environments balanced, no stale numbers left.

---

## 4. OPEN ITEM 1 — Fig. 3 / §5.3 (blocked: user is running the sweep on a VPS)

### Status
User stopped the local run — **they will run it on a VPS**. Do not restart it locally
unless they say so.

> **DO NOT SUBMIT THE PAPER IN ITS CURRENT STATE.** The fabricated assets
> `fig3_depth_sweep.{pdf,png}` are still committed in both `simulation/figures/` and
> `overleaf/figures/`, and §5.3 still describes them as measured. Compiling today therefore
> still renders invented data. Only the *generator* has been disabled so far; the *assets* and
> the surrounding prose are replaced when the VPS sweep lands (`make_depth_fig.py` overwrites
> both directories). Until then the paper is in a known-interim state.

### The scripts (both new, untracked)
- `simulation/depth_sweep.py` — measures `L∈{1,2,4,6}` at `p=2` and `p∈{1,2,3}` at `L=4`,
  3 seeds `[42,179,316]`, same env/budget/hash as `paper_results.py`, plus a Classical-HRL
  reference on the same seeds. **Checkpoints each completed `(L,p)` point** to
  `depth_sweep_points.json` (atomic via `os.replace`) and resumes on restart. A checkpoint is
  reused only if `config_hash` + seeds + train/eval all match; `--fresh` forces recompute.
  (Checkpointing exists because the machine was shut down mid-run and 4 runs were lost —
  `depth_sweep.json` is only written at the very end.)
- `simulation/make_depth_fig.py` — renders `fig3_depth_sweep.{pdf,png}` into both
  `simulation/figures/` and `overleaf/figures/` from `depth_sweep.json`, and **prints the
  table of numbers §5.3 must quote**. Plots median + min–max over seeds. No invented error bars.

### Run on VPS
```bash
python3.12 -m venv .venv_hrl
.venv_hrl/bin/pip install "numpy>=2.1" "scipy>=1.15" "scikit-learn>=1.5" "pennylane>=0.40" matplotlib
cd simulation
setsid nohup ../.venv_hrl/bin/python -u depth_sweep.py > depth_sweep.log 2>&1 &
# when done:
../.venv_hrl/bin/python make_depth_fig.py
```
No TensorFlow, no SUMO needed (that's only the old `HRL_baseline/`). First log line must
print `hash=362cacf8efd9` — if not, the environment diverged and the numbers must **not**
go into the paper.

### Partial real data already collected locally (1 of 6 points)
A **genuine** `L=1, p=2` point was measured on the laptop before the run was stopped:

```
L=1 p=2, 9 params : seeds [0.0837, 0.2152, 0.1177] -> median 0.1177, mean 0.1389 ± 0.0557, miss 0.0%
```

Treat this as an **advance signal only, not as a data point for the figure.** The checkpoint
that holds it (`simulation/depth_sweep_points.json`) is now **gitignored and untracked** — it
was briefly committed in `f6fde0ef` and removed again, precisely because the checkpoint key
records `config_hash`/seeds/budget but **not library versions**. Had it shipped, the VPS would
have silently skipped `L=1` and mixed one laptop-computed point with five VPS-computed points
in a single figure. A fresh clone now has no checkpoint, so the VPS recomputes all 6 points in
one environment automatically. Do not copy the checkpoint across machines by hand.

### How to write §5.3 — depends on what the sweep shows

The paper currently claims `L=4, p=2` is a "Pareto-efficient operating point" and that
latency "improves with L up to L=4". **The early evidence contradicts this**: `L=1` reaches
median 0.1177 s with **9 parameters**, against the 10-seed `L=4` median of 0.105 s (different
seed counts — the sweep's own `L=4,p=2` point is the fair comparison).

- **If L=1 ≈ L=4** (likely): the Pareto claim **must die**. Rewrite §5.3 to say depth does not
  materially affect latency in this range, that `L=4` was fixed a priori, and add a Limitations
  bullet. Also fix §6.3 Stage-III ("The depth sweep … corroborates Prop. 3.2") and §5.6
  ("the QAOA approximation ratio at p=2 already saturates (Fig. 3)") — both currently lean on
  the fabricated figure.
  **Upside worth considering:** if 9 params match 24 params, the footprint story gets
  *stronger* (9 vs 20,224 = **2,247×**). Do not assert this without the full sweep.
- **If latency genuinely improves with L**: keep the §5.3 argument but requote every number
  from `depth_sweep.json`.

Either way: **report what the sweep measures.** Do not reverse-engineer the figure to fit
the existing prose — that is exactly the failure being cleaned up here.

---

## 5. OPEN ITEM 2 — notation `n` and `h` (blocked: user decision)

Two symbols are still overloaded. **Not fixed — the user has not chosen a convention**, and
these are pervasive refactors touching the headline scaling claim.

- **`n`** = state dimension (`n=20`, `⌈log₂ n⌉`, `O(n·h)`, `O(L log₂ n + p)`) **and** node
  index (`Σ_{n=1}^{M_l*} c_n z_n`, `h_n`, `n^*`). Both appear in Section 4.
- **`h`** = channel gain `h_{k,e}` / Ising local field `h_n` / DQN hidden width `h` /
  altitude `h_{LAP}`. Three of these are declared in the notation table.

Recommendation given to the user (awaiting answer):
1. `h` (hidden width) → `d_h` — ~8 spots, removes the worst clash (bare `h` sits next to
   `h_n` in the same parameter-accounting section).
2. `n` (state dim) → `n_s` — keep node index `n`.
3. Or: leave both, add a disambiguating note in the notation table (cheapest, still flaggable).

Traps if you attempt this: renaming node index → `e` collides with EN index `e` and makes
`h_e` vs `h_{k,e}`; renaming state dim → `d` collides with payload `d_k`.

---

## 6. Verify after any paper edit

```bash
cd /home/ngochieu/repo/dao_hieu
.venv_hrl/bin/python - <<'EOF'
import re
tex=open('quantum_hrl_paper.tex').read(); bib=open('references.bib').read()
keys=set(re.findall(r'@\w+\{([^,]+),',bib)); cited=set()
for m in re.findall(r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}',tex):
    cited |= {k.strip() for k in m.split(',')}
labels=set(re.findall(r'\\label\{([^}]+)\}',tex))
refs={m for m in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex)}
print("missing cites :", sorted(cited-keys) or "none")
print("undefined refs:", sorted(refs-labels) or "none")
print("brace balance :", tex.count('{')-tex.count('}'))
for env in ('equation','align','table','figure','tabular','itemize'):
    o=len(re.findall(r'\\begin\{'+env+r'\*?\}',tex)); c=len(re.findall(r'\\end\{'+env+r'\*?\}',tex))
    if o!=c: print(f"UNBALANCED {env}: {o}/{c}")
EOF
```
No LaTeX toolchain is installed on this machine — the paper **cannot** be compiled here.
Compile on Overleaf.

---

## 7. Gotchas that will bite you

1. **The LuST cache is irreplaceable.** `simulation/lust_roi_trajectories.npz` (and `_24h`)
   are tracked in git — good, because the source FCD (`figure_baseline/work/window.fcd.xml`)
   was **deleted** in the `42fc5e63` cleanup. The `.npz` is now the only LuST source and
   cannot be regenerated. Never delete it.
2. **`config_hash` must match** before merging any numbers into the paper: base/24h/m1/10seed
   = `362cacf8efd9`; heavy = `a9d8b078f389` (differs because `QHRL_HEAVY_LOAD` changes config).
3. **Always use `.venv_hrl/bin/python`** — system python is 3.14 and has no wheels.
4. **Regime knobs are env vars**: `QHRL_HEAVY_LOAD=1`, `QHRL_NODE_SCALE`, `QHRL_LUST_CACHE`.
5. **Seed counts differ per regime** — 10 (base), 5 (heavy, 24h), ablations 5 (base) / 3
   (heavy, 24h). The paper now states this; keep it accurate.
6. **`visualize_results.py` is legacy and was the source of the fabricated figure.** Treat any
   hard-coded numeric array in it as suspect. `plot_convergence`/`generate_convergence_curve`
   there also synthesise curves — `fig2_convergence.pdf` is now generated by the real
   `make_convergence_fig.py` instead, but do not call the legacy functions.
7. **The run is deterministic per seed** — `L=1 seed=42` reproduced `0.0837` exactly across two
   separate runs. If numbers move without a config change, something is wrong.

---

## 8. Suggested next actions

1. Wait for the user's VPS sweep → `make_depth_fig.py` → rewrite §5.3 per §4 above → sync overleaf.
2. Get the user's answer on notation `n`/`h` (§5).
3. Nothing is committed. Ask before committing.
