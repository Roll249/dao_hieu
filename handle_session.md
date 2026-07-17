# Handoff — Quantum-HRL paper audit

Written 2026-07-17, updated same day after a follow-up session. Branch `main` (the prior
`update-for-tuanlm` work was merged via PR #1, commit `30c026b`).
New work from the follow-up session (§5b fixes) is **uncommitted** (user has not asked to
commit).

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

## 4. OPEN ITEM 1 — Fig. 3 / §5.3 — **RESOLVED 2026-07-17**

The user asked to run `depth_sweep.py` locally on this machine (not the VPS originally
planned) and gave explicit go-ahead — no `.venv_hrl` existed here yet.

### What was done
- Created `.venv_hrl` with Python 3.14 (only interpreter available on this Windows box) and
  installed `numpy 2.5.1`, `scipy 1.18.0`, `scikit-learn 1.9.0`, `pennylane 0.45.1`
  (+ `pennylane-lightning`, Windows cp314 wheels exist), `matplotlib 3.11.0`.
  `config.config_hash()` inside this venv still prints `362cacf8efd9` — environment matches.
- Smoke-tested with `--quick --fresh` first (344s, no errors), then deleted the smoke test's
  `depth_sweep.json`/`depth_sweep_points.json` so they couldn't be confused with real output.
- Ran the real sweep: `python -u depth_sweep.py` (default seeds `[42,179,316]`, `train=30
  eval=12`). First log line printed `hash=362cacf8efd9` as required. Took 4878s (~81 min) —
  faster than the ~3h estimate. Wrote `simulation/depth_sweep.json`.
- Ran `make_depth_fig.py` — regenerated `fig3_depth_sweep.{pdf,png}` in both
  `simulation/figures/` and `overleaf/figures/` from the real data (confirmed identical
  between the two copies via `cmp`), replacing the fabricated assets.

### What the sweep found — the Pareto claim was false, as anticipated

Measured medians (3 seeds each; params = `L·5 + 2p`, `q=5` qubits since `n=20`):

| | value | params | median lat (s) |
|---|---|---|---|
| $L$-slice ($p{=}2$) | $L{=}1$ | 9  | $0.118$ |
| | $L{=}2$ | 14 | $0.095$ |
| | $L{=}4$ | 24 | $0.145$ |
| | $L{=}6$ | 34 | $0.103$ |
| $p$-slice ($L{=}4$) | $p{=}1$ | 22 | $0.123$ |
| | $p{=}2$ | 24 | $0.145$ |
| | $p{=}3$ | 26 | $0.132$ |

Classical-HRL reference on the same 3 seeds: median $0.092$\,s. Per-seed std $0.015$–$0.056$.

**`L=4` (the paper's chosen operating point) is the highest-median value in both slices** —
neither axis is monotone, confirming the "likely" branch this handoff anticipated. Fixed:
- Figure caption + main §5.3 paragraph (`subsec:depth`) — rewritten to report the real
  medians, state plainly that neither axis is monotone, and **withdraw** the "Pareto-efficient
  operating point" claim. Notes that `L=1` (9 params) is numerically *not worse* than the
  24-param `L=4` point — strengthens the parameter-efficiency thesis rather than the
  depth-sensitivity one. Explicitly does not over-read the 3-seed Classical-HRL comparison
  against the powered ten-seed null of §5.1 (Welch $p=0.47$).
- §6.3 Stage-III evidence paragraph — no longer claims the sweep "corroborates
  Prop.~3.2" via saturation; states the actual non-monotone medians instead.
- §5.6 — "QAOA approximation ratio at p=2 already saturates" replaced with the actual
  medians and "no significant sensitivity to QAOA depth at this scale."
- New Limitations bullet: "Circuit depth ($L$, $p$) is fixed a priori, not empirically
  optimised" — states the finding and that the depth choice should not be read as
  Pareto-validated.

Verified after: `sim/verify_paper.py`-equivalent check (missing cites/undefined
refs/brace balance/env balance/duplicate labels) — all clean. `overleaf/main.tex` re-synced,
diff vs `quantum_hrl_paper.tex` is exactly the `\graphicspath` hunk. Figure images identical
between `simulation/figures/` and `overleaf/figures/` (`cmp`).

**The paper may now be compiled/submitted from a depth-sensitivity standpoint** — no
fabricated assets remain. `depth_sweep.log`, `depth_sweep.json` are the ground truth for any
future edits to this section; `depth_sweep_points.json` (checkpoint) stays gitignored/
machine-local as before.

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

## 5b. OPEN ITEM 3 — minor issues found — **all four fixed this session (2026-07-17)**

These were surfaced by the prior audit and left alone at the time (low severity, or the fix
was a judgement call). All four have now been fixed:

1. **Duplicate section labels — fixed.** `\label{sec:framework}` (was on the same section as
   `\label{sec:problem}`, `quantum_hrl_paper.tex:313-314`) has been dropped; its one referrer
   (contributions bullet 2, was `Section~\ref{sec:framework}`) now cites `\ref{sec:problem}`
   directly — correct, since it genuinely is the same section. Verified: `grep -c
   'label{sec:framework}'` → 0, no duplicate labels remain (checked programmatically over all
   `\label{}` in the file).

2. **`fig:scaling` caption overclaim — fixed.** Reworded to state the figure sweeps the *state
   dimension* only (per-tier node counts held fixed) and points to Table~8
   (`tab:scalability`) for the node-count axis, which is genuinely shown there. Also fixed the
   same overclaim in the body paragraph right before the figure (it said node-count scaling was
   "visualised in Fig.~\ref{fig:scaling}" — same bug, just in prose instead of the caption).

3. **Unverified LuST 50% claim — fixed, by dropping the unverifiable fraction.** The source
   FCD is deleted (see §7.1), so "% of all vehicle-records in the hour" cannot be recomputed —
   only the ROI-filtered cache exists, and it has no citywide total to divide by. Replaced with
   the real, verifiable numbers read directly from `lust_roi_trajectories.npz`: **7,845 vehicle
   trajectories (136,878 position records)**, plus the actual filter thresholds from
   `config.py` (`LUST_MIN_POINTS=6`, `LUST_MIN_TRAVEL_M=150`). No unsupported percentage left
   in the paper.

4. **`EXPERIMENT_RESULTS.md` self-contradiction — fixed.** Both occurrences of "0 of 10 /
   0/10 Classical seeds collapsed" (lines ~157-158 and ~240-241) now read "1 of 10 / 1/10 ...
   also collapsed" (seed-727, 0.4429 s, 4.6% miss — just above the 0.4 s threshold), matching
   the JSON and Experiment 7's own table. The severity framing (Quantum's worst seed 2.24 s @
   20.8% miss vs. Classical's 0.44 s @ 4.6%) is preserved so the log's honest point (Quantum's
   tail is *worse*, even though both collapse once) survives the correction.

Also re-verified during this pass that items 1–12 from Section 3 (the prior session's fixes)
are all still correctly reflected in `quantum_hrl_paper.tex` — spot-checked the `ℓ`/`q`
notation split, the topology numbers against `config.py` (altitudes 0.01/0.3/20/600 km,
coverage 0.3/1.5/50/500 km, `config_hash()` still `362cacf8efd9`), the ablation Δ% figures,
the abstract's 10-vs-5-seed scoping, and the severity-not-count framing throughout. Nothing had
drifted.

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

## 8. Next actions, in priority order

1. ~~**Fig. 3 / §5.3** (§4)~~ — **done 2026-07-17**: ran `depth_sweep.py` locally (user's
   explicit go-ahead), real data measured, `make_depth_fig.py` regenerated the figure, §5.3 +
   §6.3 Stage-III + §5.6 rewritten to match, new Limitations bullet added, fabricated assets
   replaced, overleaf synced. **No fabricated data remains for this item.**
2. **Notation `n` / `h`** (§5) — blocked on a convention decision from the user. The only
   remaining open item.
3. ~~**Minor unfixed items** (§5b)~~ — **done 2026-07-17**: all four fixed (duplicate labels,
   `fig:scaling` caption/prose overclaim, unverified LuST 50% claim, `EXPERIMENT_RESULTS.md`
   contradiction).

Item 2 is the only thing left, and it's blocked on the user's convention decision — nothing
else is actionable without their input. **Before submitting**, still worth a real LaTeX
compile on Overleaf (never done in this environment — no toolchain here) to catch anything
the regex-based verify script can't see (e.g. actual over/underfull boxes, the new Fig. 3
layout, table floats).

### Local environment now available
`.venv_hrl` (Python 3.14, Windows) was created and works end-to-end for the simulator:
`numpy 2.5.1`, `scipy 1.18.0`, `scikit-learn 1.9.0`, `pennylane 0.45.1` (+
`pennylane-lightning`), `matplotlib 3.11.0`. `config_hash()` = `362cacf8efd9`, matching. This
supersedes the old "no wheels on system python" assumption — that was about the *system*
Python (3.14, Microsoft Store), not a venv on it.

### Commit state
Everything through the prior session **is committed** on `update-for-tuanlm`, since merged
into `main` (`30c026b`, PR #1 from `Roll249/update-for-thayTuan`):
- `f6fde0ef` — user's own repo-wide commit; swept in all the paper fixes, the new sweep
  scripts, this handoff, plus logs and `EC_HRL-main/`.
- `b4908929` — untracks + gitignores `simulation/depth_sweep_points.json` (see §4), and records
  the "fabricated assets still committed" warning.
- `64f7ea0` — handoff doc update only (this file, prior session).

The 2026-07-17 session's changes (this update) are **uncommitted**: `quantum_hrl_paper.tex`,
`overleaf/main.tex`, `simulation/EXPERIMENT_RESULTS.md`, `simulation/figures/fig3_depth_sweep.*`,
`overleaf/figures/fig3_depth_sweep.*`, `simulation/depth_sweep.log`, `simulation/depth_sweep.json`
(new, untracked), and this file. Ask before committing or pushing. The user separately asked
that `paper_results.py` (the main experiment driver) **not** be run without asking first —
that has not been run this session.
