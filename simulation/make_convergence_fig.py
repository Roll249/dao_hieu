"""Regenerate fig2_convergence from the actual 10-seed training trajectories.

The previous asset was rendered from an earlier data version and could imply a
tighter Quantum-HRL band than the honest 10-seed result supports. This script
re-trains Quantum-HRL and Classical-HRL on the same LuST env and seeds as
paper_results.py, records the per-episode reward trajectory of every seed, and
plots the across-seed MEDIAN with an interquartile (25-75%) band -- robust to
the one Quantum training collapse, and faithful to the reported variance.

Run:   .venv_hrl/bin/python simulation/make_convergence_fig.py
Writes fig2_convergence.{pdf,png} to simulation/figures/ and overleaf/figures/.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from tntn_environment import TNTNEnvironment
from quantum_hrl import QuantumHRLAgent, ClassicalHRLAgent
from utils import STATE_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N_STEPS = 40
N_TRAIN = 30
SEEDS = [42 + 137 * i for i in range(10)]


def make_env(seed):
    return TNTNEnvironment(seed=seed, n_steps_per_episode=N_STEPS, mobility="lust")


def collect(build_agent):
    """Train one agent per seed; return array (n_seeds, n_train) of episode returns."""
    curves = []
    for s in SEEDS:
        agent = build_agent(s)
        agent.train(make_env(s))
        r = np.asarray(agent.metrics.episode_rewards, dtype=float)[:N_TRAIN]
        if len(r) < N_TRAIN:                      # pad defensively
            r = np.concatenate([r, np.full(N_TRAIN - len(r), r[-1])])
        curves.append(r)
        print(f"    seed={s} final_return={r[-1]:.1f}", flush=True)
    return np.vstack(curves)


def band(ax, curves, color, label):
    x = np.arange(1, curves.shape[1] + 1)
    med = np.median(curves, axis=0)
    q1 = np.percentile(curves, 25, axis=0)
    q3 = np.percentile(curves, 75, axis=0)
    ax.plot(x, med, color=color, lw=2.0, label=label)
    ax.fill_between(x, q1, q3, color=color, alpha=0.20, linewidth=0)
    return med[-1]


def main():
    print("== Classical-HRL ==", flush=True)
    c = collect(lambda s: ClassicalHRLAgent(state_dim=STATE_DIM, hidden_dim=256,
                                            n_episodes=N_TRAIN, seed=s))
    print("== Quantum-HRL ==", flush=True)
    q = collect(lambda s: QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=N_TRAIN,
                                          batch_size=16, bo_budget=20, vqc_layers=4,
                                          qaoa_depth=2, seed=s, verbose=False))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    c_last = band(ax, c, "#1f77b4", "Classical HRL (tri-DQN)")
    q_last = band(ax, q, "#d62728", "Quantum-HRL (VQC+QAOA+BO)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.01)
    fig.tight_layout()

    for d in [os.path.join(HERE, "figures"), os.path.join(REPO, "overleaf", "figures")]:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "fig2_convergence.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, "fig2_convergence.png"), dpi=150, bbox_inches="tight")
        print("saved ->", d, flush=True)

    print(f"\nfinal-episode median return: Classical={c_last:.1f}  Quantum={q_last:.1f}")
    print(f"Quantum seed IQR at last episode: "
          f"[{np.percentile(q[:,-1],25):.1f}, {np.percentile(q[:,-1],75):.1f}]  "
          f"min={q[:,-1].min():.1f} (collapse seed)")
    print(f"Classical seed IQR at last episode: "
          f"[{np.percentile(c[:,-1],25):.1f}, {np.percentile(c[:,-1],75):.1f}]  "
          f"min={c[:,-1].min():.1f}")


if __name__ == "__main__":
    main()
