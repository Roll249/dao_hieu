"""Parameter-matched classical policy-gradient control (Reviewer #2 item M1).

This baseline isolates the *quantum* contribution of Quantum-HRL from the mere
benefit of optimising a **low-dimensional** policy. It is a faithful classical
twin of the VQC high-level policy:

  * identical input pre-processing (L2-normalised state),
  * identical readout heads (softmax over 4 tier logits; Gaussian offloading
    ratio with mean = sigmoid of the collective output sum + the same 1.5
    domain-prior bias),
  * identical advantage-weighted REINFORCE training (same pg_lr / pg_lr_alpha /
    alpha_sigma / update_every / replay batch / advantage baseline),
  * identical node selection via the *exact classical* argmin (the classical
    counterpart of QAOA).

The ONLY difference from Quantum-HRL is the function approximator that maps the
state to the 5 policy outputs: a small classical MLP (state -> hidden(tanh) ->
5 outputs) instead of the amplitude-encoded variational quantum circuit.

Parameter budget (default hidden=3): 20*3 + 3 + 3*5 + 5 = 83 trainable weights,
the same order of magnitude as Quantum-HRL's 24 and ~240x smaller than the
tri-DQN's 20,224. If this control matches Quantum-HRL, the parameter-efficiency
result is attributable to low dimensionality rather than to the quantum
representation; if it does not, the quantum policy carries genuine value.
"""
from __future__ import annotations
import numpy as np
import random
from typing import Tuple, List

import config as C
from tntn_environment import TNTNEnvironment, Task
from qaoa_solver import classical_node_selection
from utils import ReplayBuffer, STATE_DIM, M_TIERS, N_TIERS


class TinyClassicalPGAgent:
    """Small classical policy-gradient controller (parameter-matched to VQC)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden: int = 3,
        n_episodes: int = 30,
        batch_size: int = 16,
        replay_capacity: int = 10000,
        seed: int = 42,
        verbose: bool = False,
    ):
        # Match the VQC's high-level policy-gradient hyperparameters exactly so
        # that only the function approximator differs (VQC vs classical MLP).
        self.pg_lr = 0.12
        self.pg_lr_alpha = 0.6
        self.alpha_sigma = 0.3
        self.alpha_bias = 1.5          # same domain prior as the VQC readout
        self.update_every = 20
        self.n_out = 5                 # mirror the VQC's 5 Pauli-Z observables
        self.state_dim = state_dim
        self.hidden = hidden
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.verbose = verbose

        np.random.seed(seed)
        random.seed(seed)

        # Small MLP: state -> hidden (tanh) -> 5 outputs
        self.W1 = np.random.randn(state_dim, hidden) * np.sqrt(1.0 / state_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, self.n_out) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(self.n_out)

        self.replay = ReplayBuffer(capacity=replay_capacity)

    # -- policy -------------------------------------------------------------
    def _forward(self, state: np.ndarray):
        """Return (raw 5-vector, cache) with the same L2 pre-processing as VQC."""
        s = state / (np.linalg.norm(state) + 1e-12)
        z1 = s @ self.W1 + self.b1
        h = np.tanh(z1)
        raw = h @ self.W2 + self.b2
        return raw, (s, h)

    def policy_forward(self, state: np.ndarray):
        raw, cache = self._forward(state)
        logits = raw[:N_TIERS] - raw[:N_TIERS].max()
        probs = np.exp(logits) / np.exp(logits).sum()
        alpha_readout = float(np.sum(raw)) + self.alpha_bias
        alpha_mean = 1.0 / (1.0 + np.exp(-np.clip(alpha_readout, -10, 10)))
        return probs, float(alpha_mean), raw, cache

    def select_action(self, state, task: Task, env: TNTNEnvironment,
                      explore: bool = False) -> Tuple[int, int, float]:
        probs, alpha_mean, _, _ = self.policy_forward(state)
        if explore:
            l_star = int(np.random.choice(N_TIERS, p=probs))
            alpha = float(np.clip(alpha_mean + self.alpha_sigma * np.random.randn(),
                                  0.01, 1.0))
        else:
            l_star = int(np.argmax(probs))
            alpha = alpha_mean
        # Exact classical node selection (the classical counterpart of QAOA).
        costs = env.get_qubo_costs(l_star, alpha, task)
        n_star = classical_node_selection(costs)
        return l_star, n_star, alpha

    # -- training -----------------------------------------------------------
    def _reinforce_update(self, states, tiers, alphas, advantages):
        """Advantage-weighted REINFORCE step, mirroring VQCAgent.reinforce_update.

        Combines the categorical tier log-prob gradient (scaled by pg_lr) and the
        Gaussian ratio log-prob gradient (scaled by pg_lr_alpha), then ascends.
        """
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        for s_raw, l, a, A in zip(states, tiers, alphas, advantages):
            probs, mu, _, (s, h) = self.policy_forward(s_raw)
            # d logpi / d raw (length n_out=5)
            d_tier = np.zeros(self.n_out)
            d_tier[:N_TIERS] = -probs
            d_tier[int(l)] += 1.0                       # onehot(l) - probs
            d_alpha = ((a - mu) / (self.alpha_sigma ** 2)) * mu * (1.0 - mu) \
                * np.ones(self.n_out)                   # readout = sum(raw)
            d_raw = float(A) * (self.pg_lr * d_tier + self.pg_lr_alpha * d_alpha)
            # backprop d_raw through the MLP (ascent gradient)
            gW2 += np.outer(h, d_raw); gb2 += d_raw
            dh = (self.W2 @ d_raw) * (1.0 - h ** 2)      # tanh'
            gW1 += np.outer(s, dh); gb1 += dh
        n = max(len(states), 1)
        self.W1 += gW1 / n; self.b1 += gb1 / n
        self.W2 += gW2 / n; self.b2 += gb2 / n
        return float(np.linalg.norm(gW2) / n)

    def _update(self):
        if len(self.replay) < self.batch_size:
            return
        states, l_stars, n_stars, alphas, R1s, _, _ = self.replay.sample(self.batch_size)
        R1 = np.array(R1s, dtype=float)
        adv = R1 - R1.mean()
        adv = adv / (adv.std() + 1e-6)
        n_pg = min(len(states), 8)
        idx = np.random.choice(len(states), size=n_pg, replace=False)
        self._reinforce_update([states[i] for i in idx],
                               [int(l_stars[i]) for i in idx],
                               [float(alphas[i]) for i in idx],
                               [float(adv[i]) for i in idx])

    def train(self, env: TNTNEnvironment):
        print("Starting TinyClassical-PG training...")
        for episode in range(self.n_episodes):
            env.reset()
            done = False
            step = 0
            lats = []
            while not done:
                task = env._generate_task()
                state = env.observe(task)              # condition on task (#15)
                l_star, n_star, alpha = self.select_action(state, task, env, explore=True)
                s_next, reward, done, info = env.step(l_star, n_star, alpha, task)
                lat, en = info['latency'], info['energy']
                F1, F2, F3 = (info['constraint_flags']['F1'],
                              info['constraint_flags']['F2'],
                              info['constraint_flags']['F3'])
                R1 = -(lat + en) - 50.0 * F1 - 50.0 * F2 - 50.0 * F3
                self.replay.push(state, l_star, n_star, alpha, R1, R1, s_next)
                if len(self.replay) >= self.batch_size and step % self.update_every == 0:
                    self._update()
                lats.append(lat)
                step += 1
            if self.verbose and episode % 20 == 0:
                print(f"  Episode {episode:4d}: lat={np.mean(lats):.4f}s")
        return self

    def param_count(self) -> int:
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size)


if __name__ == "__main__":
    a = TinyClassicalPGAgent()
    print("TinyClassical-PG trainable params:", a.param_count())
