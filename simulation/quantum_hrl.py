"""Full Quantum-HRL Agent.

Integrates:
  - VQC for high-level policy (tier + ratio selection)
  - QAOA for node selection
  - Bayesian Optimization for parameter tuning
  - Replay buffer for off-policy learning
  - Epsilon-greedy exploration

Implements Algorithm 1 from the paper.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import random

from tntn_environment import TNTNEnvironment, Task
from vqc_circuit import VQCAgent
from qaoa_solver import QAOASolver, classical_node_selection, random_node_selection
from utils import (
    ReplayBuffer, BayesianOptimizer,
    STATE_DIM, M_TIERS, M_TOTAL, N_TIERS,
    normalize_state, normalize_qubo_coefficients,
)


@dataclass
class TrainingMetrics:
    """Track training progress."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_latencies: List[float] = field(default_factory=list)
    episode_energies: List[float] = field(default_factory=list)
    vqc_losses: List[float] = field(default_factory=list)
    qaoa_energies: List[List[float]] = field(default_factory=list)
    constraint_violations: List[int] = field(default_factory=list)
    cumulative_reward: float = 0.0


class QuantumHRLAgent:
    """Quantum-HRL agent implementing Algorithm 1.

    Pipeline:
      1. Observe state s_t
      2. Amplitude encode -> VQC -> (l*, alpha)
      3. QAOA(node selection) -> n*
      4. Execute action, observe reward
      5. Store in replay buffer
      6. Periodically update VQC and QAOA params via BO
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_episodes: int = 500,
        batch_size: int = 32,
        replay_capacity: int = 10000,
        bo_budget: int = 20,
        vqc_layers: int = 4,
        qaoa_depth: int = 2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        gamma_mdp: float = 0.99,
        seed: int = 42,
        use_quantum: bool = True,
        node_random: bool = False,
        vqc_random: bool = False,
        verbose: bool = False,
    ):
        # Ablation switches: node_random -> bypass QAOA with a random node;
        # vqc_random -> bypass the VQC high-level policy with random (tier, ratio).
        self.node_random = node_random
        self.vqc_random = vqc_random
        # High-level policy-gradient hyperparameters. The ratio readout uses a
        # larger step because its single-observable signal is otherwise swamped
        # by the tier/constraint variance in the shared advantage.
        self.pg_lr = 0.12
        self.pg_lr_alpha = 0.6
        self.alpha_sigma = 0.3
        self.update_every = 20
        self.state_dim = state_dim
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.bo_budget = bo_budget
        self.vqc_layers = vqc_layers
        self.qaoa_depth = qaoa_depth
        self.gamma_mdp = gamma_mdp
        self.use_quantum = use_quantum
        self.verbose = verbose

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Set seeds
        np.random.seed(seed)
        random.seed(seed)

        # VQC agent (high-level policy)
        # Analytic statevector (shots=None) for tractable experiment turnaround;
        # the paper's finite-shot (N_shots=1024) noise is modelled separately in
        # the NISQ-noise scenario. Exact expectations also stabilise PSR gradients.
        self.vqc = VQCAgent(
            state_dim=state_dim,
            n_layers=vqc_layers,
            n_shots=None,
            seed=seed,
            lr=0.01,
        )

        # QAOA solver per tier (each tier may have different node counts)
        self.qaoa_solvers = {}
        for tier_idx, m_tier in enumerate(M_TIERS):
            self.qaoa_solvers[tier_idx] = QAOASolver(
                n_nodes=m_tier,
                depth=qaoa_depth,
                seed=seed + tier_idx,
            )

        # Bayesian Optimizer for VQC and QAOA
        self.bo_vqc = BayesianOptimizer(
            param_bounds=[(0, 2 * np.pi)] * (vqc_layers * int(np.ceil(np.log2(state_dim)))),
            n_initial=5,
            n_iterations=bo_budget,
            noise_std=0.1,
        )
        self.bo_qaoa = BayesianOptimizer(
            param_bounds=[(0, 2 * np.pi)] * (2 * qaoa_depth),
            n_initial=5,
            n_iterations=bo_budget,
            noise_std=0.1,
        )

        # Replay buffer
        self.replay = ReplayBuffer(capacity=replay_capacity)

        # Metrics
        self.metrics = TrainingMetrics()

        # TD target network (VQC parameters at previous step)
        self.vqc_params_target = self.vqc.get_params().copy()

    def select_action(
        self,
        state: np.ndarray,
        task: Task,
        env: TNTNEnvironment,
        explore: bool = False,
    ) -> Tuple[int, int, float]:
        """Select action (tier, node, alpha) using Quantum-HRL.

        Args:
            state:  Current state s_t
            task:   Current task descriptor
            env:    T-NTN environment
            explore: Whether to use epsilon-greedy exploration

        Returns:
            tier_selected:  Selected tier index [0, 3]
            node_selected:  Selected node index within tier
            alpha:         Offloading ratio [0, 1]
        """
        # Step 1: VQC high-level policy (stochastic during training)
        if self.vqc_random:
            # Ablation: random tier and ratio
            l_star = np.random.randint(0, N_TIERS)
            alpha = np.random.rand()
        else:
            probs, alpha_mean, _ = self.vqc.policy_forward(state)
            if explore:
                # On-policy sampling drives REINFORCE exploration: sample the
                # tier from the categorical policy and the ratio from the
                # Gaussian policy around its mean.
                l_star = int(np.random.choice(N_TIERS, p=probs))
                alpha = float(np.clip(alpha_mean + self.alpha_sigma * np.random.randn(), 0.01, 1.0))
            else:
                l_star = int(np.argmax(probs))
                alpha = alpha_mean

        # Step 2: QAOA node selection within selected tier
        m_tier = M_TIERS[l_star]

        if self.node_random:
            # Ablation: random node selection
            n_star = random_node_selection(m_tier)
        else:
            # QAOA node selection
            costs = env.get_qubo_costs(l_star, alpha, task)

            if self.use_quantum:
                n_star, _, qaoa_info = self.qaoa_solvers[l_star].solve(
                    costs, penalty=50.0, n_iterations=6
                )
            else:
                # Classical fallback
                n_star = classical_node_selection(costs)

        return l_star, n_star, alpha

    def update_vqc(self) -> float:
        """Update the VQC high-level policy via advantage-weighted REINFORCE.

        The VQC parameterises a categorical tier policy and a Gaussian ratio
        policy; gradients of the log-policy w.r.t. the circuit angles are
        obtained from the exact Parameter-Shift Rule (Algorithm 1, Step 4).
        Using the (bounded) policy outputs with a normalised advantage signal
        is the correctly-scaled replacement for the previous TD target, whose
        unbounded magnitude could not be represented by Pauli-Z expectations.
        """
        if self.vqc_random or len(self.replay) < self.batch_size:
            return 0.0

        states, l_stars, n_stars, alphas, R1s, R2s, next_states = \
            self.replay.sample(self.batch_size)
        R1 = np.array(R1s, dtype=float)
        # Normalised advantage (variance-reduction baseline = batch mean)
        adv = R1 - R1.mean()
        std = adv.std()
        adv = adv / (std + 1e-6)

        # Subsample to bound the per-update circuit-evaluation budget
        n_pg = min(len(states), 8)
        idx = np.random.choice(len(states), size=n_pg, replace=False)
        loss = self.vqc.reinforce_update(
            [states[i] for i in idx],
            [int(l_stars[i]) for i in idx],
            [float(alphas[i]) for i in idx],
            [float(adv[i]) for i in idx],
            lr=self.pg_lr, lr_alpha=self.pg_lr_alpha, alpha_sigma=self.alpha_sigma,
        )
        self.vqc_params_target = self.vqc.get_params().copy()
        return loss

    def update_qaoa(self) -> Dict[int, float]:
        """Update QAOA angles via BO on cached Ising energy (Algorithm 1, Step 5).

        For each tier, the BO proposes new (gamma, beta) angle vectors and
        evaluates them against the last Ising instance seen by that solver.
        The solver's angles are updated whenever BO finds an improvement.
        """
        if self.node_random or len(self.replay) < self.batch_size:
            return {}

        tier_energies = {}
        for tier_idx in range(N_TIERS):
            solver = self.qaoa_solvers[tier_idx]

            # Skip tiers whose solver hasn't seen any real problem yet
            if solver.last_h is None or solver.last_J is None:
                continue

            h, J = solver.last_h, solver.last_J

            # BO suggests a new set of angles for this tier
            bo_angles = self.bo_qaoa.suggest()

            # Evaluate QAOA energy at the suggested angles
            qnode = solver.build_qnode(h, J)
            try:
                energy = float(qnode(bo_angles))
            except Exception:
                energy = solver._classical_energy(
                    bo_angles, h, J, m=solver.n_nodes
                )

            # Report to BO (maximise negative energy = minimise energy)
            self.bo_qaoa.report(bo_angles, -energy)

            # Update solver if BO found a lower-energy configuration
            best_angles, best_val = self.bo_qaoa.get_best()
            if best_angles is not None and (-best_val) < solver.best_energy:
                solver.angles = best_angles.copy()
                solver.best_angles = best_angles.copy()
                solver.best_energy = -best_val

            tier_energies[tier_idx] = energy

        return tier_energies

    def train(self, env: TNTNEnvironment) -> TrainingMetrics:
        """Full training loop (Algorithm 1)."""
        print("Starting Quantum-HRL training...")

        for episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_latencies = []
            episode_energies = []
            done = False
            step = 0

            while not done:
                # Generate task
                task = env._generate_task()

                # Select action
                l_star, n_star, alpha = self.select_action(
                    state, task, env, explore=True
                )

                # Execute action
                s_next, reward, done, info = env.step(l_star, n_star, alpha, task)

                # Compute R1 and R2 rewards (Section 5.6, Step 2)
                lat = info['latency']
                en = info['energy']
                beta1, beta2 = 1.0, 1.0
                w1, w2, w3 = 50.0, 50.0, 50.0
                F1, F2, F3 = (info['constraint_flags']['F1'],
                               info['constraint_flags']['F2'],
                               info['constraint_flags']['F3'])

                R1 = -(beta1 * lat + beta2 * en) - w1 * F1 - w2 * F2 - w3 * F3
                R2 = -(beta1 * lat + beta2 * en) - w2 * F2 - w3 * F3

                # Store in replay buffer
                self.replay.push(state, l_star, n_star, alpha, R1, R2, s_next)

                # Periodic updates
                if len(self.replay) >= self.batch_size and step % self.update_every == 0:
                    vqc_loss = self.update_vqc()
                    self.metrics.vqc_losses.append(vqc_loss)
                    _ = self.update_qaoa()

                episode_reward += reward
                episode_latencies.append(lat)
                episode_energies.append(en)
                state = s_next
                step += 1

            # Epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Record metrics
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_latencies.append(np.mean(episode_latencies))
            self.metrics.episode_energies.append(np.mean(episode_energies))
            self.metrics.cumulative_reward = episode_reward

            if episode % 20 == 0 and self.verbose:
                print(f"  Episode {episode:4d}: reward={episode_reward:.2f}, "
                      f"lat={np.mean(episode_latencies):.4f}s, "
                      f"energy={np.mean(episode_energies):.4f}J, "
                      f"eps={self.epsilon:.3f}")

        return self.metrics


# =============================================================================
# CLASSICAL BASELINE AGENTS
# =============================================================================

class ClassicalHRLAgent:
    """Classical tri-DQN HRL baseline (Shinde & Tarchi, 2024).

    Uses three separate DQN networks:
      - P1: tier selection
      - P2: node selection
      - P3: ratio regression
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
        n_episodes: int = 500,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_episodes = n_episodes

        np.random.seed(seed)
        random.seed(seed)

        self.n_tiers = N_TIERS
        self.m_tiers = M_TIERS
        self.epsilon = 1.0

        # Three DQN networks (simplified MLP approximation)
        self.dqn_tier = self._build_mlp(state_dim, hidden_dim, N_TIERS, seed)
        self.dqn_node = self._build_mlp(state_dim, hidden_dim, max(M_TIERS), seed + 1)
        self.dqn_ratio = self._build_mlp(state_dim, hidden_dim, 10, seed + 2)

        self.target_tier = [w.copy() for w in self.dqn_tier]
        self.target_node = [w.copy() for w in self.dqn_node]
        self.target_ratio = [w.copy() for w in self.dqn_ratio]

        self.replay = ReplayBuffer(capacity=10000)
        self.metrics = TrainingMetrics()

    def _build_mlp(self, input_dim: int, hidden: int, output_dim: int, seed: int) -> List[np.ndarray]:
        """Build simple MLP weights."""
        np.random.seed(seed)
        W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(hidden)
        W2 = np.random.randn(hidden, output_dim) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(output_dim)
        return [W1, b1, W2, b2]

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _forward(self, state: np.ndarray, weights: List[np.ndarray]) -> np.ndarray:
        """Forward pass through MLP."""
        W1, b1, W2, b2 = weights
        h = self._relu(state @ W1 + b1)
        return h @ W2 + b2

    def _td_update(self, weights, target_weights, states, actions, targets, lr=1e-3):
        """One mini-batch semi-gradient TD update on a single Q-head (numpy backprop).

        Loss = mean_i (Q(s_i)[a_i] - y_i)^2 with y_i = R + gamma*max_a Q_target(s'_i).
        Standard DQN update; makes the classical baseline a genuine learner.
        """
        W1, b1, W2, b2 = weights
        B = len(states)
        gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)
        gW2 = np.zeros_like(W2); gb2 = np.zeros_like(b2)
        for i in range(B):
            s = states[i]
            z1 = s @ W1 + b1
            h = self._relu(z1)
            q = h @ W2 + b2
            a = int(actions[i])
            dq = np.zeros_like(q)
            dq[a] = 2.0 * (q[a] - targets[i]) / B
            gW2 += np.outer(h, dq); gb2 += dq
            dh = W2 @ dq
            dh[z1 <= 0] = 0.0
            gW1 += np.outer(s, dh); gb1 += dh
        # gradient clipping for stability under the noisy reward scale
        for g in (gW1, gb1, gW2, gb2):
            np.clip(g, -1.0, 1.0, out=g)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

    def _learn(self, batch_size: int = 64):
        """Sample a mini-batch and update all three Q-heads (tier, node, ratio)."""
        if len(self.replay) < batch_size:
            return
        s, l_star, n_star, alpha, R1, R2, s_next = self.replay.sample(batch_size)
        s = np.array(s); s_next = np.array(s_next)
        R = np.array(R1, dtype=float)
        gamma = 0.95
        # Bellman targets bootstrapped from the frozen target networks
        y_tier = R + gamma * np.array([self._forward(sn, self.target_tier).max() for sn in s_next])
        y_node = R + gamma * np.array([self._forward(sn, self.target_node).max() for sn in s_next])
        y_ratio = R + gamma * np.array([self._forward(sn, self.target_ratio).max() for sn in s_next])
        a_tier = [int(x) for x in l_star]
        a_node = [min(int(x), max(M_TIERS) - 1) for x in n_star]
        a_ratio = [int(np.clip(round(a * 10) - 1, 0, 9)) for a in alpha]
        self._td_update(self.dqn_tier, self.target_tier, s, a_tier, y_tier)
        self._td_update(self.dqn_node, self.target_node, s, a_node, y_node)
        self._td_update(self.dqn_ratio, self.target_ratio, s, a_ratio, y_ratio)

    def select_action(
        self,
        state: np.ndarray,
        task: Task,
        env: TNTNEnvironment,
        explore: bool = False,
    ) -> Tuple[int, int, float]:
        """Select action using classical HRL."""
        if explore and np.random.rand() < self.epsilon:
            l_star = np.random.randint(0, self.n_tiers)
            n_star = np.random.randint(0, M_TIERS[l_star])
            alpha = np.random.rand()
        else:
            # Tier selection
            tier_qs = self._forward(state, self.dqn_tier)
            l_star = int(np.argmax(tier_qs))

            # Node selection
            node_qs = self._forward(state, self.dqn_node)
            n_star = int(np.argmax(node_qs[:M_TIERS[l_star]]))
            n_star = min(n_star, M_TIERS[l_star] - 1)

            # Ratio regression
            ratio_qs = self._forward(state, self.dqn_ratio)
            alpha_idx = int(np.argmax(ratio_qs))
            alpha = (alpha_idx + 1) / 10.0

        return l_star, n_star, alpha

    def train(self, env: TNTNEnvironment) -> TrainingMetrics:
        """Train the classical HRL agent."""
        print("Starting Classical HRL training...")

        for episode in range(self.n_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_latencies = []
            done = False
            step = 0

            while not done:
                task = env._generate_task()
                l_star, n_star, alpha = self.select_action(state, task, env, explore=True)
                s_next, reward, done, info = env.step(l_star, n_star, alpha, task)

                self.replay.push(state, l_star, n_star, alpha, reward, reward, s_next)
                episode_reward += reward
                episode_latencies.append(info['latency'])

                # Off-policy TD learning every few steps
                if step % 4 == 0:
                    self._learn(batch_size=64)

                state = s_next
                step += 1

            # Periodically sync target networks (DQN-style frozen targets)
            if episode % 5 == 0:
                self.target_tier = [w.copy() for w in self.dqn_tier]
                self.target_node = [w.copy() for w in self.dqn_node]
                self.target_ratio = [w.copy() for w in self.dqn_ratio]

            self.epsilon = max(0.05, self.epsilon * 0.99)
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_latencies.append(np.mean(episode_latencies))

            if episode % 20 == 0:
                print(f"  Episode {episode:4d}: reward={episode_reward:.2f}, "
                      f"lat={np.mean(episode_latencies):.4f}s")

        return self.metrics


class SingleDQNAgent:
    """Flat single-DQN baseline with a joint (tier, node, ratio) action space.

    One network maps the state to Q-values over all valid composite actions,
    illustrating the parameter blow-up of a non-hierarchical agent.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128,
                 n_episodes: int = 500, n_ratio_bins: int = 10, seed: int = 42):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_episodes = n_episodes
        self.n_ratio_bins = n_ratio_bins
        np.random.seed(seed); random.seed(seed)
        # Enumerate composite actions: (tier, node-within-tier) x ratio bin
        self.actions = []
        for l in range(N_TIERS):
            for nd in range(int(M_TIERS[l])):
                for rb in range(n_ratio_bins):
                    self.actions.append((l, nd, (rb + 1) / n_ratio_bins))
        self.n_actions = len(self.actions)
        self.epsilon = 1.0
        np.random.seed(seed)
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, self.n_actions) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(self.n_actions)
        self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
        self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()
        self.replay = ReplayBuffer(capacity=10000)
        self.metrics = TrainingMetrics()

    def _fwd(self, s, target=False):
        W1, b1, W2, b2 = ((self.tW1, self.tb1, self.tW2, self.tb2) if target
                          else (self.W1, self.b1, self.W2, self.b2))
        h = np.maximum(0, s @ W1 + b1)
        return h @ W2 + b2

    def select_action(self, state, task, env, explore=False):
        if explore and np.random.rand() < self.epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            idx = int(np.argmax(self._fwd(state)))
        return self.actions[idx]

    def _learn(self, batch_size=64):
        if len(self.replay) < batch_size:
            return
        s, l_star, n_star, alpha, R1, R2, s_next = self.replay.sample(batch_size)
        s = np.array(s); s_next = np.array(s_next); R = np.array(R1, dtype=float)
        gamma = 0.95
        y = R + gamma * np.array([self._fwd(sn, target=True).max() for sn in s_next])
        # map (l,n,alpha) back to composite action index
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        for i in range(batch_size):
            rb = int(np.clip(round(alpha[i] * self.n_ratio_bins) - 1, 0, self.n_ratio_bins - 1))
            try:
                a = self.actions.index((int(l_star[i]), int(n_star[i]), (rb + 1) / self.n_ratio_bins))
            except ValueError:
                continue
            z1 = s[i] @ self.W1 + self.b1
            h = np.maximum(0, z1)
            q = h @ self.W2 + self.b2
            dq = np.zeros_like(q)
            dq[a] = 2.0 * (q[a] - y[i]) / batch_size
            gW2 += np.outer(h, dq); gb2 += dq
            dh = self.W2 @ dq; dh[z1 <= 0] = 0.0
            gW1 += np.outer(s[i], dh); gb1 += dh
        for g in (gW1, gb1, gW2, gb2):
            np.clip(g, -1.0, 1.0, out=g)
        self.W1 -= 1e-3 * gW1; self.b1 -= 1e-3 * gb1
        self.W2 -= 1e-3 * gW2; self.b2 -= 1e-3 * gb2

    def train(self, env: TNTNEnvironment) -> TrainingMetrics:
        print("Starting Single-DQN training...")
        for episode in range(self.n_episodes):
            state = env.reset(); done = False; step = 0
            ep_reward = 0.0; lats = []
            while not done:
                task = env._generate_task()
                l, n, a = self.select_action(state, task, env, explore=True)
                s_next, r, done, info = env.step(l, n, a, task)
                self.replay.push(state, l, n, a, r, r, s_next)
                ep_reward += r; lats.append(info['latency'])
                if step % 4 == 0:
                    self._learn()
                state = s_next; step += 1
            if episode % 5 == 0:
                self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
                self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()
            self.epsilon = max(0.05, self.epsilon * 0.99)
            self.metrics.episode_rewards.append(ep_reward)
            self.metrics.episode_latencies.append(np.mean(lats))
        return self.metrics


class RandomAgent:
    """Random action baseline."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def select_action(self, state, task, env, explore=False):
        l_star = np.random.randint(0, N_TIERS)
        n_star = np.random.randint(0, M_TIERS[l_star])
        alpha = np.random.rand()
        return l_star, n_star, alpha

    def train(self, env, n_episodes=100):
        print("Running Random baseline...")
        metrics = TrainingMetrics()
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            ep_reward = 0.0
            lats = []
            ens = []
            while not done:
                task = env._generate_task()
                l, n, a = self.select_action(state, task, env)
                _, r, done, info = env.step(l, n, a, task)
                ep_reward += r
                lats.append(info['latency'])
                ens.append(info['energy'])
                state = env._get_state()
            metrics.episode_rewards.append(ep_reward)
            metrics.episode_latencies.append(np.mean(lats))
            metrics.episode_energies.append(np.mean(ens))
        return metrics


class GreedyAgent:
    """Greedy baseline: always select nearest RSU with alpha=1."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def select_action(self, state, task, env, explore=False):
        return 0, 0, 1.0  # Always RSU, first node, full offload

    def train(self, env, n_episodes=100):
        print("Running Greedy baseline...")
        metrics = TrainingMetrics()
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            ep_reward = 0.0
            lats = []
            ens = []
            while not done:
                task = env._generate_task()
                l, n, a = self.select_action(state, task, env)
                _, r, done, info = env.step(l, n, a, task)
                ep_reward += r
                lats.append(info['latency'])
                ens.append(info['energy'])
                state = env._get_state()
            metrics.episode_rewards.append(ep_reward)
            metrics.episode_latencies.append(np.mean(lats))
            metrics.episode_energies.append(np.mean(ens))
        return metrics
