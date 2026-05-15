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
        verbose: bool = False,
    ):
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
        self.vqc = VQCAgent(
            state_dim=state_dim,
            n_layers=vqc_layers,
            n_shots=1000,
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
        # Step 1: VQC high-level policy
        if explore and np.random.rand() < self.epsilon:
            # Exploration: random tier and alpha
            l_star = np.random.randint(0, N_TIERS)
            alpha = np.random.rand()
        else:
            # Exploitation: VQC forward pass
            l_star, alpha, _ = self.vqc.forward(state)

        # Step 2: QAOA node selection within selected tier
        m_tier = M_TIERS[l_star]

        if explore and np.random.rand() < self.epsilon:
            # Random node selection during exploration
            n_star = random_node_selection(m_tier)
        else:
            # QAOA node selection
            costs = env.get_qubo_costs(l_star, alpha, task)

            if self.use_quantum:
                n_star, _, qaoa_info = self.qaoa_solvers[l_star].solve(
                    costs, penalty=50.0, n_iterations=50
                )
            else:
                # Classical fallback
                n_star = classical_node_selection(costs)

        return l_star, n_star, alpha

    def update_vqc(self) -> float:
        """Update VQC parameters via PSR gradient + BO (Algorithm 1, Step 4).

        Uses Bellman TD target r + gamma*V(s') then runs one BO step to
        search for better parameter vectors beyond the local PSR gradient.
        """
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = self.replay.sample(self.batch_size)
        states, l_stars, n_stars, alphas, R1s, R2s, next_states = batch

        # --- Step 1: Compute proper Bellman TD targets ---
        targets = []
        for i in range(len(states)):
            s_next = next_states[i]
            # Bootstrap from target network (frozen copy) for stability
            old_params = self.vqc.get_params().copy()
            self.vqc.set_params(
                self.vqc_params_target.reshape(self.vqc.params.shape)
            )
            _, _, next_outputs = self.vqc.forward(s_next)
            self.vqc.set_params(old_params)
            next_value = float(np.max(next_outputs[:self.vqc.n_tiers]))
            targets.append(float(R1s[i]) + self.gamma_mdp * next_value)

        # --- Step 2: PSR gradient steps on the batch ---
        losses = []
        for i in range(len(states)):
            loss = self.vqc.train_step(states[i], targets[i])
            losses.append(loss)
        mean_loss = float(np.mean(losses)) if losses else 0.0

        # --- Step 3: BO proposes a candidate parameter vector ---
        # Seed BO with the current (PSR-updated) params on first call
        if len(self.bo_vqc.X_history) == 0:
            self.bo_vqc.report(self.vqc.get_params().flatten(), -mean_loss)

        bo_candidate = self.bo_vqc.suggest()
        old_params = self.vqc.get_params().copy()
        self.vqc.set_params(bo_candidate.reshape(self.vqc.params.shape))

        # Evaluate BO candidate loss on a small subset (keeps overhead low)
        n_eval = min(len(states), 8)
        bo_loss = 0.0
        for i in range(n_eval):
            _, _, outputs = self.vqc.forward(states[i])
            tier_logits = outputs[:self.vqc.n_tiers]
            tier_target_vec = np.zeros(self.vqc.n_tiers)
            tier_target_vec[int(l_stars[i])] = targets[i]
            bo_loss += float(np.mean((tier_logits / 2.0 + 0.5 - tier_target_vec) ** 2))
        bo_loss /= n_eval

        self.bo_vqc.report(bo_candidate, -bo_loss)  # maximise negative loss

        # Keep whichever is better: BO candidate or PSR result
        best_bo_params, best_bo_val = self.bo_vqc.get_best()
        if best_bo_params is not None and (-best_bo_val) < mean_loss:
            self.vqc.set_params(best_bo_params.reshape(self.vqc.params.shape))
        else:
            self.vqc.set_params(old_params)  # revert to PSR result

        # Soft-update target network
        self.vqc_params_target = self.vqc.get_params().copy()

        return mean_loss

    def update_qaoa(self) -> Dict[int, float]:
        """Update QAOA angles via BO on cached Ising energy (Algorithm 1, Step 5).

        For each tier, the BO proposes new (gamma, beta) angle vectors and
        evaluates them against the last Ising instance seen by that solver.
        The solver's angles are updated whenever BO finds an improvement.
        """
        if len(self.replay) < self.batch_size:
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
                if len(self.replay) >= self.batch_size and step % self.bo_budget == 0:
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

                state = s_next
                step += 1

            self.epsilon = max(0.05, self.epsilon * 0.99)
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.episode_latencies.append(np.mean(episode_latencies))

            if episode % 20 == 0:
                print(f"  Episode {episode:4d}: reward={episode_reward:.2f}, "
                      f"lat={np.mean(episode_latencies):.4f}s")

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
