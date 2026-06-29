"""Variational quantum high-level policy used by the scenario experiments.

This module is intentionally independent from the Classical HRL implementation.
It exposes the small subset of the baseline Agent API needed by the scenario
runner (act/store/retrain/align target), so replacing the high-level DQN does
not require editing ``hrl_core.py``.
"""
from collections import deque
import random

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

FEATURE_MAX = np.array([3.0, 2.0, 3.0, 3.0])


class VQCPolicy:
    """Four-qubit VQC with angle encoding and two entangling layers."""

    def __init__(self, n_qubits=4, n_layers=2, n_actions=4, lr=0.05,
                 gamma=0.7, epsilon=0.1, seed=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.experience_replay = deque(maxlen=50000)
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        rng = np.random.default_rng(seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.params = pnp.array(rng.uniform(-0.05, 0.05, size=shape), requires_grad=True)
        self.target_params = pnp.array(self.params, requires_grad=False)
        self.n_params = int(np.prod(shape))
        self.opt = qml.AdamOptimizer(stepsize=lr)
        self.last_loss = float("nan")
        self._cache = {}

        @qml.qnode(self.dev, diff_method="adjoint")
        def circuit(x, params):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_actions)]

        self.circuit = circuit

    def _encode(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)[:self.n_qubits]
        return (state / FEATURE_MAX[:self.n_qubits]) * np.pi

    @staticmethod
    def _key(state):
        return tuple(np.asarray(state, dtype=float).reshape(-1).tolist())

    def q_values(self, state, target=False):
        if target:
            return np.asarray(self.circuit(self._encode(state), self.target_params), dtype=float)
        key = self._key(state)
        if key not in self._cache:
            self._cache[key] = np.asarray(self.circuit(self._encode(state), self.params), dtype=float)
        return self._cache[key].copy()

    def act(self, state):
        # Keep the baseline epsilon value, but use conventional epsilon-greedy
        # because the VQC fully replaces (rather than augments) the DQN policy.
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def store(self, state, action, reward, next_state):
        self.experience_replay.append((np.asarray(state), int(action), float(reward),
                                       np.asarray(next_state)))

    def align_target_model(self):
        self.target_params = pnp.array(self.params, requires_grad=False)

    def retrain(self, batch_size):
        if len(self.experience_replay) < batch_size:
            return self.last_loss
        minibatch = random.sample(self.experience_replay, batch_size)
        encoded = [pnp.array(self._encode(s), requires_grad=False) for s, _, _, _ in minibatch]
        actions = [a for _, a, _, _ in minibatch]
        targets = pnp.array([
            reward + self.gamma * float(np.max(self.q_values(next_state, target=True)))
            for _, _, reward, next_state in minibatch
        ], requires_grad=False)

        def cost(params):
            loss = 0.0
            for x, action, target in zip(encoded, actions, targets):
                loss = loss + (self.circuit(x, params)[action] - target) ** 2
            return loss / len(encoded)

        self.params, loss = self.opt.step_and_cost(cost, self.params)
        self.last_loss = float(loss)
        self._cache.clear()
        return self.last_loss

    def avg_max_q(self, states):
        return float(np.mean([np.max(self.q_values(state)) for state in states]))
