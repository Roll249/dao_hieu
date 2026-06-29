"""QAOA node selector: binary variables -> QUBO -> Ising -> QAOA.

The simulator is a compact NumPy statevector implementation.  Keeping it here
avoids coupling the experiment to a cloud backend and makes every QUBO/Ising
coefficient directly testable.
"""
import numpy as np

MAX_QUBITS = 6
PENALTY = 5.0
P = 1
GAMMA_GRID = np.linspace(0.05, np.pi, 9)
BETA_GRID = np.linspace(0.05, np.pi / 2, 7)


def build_qubo(costs, penalty=PENALTY):
    """Return upper-triangular Q for min x'Qx with one-hot penalty."""
    costs = np.asarray(costs, dtype=float)
    n = costs.size
    Q = np.zeros((n, n), dtype=float)
    np.fill_diagonal(Q, costs - penalty)
    for i in range(n):
        Q[i, i + 1:] = 2.0 * penalty
    return Q


def qubo_energy(Q, bits):
    bits = np.asarray(bits, dtype=float)
    return float(bits @ Q @ bits)


def qubo_to_ising(Q):
    """Map x=(1-Z)/2 and return h, J, offset for H_C."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        offset += Q[i, i] / 2.0
        h[i] -= Q[i, i] / 2.0
        for j in range(i + 1, n):
            q = Q[i, j]
            offset += q / 4.0
            h[i] -= q / 4.0
            h[j] -= q / 4.0
            J[i, j] = q / 4.0
    return h, J, float(offset)


def ising_energies(h, J, offset):
    n = len(h)
    indices = np.arange(2 ** n, dtype=np.uint64)
    bits = ((indices[:, None] >> np.arange(n - 1, -1, -1, dtype=np.uint64)) & 1).astype(float)
    z = 1.0 - 2.0 * bits
    energies = offset + z @ h
    for i in range(n):
        for j in range(i + 1, n):
            energies += J[i, j] * z[:, i] * z[:, j]
    return energies, bits


def _apply_rx_all(state, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    out = state.copy()
    for qubit in range(n):
        stride = 2 ** (n - qubit - 1)
        block = 2 * stride
        for start in range(0, out.size, block):
            a = out[start:start + stride].copy()
            b = out[start + stride:start + block].copy()
            out[start:start + stride] = c * a + s * b
            out[start + stride:start + block] = s * a + c * b
    return out


def qaoa_probabilities(Q, gamma, beta):
    h, J, offset = qubo_to_ising(Q)
    energies, _ = ising_energies(h, J, offset)
    state = np.ones(energies.size, dtype=complex) / np.sqrt(energies.size)
    state *= np.exp(-1j * gamma * energies)
    state = _apply_rx_all(state, beta, Q.shape[0])
    return np.abs(state) ** 2


class QAOANodeSelector:
    def __init__(self):
        self.cache = {}
        self.last_angles = None

    def select(self, feasible_idx, costs):
        feasible_idx = list(map(int, feasible_idx))
        if not feasible_idx:
            return None
        if len(feasible_idx) == 1:
            return feasible_idx[0]
        costs = np.asarray(costs, dtype=float)
        if len(feasible_idx) > MAX_QUBITS:
            keep = np.argsort(costs)[:MAX_QUBITS]
            feasible_idx = [feasible_idx[i] for i in keep]
            costs = costs[keep]
        scaled = costs - costs.min()
        if scaled.max() > 0:
            scaled /= scaled.max()
        key = (tuple(feasible_idx), tuple(np.round(scaled, 6)))
        if key in self.cache:
            node, angles = self.cache[key]
            self.last_angles = angles
            return node

        Q = build_qubo(scaled)
        h, J, offset = qubo_to_ising(Q)
        energies, _ = ising_energies(h, J, offset)
        best = None
        for gamma in GAMMA_GRID:
            for beta in BETA_GRID:
                probs = qaoa_probabilities(Q, gamma, beta)
                expectation = float(probs @ energies)
                if best is None or expectation < best[0]:
                    best = (expectation, float(gamma), float(beta), probs)
        _, gamma, beta, probs = best
        one_hot_indices = [1 << (len(feasible_idx) - 1 - i) for i in range(len(feasible_idx))]
        local = int(np.argmax(probs[one_hot_indices]))
        node = feasible_idx[local]
        self.last_angles = (gamma, beta)
        self.cache[key] = (node, self.last_angles)
        return node


_DEFAULT = QAOANodeSelector()


def qaoa_select(feasible_idx, costs, seed=0):
    del seed
    return _DEFAULT.select(feasible_idx, costs)


def n_trainable_params():
    return 2 * P
