"""QAOA Solver for Node Selection.

Implements:
  - QUBO formulation for node selection (Section 4.5, Eq. 27)
  - QUBO-to-Ising Hamiltonian mapping (Section 3.5, Eq. 13)
  - QAOA circuit (Section 3.6, Definition 3.6, Eq. 14)
  - Optimization of QAOA angles (gamma, beta)

The QAOA replaces the classical flat Q-value lookup over M_l discrete actions,
providing near-optimal combinatorial node selection.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple, Optional, Dict, List
try:
    import sympy as sp   # optional; not required at runtime
except ModuleNotFoundError:
    sp = None


# ============================================================================
# QUBO-TO-ISING MAPPING (with CORRECTED signs from peer review)
# ============================================================================

def qubo_to_ising(q: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO to Ising Hamiltonian (Section 3.5, Proposition 3.5).

    QUBO: H(z) = sum_n c_n z_n + A*(sum_n z_n - 1)^2
    Expanding with z_n^2 = z_n (binary):
      H(z) = sum_n (c_n - A)*z_n + 2A*sum_{i<j} z_i z_j + A

    Substituting z_n = (1 - sigma_z^n)/2  [sigma_z=+1 -> z=0, sigma_z=-1 -> z=1]:
      h_n  = -c_n/2 + A*(1 - M/2)   (local field)
      J_ij = +A/2                    (POSITIVE coupling — penalises co-selection)
      E0   = sum_n c_n/2 + A*(1 - M/2 + M*(M-1)/4)

    NOTE: Paper Eq.(28) J_ij = +A/4 is wrong magnitude.
          PAPER_IMPROVEMENTS.md correction of -A/2 is wrong sign AND magnitude.
          Verified analytically for M=2,3,5 against direct QUBO evaluation.

    Args:
        q: QUBO matrix (m x m) from build_qubo_matrix() — used to recover A
        c: Original node cost vector (m,) before penalty embedding

    Returns:
        h:  Local field array (m,)
        J:  Coupling matrix (m x m), symmetric
        E0: Constant energy offset
    """
    m = len(c)
    h = np.zeros(m)
    J = np.zeros((m, m))

    # Recover penalty A from QUBO off-diagonal (Q_ij = 2A)
    A = float(q[0, 1]) / 2.0 if m >= 2 else 0.0

    # Use original costs directly (no normalisation — would break absolute E0)
    for n in range(m):
        h[n] = -float(c[n]) / 2.0 + A * (1.0 - m / 2.0)

    for i in range(m):
        for j in range(i + 1, m):
            J[i, j] = A / 2.0
            J[j, i] = A / 2.0

    E0 = float(np.sum(c)) / 2.0 + A * (1.0 - m / 2.0 + m * (m - 1) / 4.0)

    return h, J, E0


def build_qubo_matrix(
    costs: np.ndarray,
    penalty: float = 50.0,
) -> np.ndarray:
    """Build the QUBO matrix for node selection (Section 4.5, Eq. 27).

    H(z) = sum_n c_n z_n + A * (sum_n z_n - 1)^2

    This encodes:
      1. Linear costs c_n for selecting each node
      2. One-hot constraint: exactly one node must be selected
    """
    m = len(costs)
    Q = np.zeros((m, m))

    # Diagonal: expanding A*(Σz_n-1)² gives coefficient (c_n - A) on each z_n
    # because z_n²=z_n collapses the squared term: -A per node from the penalty
    for n in range(m):
        Q[n, n] = costs[n] - penalty  # c_n - A (not +A — see one-hot expansion)

    # Off-diagonal: 2*A from -2*A*z_i*z_j
    for i in range(m):
        for j in range(i + 1, m):
            Q[i, j] = 2.0 * penalty
            Q[j, i] = 2.0 * penalty

    return Q


# ============================================================================
# QAOA CIRCUIT
# ============================================================================

def build_qaoa_cost_hamiltonian(h: np.ndarray, J: np.ndarray) -> qml.Hamiltonian:
    """Build the QAOA cost Hamiltonian from Ising parameters.

    H_C = sum_n h_n sigma_z^n + sum_{i<j} J_ij sigma_z^i sigma_z^j
    """
    m = len(h)
    coeffs = []
    observables = []

    # Local field terms
    for n in range(m):
        if abs(h[n]) > 1e-12:
            coeffs.append(h[n])
            observables.append(qml.PauliZ(n))

    # Coupling terms
    for i in range(m):
        for j in range(i + 1, m):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(J[i, j])
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, observables)


def build_qaoa_mixer(m: int) -> qml.Hamiltonian:
    """Build the QAOA mixer Hamiltonian.

    H_M = sum_n sigma_x^n  (transverse field mixer)
    """
    coeffs = [1.0] * m
    observables = [qml.PauliX(n) for n in range(m)]
    return qml.Hamiltonian(coeffs, observables)


def build_qaoa_circuit(m: int, depth: int = 2):
    """Build the QAOA circuit as per paper's Eq. (36).

    |gamma, beta> = prod_{q=1}^p e^{-i beta_q H_M} e^{-i gamma_q H_C} |+>^{otimes m}

    Args:
        m:      Number of qubits (one per node)
        depth:  QAOA depth p

    Returns:
        QNode that evaluates the QAOA energy expectation
    """
    wires = list(range(m))

    def circuit_fn(gamma_beta, h, J):
        """QAOA circuit.

        Args:
            gamma_beta: Array of 2p angles [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            h:         Ising local fields (m,)
            J:         Ising couplings (m, m)

        Returns:
            Expectation value of the cost Hamiltonian
        """
        # Initial superposition (Hadamard on all qubits)
        for wire in wires:
            qml.Hadamard(wires=wire)

        # Build Hamiltonians
        H_C = build_qaoa_cost_hamiltonian(h, J)
        H_M = build_qaoa_mixer(m)

        # Apply p QAOA layers
        for q in range(depth):
            gamma_q = gamma_beta[q]
            beta_q = gamma_beta[depth + q]

            # Cost unitary: exp(-i gamma_q H_C)
            qml.qaoa.cost_layer(gamma_q, H_C)

            # Mixer unitary: exp(-i beta_q H_M)
            qml.qaoa.mixer_layer(beta_q, H_M)

        # Return cost Hamiltonian expectation
        return qml.expval(H_C)

    return circuit_fn


# ============================================================================
# QAOA SOLVER CLASS
# ============================================================================

class QAOASolver:
    """QAOA-based node selector for Quantum-HRL.

    Replaces the flat Q-value lookup over M_l discrete actions with
    a quantum combinatorial optimizer that exploits the problem's
    algebraic structure.
    """

    def __init__(
        self,
        n_nodes: int,
        depth: int = 2,
        n_shots: int = 1000,
        seed: int = 42,
    ):
        self.n_nodes = n_nodes
        self.depth = depth
        self.n_shots = n_shots
        self.rng = np.random.RandomState(seed)

        # Number of QAOA parameters: 2p
        self.n_params = 2 * depth

        # Initialize angles randomly
        self.angles = self.rng.uniform(0, 2 * np.pi, size=self.n_params)
        self.best_angles = self.angles.copy()
        self.best_energy = np.inf

        # Track history
        self.energy_history = []
        self.bitstring_history = []

        # Fallback accounting (urgent_need_to_fixed.md #5): count how often the
        # quantum decode is unusable (flat distribution / exception) and we fall
        # back to the classical min-cost node.
        self.n_extract = 0
        self.n_fallback = 0

        # Cache last Ising params so update_qaoa() can re-evaluate angles
        self.last_h: Optional[np.ndarray] = None
        self.last_J: Optional[np.ndarray] = None

        # Build quantum device. The inner variational loop uses analytic
        # (statevector) expectation values for tractable simulation; finite-shot
        # and hardware-noise effects are studied separately in the noise
        # scenario. ``n_shots`` is retained for reference/reporting.
        self.dev = qml.device('default.qubit', wires=n_nodes, shots=None)

    def build_qnode(self, h: np.ndarray, J: np.ndarray) -> qml.QNode:
        """Build the QAOA QNode for specific Ising parameters."""
        m = self.n_nodes
        wires = list(range(m))

        H_C = build_qaoa_cost_hamiltonian(h, J)
        H_M = build_qaoa_mixer(m)

        @qml.qnode(self.dev, interface='autograd')
        def qaoa_forward(params):
            # Initialize in |+>^{otimes m}
            for wire in wires:
                qml.Hadamard(wires=wire)

            # Apply p QAOA layers
            for q in range(self.depth):
                gamma_q = params[q]
                beta_q = params[self.depth + q]

                qml.qaoa.cost_layer(gamma_q, H_C)
                qml.qaoa.mixer_layer(beta_q, H_M)

            return qml.expval(H_C)

        return qaoa_forward

    def solve(
        self,
        costs: np.ndarray,
        penalty: float = 2.0,
        n_iterations: int = 100,
    ) -> Tuple[int, float, Dict]:
        """Solve the node selection QUBO using QAOA.

        Args:
            costs:   Node costs c_n (lower is better)
            penalty: One-hot constraint penalty A (relative to normalised costs)
            n_iterations: Number of classical optimization iterations

        Returns:
            selected_node: Index of the selected node
            energy:       QAOA energy (lower is better)
            info:         Additional diagnostics
        """
        costs = np.asarray(costs, dtype=float)
        # Normalise costs to [0,1] so the Ising coefficients stay O(1) and the
        # QAOA angle landscape is well-conditioned (node selection only depends
        # on relative costs, so this preserves the optimum / argmin).
        cmin, cmax = float(costs.min()), float(costs.max())
        cnorm = (costs - cmin) / (cmax - cmin) if (cmax - cmin) > 1e-12 else np.zeros_like(costs)

        # Build QUBO matrix on normalised costs
        Q = build_qubo_matrix(cnorm, penalty)

        # Convert to Ising (with correct signs from peer review)
        h, J, E0 = qubo_to_ising(Q, cnorm)

        # Build QNode
        qnode = self.build_qnode(h, J)

        # Optimize angles by per-decision gradient descent. Angles must be a
        # pennylane-trainable array or qml.grad silently returns nothing
        # (autograd needs requires_grad=True), which previously made this loop a
        # no-op (see urgent_need_to_fixed.md #4).
        angles = pnp.array(self.angles.copy(), requires_grad=True)

        energies = []
        for iteration in range(n_iterations):
            try:
                energy = qnode(angles)
            except Exception:
                # Fallback: use classical evaluation
                energy = self._classical_energy(angles, h, J, m=self.n_nodes)

            energies.append(float(energy))

            # Gradient descent update
            try:
                grad = qml.grad(qnode)(angles)
                angles = pnp.array(angles - 0.1 * grad, requires_grad=True)
            except Exception:
                # Random perturbation if gradient fails
                angles = pnp.array(
                    angles + self.rng.uniform(-0.1, 0.1, size=len(angles)),
                    requires_grad=True)

            # Clip to reasonable range
            angles = pnp.array(np.clip(angles, 0, 2 * np.pi), requires_grad=True)

            # Track best
            if float(energy) < self.best_energy:
                self.best_energy = float(energy)
                self.best_angles = angles.copy()

        self.angles = angles
        self.energy_history.append(energies)

        # Cache Ising params for BO-based angle update in update_qaoa()
        self.last_h = h.copy()
        self.last_J = J.copy()

        # Extract solution from the optimised QAOA measurement distribution
        selected_node, fell_back = self._extract_solution(h, J, angles, costs=costs)

        info = {
            'final_energy': float(energy),
            'best_energy': self.best_energy,
            'energy_curve': energies,
            'angles': angles.copy(),
            'h': h,
            'J': J,
            'E0': E0,
            'fell_back': fell_back,
            'fallback_rate': self.fallback_rate(),
        }

        return selected_node, self.best_energy, info

    def fallback_rate(self) -> float:
        """Fraction of node decodes that fell back to the classical heuristic."""
        return self.n_fallback / max(self.n_extract, 1)

    def _extract_solution(self, h: np.ndarray, J: np.ndarray, angles: np.ndarray,
                          costs: Optional[np.ndarray] = None) -> Tuple[int, bool]:
        """Decode the selected node from the QAOA measurement distribution.

        The device is analytic (shots=None), so we read the EXACT measurement
        probability vector (qml.probs) — this is the ideal-measurement QAOA
        readout (finite-shot noise is studied separately in the noise scenario).
        The node is the qubit with the highest marginal selection probability.
        If the distribution carries no signal (flat) or the circuit errors, we
        fall back to the classical min-cost node and flag it.
        """
        m = self.n_nodes
        wires = list(range(m))
        H_C = build_qaoa_cost_hamiltonian(h, J)
        H_M = build_qaoa_mixer(m)

        @qml.qnode(self.dev, interface='autograd')
        def probs_circuit(params):
            for wire in wires:
                qml.Hadamard(wires=wire)
            for q in range(self.depth):
                qml.qaoa.cost_layer(params[q], H_C)
                qml.qaoa.mixer_layer(params[self.depth + q], H_M)
            return qml.probs(wires=wires)

        self.n_extract += 1

        def _classical_node():
            if costs is not None:
                return int(np.argmin(costs))
            return int(np.argmax(h))    # min cost <-> max local field

        try:
            p = np.asarray(probs_circuit(angles), dtype=float).ravel()  # len 2^m
            idx = np.arange(2 ** m)
            # PennyLane orders basis states with the first wire as MSB.
            marg = np.array([
                p[((idx >> (m - 1 - n)) & 1) == 1].sum() for n in range(m)
            ])
            if float(marg.max() - marg.min()) < 1e-6:
                self.n_fallback += 1
                return _classical_node(), True
            return int(np.argmax(marg)), False
        except Exception:
            self.n_fallback += 1
            return _classical_node(), True

    def _classical_energy(self, angles: np.ndarray, h: np.ndarray, J: np.ndarray, m: int) -> float:
        """Classical approximation of QAOA energy.

        For m small, compute exact energy by summing over all 2^m bitstrings.
        """
        if m > 10:
            return 0.0

        best_e = np.inf
        best_z = None

        for z_int in range(2 ** m):
            z = np.array([(z_int >> i) & 1 for i in range(m)], dtype=float)
            e = self._ising_energy(z, h, J)
            if e < best_e:
                best_e = e
                best_z = z

        self.bitstring_history.append(best_z)
        return best_e

    @staticmethod
    def _ising_energy(z: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
        """Compute Ising energy for a given bitstring."""
        m = len(z)
        # z_n = (1 - sigma_z^n) / 2  =>  sigma_z^n = 1 - 2*z_n
        sigma_z = 1.0 - 2.0 * z

        energy = 0.0
        for n in range(m):
            energy += h[n] * sigma_z[n]
        for i in range(m):
            for j in range(i + 1, m):
                energy += J[i, j] * sigma_z[i] * sigma_z[j]

        return energy


# ============================================================================
# CLASSICAL NODE SELECTION (for comparison)
# ============================================================================

def classical_node_selection(costs: np.ndarray) -> int:
    """Classical greedy node selection: pick node with minimum cost.

    This is the classical baseline corresponding to the flat Q-value lookup.
    """
    return int(np.argmin(costs))


def random_node_selection(m: int) -> int:
    """Random node selection (uniform)."""
    return int(np.random.randint(0, m))
