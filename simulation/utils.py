"""Shared utilities for the Quantum-HRL simulation framework.

All paper-facing constants are sourced from ``config.py`` (single source of
truth). This module keeps the physics helpers plus a few derived aliases.
"""

import random
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.stats import norm as _scipy_norm

import config as _cfg

# ============================================================================
# CONSTANTS — imported from config.py (do NOT redefine here)
# ============================================================================
CLIGHT = 3e8          # Speed of light (m/s)

ETA0 = _cfg.ETA0
DELTA_PER_TIER = _cfg.DELTA_PER_TIER
B_TIER = list(_cfg.TIER_BANDWIDTH_HZ)          # RSU, LAP, HAP, LEO
N0 = _cfg.N0
PK_VEHICLE = _cfg.PK_VEHICLE
P_RX = _cfg.P_RX
FLOC_VEHICLE = _cfg.FLOC_VEHICLE
EPS_LOCAL = _cfg.EPS_LOCAL
EPS_EDGE_PER_TIER = _cfg.EPS_EDGE_PER_TIER
W_VU = _cfg.W_VU
W_EN = _cfg.W_EN

# --- Legacy constants kept for backward compatibility ---
KAPPA = 1e-27         # (no longer used in main formulas)
SIGMA2 = 1e-11        # (no longer used in main formulas)

# Bandwidth aliases (kept for backward compat)
B_RSU = B_TIER[0]
B_LAP = B_TIER[1]
B_HAP = B_TIER[2]
B_LEO = B_TIER[3]

# Per-tier edge compute frequencies (cycles/s), keyed by tier name
F_EDGE = {name: freq for name, freq in
          zip(_cfg.TIER_NAMES, _cfg.TIER_COMPUTE_HZ)}

# Tier names for indexing
TIER_NAMES = list(_cfg.TIER_NAMES)
TIER_LABELS = list(_cfg.TIER_NAMES)
N_TIERS = _cfg.N_TIERS

# Per-tier node counts
M_TIERS = np.array(_cfg.M_TIERS, dtype=int)  # M1, M2, M3, M4
M_TOTAL = int(M_TIERS.sum())

# State dimension
STATE_DIM = _cfg.STATE_DIM


# =============================================================================
# STATE NORMALIZATION
# =============================================================================

def normalize_state(s: np.ndarray) -> np.ndarray:
    """L2-normalize state vector for amplitude encoding (Corollary 3.1)."""
    norm = np.linalg.norm(s)
    if norm < 1e-12:
        return np.zeros_like(s)
    return s / norm


def normalize_state_components(s: np.ndarray) -> np.ndarray:
    """Component-wise normalization with clipping for stability."""
    s_norm = s.copy()
    s_norm = np.clip(s_norm, -10, 10)
    s_min = s_norm.min()
    s_max = s_norm.max()
    if s_max - s_min > 1e-12:
        s_norm = (s_norm - s_min) / (s_max - s_min)
    return s_norm


# =============================================================================
# UNIFIED CHANNEL MODEL  (Section 4.2, Shinde & Tarchi 2024)
#   h_{k,e}(i) = eta_0 * d_{k,e}(i)^{-delta}
# =============================================================================

def compute_channel_gain(tier_idx: int, d_km: float) -> float:
    """Unified channel gain for ALL tiers (Eq. 18 unified form).

    h = ETA0 * d^{-delta}   where delta varies by link type.

    Args:
        tier_idx: 0=RSU, 1=LAP, 2=HAP, 3=LEO
        d_km:     3-D distance from vehicle to node (km)

    Returns:
        Channel gain (linear, dimensionless).
    """
    d_m = max(d_km * 1000.0, 1.0)   # convert to metres, enforce >= 1 m
    delta = DELTA_PER_TIER[tier_idx]
    return ETA0 * d_m ** (-delta)


def get_bandwidth(tier_idx: int) -> float:
    """Get bandwidth for a tier (Hz)."""
    return B_TIER[tier_idx]


def get_prop_delay(tier_idx: int) -> float:
    """Get one-way propagation delay for a tier (s)."""
    tier_name = TIER_NAMES[tier_idx]
    return PROP_DELAY[tier_name]


def shannon_rate(bandwidth: float, snr_linear: float) -> float:
    """Shannon capacity (Eq. 17).  R = B * log2(1 + SNR)."""
    return bandwidth * np.log2(1.0 + np.clip(snr_linear, 0, 1e9))


# =============================================================================
# LATENCY MODEL — PARALLEL offloading  (Section 4.3, Shinde & Tarchi 2024)
#
#   T_off = T_tx + T_w + T_c_edge + T_rx   (T_w ≈ 0, T_rx = T_tx)
#   T_loc = (1-alpha) * c_k / f_{v,k}
#   T_total = max(T_off, T_loc)             ← parallel, NOT sum
# =============================================================================

def compute_latency(
    d_bits: float,
    c_cycles: float,
    alpha: float,
    tier_idx: int,
    node_idx: int,
    env_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute total task latency with parallel offloading (max model).

    T_k = max(T_off, T_loc)
    where
      T_off = alpha * d_bits / R  +  alpha * c_k / f_edge  +  alpha * d_bits / R
            = T_tx + T_c_edge + T_rx
      T_loc = (1 - alpha) * c_k / f_loc
    """
    tier_name = TIER_NAMES[tier_idx]
    bandwidth = get_bandwidth(tier_idx)

    # Channel gain from environment state (pre-computed)
    g = env_state['channel_gains'][tier_idx, node_idx]
    snr = PK_VEHICLE * g / N0
    R = shannon_rate(bandwidth, snr)
    R = max(R, 1.0)   # guard against zero rate

    # Offloading branch
    t_tx = alpha * d_bits / R
    t_rx = t_tx                              # symmetric channel
    t_c_edge = alpha * c_cycles / F_EDGE[tier_name]
    # Queueing/waiting time: proportional to the target node's utilisation.
    # Zero unless the environment supplies per-node loads (heavy-load regime),
    # so the light-load results are unchanged.
    t_w = 0.0
    loads = env_state.get('node_loads') if isinstance(env_state, dict) else None
    if loads is not None:
        load = float(loads[tier_idx, node_idx])
        t_w = _cfg.QUEUE_BASE_S * (load ** 2) if alpha > 0 else 0.0
    t_off = t_tx + t_w + t_c_edge + t_rx

    # Local branch
    t_c_local = c_cycles / FLOC_VEHICLE
    t_loc = (1.0 - alpha) * t_c_local

    # Parallel processing → max
    total_latency = max(t_off, t_loc)

    components = {
        't_tx': t_tx,
        't_rx': t_rx,
        't_c_edge': t_c_edge,
        't_off': t_off,
        't_c_local': t_c_local,
        't_loc': t_loc,
        't_total': total_latency,
    }
    return total_latency, components


# =============================================================================
# ENERGY MODEL — includes EN energy  (Section 4.3, Shinde & Tarchi 2024)
#
#   E = w_k*(E_tx + E_rx) + E_local + w_e*(E_wait + E_c_edge)
#   E_tx   = P_k * T_tx
#   E_rx   = P_rx * T_rx
#   E_local = (1-alpha) * c_k * eps_local
#   E_c_edge = alpha * c_k * eps_edge[tier]
# =============================================================================

def compute_energy(
    d_bits: float,
    c_cycles: float,
    alpha: float,
    tier_idx: int,
    node_idx: int,
    env_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute total energy consumption with EN energy (Eq. 22 corrected).

    E = W_VU*(E_tx + E_rx) + E_local + W_EN*E_c_edge
    """
    tier_name = TIER_NAMES[tier_idx]
    bandwidth = get_bandwidth(tier_idx)

    g = env_state['channel_gains'][tier_idx, node_idx]
    snr = PK_VEHICLE * g / N0
    R = shannon_rate(bandwidth, snr)
    R = max(R, 1.0)

    t_tx = alpha * d_bits / R
    t_rx = t_tx

    e_tx = PK_VEHICLE * t_tx           # Vehicle transmit energy
    e_rx = P_RX * t_rx                 # Vehicle receive energy
    e_local = (1.0 - alpha) * c_cycles * EPS_LOCAL           # Local computation energy
    e_c_edge = alpha * c_cycles * EPS_EDGE_PER_TIER[tier_idx] # EN computation energy
    # E_wait simplified to 0 for light load

    total_energy = W_VU * (e_tx + e_rx) + e_local + W_EN * e_c_edge

    components = {
        'e_tx': e_tx,
        'e_rx': e_rx,
        'e_local': e_local,
        'e_c_edge': e_c_edge,
        'e_total': total_energy,
    }
    return total_energy, components


# =============================================================================
# REWARD COMPUTATION (Section 4.6)
#
#   F3 = 1  if  E > w_k * E_local_full  else 0
#   where E_local_full = c_k * eps_local  (energy if whole task is local)
# =============================================================================

def compute_reward(
    latency: float,
    energy: float,
    Tsoj: float,
    Tmax: float,
    c_cycles: float,           # task workload (replaces Emax)
    beta1: float = _cfg.BETA1,
    beta2: float = _cfg.BETA2,
    w1: float = _cfg.W_F1,
    w2: float = _cfg.W_F2,
    w3: float = _cfg.W_F3,
) -> Tuple[float, Dict[str, int]]:
    """Compute reward (Eq. 32).

    R_t = -(beta1*T_k + beta2*E_k) - w1*F1 - w2*F2 - w3*F3

    F1 = 1 if latency > Tsoj (sojourn constraint violated)
    F2 = 1 if latency > Tmax (deadline constraint violated)
    F3 = 1 if energy > w_k * E_local_full (energy exceeds local-only baseline)
    """
    F1 = 1 if latency > Tsoj else 0
    F2 = 1 if latency > Tmax else 0

    # F3: compare against local-only energy baseline (not an arbitrary budget)
    e_local_full = W_VU * (c_cycles * EPS_LOCAL)
    F3 = 1 if energy > e_local_full else 0

    base_reward = -(beta1 * latency + beta2 * energy)
    penalty = w1 * F1 + w2 * F2 + w3 * F3

    reward = base_reward - penalty

    flags = {'F1': F1, 'F2': F2, 'F3': F3}
    return reward, flags


# =============================================================================
# QUBO COEFFICIENT NORMALIZATION
# =============================================================================

def normalize_qubo_coefficients(costs: np.ndarray, penalty_A: float) -> np.ndarray:
    """Normalize QUBO costs to [0, 1] for stable Ising mapping."""
    c_min = costs.min()
    c_max = costs.max()
    c_range = c_max - c_min
    if c_range < 1e-12:
        return np.zeros_like(costs)
    return (costs - c_min) / c_range


# =============================================================================
# BAYESIAN OPTIMIZATION WRAPPER
# =============================================================================

class BayesianOptimizer:
    """Wrapper for Bayesian Optimization using scikit-optimize."""

    def __init__(
        self,
        param_bounds: list,
        n_initial: int = 5,
        n_iterations: int = 20,
        noise_std: float = 0.01,
    ):
        self.param_bounds = param_bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.noise_std = noise_std
        self.X_history = []
        self.y_history = []
        self.gp_model = None
        # Sliding window over observations keeps each GP fit O(1) in wall-clock
        # (a standard moving-window GP surrogate); avoids the cubic blow-up of
        # an ever-growing design matrix without changing the optimisation logic.
        self.max_history = 40
        self.n_acquisition_samples = 24

    def _build_gp(self):
        """Build Gaussian Process surrogate model."""
        if len(self.X_history) < 2:
            return None
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X = np.array(self.X_history)
        y = np.array(self.y_history).ravel()

        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.noise_std ** 2)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            random_state=42,
            normalize_y=True,
        )
        gp.fit(X, y)
        return gp

    def _expected_improvement(self, gp, X_query: np.ndarray, xi: float = 0.01) -> float:
        """Compute Expected Improvement acquisition function."""
        mu, sigma = gp.predict(X_query.reshape(1, -1), return_std=True)
        mu = float(np.ravel(mu)[0])
        sigma = float(np.ravel(sigma)[0])
        sigma = max(sigma, 1e-9)

        f_best = max(self.y_history)
        with np.errstate(divide='ignore'):
            z = (mu - f_best - xi) / sigma
            ei = (mu - f_best - xi) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            if sigma < 1e-9:
                ei = 0.0
        return float(ei)

    @staticmethod
    def _norm_cdf(x):
        return _scipy_norm.cdf(x)

    @staticmethod
    def _norm_pdf(x):
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def _acquisition_optimize(self, gp) -> np.ndarray:
        """Maximize EI over parameter space."""
        best_ei = -np.inf
        best_x = None
        for _ in range(self.n_acquisition_samples):
            x = np.array([
                np.random.uniform(b[0], b[1]) for b in self.param_bounds
            ])
            ei = self._expected_improvement(gp, x)
            if ei > best_ei:
                best_ei = ei
                best_x = x
        return best_x

    def suggest(self) -> np.ndarray:
        """Suggest next query point based on EI."""
        if len(self.X_history) < self.n_initial:
            return np.array([
                np.random.uniform(b[0], b[1]) for b in self.param_bounds
            ])

        gp = self._build_gp()
        if gp is None:
            return np.array([
                np.random.uniform(b[0], b[1]) for b in self.param_bounds
            ])

        return self._acquisition_optimize(gp)

    def report(self, x: np.ndarray, y: float):
        """Record observation (retaining only the most recent ``max_history``)."""
        self.X_history.append(x.tolist())
        self.y_history.append(float(y))
        if len(self.X_history) > self.max_history:
            self.X_history = self.X_history[-self.max_history:]
            self.y_history = self.y_history[-self.max_history:]

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Return best observed point and value."""
        if not self.y_history:
            return None, -np.inf
        idx = int(np.argmax(self.y_history))
        return np.array(self.X_history[idx]), float(self.y_history[idx])


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        s: np.ndarray,
        l_star: int,
        n_star: int,
        alpha: float,
        R1: float,
        R2: float,
        s_next: np.ndarray,
    ):
        """Store a transition tuple."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, l_star, n_star, alpha, R1, R2, s_next))
        else:
            self.buffer[self.position] = (s, l_star, n_star, alpha, R1, R2, s_next)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        """Sample a random mini-batch."""
        return list(zip(*random.choices(self.buffer, k=batch_size)))

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# PARAMETER COUNT UTILITIES
# =============================================================================

def classical_hrl_params(n: int, h: int, n_actions: list, n_layers: int = 2) -> int:
    """Compute DQN parameter count (Eq. 46).

    W_DQN = n*h + (n_layers-2)*h^2 + h*sum(n_actions)
    """
    return n * h + (n_layers - 2) * h * h + h * sum(n_actions)


def vqc_params(L: int, q: int) -> int:
    """Compute VQC parameter count (Eq. 49). W_VQC = L * q."""
    return L * q


def qaoa_params(p: int) -> int:
    """Compute QAOA parameter count (Eq. 50). W_QAOA = 2p."""
    return 2 * p
