"""Shared utilities for the Quantum-HRL simulation framework."""

import numpy as np
from typing import Tuple, Dict, Any, Optional


# ============================================================================
# CONSTANTS matching the paper
# ============================================================================
CLIGHT = 3e8          # Speed of light (m/s)
D0_RSU = 10.0         # Reference distance for RSU path loss (m)
G0 = 1.0              # Reference gain at d0
DELTA_RSU = 2.5       # Path loss exponent for RSU
D0_LAP = 50.0         # Reference distance for LAP (m)
D0_HAP = 100.0        # Reference distance for HAP/LEO (m)
FC_HAP = 2e9          # Carrier frequency HAP (Hz)
FC_LEO = 12e9         # Carrier frequency LEO (Hz)
B_RSU = 20e6          # Bandwidth RSU (Hz)
B_LAP = 40e6          # Bandwidth LAP (Hz)
B_HAP = 60e6          # Bandwidth HAP (Hz)
B_LEO = 100e6         # Bandwidth LEO (Hz)
KAPPA = 1e-27         # Effective switched capacitance (J/(cycle^3))
PK_VEHICLE = 0.1      # Vehicle transmit power (W)
SIGMA2 = 1e-11        # Noise power (W)
FLOC_VEHICLE = 1e9    # Local CPU frequency (cycles/s)

# Per-tier edge compute frequencies (cycles/s)
F_EDGE = {
    'RSU': 5e9,
    'LAP': 10e9,
    'HAP': 20e9,
    'LEO': 40e9,
}

# Per-tier one-way propagation delays (s)
PROP_DELAY = {
    'RSU': 0.0001,   # ~30 us
    'LAP': 0.001,    # ~1 ms
    'HAP': 0.067,    # ~20 km / c
    'LEO': 0.004,    # ~1200 km / c
}

# Tier names for indexing
TIER_NAMES = ['RSU', 'LAP', 'HAP', 'LEO']
TIER_LABELS = ['RSU', 'LAP', 'HAP', 'LEO']
N_TIERS = len(TIER_NAMES)

# Per-tier node counts
M_TIERS = np.array([5, 3, 2, 2], dtype=int)  # M1, M2, M3, M4
M_TOTAL = int(M_TIERS.sum())

# State dimension
STATE_DIM = 20

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
    # Clip to [-10, 10] to avoid extreme values
    s_norm = np.clip(s_norm, -10, 10)
    # Min-max normalize each component to [0, 1]
    s_min = s_norm.min()
    s_max = s_norm.max()
    if s_max - s_min > 1e-12:
        s_norm = (s_norm - s_min) / (s_max - s_min)
    return s_norm


# =============================================================================
# CHANNEL MODELS (Section 4.2)
# =============================================================================

def channel_rsu(d_km: np.ndarray) -> np.ndarray:
    """RSU ground-to-vehicle path loss (Eq. 18)."""
    d = np.maximum(d_km * 1000.0, D0_RSU)  # Convert to meters, enforce d >= d0
    return G0 * (D0_RSU / d) ** DELTA_RSU


def channel_lap(d_km: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """LAP air-to-ground Rician channel (Eq. 19)."""
    theta = np.deg2rad(theta_deg)
    # Elevation-dependent LoS probability
    p_los = 1.0 / (1.0 + 0.1 * np.exp(-0.1 * (theta_deg - 30)))
    # Mean gains (relative units, will be scaled by SNR)
    g_los = 0.8 + 0.2 * np.cos(theta) ** 2
    g_nlos = 0.3
    return p_los * g_los + (1 - p_los) * g_nlos


def channel_hap_leo(d_km: np.ndarray, fc_hz: float) -> np.ndarray:
    """HAP/LEO free-space path loss (Eq. 20)."""
    d = np.maximum(d_km * 1000.0, D0_HAP)
    wavelength = CLIGHT / fc_hz
    return (wavelength / (4 * np.pi * d)) ** 2


def shannon_rate(bandwidth: float, snr_linear: float) -> float:
    """Shannon capacity (Eq. 17)."""
    return bandwidth * np.log2(1.0 + np.clip(snr_linear, 0, 100))


def compute_channel_gain(tier_idx: int, d_km: float, theta_deg: float) -> float:
    """Compute channel gain for a specific tier."""
    if tier_idx == 0:  # RSU
        return channel_rsu(np.array([d_km]))[0]
    elif tier_idx == 1:  # LAP
        return channel_lap(np.array([d_km]), np.array([theta_deg]))[0]
    elif tier_idx == 2:  # HAP
        return channel_hap_leo(np.array([d_km]), FC_HAP)[0]
    else:  # LEO
        return channel_hap_leo(np.array([d_km]), FC_LEO)[0]


def get_bandwidth(tier_idx: int) -> float:
    """Get bandwidth for a tier."""
    return [B_RSU, B_LAP, B_HAP, B_LEO][tier_idx]


def get_prop_delay(tier_idx: int) -> float:
    """Get one-way propagation delay for a tier."""
    return [PROP_DELAY['RSU'], PROP_DELAY['LAP'], PROP_DELAY['HAP'], PROP_DELAY['LEO']][tier_idx]


# =============================================================================
# LATENCY AND ENERGY MODELS (Section 4.3)
# =============================================================================

def compute_latency(
    d_bits: float,
    c_cycles: float,
    alpha: float,
    tier_idx: int,
    node_idx: int,
    env_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute total task latency (Eq. 21).

    T_k = alpha * d_k / R_k,n + 2*tau_l + alpha * c_k / f_l,n + (1-alpha) * c_k / f_loc
    """
    tier_name = TIER_NAMES[tier_idx]
    bandwidth = get_bandwidth(tier_idx)
    prop_delay = get_prop_delay(tier_idx)

    # Channel gain from environment state
    g = env_state['channel_gains'][tier_idx, node_idx]
    snr = PK_VEHICLE * g / SIGMA2
    R = shannon_rate(bandwidth, snr)

    # Transmission time for offloaded fraction
    t_tx = alpha * d_bits / max(R, 1.0)
    t_proc_edge = alpha * c_cycles / F_EDGE[tier_name]
    t_proc_local = (1.0 - alpha) * c_cycles / FLOC_VEHICLE
    t_rtt = 2.0 * prop_delay

    total_latency = t_tx + t_proc_edge + t_proc_local + t_rtt

    components = {
        't_tx': t_tx,
        't_proc_edge': t_proc_edge,
        't_proc_local': t_proc_local,
        't_rtt': t_rtt,
        't_total': total_latency,
    }
    return total_latency, components


def compute_energy(
    d_bits: float,
    c_cycles: float,
    alpha: float,
    tier_idx: int,
    node_idx: int,
    env_state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Compute total energy consumption (Eq. 22).

    E_k = P_k * alpha * d_k / R_k,n + kappa * [(1-alpha) * c_k]^3
    """
    tier_name = TIER_NAMES[tier_idx]
    bandwidth = get_bandwidth(tier_idx)

    g = env_state['channel_gains'][tier_idx, node_idx]
    snr = PK_VEHICLE * g / SIGMA2
    R = shannon_rate(bandwidth, snr)

    e_tx = PK_VEHICLE * alpha * d_bits / max(R, 1.0)
    e_local = KAPPA * ((1.0 - alpha) * c_cycles) ** 3

    total_energy = e_tx + e_local

    components = {
        'e_tx': e_tx,
        'e_local': e_local,
        'e_total': total_energy,
    }
    return total_energy, components


# =============================================================================
# REWARD COMPUTATION (Section 4.6)
# =============================================================================

def compute_reward(
    latency: float,
    energy: float,
    Tsoj: float,
    Tmax: float,
    Emax: float,
    beta1: float = 1.0,
    beta2: float = 1.0,
    w1: float = 50.0,
    w2: float = 50.0,
    w3: float = 50.0,
) -> Tuple[float, Dict[str, int]]:
    """Compute reward (Eq. 32).

    R_t = -(beta1*T_k + beta2*E_k) - w1*F1 - w2*F2 - w3*F3
    """
    F1 = 1 if latency > Tsoj else 0
    F2 = 1 if latency > Tmax else 0
    F3 = 1 if energy > Emax else 0

    base_reward = -(beta1 * latency + beta2 * energy)
    penalty = w1 * F1 + w2 * F2 + w3 * F3

    reward = base_reward - penalty

    flags = {'F1': F1, 'F2': F2, 'F3': F3}
    return reward, flags


# =============================================================================
# QUBO COEFFICIENT NORMALIZATION
# =============================================================================

def normalize_qubo_coefficients(costs: np.ndarray, penalty_A: float) -> np.ndarray:
    """Normalize QUBO costs to [0, 1] for stable Ising mapping.

    Without normalization, the one-hot penalty A may not dominate properly
    when cost magnitudes differ by orders of magnitude (e.g., latency ~ms vs energy ~J).
    """
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

    def _build_gp(self):
        """Build Gaussian Process surrogate model."""
        if len(self.X_history) < 2:
            return None
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X = np.array(self.X_history)
        y = np.array(self.y_history).reshape(-1, 1)

        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.noise_std ** 2)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        )
        gp.fit(X, y)
        return gp

    def _expected_improvement(self, gp, X_query: np.ndarray, xi: float = 0.01) -> float:
        """Compute Expected Improvement acquisition function."""
        mu, sigma = gp.predict(X_query.reshape(1, -1), return_std=True)
        sigma = max(float(sigma), 1e-9)

        f_best = max(self.y_history)
        with np.errstate(divide='ignore'):
            z = (mu - f_best - xi) / sigma
            ei = (mu - f_best - xi) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
            ei[sigma < 1e-9] = 0.0
        return float(ei)

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + np.vectorize(lambda t: float(__import__('scipy').stats.norm.cdf(t)))(x))

    @staticmethod
    def _norm_pdf(x):
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    def _acquisition_optimize(self, gp) -> np.ndarray:
        """Maximize EI over parameter space."""
        best_ei = -np.inf
        best_x = None
        for _ in range(500):
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
            # Random exploration
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
        """Record observation."""
        self.X_history.append(x.tolist())
        self.y_history.append(float(y))

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


import random


# =============================================================================
# PARAMETER COUNT UTILITIES
# =============================================================================

def classical_hrl_params(n: int, h: int, n_actions: list, n_layers: int = 2) -> int:
    """Compute DQN parameter count (Eq. 46).

    W_DQN = n*h + (n_layers-2)*h^2 + h*sum(n_actions)

    Note: n_layers includes input AND output layers.
    A DQN with "single hidden layer" => n_layers=2 (input + 1 hidden + output).
    A DQN with "four hidden layers" => n_layers=5 (input + 4 hidden + output).
    The paper's 144K baseline uses n_layers=2 per DQN.
    The paper's 823K figure uses n_layers=5 per DQN.
    """
    return n * h + (n_layers - 2) * h * h + h * sum(n_actions)


def vqc_params(L: int, q: int) -> int:
    """Compute VQC parameter count (Eq. 49). W_VQC = L * q."""
    return L * q


def qaoa_params(p: int) -> int:
    """Compute QAOA parameter count (Eq. 50). W_QAOA = 2p."""
    return 2 * p
