"""T-NTN (Terrestrial-Non-Terrestrial Network) Environment.

Implements the multi-tier vehicular edge computing environment described in
the paper's Section 4, including:
  - Four-tier network topology (RSU, LAP, HAP, LEO)
  - Channel models per tier (Eqs. 18-20)
  - Latency and energy cost computation (Eqs. 21-22)
  - MDP state/action/reward interface
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from utils import (
    TIER_NAMES, TIER_LABELS, M_TIERS, M_TOTAL, STATE_DIM,
    compute_channel_gain, compute_latency, compute_energy, compute_reward,
    PK_VEHICLE, KAPPA, SIGMA2, FLOC_VEHICLE,
    normalize_state,
)

Z_SERVICES = 6   # Number of service types xi_k in {0, ..., Z-1}


@dataclass
class Task:
    """Task descriptor (Section 4.1)."""
    d_k: float      # Payload size in bits
    c_k: float      # Workload in CPU cycles
    Tmax_k: float   # Hard deadline in seconds
    vehicle_id: int = 0
    arrive_time: float = 0.0
    xi_k: int = 0   # Service type in {0, ..., Z_SERVICES-1}


@dataclass
class Vehicle:
    """Vehicle with mobility state."""
    x: float = 0.0          # x position (km)
    y: float = 0.0          # y position (km)
    vx: float = 16.67       # Velocity x (m/s), default ~60 km/h
    vy: float = 0.0         # Velocity y (m/s)
    speed_kmh: float = 60.0  # Speed in km/h
    heading: float = 0.0     # Heading angle (radians)
    energy_budget: float = 100.0  # Energy budget (J)


@dataclass
class Node:
    """Edge compute node at a specific tier."""
    tier_idx: int     # 0=RSU, 1=LAP, 2=HAP, 3=LEO
    node_idx: int     # Index within tier
    x: float = 0.0   # x position (km)
    y: float = 0.0   # y position (km)
    z: float = 0.0   # z position (km)
    cpu_load: float = 0.0  # Current CPU load [0, 1]
    compute_capacity: float = 1e9  # CPU cycles/s


class TNTNEnvironment:
    """T-NTN vehicular edge computing environment.

    Implements the MDP (S, A, R, gamma) from Section 4.6.
    """

    def __init__(
        self,
        area_km: float = 2.0,
        n_steps_per_episode: int = 200,
        dt: float = 0.1,
        seed: int = 42,
    ):
        self.area_km = area_km
        self.n_steps = n_steps_per_episode
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        # Tiers: (name, node_count, z_height_km, cpu_cycles_per_s)
        self.tier_configs = [
            ('RSU', M_TIERS[0], 0.01, 5e9),
            ('LAP', M_TIERS[1], 0.3, 10e9),
            ('HAP', M_TIERS[2], 20.0, 20e9),
            ('LEO', M_TIERS[3], 600.0, 40e9),
        ]

        # Build network topology
        self.nodes = []
        self._build_topology()

        # Pre-compute channel gains for all vehicle positions
        self.current_step = 0
        self.vehicle = Vehicle()

        # State history for MDP
        self._state = np.zeros(STATE_DIM)

        # Tracking metrics
        self.episode_metrics = {
            'latencies': [],
            'energies': [],
            'rewards': [],
            'constraint_violations': [],
        }

    def _build_topology(self):
        """Place nodes across the 4 tiers."""
        self.nodes = []
        node_id = 0
        for tier_idx, (name, count, z, cpu) in enumerate(self.tier_configs):
            for n_idx in range(count):
                if tier_idx == 0:  # RSU: distributed along roads
                    x = (n_idx + 1) * self.area_km / (count + 1)
                    y = self.area_km * 0.5 + (0.2 * self.area_km * (n_idx % 2 - 0.5))
                    pos_z = 0.01
                elif tier_idx == 1:  # LAP: aerial above grid
                    grid_dim = int(np.ceil(np.sqrt(count)))
                    xi = n_idx % grid_dim
                    yi = n_idx // grid_dim
                    x = (xi + 1) * self.area_km / (grid_dim + 1)
                    y = (yi + 1) * self.area_km / (grid_dim + 1)
                    pos_z = 0.3
                elif tier_idx == 2:  # HAP: wide coverage
                    x = self.area_km * (0.3 + 0.4 * (n_idx % 2))
                    y = self.area_km * (0.3 + 0.4 * (n_idx // 2))
                    pos_z = 20.0
                else:  # LEO: satellite constellation
                    # Simulate orbital positions
                    angle = 2 * np.pi * n_idx / count + 0.3
                    orbit_r = self.area_km * 0.6
                    x = self.area_km * 0.5 + orbit_r * np.cos(angle)
                    y = self.area_km * 0.5 + orbit_r * np.sin(angle)
                    pos_z = 600.0

                node = Node(
                    tier_idx=tier_idx,
                    node_idx=n_idx,
                    x=x,
                    y=y,
                    z=pos_z,
                    compute_capacity=cpu,
                )
                self.nodes.append(node)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0

        # Random vehicle start
        self.vehicle.x = self.rng.uniform(0.1, self.area_km - 0.1)
        self.vehicle.y = self.rng.uniform(0.1, self.area_km - 0.1)
        self.vehicle.speed_kmh = max(20.0, min(90.0,
            self.rng.normal(60.0, 15.0)))
        heading = self.rng.uniform(0, 2 * np.pi)
        self.vehicle.heading = heading
        self.vehicle.vx = self.vehicle.speed_kmh / 3.6 * np.cos(heading)
        self.vehicle.vy = self.vehicle.speed_kmh / 3.6 * np.sin(heading)

        # Reset node CPU loads
        for node in self.nodes:
            node.cpu_load = self.rng.uniform(0.1, 0.8)

        self._state = self._get_state()
        return self._state.copy()

    def _get_state(self) -> np.ndarray:
        """Construct MDP state vector s_t in R^20 (Section 4.6).

        Components:
          [0,1]:   vehicle position (x, y)
          [2]:      vehicle speed (normalized)
          [3,4,5,6]: per-tier avg CPU load (4 dims)
          [7,8,9,10]: best channel gain per tier (4 dims)
          [11,12,13]: avg VU count per tier (3 tiers, HAP/LEO grouped)
          [14,15,16]: mobility estimate (vx, vy, heading)
          [17,18,19]: task parameters (d_k, c_k, Tmax_k) / max values
        """
        s = np.zeros(STATE_DIM)

        # Vehicle position (normalized)
        s[0] = self.vehicle.x / self.area_km
        s[1] = self.vehicle.y / self.area_km

        # Vehicle speed (normalized)
        s[2] = self.vehicle.speed_kmh / 90.0

        # Per-tier avg CPU load
        for t in range(4):
            tier_nodes = [n for n in self.nodes if n.tier_idx == t]
            if tier_nodes:
                s[3 + t] = np.mean([n.cpu_load for n in tier_nodes])

        # Per-tier best channel gain
        vx_km = self.vehicle.x
        vy_km = self.vehicle.y
        for t in range(4):
            best_g = 0.0
            for n in self.nodes:
                if n.tier_idx == t:
                    # 3-D distance in km (includes altitude z)
                    d = np.sqrt((n.x - vx_km) ** 2 + (n.y - vy_km) ** 2 + n.z ** 2)
                    g = compute_channel_gain(t, d)
                    if g > best_g:
                        best_g = g
            # Log-scale normalization for channel gains
            s[7 + t] = np.log1p(best_g) / 10.0

        # Avg VU count per tier (simulated)
        for t in range(3):
            s[11 + t] = self.rng.uniform(0.2, 0.8)

        # Mobility estimate
        s[14] = self.vehicle.vx / 25.0
        s[15] = self.vehicle.vy / 25.0
        s[16] = self.vehicle.heading / (2 * np.pi)

        # Task parameters (will be overwritten per step)
        s[17] = 2.5 / 5.0        # normalized d_k
        s[18] = 550 / 1000.0     # normalized c_k
        s[19] = 1.25 / 2.0       # normalized Tmax

        self._state = s
        return s

    def _generate_task(self) -> Task:
        """Generate a random task (Section 4.1)."""
        d_k = self.rng.uniform(0.5, 5.0)  # Mbits
        c_k = self.rng.uniform(100, 1000) * 1e6  # CPU cycles
        Tmax_k = self.rng.uniform(0.5, 2.0)  # seconds
        xi_k = int(self.rng.randint(0, Z_SERVICES))  # service type
        return Task(d_k=d_k * 1e6, c_k=c_k, Tmax_k=Tmax_k, xi_k=xi_k)

    def step(
        self,
        action_l: int,
        action_n: int,
        action_alpha: float,
        task: Optional[Task] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action_l:   Selected tier index [0, 3]
            action_n:   Selected node index within tier
            action_alpha: Offloading ratio in [0, 1]
            task:       Task to offload (auto-generated if None)

        Returns:
            s_next:     Next state
            reward:     Scalar reward (Eq. 32)
            done:       Episode termination flag
            info:       Diagnostic dict
        """
        if task is None:
            task = self._generate_task()

        # Update state with actual task params
        self._state[17] = task.d_k / 5e6
        self._state[18] = task.c_k / 1e9
        self._state[19] = task.Tmax_k / 2.0

        # Compute sojourn time (time before vehicle exits coverage)
        # Simplified: Tsoj = coverage_radius / vehicle_speed
        tier_node = self._get_node(action_l, action_n)
        coverage_radius = self._coverage_radius(action_l)  # km
        v_ms = self.vehicle.speed_kmh / 3.6               # m/s
        Tsoj = coverage_radius * 1000.0 / max(v_ms, 1.0)  # convert km→m then divide by m/s

        # Build environment state dict for cost computation
        channel_gains = self._compute_all_channel_gains()
        env_state = {'channel_gains': channel_gains}

        # Compute latency and energy
        latency, lat_components = compute_latency(
            task.d_k, task.c_k, action_alpha,
            action_l, action_n, env_state,
        )
        energy, en_components = compute_energy(
            task.d_k, task.c_k, action_alpha,
            action_l, action_n, env_state,
        )

        # Compute reward with penalty terms
        # F3 is now compared to local-only energy baseline (c_k * eps_local)
        reward, flags = compute_reward(
            latency, energy, Tsoj, task.Tmax_k,
            task.c_k,    # workload in cycles, used to compute E_local_full
            w1=50.0, w2=50.0, w3=50.0,
        )

        # Update node CPU load based on processing
        if action_l < len(self.nodes):
            load_increase = action_alpha * task.c_k / (tier_node.compute_capacity * self.dt)
            tier_node.cpu_load = min(1.0, tier_node.cpu_load + load_increase * 1e-9)

        # Move vehicle
        self._move_vehicle(self.dt)

        # Update state
        s_next = self._get_state()

        # Check termination
        self.current_step += 1
        done = self.current_step >= self.n_steps

        # Record metrics
        self.episode_metrics['latencies'].append(latency)
        self.episode_metrics['energies'].append(energy)
        self.episode_metrics['rewards'].append(reward)
        self.episode_metrics['constraint_violations'].append(sum(flags.values()))

        info = {
            'latency': latency,
            'latency_components': lat_components,
            'energy': energy,
            'energy_components': en_components,
            'constraint_flags': flags,
            'tier_selected': action_l,
            'node_selected': action_n,
            'alpha': action_alpha,
            'Tsoj': Tsoj,
            'Tmax': task.Tmax_k,
        }
        return s_next, reward, done, info

    def _move_vehicle(self, dt: float):
        """Update vehicle position with random mobility model."""
        # Speed perturbations (truncated Gaussian)
        self.vehicle.speed_kmh = max(20.0, min(90.0,
            self.vehicle.speed_kmh + self.rng.normal(0, 2.0)))
        speed_ms = self.vehicle.speed_kmh / 3.6

        # Heading perturbations
        self.vehicle.heading += self.rng.normal(0, 0.05)
        self.vehicle.heading = self.vehicle.heading % (2 * np.pi)

        # Update position
        self.vehicle.vx = speed_ms * np.cos(self.vehicle.heading)
        self.vehicle.vy = speed_ms * np.sin(self.vehicle.heading)
        self.vehicle.x += self.vehicle.vx * dt / 1000.0  # m -> km
        self.vehicle.y += self.vehicle.vy * dt / 1000.0

        # Boundary reflection
        if self.vehicle.x < 0:
            self.vehicle.x = 0
            self.vehicle.heading = np.pi - self.vehicle.heading
        elif self.vehicle.x > self.area_km:
            self.vehicle.x = self.area_km
            self.vehicle.heading = np.pi - self.vehicle.heading
        if self.vehicle.y < 0:
            self.vehicle.y = 0
            self.vehicle.heading = -self.vehicle.heading
        elif self.vehicle.y > self.area_km:
            self.vehicle.y = self.area_km
            self.vehicle.heading = -self.vehicle.heading

        # Energy consumption
        self.vehicle.energy_budget -= 0.01 * dt  # Background drain

    def _coverage_radius(self, tier_idx: int) -> float:
        """Approximate coverage radius per tier (km)."""
        radii = {0: 0.3, 1: 1.5, 2: 50.0, 3: 500.0}
        return radii.get(tier_idx, 1.0)

    def _get_node(self, tier_idx: int, node_idx: int) -> Node:
        """Get node by tier and index."""
        for n in self.nodes:
            if n.tier_idx == tier_idx and n.node_idx == node_idx:
                return n
        return self.nodes[0]

    def _compute_all_channel_gains(self) -> np.ndarray:
        """Pre-compute channel gains for all nodes. Shape: (4, max(M_TIERS))."""
        g = np.zeros((4, max(M_TIERS)))
        for node in self.nodes:
            tier_idx = node.tier_idx
            # 3-D distance in km (includes altitude z)
            d = np.sqrt((node.x - self.vehicle.x) ** 2 +
                         (node.y - self.vehicle.y) ** 2 +
                         node.z ** 2)
            g[tier_idx, node.node_idx] = compute_channel_gain(tier_idx, d)
        return g

    def get_qubo_costs(
        self,
        tier_idx: int,
        alpha: float,
        task: Task,
    ) -> np.ndarray:
        """Compute per-node QUBO costs c_n for a given tier (Section 4.5)."""
        node_list = [n for n in self.nodes if n.tier_idx == tier_idx]
        m_l = len(node_list)
        costs = np.zeros(m_l)

        channel_gains = self._compute_all_channel_gains()
        env_state = {'channel_gains': channel_gains}

        for i, node in enumerate(node_list):
            lat, _ = compute_latency(task.d_k, task.c_k, alpha, tier_idx, node.node_idx, env_state)
            en, _ = compute_energy(task.d_k, task.c_k, alpha, tier_idx, node.node_idx, env_state)
            costs[i] = lat + en  # beta1=beta2=1.0

        return costs

    def render(self) -> Dict[str, Any]:
        """Return current environment state for visualization."""
        return {
            'vehicle_pos': (self.vehicle.x, self.vehicle.y),
            'nodes': [(n.x, n.y, n.z, TIER_LABELS[n.tier_idx]) for n in self.nodes],
            'state': self._state.copy(),
        }
