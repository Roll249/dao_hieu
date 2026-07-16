"""T-NTN (Terrestrial-Non-Terrestrial Network) Environment.

Implements the multi-tier vehicular edge computing environment described in
the paper's Section 4, including:
  - Four-tier network topology (RSU, LAP, HAP, LEO)
  - Channel models per tier (Eqs. 18-20)
  - Latency and energy cost computation (Eqs. 21-22)
  - MDP state/action/reward interface

Mobility layer (mobility='lust') replays real LuST SUMO FCD vehicle
trajectories within a dense urban region of interest; mobility='synthetic'
keeps the legacy random-walk model. All network/channel/workload/reward
constants come from config.py (single source of truth).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import config as C
from utils import (
    TIER_NAMES, TIER_LABELS, M_TIERS, M_TOTAL, STATE_DIM,
    compute_channel_gain, compute_latency, compute_energy, compute_reward,
    PK_VEHICLE, FLOC_VEHICLE,
    normalize_state,
)
from lust_mobility import LuSTMobilityProvider

Z_SERVICES = C.Z_SERVICES   # Number of service types xi_k in {0, ..., Z-1}

# Load-model window (s): converts offloaded cycles into a bounded utilisation.
_LOAD_WINDOW_S = 1.0

# Cache one LuST provider per cache-file so many env instances share it cheaply.
_LUST_PROVIDER_CACHE: Dict[str, LuSTMobilityProvider] = {}


def _get_lust_provider() -> LuSTMobilityProvider:
    key = C.LUST_CACHE
    if key not in _LUST_PROVIDER_CACHE:
        _LUST_PROVIDER_CACHE[key] = LuSTMobilityProvider()
    return _LUST_PROVIDER_CACHE[key]


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
        area_km: Optional[float] = None,
        n_steps_per_episode: int = 200,
        dt: float = 0.1,
        seed: int = 42,
        mobility: str = "lust",
    ):
        self.mobility = mobility
        if mobility == "lust":
            self.lust = _get_lust_provider()
            self.area_km = self.lust.area_km
        else:
            self.lust = None
            self.area_km = area_km if area_km is not None else 2.0
        self.n_steps = n_steps_per_episode
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        # Tiers: (name, node_count, z_height_km, cpu_cycles_per_s) from config
        self.tier_configs = [
            (C.TIER_NAMES[t], C.M_TIERS[t], C.TIER_ALTITUDE_KM[t], C.TIER_COMPUTE_HZ[t])
            for t in range(C.N_TIERS)
        ]

        # Build network topology
        self.nodes = []
        self._build_topology()

        self.current_step = 0
        self.vehicle = Vehicle()
        self.current_task: Optional[Task] = None

        # LuST episode trajectory (set in reset when mobility='lust')
        self._traj_x = self._traj_y = self._traj_sp = None

        # Cached environment feature block s[0:17] (recomputed once per step)
        self._env_feat = np.zeros(17)
        self._state = np.zeros(STATE_DIM)

        self.episode_metrics = {
            'latencies': [], 'energies': [], 'rewards': [], 'constraint_violations': [],
        }

    def _build_topology(self):
        """Place nodes across the 4 tiers within the operational area."""
        self.nodes = []
        for tier_idx, (name, count, z, cpu) in enumerate(self.tier_configs):
            for n_idx in range(count):
                if tier_idx == 0:  # RSU: distributed grid on the ground
                    grid_dim = int(np.ceil(np.sqrt(count)))
                    xi = n_idx % grid_dim
                    yi = n_idx // grid_dim
                    x = (xi + 1) * self.area_km / (grid_dim + 1)
                    y = (yi + 1) * self.area_km / (grid_dim + 1)
                    pos_z = z
                elif tier_idx == 1:  # LAP: aerial above grid
                    grid_dim = int(np.ceil(np.sqrt(count)))
                    xi = n_idx % grid_dim
                    yi = n_idx // grid_dim
                    x = (xi + 1) * self.area_km / (grid_dim + 1)
                    y = (yi + 1) * self.area_km / (grid_dim + 1)
                    pos_z = z
                elif tier_idx == 2:  # HAP: wide coverage
                    x = self.area_km * (0.3 + 0.4 * (n_idx % 2))
                    y = self.area_km * (0.3 + 0.4 * (n_idx // 2))
                    pos_z = z
                else:  # LEO: satellite constellation
                    angle = 2 * np.pi * n_idx / count + 0.3
                    orbit_r = self.area_km * 0.6
                    x = self.area_km * 0.5 + orbit_r * np.cos(angle)
                    y = self.area_km * 0.5 + orbit_r * np.sin(angle)
                    pos_z = z

                self.nodes.append(Node(tier_idx=tier_idx, node_idx=n_idx,
                                       x=x, y=y, z=pos_z, compute_capacity=cpu))

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_step = 0
        self.current_task = None

        if self.mobility == "lust":
            X, Y, SP = self.lust.sample_episode(self.rng, self.n_steps)
            self._traj_x, self._traj_y, self._traj_sp = X, Y, SP
            self._set_vehicle_from_traj(0)
        else:
            self.vehicle.x = self.rng.uniform(0.1, self.area_km - 0.1)
            self.vehicle.y = self.rng.uniform(0.1, self.area_km - 0.1)
            self.vehicle.speed_kmh = max(20.0, min(90.0, self.rng.normal(60.0, 15.0)))
            heading = self.rng.uniform(0, 2 * np.pi)
            self.vehicle.heading = heading
            self.vehicle.vx = self.vehicle.speed_kmh / 3.6 * np.cos(heading)
            self.vehicle.vy = self.vehicle.speed_kmh / 3.6 * np.sin(heading)

        lo, hi = (C.HEAVY_LOAD_INIT if C.HEAVY_LOAD else (0.1, 0.5))
        for node in self.nodes:
            node.cpu_load = self.rng.uniform(lo, hi)

        self._compute_env_features()
        return self.observe(None)

    def _set_vehicle_from_traj(self, k: int):
        """Place the ego vehicle at trajectory index k (LuST replay)."""
        k = int(np.clip(k, 0, len(self._traj_x) - 1))
        x, y, sp = float(self._traj_x[k]), float(self._traj_y[k]), float(self._traj_sp[k])
        if k > 0:
            dx = self._traj_x[k] - self._traj_x[k - 1]
            dy = self._traj_y[k] - self._traj_y[k - 1]
            heading = float(np.arctan2(dy, dx)) if (dx or dy) else self.vehicle.heading
        else:
            heading = 0.0
        self.vehicle.x = x
        self.vehicle.y = y
        self.vehicle.speed_kmh = sp * 3.6
        self.vehicle.heading = heading
        self.vehicle.vx = sp * np.cos(heading)
        self.vehicle.vy = sp * np.sin(heading)

    # ------------------------------------------------------------- observation
    def _compute_env_features(self):
        """Compute the environment-only feature block s[0:17] once per step.

        Task-dependent features s[17:20] are filled by observe(task); keeping
        the two separate lets the agent condition on the CURRENT task
        (urgent_need_to_fixed.md #15) rather than a stale descriptor.
        """
        f = np.zeros(17)
        f[0] = self.vehicle.x / self.area_km
        f[1] = self.vehicle.y / self.area_km
        f[2] = min(self.vehicle.speed_kmh / 90.0, 1.0)

        for t in range(4):
            tier_nodes = [n for n in self.nodes if n.tier_idx == t]
            if tier_nodes:
                f[3 + t] = np.mean([n.cpu_load for n in tier_nodes])

        vx_km, vy_km = self.vehicle.x, self.vehicle.y
        for t in range(4):
            best_g = 0.0
            for n in self.nodes:
                if n.tier_idx == t:
                    d = np.sqrt((n.x - vx_km) ** 2 + (n.y - vy_km) ** 2 + n.z ** 2)
                    g = compute_channel_gain(t, d)
                    if g > best_g:
                        best_g = g
            f[7 + t] = np.log1p(best_g) / 10.0

        for t in range(3):
            f[11 + t] = self.rng.uniform(0.2, 0.8)

        f[14] = self.vehicle.vx / 25.0
        f[15] = self.vehicle.vy / 25.0
        f[16] = self.vehicle.heading / (2 * np.pi)
        self._env_feat = f

    def observe(self, task: Optional[Task]) -> np.ndarray:
        """Build the full observation s in R^20 for a given current task.

        Sets self.current_task and embeds its (d_k, c_k, Tmax) descriptor into
        s[17:20]. Call this immediately before select_action so the policy sees
        the task it is about to act on.
        """
        if task is not None:
            self.current_task = task
        s = np.zeros(STATE_DIM)
        s[:17] = self._env_feat
        t = self.current_task
        if t is not None:
            s[17] = t.d_k / 5e6
            s[18] = t.c_k / 1e9
            s[19] = t.Tmax_k / 2.0
        self._state = s
        return s.copy()

    def _get_state(self) -> np.ndarray:
        """Backward-compatible full-state accessor (uses current task)."""
        return self.observe(self.current_task)

    # --------------------------------------------------------------- task gen
    def _generate_task(self) -> Task:
        """Generate a random task (Section 4.1) per config distribution."""
        d_units = self.rng.uniform(C.D_MIN, C.D_MAX)               # Mbit
        c_k = self.rng.uniform(C.C_MIN, C.C_MAX) * 1e6            # cycles
        Tmax_k = self.rng.uniform(C.TMAX_MIN, C.TMAX_MAX)         # seconds
        xi_k = int(self.rng.randint(0, Z_SERVICES))              # service type
        return Task(d_k=d_units * C.PAYLOAD_UNIT_BITS, c_k=c_k, Tmax_k=Tmax_k, xi_k=xi_k)

    # ------------------------------------------------------------------- step
    def step(
        self,
        action_l: int,
        action_n: int,
        action_alpha: float,
        task: Optional[Task] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if task is None:
            task = self.current_task if self.current_task is not None else self._generate_task()
        self.current_task = task

        tier_node = self._get_node(action_l, action_n)
        coverage_radius = self._coverage_radius(action_l)         # km
        v_ms = max(self.vehicle.speed_kmh / 3.6, 0.1)             # m/s
        Tsoj = coverage_radius * 1000.0 / v_ms                    # s

        env_state = self._env_state()

        latency, lat_components = compute_latency(
            task.d_k, task.c_k, action_alpha, action_l, action_n, env_state)
        energy, en_components = compute_energy(
            task.d_k, task.c_k, action_alpha, action_l, action_n, env_state)

        reward, flags = compute_reward(
            latency, energy, Tsoj, task.Tmax_k, task.c_k)

        # Update node CPU load: bounded EMA of per-task utilisation. In the
        # light-load regime no queueing latency is added (urgent_need_to_fixed.md
        # #14); in the heavy-load regime a load-dependent waiting time is applied
        # in compute_latency and background traffic loads the other nodes.
        for n in self.nodes:
            n.cpu_load = max(0.0, n.cpu_load * 0.98)
        util = action_alpha * task.c_k / (tier_node.compute_capacity * _LOAD_WINDOW_S)
        tier_node.cpu_load = float(np.clip(0.9 * tier_node.cpu_load + 0.5 * util, 0.0, 1.0))
        if C.HEAVY_LOAD:
            for n in self.nodes:
                n.cpu_load = float(np.clip(
                    n.cpu_load + self.rng.normal(0.0, C.HEAVY_BG_STEP), 0.05, 1.0))

        # Advance mobility
        self.current_step += 1
        self._move_vehicle(self.dt)
        self._compute_env_features()
        s_next = self.observe(task)

        done = self.current_step >= self.n_steps

        self.episode_metrics['latencies'].append(latency)
        self.episode_metrics['energies'].append(energy)
        self.episode_metrics['rewards'].append(reward)
        self.episode_metrics['constraint_violations'].append(sum(flags.values()))

        info = {
            'latency': latency, 'latency_components': lat_components,
            'energy': energy, 'energy_components': en_components,
            'constraint_flags': flags, 'tier_selected': action_l,
            'node_selected': action_n, 'alpha': action_alpha,
            'Tsoj': Tsoj, 'Tmax': task.Tmax_k,
        }
        return s_next, reward, done, info

    def _move_vehicle(self, dt: float):
        """Advance the ego vehicle: LuST replay or synthetic random walk."""
        if self.mobility == "lust":
            self._set_vehicle_from_traj(self.current_step)
            self.vehicle.energy_budget -= 0.01 * dt
            return

        # --- synthetic random-walk fallback ---
        self.vehicle.speed_kmh = max(20.0, min(90.0,
            self.vehicle.speed_kmh + self.rng.normal(0, 2.0)))
        speed_ms = self.vehicle.speed_kmh / 3.6
        self.vehicle.heading += self.rng.normal(0, 0.05)
        self.vehicle.heading = self.vehicle.heading % (2 * np.pi)
        self.vehicle.vx = speed_ms * np.cos(self.vehicle.heading)
        self.vehicle.vy = speed_ms * np.sin(self.vehicle.heading)
        self.vehicle.x += self.vehicle.vx * dt / 1000.0
        self.vehicle.y += self.vehicle.vy * dt / 1000.0
        if self.vehicle.x < 0:
            self.vehicle.x = 0; self.vehicle.heading = np.pi - self.vehicle.heading
        elif self.vehicle.x > self.area_km:
            self.vehicle.x = self.area_km; self.vehicle.heading = np.pi - self.vehicle.heading
        if self.vehicle.y < 0:
            self.vehicle.y = 0; self.vehicle.heading = -self.vehicle.heading
        elif self.vehicle.y > self.area_km:
            self.vehicle.y = self.area_km; self.vehicle.heading = -self.vehicle.heading
        self.vehicle.energy_budget -= 0.01 * dt

    def _coverage_radius(self, tier_idx: int) -> float:
        """Coverage radius per tier (km) from config."""
        return C.TIER_COVERAGE_KM[tier_idx]

    def _get_node(self, tier_idx: int, node_idx: int) -> Node:
        for n in self.nodes:
            if n.tier_idx == tier_idx and n.node_idx == node_idx:
                return n
        return self.nodes[0]

    def _compute_all_channel_gains(self) -> np.ndarray:
        g = np.zeros((4, max(M_TIERS)))
        for node in self.nodes:
            d = np.sqrt((node.x - self.vehicle.x) ** 2 +
                        (node.y - self.vehicle.y) ** 2 + node.z ** 2)
            g[node.tier_idx, node.node_idx] = compute_channel_gain(node.tier_idx, d)
        return g

    def _node_load_array(self) -> Optional[np.ndarray]:
        """Per-node CPU load (4, maxM); only used when heavy-load queueing is on."""
        if not C.HEAVY_LOAD:
            return None
        L = np.zeros((4, max(M_TIERS)))
        for node in self.nodes:
            L[node.tier_idx, node.node_idx] = node.cpu_load
        return L

    def _env_state(self) -> Dict[str, Any]:
        return {'channel_gains': self._compute_all_channel_gains(),
                'node_loads': self._node_load_array()}

    def get_qubo_costs(self, tier_idx: int, alpha: float, task: Task) -> np.ndarray:
        """Per-node QUBO costs c_n = BETA1*latency + BETA2*energy (Section 4.5)."""
        node_list = [n for n in self.nodes if n.tier_idx == tier_idx]
        costs = np.zeros(len(node_list))
        env_state = self._env_state()
        for i, node in enumerate(node_list):
            lat, _ = compute_latency(task.d_k, task.c_k, alpha, tier_idx, node.node_idx, env_state)
            en, _ = compute_energy(task.d_k, task.c_k, alpha, tier_idx, node.node_idx, env_state)
            costs[i] = C.BETA1 * lat + C.BETA2 * en
        return costs

    def render(self) -> Dict[str, Any]:
        return {
            'vehicle_pos': (self.vehicle.x, self.vehicle.y),
            'nodes': [(n.x, n.y, n.z, TIER_LABELS[n.tier_idx]) for n in self.nodes],
            'state': self._state.copy(),
        }
