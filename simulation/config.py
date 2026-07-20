"""Single source of truth for the T-NTN experiment configuration.

Every paper-facing quantity (network topology, channel model, workload
distribution, reward objective, LuST mobility window) is defined here ONCE so
that the code and the paper's configuration table cannot drift apart
(addresses urgent_need_to_fixed.md issues #7, #8, #9, #20).

`CONFIG` is a plain dict serialised into every results_manifest.json, and
`config_hash()` gives a short fingerprint logged next to every result so a
table of numbers is always traceable to the exact configuration that produced
it.

The altitude/coverage values are the physically-defensible engineering values
(HAP ~20 km, LEO ~600 km, RSU/LAP/HAP/LEO coverage 0.3/1.5/50/500 km); the
paper's Section-6 table is updated to match THESE values, not the reverse.
"""
from __future__ import annotations
import hashlib
import json
import os
import numpy as np

# =============================================================================
# NETWORK TOPOLOGY (4 tiers: RSU, LAP, HAP, LEO)
# =============================================================================
TIER_NAMES = ["RSU", "LAP", "HAP", "LEO"]
N_TIERS = 4

# Node count per tier  (M_l).  Node-selection search space is max(M_TIERS)=5.
# QHRL_NODE_SCALE (int env var, default 1) multiplies every tier's node count to
# probe the regime where QAOA's combinatorial node selection actually matters
# (larger M_l => the flat/random selector degrades, QAOA earns its place).
_NODE_SCALE = int(os.environ.get("QHRL_NODE_SCALE", "1"))
M_TIERS = [max(1, m * _NODE_SCALE) for m in [5, 3, 2, 2]]

# QHRL_HEAVY_LOAD (int env var, default 0): enable queueing + higher background
# load so node costs become load-coupled and deadlines bind; a naive offload-to-
# nearest heuristic then saturates its target and fails, restoring the value of
# a load-aware learned policy.
HEAVY_LOAD = bool(int(os.environ.get("QHRL_HEAVY_LOAD", "0")))
QUEUE_BASE_S = 1.2        # waiting time (s) at full node utilisation
HEAVY_LOAD_INIT = (0.55, 0.95)   # initial node-load range under heavy load
HEAVY_BG_STEP = 0.08     # background load random-walk std per slot (heavy load)

# Node altitude per tier (km above ground).
TIER_ALTITUDE_KM = [0.01, 0.3, 20.0, 600.0]

# Nominal coverage radius per tier (km) used for the sojourn-time model.
TIER_COVERAGE_KM = [0.3, 1.5, 50.0, 500.0]

# Per-tier uplink bandwidth (Hz).
TIER_BANDWIDTH_HZ = [20e6, 40e6, 60e6, 100e6]

# Per-tier edge compute capacity (CPU cycles / s).
TIER_COMPUTE_HZ = [5e9, 10e9, 20e9, 40e9]

# =============================================================================
# CHANNEL MODEL  h = ETA0 * d_m^{-delta}   (Section 4.2)
# =============================================================================
ETA0 = 1e-3                                   # reference gain at 1 m
DELTA_PER_TIER = [2.5, 2.2, 2.0, 2.0]         # path-loss exponent per tier
N0 = 1e-13                                     # noise power (W)
PK_VEHICLE = 0.1                               # vehicle transmit power (W)
P_RX = 0.02                                    # vehicle receive power (W)

# =============================================================================
# COMPUTE / ENERGY
# =============================================================================
FLOC_VEHICLE = 1e9                             # local CPU frequency (cycles/s)
EPS_LOCAL = 5e-9                               # local energy (J/cycle)
EPS_EDGE_PER_TIER = [1e-9, 2e-9, 3e-9, 4e-9]   # edge energy (J/cycle) per tier
W_VU = 1.0                                     # vehicle (tx+rx) energy weight
W_EN = 0.5                                     # edge-node compute energy weight

# =============================================================================
# WORKLOAD / TASK DISTRIBUTION (Section 4.1)
#   payload d_k ~ U(D_MIN, D_MAX) * PAYLOAD_UNIT_BITS  bits
#   workload c_k ~ U(C_MIN, C_MAX) * 1e6  CPU cycles
#   deadline Tmax ~ U(TMAX_MIN, TMAX_MAX) s
# =============================================================================
PAYLOAD_UNIT = "Mbit"          # UNIT ACTUALLY USED (paper table cites Mbit)
PAYLOAD_UNIT_BITS = 1e6        # 1 Mbit = 1e6 bits
D_MIN, D_MAX = 0.5, 5.0        # payload in Mbit
C_MIN, C_MAX = 100.0, 1000.0   # workload in Mcycles
TMAX_MIN, TMAX_MAX = 0.5, 2.0  # deadline in seconds
Z_SERVICES = 6                 # number of service types xi_k

# =============================================================================
# REWARD / OBJECTIVE  (Eq. 32)
#   R = -(BETA1*T + BETA2*E) - W_F1*F1 - W_F2*F2 - W_F3*F3
# BETA1=BETA2=0.5 matches the paper's objective P1 weighting.
# =============================================================================
BETA1 = 0.5
BETA2 = 0.5
W_F1 = 50.0   # sojourn-violation penalty
W_F2 = 50.0   # deadline-violation penalty
W_F3 = 50.0   # energy-above-local penalty

# QAOA node-selection QUBO one-hot penalty. Applied to per-node costs that are
# normalised to [0,1] inside the solver, so a modest relative value keeps the
# Ising landscape well-conditioned (a huge penalty makes QAOA ill-posed).
QAOA_ONEHOT_PENALTY = 2.0

# =============================================================================
# LuST MOBILITY  (17:00-18:00 window, densest 3 km urban region of interest)
#   Coordinates are the LuST network frame (metres); the ROI below holds
#   ~50% of all vehicle-records in the hour.  See simulation/lust_mobility.py.
# =============================================================================
LUST_FCD = "figure_baseline/work/window.fcd.xml"
# Trajectory cache; override with QHRL_LUST_CACHE (e.g. the full-24h pooled cache).
LUST_CACHE = os.environ.get("QHRL_LUST_CACHE", "simulation/lust_roi_trajectories.npz")
LUST_ROI_X = [4702.0, 7702.0]     # metres
LUST_ROI_Y = [4640.0, 7640.0]     # metres
LUST_MIN_POINTS = 6               # min in-ROI FCD snapshots to keep a vehicle
LUST_MIN_TRAVEL_M = 150.0         # min in-ROI path length (drop queued/parked)
LUST_WINDOW_CLOCK = "17:00-18:00"

# Operational area = ROI extent (km). Nodes are placed inside this frame.
AREA_KM = round((LUST_ROI_X[1] - LUST_ROI_X[0]) / 1000.0, 3)   # 3.0 km

STATE_DIM = 20


def M_total() -> int:
    return int(sum(M_TIERS))


CONFIG = {
    "topology": {
        "tiers": TIER_NAMES,
        "M_TIERS": M_TIERS,
        "altitude_km": TIER_ALTITUDE_KM,
        "coverage_km": TIER_COVERAGE_KM,
        "bandwidth_hz": TIER_BANDWIDTH_HZ,
        "compute_hz": TIER_COMPUTE_HZ,
        "area_km": AREA_KM,
    },
    "channel": {
        "eta0": ETA0, "delta_per_tier": DELTA_PER_TIER, "N0": N0,
        "pk_vehicle": PK_VEHICLE, "p_rx": P_RX,
    },
    "compute_energy": {
        "floc_vehicle": FLOC_VEHICLE, "eps_local": EPS_LOCAL,
        "eps_edge_per_tier": EPS_EDGE_PER_TIER, "w_vu": W_VU, "w_en": W_EN,
    },
    "workload": {
        "payload_unit": PAYLOAD_UNIT, "d_range_mbit": [D_MIN, D_MAX],
        "c_range_mcycles": [C_MIN, C_MAX], "tmax_range_s": [TMAX_MIN, TMAX_MAX],
        "z_services": Z_SERVICES,
    },
    "reward": {
        "beta1": BETA1, "beta2": BETA2,
        "w_f1": W_F1, "w_f2": W_F2, "w_f3": W_F3,
    },
    "mobility": {
        "source": "LuST",
        "fcd": LUST_FCD,
        "window_clock": LUST_WINDOW_CLOCK,
        "roi_x_m": LUST_ROI_X, "roi_y_m": LUST_ROI_Y,
        "min_points": LUST_MIN_POINTS,
    },
    "regime": {"node_scale": _NODE_SCALE, "heavy_load": HEAVY_LOAD},
}


def config_hash() -> str:
    """Short deterministic fingerprint of CONFIG."""
    blob = json.dumps(CONFIG, sort_keys=True, default=float).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


if __name__ == "__main__":
    print(json.dumps(CONFIG, indent=2, default=float))
    print("config_hash:", config_hash())
    print("M_total:", M_total(), "AREA_KM:", AREA_KM)
