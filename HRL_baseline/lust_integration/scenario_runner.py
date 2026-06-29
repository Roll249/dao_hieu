"""Reproducible LuST-24h scenario and ablation experiments.

The Classical HRL and LuST preprocessing modules are imported read-only.  This
runner changes only the requested decision component in each configuration.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-scenario")

import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Seed before importing hrl_core because the original module creates service
# allocations at import time.
random.seed(2024)
np.random.seed(2024)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCENARIO_DIR = REPO / "scenario"
RESULTS = SCENARIO_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "quantum"))

import tensorflow as tf
from scipy import stats

import hrl_core as H
import run_baseline as RB
import run_baseline_full as RBF
import bo_offload
import qaoa_node
import vqc_policy

tf.keras.utils.disable_interactive_logging()

FULL_POOL = REPO / "figure_baseline" / "results" / "lust_speed_pool_full.npy"
if not FULL_POOL.exists():
    raise SystemExit(f"Missing LuST-24h speed pool: {FULL_POOL}")
RB.SPEED_POOL = np.load(FULL_POOL)

SCHEMA_VERSION = 2
SEEDS = [2024, 2025, 2026, 2027, 2028]
EPISODES = RB.NUM_EPISODES
TIMESTEPS_TRAIN = RB.TIMESTEPS_TRAIN
TIMESTEPS_EVAL = RB.TIMESTEPS_EVAL
V_TRAIN = RB.V_TRAIN
V_LIST = list(RB.V_LIST)
EVAL_REPEATS = RB.RP

CONFIGS = {
    "Classical": {"layer": "classical", "node": "classical", "ratio": "classical"},
    "HRL+VQC": {"layer": "vqc", "node": "classical", "ratio": "classical"},
    "HRL+VQC+QAOA": {"layer": "vqc", "node": "qaoa", "ratio": "classical"},
    "Full-QHRL": {"layer": "vqc", "node": "qaoa", "ratio": "bo"},
    "w/o VQC": {"layer": "classical", "node": "qaoa", "ratio": "bo"},
    "w/o QAOA": {"layer": "vqc", "node": "classical", "ratio": "bo"},
}
SCENARIO_ORDER = ["Classical", "HRL+VQC", "HRL+VQC+QAOA", "Full-QHRL"]
ORDER = SCENARIO_ORDER + ["w/o VQC", "w/o QAOA"]
ABLATION_ORDER = ["Full-QHRL", "w/o VQC", "w/o QAOA", "HRL+VQC+QAOA"]

LAYER_OFFSETS = [0, H.RSU_T, H.RSU_T + H.UAV_T,
                 H.RSU_T + H.UAV_T + H.HAP_T]
MAX_USERS = [RB.MAX_U_RSU, RB.MAX_U_UAV, RB.MAX_U_HAP, RB.MAX_U_LEO]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_agents(seed, use_vqc):
    set_seed(seed)
    high = (vqc_policy.VQCPolicy(gamma=RB.GAMMA_H, epsilon=RB.EPSILON_H,
                                 seed=seed)
            if use_vqc else
            RB.FastAgent(RB.OPTIMIZER, len(H.State_Space_H), len(H.Action_Space_H),
                         RB.GAMMA_H, RB.EPSILON_H))
    middle = RB.FastAgent(RB.OPTIMIZER, len(H.State_Space_M),
                          H.RSU_T + H.UAV_T + H.HAP_T + H.LEO_T,
                          RB.GAMMA_M, RB.EPSILON_M)
    low = RB.FastAgent(RB.OPTIMIZER, len(H.State_Space_L), len(H.Action_Space_L),
                       RB.GAMMA_L, RB.EPSILON_L)
    return high, middle, low


def layer_arrays(E, layer):
    assignments = [E["VN_RSU_asign"], E["VN_UAV_asign"],
                   E["VN_HAP_asign"], E["VN_LEO_asign"]]
    sojourn = [E["VN_RSU_Soj"], E["VN_UAV_Soj"],
               E["VN_HAP_Soj"], E["VN_LEO_Soj"]]
    loads = [E["NO_VUs_RSU"], E["NO_VUs_UAV"],
             E["NO_VUs_HAP"], E["NO_VUs_LEO"]]
    return assignments[layer], sojourn[layer], loads[layer]


def choose_node(E, layer, vehicle, state_index_m, q_values_m, selector):
    M_A = H.M_Action_Space(layer, vehicle, E["VN_RSU_asign"], E["VN_UAV_asign"],
                           E["VN_HAP_asign"], E["VN_LEO_asign"])
    feasible = np.nonzero(M_A)[1]
    global_ids = LAYER_OFFSETS[layer] + feasible
    if selector is None:
        action = H.get_next_action_M(state_index_m, global_ids, RB.EPSILON_M,
                                     M_A, q_values_m)
    else:
        _, sojourn, loads = layer_arrays(E, layer)
        normalized_load = loads[feasible] / MAX_USERS[layer]
        residence = np.asarray(sojourn[vehicle])[feasible]
        residence = residence / (residence.max() + 1e-12)
        node = selector.select(feasible, normalized_load - 0.3 * residence)
        action = np.zeros(M_A.shape[1], dtype=float)
        action[int(node)] = 1.0
    return action, global_ids


def next_middle_state(E, layer, node_action, vehicle, state_index_m, request=None):
    del state_index_m
    request = E["V_Ser_Req"][vehicle] if request is None else request
    return H.Next_State_M(
        layer, node_action, E["VN_spd"][vehicle], request,
        E["NO_VUs_RSU"], RB.MAX_U_RSU, H.service_allocation_RSU, E["VN_RSU_Soj"][vehicle], H.RSU_r,
        E["NO_VUs_UAV"], RB.MAX_U_UAV, H.service_allocation_UAV, E["VN_UAV_Soj"][vehicle], H.UAV_r,
        E["NO_VUs_HAP"], RB.MAX_U_HAP, H.service_allocation_HAP, E["VN_HAP_Soj"][vehicle], H.HAP_r,
        E["NO_VUs_LEO"], RB.MAX_U_LEO, H.service_allocation_LEO, E["VN_LEO_Soj"][vehicle], H.LEO_r)


def eval_pipeline(E, decisions):
    V = E["V"]
    counts = [np.zeros(H.RSU_T), np.zeros(H.UAV_T),
              np.zeros(H.HAP_T), np.zeros(H.LEO_T)]
    for vehicle in range(V):
        counts[int(decisions[vehicle, 0])][int(decisions[vehicle, 1])] += 1
    ra = H.Resource_Allocation(E["IP"], decisions, *counts)
    rates = H.Data_Rate(E["IP"], decisions, *ra[:4])
    return H.Task_Proc_Main(E["IP"], ra[4], rates[0], ra[5], rates[1],
                            ra[6], rates[2], ra[7], rates[3],
                            RB.GAMMA_1, RB.GAMMA_2, decisions)


def metrics_from_cost(cost, V):
    latency = float(np.mean(cost[0]))
    energy = float(np.mean(cost[1]))
    total_cost = float(np.mean(cost[8]))
    return {
        "reward": -total_cost,
        "latency": latency,
        "energy": energy,
        "cost": total_cost,
        "deadline_miss": float(cost[9]) / V,
    }


def apply_bo_ratios(E, decisions):
    anchors = decisions.copy()
    anchors[:, 2] = 1.0
    offload = eval_pipeline(E, anchors)
    anchors[:, 2] = 0.0
    local = eval_pipeline(E, anchors)
    decisions[:, 2] = bo_offload.optimal_ratios_bo_batch(
        np.asarray(offload[0]).ravel(), np.asarray(local[0]).ravel(),
        np.asarray(offload[1]).ravel(), np.asarray(local[1]).ravel(),
        RB.GAMMA_1, RB.GAMMA_2)
    return decisions


def evaluate_quantum_once(V, agents, qtabs, rows, rng, cfg, selector):
    high, _, low = agents
    qh, qm, ql = qtabs
    row_h, row_m, row_l = rows
    E = RB.build_env(V, RB.lust_speeds(V, rng), "eval")
    decisions = np.zeros((V, 3), dtype=float)
    t_decision = time.perf_counter()
    for vehicle in range(V):
        state_index_h = H.get_starting_location(row_h)
        state_h = H.State_Space_H[state_index_h]
        state_index_m = H.get_starting_location(row_m)
        state_index_l = H.get_starting_location(row_l)
        state_l = H.State_Space_L[state_index_l]
        H.v_id = vehicle
        layer, node_action, ratio = 0, None, 0.0
        for _ in range(TIMESTEPS_EVAL):
            layer = high.act(state_h) if cfg["layer"] == "vqc" else high.act(state_h, qh)
            new_state_h, new_state_index_h = H.Next_State_H(
                layer, E["V_Ser_Req"][vehicle], H.RSU_T, E["NO_VUs_RSU"], RB.MAX_U_RSU,
                H.UAV_T, E["NO_VUs_UAV"], RB.MAX_U_UAV, H.HAP_T, E["NO_VUs_HAP"], RB.MAX_U_HAP,
                H.LEO_T, E["NO_VUs_LEO"], RB.MAX_U_LEO)
            node_action, _ = choose_node(E, layer, vehicle, state_index_m, qm, selector)
            new_state_m, new_state_index_m, _, t_soj = next_middle_state(
                E, layer, node_action, vehicle, state_index_m)
            ratio_id = low.act(state_l, ql)
            ratio = float(H.Action_Space_L[int(ratio_id)])
            H.v_id = vehicle
            learning = H.Learning_Cost(
                E["IP"], E["DL_Req"], t_soj, E["RSU_C_RA"], E["Rate_VR_R"],
                E["UAV_C_RA"], E["Rate_VU_R"], E["HAP_C_RA"], E["Rate_VH_R"],
                E["LEO_C_RA"], E["Rate_VL_R"], layer, node_action, ratio)
            new_state_l = [int(learning[6] > 0 or learning[8] > 0), int(learning[7] > 0), 0]
            state_index_l = int(np.argwhere((H.State_Space_L == new_state_l).all(axis=1))[0, 0])
            state_h = H.State_Space_H[new_state_index_h]
            state_index_h, state_index_m = new_state_index_h, new_state_index_m
            state_l = H.State_Space_L[state_index_l]
        decisions[vehicle] = [layer, int(np.nonzero(node_action)[0][0]), ratio]
    if cfg["ratio"] == "bo":
        decisions = apply_bo_ratios(E, decisions)
    decision_latency_ms = 1000.0 * (time.perf_counter() - t_decision) / V
    result = metrics_from_cost(eval_pipeline(E, decisions), V)
    result["decision_latency_ms"] = decision_latency_ms
    return result


def train_quantum(agents, qtabs, rows, cfg, seed, selector):
    high, middle, low = agents
    qh, qm, ql = qtabs
    row_h, row_m, row_l = rows
    E = RB.build_env(V_TRAIN, RB.lust_speeds(V_TRAIN, np.random.default_rng(seed)), "train")
    records = []
    for episode in range(EPISODES):
        t_episode = time.perf_counter()
        state_index_h = H.get_starting_location(row_h)
        state_h = H.State_Space_H[state_index_h]
        state_index_m = H.get_starting_location(row_m)
        state_index_l = H.get_starting_location(row_l)
        H.v_id = 0
        request = random.choice([1, 2, 3, 4, 5, 6])
        for _ in range(TIMESTEPS_TRAIN):
            layer = high.act(state_h) if cfg["layer"] == "vqc" else high.act(state_h, qh)
            new_state_h, new_state_index_h = H.Next_State_H(
                layer, request, H.RSU_T, E["NO_VUs_RSU"], RB.MAX_U_RSU,
                H.UAV_T, E["NO_VUs_UAV"], RB.MAX_U_UAV, H.HAP_T, E["NO_VUs_HAP"], RB.MAX_U_HAP,
                H.LEO_T, E["NO_VUs_LEO"], RB.MAX_U_LEO)
            node_action, global_ids = choose_node(E, layer, 0, state_index_m, qm, selector)
            new_state_m, new_state_index_m, _, t_soj = next_middle_state(
                E, layer, node_action, 0, state_index_m, request=request)
            ratio_id = H.get_next_action(state_index_l, RB.EPSILON_L, H.Action_Space_L, ql)
            ratio = float(H.Action_Space_L[int(ratio_id)])
            H.v_id = 0
            learning = H.Learning_Cost(
                E["IP"], E["DL_Req"], t_soj, E["RSU_C_RA"], E["Rate_VR_R"],
                E["UAV_C_RA"], E["Rate_VU_R"], E["HAP_C_RA"], E["Rate_VH_R"],
                E["LEO_C_RA"], E["Rate_VL_R"], layer, node_action, ratio)
            reward = 0.5 * learning[0] + 0.5 * learning[1]
            new_state_l = [int(learning[6] > 0 or learning[8] > 0), int(learning[7] > 0), 0]
            new_state_index_l = int(np.argwhere((H.State_Space_L == new_state_l).all(axis=1))[0, 0])
            low.store(H.State_Space_L[state_index_l], ratio_id, reward, np.asarray(new_state_l))
            middle.store(H.State_Space_M[state_index_m], global_ids, reward, new_state_m)
            high.store(H.State_Space_H[state_index_h], layer, reward, new_state_h)
            state_index_h, state_index_m, state_index_l = (
                new_state_index_h, new_state_index_m, new_state_index_l)
            state_h = H.State_Space_H[state_index_h]
        if cfg["layer"] == "vqc":
            high.align_target_model()
            loss = high.retrain(RB.BATCH_SIZE)
        else:
            high.alighn_target_model()
            loss = high.retrain(RB.BATCH_SIZE) if len(high.experience_replay) > RB.BATCH_SIZE else math.nan
        rollout = evaluate_quantum_once(
            V_TRAIN, agents, qtabs, rows, np.random.default_rng(seed + 100), cfg, selector)
        rollout.update({
            "episode": episode + 1,
            "loss": float(loss),
            "avg_q": (high.avg_max_q(H.State_Space_H) if cfg["layer"] == "vqc"
                      else RBF.avg_q_value(high)),
            "wall_clock_s": time.perf_counter() - t_episode,
        })
        records.append(rollout)
        print(f"  ep {episode + 1:02d}/{EPISODES} reward={rollout['reward']:.4f} "
              f"lat={rollout['latency']:.4f}s E={rollout['energy']:.4f}J "
              f"loss={rollout['loss']:.4g} t={rollout['wall_clock_s']:.2f}s", flush=True)
    return records


def normalize_classical_log(log):
    return [{
        "episode": int(r["episode"]), "reward": float(r["reward"]),
        "loss": float(r["loss"]), "latency": float(r["latency"]),
        "energy": float(r["energy"]), "cost": float(r["cost"]),
        "deadline_miss": None, "avg_q": float(r["avg_q_value"]),
        "decision_latency_ms": None, "wall_clock_s": float(r["wall_clock_s"]),
    } for r in log]


def parameter_counts(cfg):
    def dense_params(states, actions):
        return states * 10 + 10 * 50 + 50 + 50 * 50 + 50 + 50 * actions + actions
    classical_high = dense_params(len(H.State_Space_H), len(H.Action_Space_H))
    middle = dense_params(len(H.State_Space_M), H.RSU_T + H.UAV_T + H.HAP_T + H.LEO_T)
    low = dense_params(len(H.State_Space_L), len(H.Action_Space_L))
    high = 24 if cfg["layer"] == "vqc" else classical_high
    qaoa = qaoa_node.n_trainable_params() if cfg["node"] == "qaoa" else 0
    return {
        "high_level": high, "middle_level": middle, "low_level": low,
        "qaoa_variational": qaoa, "bo": 0,
        "total_trainable": high + middle + low + qaoa,
        "classical_high_reference": classical_high,
    }


def run_seed(name, seed):
    cfg = CONFIGS[name]
    set_seed(seed)
    agents = make_agents(seed, cfg["layer"] == "vqc")
    qtabs, rows = RB.make_q_tables()
    selector = qaoa_node.QAOANodeSelector() if cfg["node"] == "qaoa" else None
    t_train = time.perf_counter()
    if name == "Classical":
        # Exact existing baseline training loop; only seed and LuST pool are injected.
        RBF.SEED = seed
        RB.SEED = seed
        log = normalize_classical_log(RBF.train_full(agents, qtabs, rows))
    else:
        log = train_quantum(agents, qtabs, rows, cfg, seed, selector)
    training_time_s = time.perf_counter() - t_train

    evaluations = []
    rng = np.random.default_rng(seed + 1000)
    for V in V_LIST:
        for repeat in range(EVAL_REPEATS):
            if name == "Classical":
                started = time.perf_counter()
                base = RB.eval_hrl_once(V, agents, qtabs, rows, rng)
                metric = {
                    "reward": -base["cost"], "latency": base["latency"],
                    "energy": base["energy"], "cost": base["cost"],
                    "deadline_miss": base["ser_fail"] / V,
                    "decision_latency_ms": 1000.0 * (time.perf_counter() - started) / V,
                }
            else:
                metric = evaluate_quantum_once(V, agents, qtabs, rows, rng, cfg, selector)
            metric.update({"V": V, "repeat": repeat + 1})
            evaluations.append(metric)
    final = {key: float(np.mean([r[key] for r in evaluations])) for key in
             ["reward", "latency", "energy", "cost", "deadline_miss", "decision_latency_ms"]}
    final["avg_q"] = float(log[-1]["avg_q"])
    final["training_time_s"] = float(training_time_s)
    return {
        "name": name, "seed": seed, "config": cfg, "training_log": log,
        "evaluations": evaluations, "final": final,
        "parameters": parameter_counts(cfg),
    }


def checkpoint_path(name):
    return RESULTS / ("_v2_" + name.replace(" ", "_").replace("/", "_") + ".json")


def run_one(name):
    path = checkpoint_path(name)
    data = {"schema_version": SCHEMA_VERSION, "runs": []}
    if path.exists():
        loaded = json.loads(path.read_text())
        if loaded.get("schema_version") == SCHEMA_VERSION:
            data = loaded
    completed = {r["seed"] for r in data["runs"]}
    for seed in SEEDS:
        if seed in completed:
            continue
        print(f"\n[{name}] seed={seed}", flush=True)
        run = run_seed(name, seed)
        data["runs"].append(run)
        path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        f = run["final"]
        print(f"[{name}] seed={seed} DONE | reward={f['reward']:.4f} "
              f"lat={f['latency']:.4f}s E={f['energy']:.4f}J "
              f"miss={f['deadline_miss']:.4f} train={f['training_time_s']:.1f}s", flush=True)


def summarize(runs):
    result = {}
    keys = ["reward", "latency", "energy", "cost", "deadline_miss", "avg_q",
            "training_time_s", "decision_latency_ms"]
    for name in ORDER:
        group = [r for r in runs if r["name"] == name]
        row = {"n_seeds": len(group)}
        for key in keys:
            values = np.asarray([r["final"][key] for r in group], dtype=float)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["n_params"] = group[0]["parameters"]["total_trainable"]
        row["high_level_params"] = group[0]["parameters"]["high_level"]
        result[name] = row
    return result


def print_summary_table(name, group):
    keys = ["reward", "latency", "energy", "cost", "deadline_miss", "training_time_s"]
    print(f"\n=== TÓM TẮT SAU {name} ({len(group)} seeds) ===")
    print("Metric                 Mean           Std")
    print("------------------------------------------")
    for key in keys:
        values = np.asarray([r["final"][key] for r in group])
        print(f"{key:20s} {values.mean():12.5f} {values.std(ddof=1):12.5f}")
    print(f"trainable_parameters {group[0]['parameters']['total_trainable']:12d}", flush=True)


def welch_tests(runs):
    tests = {}
    for key in ["reward", "latency", "energy", "cost", "deadline_miss"]:
        a = np.asarray([r["final"][key] for r in runs if r["name"] == "Classical"])
        b = np.asarray([r["final"][key] for r in runs if r["name"] == "Full-QHRL"])
        test = stats.ttest_ind(a, b, equal_var=False)
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                           (len(a) + len(b) - 2))
        d = float((a.mean() - b.mean()) / pooled) if pooled else math.nan
        correction = 1.0 - 3.0 / (4.0 * (len(a) + len(b)) - 9.0)
        tests[key] = {
            "classical_mean": float(a.mean()), "full_qhrl_mean": float(b.mean()),
            "t_stat": float(test.statistic), "df": float(test.df), "p_value": float(test.pvalue),
            "hedges_g": d * correction, "significant_0p05": bool(test.pvalue < 0.05),
            "shapiro_classical_p": float(stats.shapiro(a).pvalue),
            "shapiro_full_qhrl_p": float(stats.shapiro(b).pvalue),
            "levene_p": float(stats.levene(a, b, center="median").pvalue),
        }
    return {"comparison": "Classical vs Full-QHRL", "test": "two-sided Welch t-test",
            "alpha": 0.05, "n_per_group": len(SEEDS), "tests": tests}


def export_results(runs, elapsed):
    summary = summarize(runs)
    raw = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "dataset": "LuST 24h", "speed_pool": FULL_POOL.name,
            "seeds": SEEDS, "episodes": EPISODES, "timesteps_train": TIMESTEPS_TRAIN,
            "timesteps_eval": TIMESTEPS_EVAL, "V_train": V_TRAIN,
            "V_eval": V_LIST, "eval_repeats": EVAL_REPEATS, "configs": CONFIGS,
            "baseline_hyperparameters_changed": False,
        },
        "runs": runs, "summary": summary, "total_orchestration_wall_clock_s": elapsed,
    }
    (RESULTS / "raw_results.json").write_text(json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8")

    columns = ["scenario", "reward_mean", "reward_std", "latency_mean", "latency_std",
               "energy_mean", "energy_std", "cost_mean", "cost_std",
               "deadline_miss_mean", "deadline_miss_std", "avg_q_mean", "avg_q_std",
               "training_time_s_mean", "training_time_s_std", "decision_latency_ms_mean",
               "decision_latency_ms_std", "high_level_params", "n_params", "n_seeds"]
    with (RESULTS / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name in ORDER:
            writer.writerow({"scenario": name, **{k: summary[name][k] for k in columns[1:]}})

    with (RESULTS / "parameter_count.csv").open("w", newline="") as f:
        columns_p = ["scenario", "layer_mechanism", "high_level", "middle_level", "low_level",
                     "qaoa_variational", "bo", "total_trainable", "classical_high_reference"]
        writer = csv.DictWriter(f, fieldnames=columns_p)
        writer.writeheader()
        for name in ORDER:
            run = next(r for r in runs if r["name"] == name)
            writer.writerow({"scenario": name, "layer_mechanism": CONFIGS[name]["layer"],
                             **run["parameters"]})

    labels = {"Full-QHRL": "Full Quantum-HRL", "w/o VQC": "w/o VQC",
              "w/o QAOA": "w/o QAOA", "HRL+VQC+QAOA": "w/o Bayesian Optimization"}
    with (RESULTS / "ablation_results.csv").open("w", newline="") as f:
        columns_a = ["ablation", "reward_mean", "reward_std", "latency_mean", "latency_std",
                     "energy_mean", "energy_std", "cost_mean", "cost_std",
                     "deadline_miss_mean", "deadline_miss_std"]
        writer = csv.DictWriter(f, fieldnames=columns_a)
        writer.writeheader()
        for name in ABLATION_ORDER:
            writer.writerow({"ablation": labels[name],
                             **{k: summary[name][k] for k in columns_a[1:]}})

    (RESULTS / "statistical_tests.json").write_text(
        json.dumps(welch_tests(runs), indent=2, allow_nan=False), encoding="utf-8")


def main():
    started = time.perf_counter()
    print(f"LuST-24h pool: {RB.SPEED_POOL.size} samples; mean={RB.SPEED_POOL.mean():.3f} m/s")
    print(f"Experiment: 6 configs × {len(SEEDS)} seeds × {EPISODES} episodes × "
          f"{TIMESTEPS_TRAIN} timesteps")
    for name in ORDER:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name], check=True)
        checkpoint = json.loads(checkpoint_path(name).read_text())
        if len(checkpoint["runs"]) != len(SEEDS):
            raise RuntimeError(f"{name}: only {len(checkpoint['runs'])}/{len(SEEDS)} seeds complete")
        print_summary_table(name, checkpoint["runs"])
    runs = []
    for name in ORDER:
        runs.extend(json.loads(checkpoint_path(name).read_text())["runs"])
    export_results(runs, time.perf_counter() - started)
    print("\nAll scenario and ablation result files have been exported.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        run_one(sys.argv[2])
    else:
        main()
