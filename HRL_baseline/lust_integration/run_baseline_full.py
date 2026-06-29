"""
run_baseline_full.py
====================
Train baseline HRL trên TOÀN BỘ mobility LuST 24h (pool tốc độ gộp từ mobility_24h.py).

NGUYÊN TẮC GIỮ NGUYÊN BASELINE:
  * Thuật toán/agent/không gian trạng thái: import nguyên văn từ hrl_core (qua run_baseline).
  * Hyperparameter: dùng đúng hằng số trong run_baseline (cell 23 + cell 28 notebook gốc).
  * KHÔNG sửa reward / update rule / DQN / hyperparameter. CHỈ thêm logging + export.

Điểm tích hợp LuST DUY NHẤT: VN_spd lấy mẫu từ pool tốc độ 24h thật (lust_speed_pool_full.npy).

Log mỗi episode (yêu cầu #6): reward, loss, latency, energy, cost, avg_q_value, wall_clock_time.
"""
import os
import sys
import json
import time
import random
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "figure_baseline", "results")
sys.path.insert(0, HERE)

import run_baseline as RB   # tái dùng FastAgent/build_env/eval_hrl_once/make_q_tables + hyperparameter
import hrl_core as H
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

# --- Nạp pool tốc độ 24h và GHI ĐÈ pool của run_baseline (đó là điểm tiêm mobility) ---
FULL_POOL_PATH = os.path.join(RESULTS, "lust_speed_pool_full.npy")
if not os.path.exists(FULL_POOL_PATH):
    raise SystemExit("Thiếu results/lust_speed_pool_full.npy — hãy chạy mobility_24h.py trước.")
FULL_POOL = np.load(FULL_POOL_PATH)
RB.SPEED_POOL = FULL_POOL   # eval_hrl_once / build_env sẽ lấy mẫu VN_spd từ pool 24h này
print(f"[LuST-24h] speed pool: {FULL_POOL.size} mẫu, mean={FULL_POOL.mean():.2f} m/s, "
      f"median={np.median(FULL_POOL):.2f} m/s", flush=True)

# Hyperparameter (lấy lại từ run_baseline để đảm bảo ĐỒNG NHẤT với baseline gốc)
NUM_EPISODES = RB.NUM_EPISODES
TIMESTEPS_TRAIN = RB.TIMESTEPS_TRAIN
EPSILON_M, EPSILON_L = RB.EPSILON_M, RB.EPSILON_L
MAX_U_RSU, MAX_U_UAV, MAX_U_HAP, MAX_U_LEO = RB.MAX_U_RSU, RB.MAX_U_UAV, RB.MAX_U_HAP, RB.MAX_U_LEO
BATCH_SIZE = RB.BATCH_SIZE
V_TRAIN = RB.V_TRAIN
SEED = RB.SEED


def avg_q_value(agent):
    """Chỉ số quan trắc (không thuộc thuật toán): trung bình max-Q của q_network trên State_Space_H."""
    qs = []
    for s in H.State_Space_H:
        out = agent.q_network(s, training=False).numpy()
        qs.append(float(np.max(out[0])))
    return float(np.mean(qs))


def train_full(agents, qtabs, rows):
    """Vòng huấn luyện cell-24 (LS-DQN 0.5/0.5) NGUYÊN VĂN + logging đầy đủ mỗi episode."""
    agent_h, agent_m, agent_l = agents
    q_values_h, q_values_m, q_values_l = qtabs
    environment_rows_h, environment_rows_m, environment_rows_l = rows

    E = RB.build_env(V_TRAIN, RB.lust_speeds(V_TRAIN, np.random.default_rng(SEED)), "train")
    IP = E["IP"]
    log = []
    for episode in range(NUM_EPISODES):
        t_ep = time.time()
        state_index_h = H.get_starting_location(environment_rows_h)
        state_h = H.State_Space_H[state_index_h]
        state_index_m = H.get_starting_location(environment_rows_m)
        state_index_l = H.get_starting_location(environment_rows_l)
        v_id = 0; H.v_id = v_id
        V_spd = E["VN_spd"][v_id]
        Req_Ser = random.choice([1, 2, 3, 4, 5, 6])

        for timestep in range(TIMESTEPS_TRAIN):
            action_index_h = agent_h.act(state_h, q_values_h)
            new_state_h, new_state_index_h = H.Next_State_H(
                action_index_h, Req_Ser, H.RSU_T, E["NO_VUs_RSU"], MAX_U_RSU,
                H.UAV_T, E["NO_VUs_UAV"], MAX_U_UAV, H.HAP_T, E["NO_VUs_HAP"], MAX_U_HAP,
                H.LEO_T, E["NO_VUs_LEO"], MAX_U_LEO)
            M_A = H.M_Action_Space(action_index_h, v_id, E["VN_RSU_asign"],
                                   E["VN_UAV_asign"], E["VN_HAP_asign"], E["VN_LEO_asign"])
            nz = np.nonzero(M_A)[1]
            off = [0, H.RSU_T, H.RSU_T + H.UAV_T, H.RSU_T + H.UAV_T + H.HAP_T][action_index_h]
            nonzero_indices_M_A_l = off + nz
            action_index_m = H.get_next_action_M(state_index_m, nonzero_indices_M_A_l,
                                                 EPSILON_M, M_A, q_values_m)
            new_state_m, new_state_index_m, col_index_m, T_soj_me = H.Next_State_M(
                action_index_h, action_index_m, V_spd, Req_Ser,
                E["NO_VUs_RSU"], MAX_U_RSU, H.service_allocation_RSU, E["VN_RSU_Soj"][v_id], H.RSU_r,
                E["NO_VUs_UAV"], MAX_U_UAV, H.service_allocation_UAV, E["VN_UAV_Soj"][v_id], H.UAV_r,
                E["NO_VUs_HAP"], MAX_U_HAP, H.service_allocation_HAP, E["VN_HAP_Soj"][v_id], H.HAP_r,
                E["NO_VUs_LEO"], MAX_U_LEO, H.service_allocation_LEO, E["VN_LEO_Soj"][v_id], H.LEO_r)
            action_index_l_id = H.get_next_action(state_index_l, EPSILON_L, H.Action_Space_L, q_values_l)
            action_l = H.Action_Space_L[int(action_index_l_id)]
            H.v_id = v_id
            out_l = H.Learning_Cost(IP, E["DL_Req"], T_soj_me,
                                    E["RSU_C_RA"], E["Rate_VR_R"], E["UAV_C_RA"], E["Rate_VU_R"],
                                    E["HAP_C_RA"], E["Rate_VH_R"], E["LEO_C_RA"], E["Rate_VL_R"],
                                    action_index_h, action_index_m, action_l)
            Tot_LC, Tot_EC = out_l[0], out_l[1]
            F1_V, F2_V, F3_V = out_l[6], out_l[7], out_l[8]
            new_state_l = [0, 0, 0]
            if F1_V > 0: new_state_l[0] = 1
            if F2_V > 0: new_state_l[1] = 1
            if F3_V > 0: new_state_l[0] = 1
            new_state_index_l = np.argwhere((H.State_Space_L == new_state_l).all(axis=1))[0, 0]
            reward_l = 0.5 * Tot_LC + 0.5 * Tot_EC
            agent_l.store(H.State_Space_L[state_index_l], action_index_l_id, reward_l, np.array(new_state_l))
            agent_m.store(H.State_Space_M[state_index_m], nonzero_indices_M_A_l, reward_l, new_state_m)
            agent_h.store(H.State_Space_H[state_index_h], action_index_h, reward_l, new_state_h)
            state_index_h, state_index_m, state_index_l = new_state_index_h, new_state_index_m, new_state_index_l
            state_h = H.State_Space_H[state_index_h]
            if timestep == TIMESTEPS_TRAIN - 1:
                agent_h.alighn_target_model()
                if len(agent_h.experience_replay) > BATCH_SIZE:
                    agent_h.retrain(BATCH_SIZE)

        # --- Logging mỗi episode (instrumentation; reward đo bằng pipeline đánh giá thật) ---
        eval_rng = np.random.default_rng(SEED + 100)
        roll = RB.eval_hrl_once(V_TRAIN, agents, qtabs, rows, eval_rng)
        rec = dict(
            episode=episode + 1,
            reward=float(-roll["cost"]),
            loss=float(getattr(agent_h, "last_loss", np.nan)),
            latency=float(roll["latency"]),
            energy=float(roll["energy"]),
            cost=float(roll["cost"]),
            avg_q_value=avg_q_value(agent_h),
            wall_clock_s=round(time.time() - t_ep, 3),
        )
        log.append(rec)
        print(f"  [full] ep {rec['episode']:2d}/{NUM_EPISODES} | reward={rec['reward']:.3f} "
              f"loss={rec['loss']:.3g} lat={rec['latency']:.3f}s E={rec['energy']:.3f}J "
              f"cost={rec['cost']:.3f} avgQ={rec['avg_q_value']:.2f} t={rec['wall_clock_s']}s",
              flush=True)
    return log


def main():
    t0 = time.time()
    state_size_h = len(H.State_Space_H); action_size_h = len(H.Action_Space_H)
    state_size_m = len(H.State_Space_M); action_size_m = H.RSU_T + H.UAV_T + H.HAP_T + H.LEO_T
    state_size_l = len(H.State_Space_L); action_size_l = len(H.Action_Space_L)
    agent_h = RB.FastAgent(RB.OPTIMIZER, state_size_h, action_size_h, RB.GAMMA_H, RB.EPSILON_H)
    agent_m = RB.FastAgent(RB.OPTIMIZER, state_size_m, action_size_m, RB.GAMMA_M, RB.EPSILON_M)
    agent_l = RB.FastAgent(RB.OPTIMIZER, state_size_l, action_size_l, RB.GAMMA_L, RB.EPSILON_L)
    agents = (agent_h, agent_m, agent_l)
    qtabs, rows = RB.make_q_tables()

    print("\n=== TRAIN baseline HRL trên TOÀN BỘ LuST 24h ===", flush=True)
    log = train_full(agents, qtabs, rows)

    # ---- Export ----
    keys = ["episode", "reward", "loss", "latency", "energy", "cost", "avg_q_value", "wall_clock_s"]
    with open(os.path.join(RESULTS, "training_log_full_lust.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in log: w.writerow(r)

    arr = {k: [r[k] for r in log] for k in keys}
    last = log[-1]; tail = log[max(0, len(log)-10):]
    def tmean(k): return float(np.mean([r[k] for r in tail]))
    mob = {}
    for fn in ("mobility_full_summary.json", "per_window_mobility.json"):
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p):
            mob[fn] = json.load(open(p, encoding="utf-8"))
    raw = dict(
        config=dict(num_episodes=NUM_EPISODES, timesteps_per_episode=TIMESTEPS_TRAIN,
                    V_train=V_TRAIN, seed=SEED,
                    gamma=dict(H=RB.GAMMA_H, M=RB.GAMMA_M, L=RB.GAMMA_L),
                    epsilon=dict(H=RB.EPSILON_H, M=EPSILON_M, L=EPSILON_L),
                    optimizer=RB.OPTIMIZER, batch_size=BATCH_SIZE,
                    speed_source="lust_speed_pool_full.npy (LuST 24h, FCD period=60)"),
        per_episode=arr,
        final_last_episode=last,
        final_mean_last10=dict(reward=tmean("reward"), latency=tmean("latency"),
                               energy=tmean("energy"), cost=tmean("cost"),
                               avg_q_value=tmean("avg_q_value")),
        total_wall_clock_s=round(time.time() - t0, 1),
        mobility=mob.get("mobility_full_summary.json", {}),
    )
    with open(os.path.join(RESULTS, "raw_results_full_lust.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    # summary_full_lust.csv = bảng mobility theo từng cửa sổ (low/med/high)
    pw = mob.get("per_window_mobility.json")
    if pw:
        cols = ["window", "clock", "traffic_class", "n_vehicles_departing",
                "n_distinct_fcd", "concurrent_mean", "concurrent_max",
                "speed_mean", "speed_median", "n_speed_samples"]
        with open(os.path.join(RESULTS, "summary_full_lust.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in pw:
                w.writerow([r.get(c, "") for c in cols])

    print(f"\nXong train full trong {(time.time()-t0)/60:.1f} phút.")
    print("  -> results/training_log_full_lust.csv, raw_results_full_lust.json, summary_full_lust.csv")


if __name__ == "__main__":
    main()
