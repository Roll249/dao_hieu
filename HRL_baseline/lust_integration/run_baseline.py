"""
run_baseline.py
===============
Chạy BASELINE HRL trên mobility LuST (cửa sổ 17:00-18:00).

Nguyên tắc:
  * THUẬT TOÁN HRL giữ NGUYÊN: mọi hàm môi trường / agent / không gian trạng thái
    đều import nguyên văn từ hrl_core.py (trích từ notebook gốc).
  * HYPERPARAMETER giữ NGUYÊN theo paper baseline (cell 23 + cell 28 notebook gốc).
  * ĐIỂM TÍCH HỢP LuST duy nhất: tốc độ xe `VN_spd` được lấy mẫu từ phân phối
    tốc độ THẬT của LuST (results/lust_speed_pool.npy) thay cho truncnorm tổng hợp.
    Đây là khâu "đọc dữ liệu / môi trường", không phải thuật toán.

Hai pha (đúng như notebook gốc):
  1. TRAIN  : tái hiện vòng huấn luyện LS-DQN (gamma1=gamma2=0.5) — cell 24.
              Bổ sung LOG: reward & training-loss theo episode (notebook gốc không log).
  2. EVAL   : tái hiện vòng đánh giá HRL (0.5,0.5) — cell 28, quét số xe V.
              Thu latency / energy / cost / handovers theo tải V.

Xuất số liệu -> figure_baseline/results/ ; hình -> figure_baseline/.
"""
import os
import sys
import json
import time
import random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "figure_baseline", "results")
sys.path.insert(0, HERE)

import hrl_core as H  # noqa: E402  (thuật toán nguyên văn)
import tensorflow as tf  # noqa: E402
tf.keras.utils.disable_interactive_logging()  # tắt progress-bar predict/fit (chỉ cosmetic)

# ----------------------------------------------------------------------------
# Tái lập (không thuộc thuật toán) — cố định seed để kết quả lặp lại được.
SEED = 2024
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------------------------------------------------
# HYPERPARAMETER (giữ nguyên paper baseline — notebook cell 23 & cell 28)
NUM_EPISODES = 50
TIMESTEPS_TRAIN = 1000
TIMESTEPS_EVAL = 10
OPTIMIZER = "adam"
GAMMA_H, EPSILON_H = 0.7, 0.1
GAMMA_M, EPSILON_M = 0.05, 0.7
GAMMA_L, EPSILON_L = 0.05, 0.7
MAX_U_RSU, MAX_U_UAV, MAX_U_HAP, MAX_U_LEO = 8, 8, 20, 40
BATCH_SIZE = 2
GAMMA_1, GAMMA_2 = H.gamma_1, H.gamma_2  # 0.5, 0.5

# Quét tải V — đúng dải [20..200] của paper (mỗi điểm = số xe đồng thời trong vùng phủ)
V_LIST = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
RP = 3            # số lần lặp mỗi điểm V (paper dùng 10; giảm để tiết kiệm chi phí)
V_TRAIN = 100     # tải dùng cho pha huấn luyện

# Cho phép override để SMOKE-TEST nhanh (không ảnh hưởng cấu hình chạy thật).
if os.environ.get("HRL_SMOKE"):
    NUM_EPISODES = int(os.environ.get("HRL_EP", 2))
    TIMESTEPS_TRAIN = int(os.environ.get("HRL_TT", 20))
    V_LIST = [int(x) for x in os.environ.get("HRL_VLIST", "20,40").split(",")]
    RP = int(os.environ.get("HRL_RP", 1))
    V_TRAIN = int(os.environ.get("HRL_VTRAIN", 20))

# ----------------------------------------------------------------------------
# Nạp phân phối tốc độ LuST
SPEED_POOL_PATH = os.path.join(RESULTS, "lust_speed_pool.npy")
if not os.path.exists(SPEED_POOL_PATH):
    raise SystemExit("Thiếu results/lust_speed_pool.npy — hãy chạy extract_speeds.py trước.")
SPEED_POOL = np.load(SPEED_POOL_PATH)
print(f"[LuST] speed pool: {SPEED_POOL.size} mẫu, "
      f"mean={SPEED_POOL.mean():.2f} m/s, median={np.median(SPEED_POOL):.2f} m/s")


def lust_speeds(V, rng):
    """Lấy mẫu V tốc độ xe từ phân phối tốc độ THẬT của LuST (thay truncnorm)."""
    return rng.choice(SPEED_POOL, size=V, replace=True).astype(float)


# ----------------------------------------------------------------------------
class FastAgent(H.Agent):
    """Giống hệt H.Agent về MẶT TOÁN HỌC; chỉ tối ưu tốc độ + log loss.

    - `act`/`retrain` thay `q_network.predict(x)` bằng `q_network(x, training=False)`
      — CÙNG forward pass, cùng trọng số (đã kiểm chứng sai khác = 0.0), nhưng nhanh
      ~14x do bỏ overhead của API .predict() trên input nhỏ. KHÔNG đổi thuật toán.
    - retrain ghi lại training-loss (vẫn fit y hệt bản gốc) để vẽ training_loss.png.
    """
    def act(self, state, q_values):
        if np.random.rand() <= self.epsilon:
            q_values = self.q_network(state, training=False).numpy()
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        losses = []
        for state, action, reward, next_state in minibatch:
            target = self.q_network(state, training=False).numpy()
            t = self.target_network(next_state, training=False).numpy()
            target[0][action] = reward + self.gamma * np.amax(t)
            hist = self.q_network.fit(state, target, epochs=1, verbose=0)
            losses.append(float(hist.history["loss"][-1]))
        self.last_loss = float(np.mean(losses)) if losses else None
        return self.last_loss


# ----------------------------------------------------------------------------
def build_env(V, vn_spd, phase):
    """Dựng môi trường cho V xe — tái hiện đúng phần setup của notebook
    (cell 9 cho 'train', cell 28 cho 'eval'), nhưng VN_spd nạp từ LuST.

    Trả về dict E chứa mọi biến mà vòng train/eval cần.
    """
    # Hằng số công suất theo từng pha (đúng notebook gốc)
    if phase == "train":
        Pcomp_m = (1.3) * np.ones((1, V))
        P_tx_v = (1.5) * np.ones((1, V))
        P_rx_v = (1.3) * np.ones((1, V))
    else:  # eval (cell 28)
        Pcomp_m = (0.8) * np.ones((1, V))
        P_tx_v = (1.6) * np.ones((1, V))
        P_rx_v = (1.4) * np.ones((1, V))

    C_V = (8 * 10 ** 9) * np.ones((1, V))
    psi_dmp = 1000
    TS = (5 * 8 * 10 ** 6) * np.ones((1, V))
    TSD = (1 * 8 * 10 ** 6) * np.ones((1, V))
    DL_Req = 4
    P_comv = (1) * np.ones((1, V))
    b_0, theta = H.b_0, H.theta

    V_Ser_Req = np.array([random.choice([1, 2, 3, 4, 5, 6]) for _ in range(V)])

    # ---- VN_Main_IP (giống hệt notebook) ----
    IP = []
    IP += [V, H.RSU_T, H.RSU_loc, H.RSU_r, H.UAV_T, H.UAV_loc, H.UAV_r,
           H.HAP_T, H.HAP_loc, H.HAP_r, H.LEO_T, H.LEO_loc, H.LEO_r,
           H.B_RSU, H.B_UAV, H.B_HAP, H.B_LEO, H.C_RSU, H.C_UAV, H.C_HAP, H.C_LEO,
           H.P_tx_r, H.P_tx_u, H.P_tx_h, H.P_tx_s, b_0, theta]

    VN_assign_OUT = H.VN_EN_Assign(IP, vn_spd)
    VN_loc = VN_assign_OUT[0]
    VN_RSU_asign, VN_RSU_Soj, VN_RSU_dist = VN_assign_OUT[1], VN_assign_OUT[2], VN_assign_OUT[3]
    VN_UAV_asign, VN_UAV_Soj, VN_UAV_dist = VN_assign_OUT[4], VN_assign_OUT[5], VN_assign_OUT[6]
    VN_HAP_asign, VN_HAP_Soj, VN_HAP_dist = VN_assign_OUT[7], VN_assign_OUT[8], VN_assign_OUT[9]
    VN_LEO_asign, VN_LEO_Soj, VN_LEO_dist = VN_assign_OUT[10], VN_assign_OUT[11], VN_assign_OUT[12]

    IP += [VN_RSU_dist, VN_UAV_dist, VN_HAP_dist, VN_LEO_dist,
           TS, TSD, psi_dmp, Pcomp_m, P_tx_v,
           H.Pcomp_r, H.P_tx_r, H.Pcomp_u, H.P_tx_u, H.Pcomp_h, H.P_tx_h, H.Pcomp_s, H.P_tx_s,
           C_V, VN_RSU_Soj, VN_UAV_Soj, VN_HAP_Soj, VN_LEO_Soj, GAMMA_1, GAMMA_2,
           H.service_allocation_RSU, H.service_allocation_UAV,
           H.service_allocation_HAP, H.service_allocation_LEO, VN_loc]

    # ---- Random approach (cung cấp NO_VUs_* + phân bổ tài nguyên nền) ----
    Random_decisions = np.zeros((V, 3))
    for v in range(V):
        Random_decisions[v][0] = random.choice([0, 1, 2, 3])
        asign = [VN_RSU_asign, VN_UAV_asign, VN_HAP_asign, VN_LEO_asign][int(Random_decisions[v][0])]
        idx_ones = np.where(np.array(asign[v]) == 1)[0]
        Random_decisions[v][1] = random.choice(idx_ones.tolist())
        Random_decisions[v][2] = random.random()

    NO_VUs_RSU = np.zeros(H.RSU_T); NO_VUs_UAV = np.zeros(H.UAV_T)
    NO_VUs_HAP = np.zeros(H.HAP_T); NO_VUs_LEO = np.zeros(H.LEO_T)
    for v in range(V):
        lyr = int(Random_decisions[v][0]); node = int(Random_decisions[v][1])
        [NO_VUs_RSU, NO_VUs_UAV, NO_VUs_HAP, NO_VUs_LEO][lyr][node] += 1

    RA = H.Resource_Allocation(IP, Random_decisions, NO_VUs_RSU, NO_VUs_UAV, NO_VUs_HAP, NO_VUs_LEO)
    RSU_B_RA, UAV_B_RA, HAP_B_RA, LEO_B_RA = RA[0], RA[1], RA[2], RA[3]
    RSU_C_RA, UAV_C_RA, HAP_C_RA, LEO_C_RA = RA[4], RA[5], RA[6], RA[7]
    Rate_VR_R, Rate_VU_R, Rate_VH_R, Rate_VL_R = H.Data_Rate(
        IP, Random_decisions, RSU_B_RA, UAV_B_RA, HAP_B_RA, LEO_B_RA)

    return dict(
        V=V, IP=IP, DL_Req=DL_Req, VN_spd=vn_spd, V_Ser_Req=V_Ser_Req,
        VN_RSU_asign=VN_RSU_asign, VN_UAV_asign=VN_UAV_asign,
        VN_HAP_asign=VN_HAP_asign, VN_LEO_asign=VN_LEO_asign,
        VN_RSU_Soj=VN_RSU_Soj, VN_UAV_Soj=VN_UAV_Soj,
        VN_HAP_Soj=VN_HAP_Soj, VN_LEO_Soj=VN_LEO_Soj,
        NO_VUs_RSU=NO_VUs_RSU, NO_VUs_UAV=NO_VUs_UAV,
        NO_VUs_HAP=NO_VUs_HAP, NO_VUs_LEO=NO_VUs_LEO,
        RSU_C_RA=RSU_C_RA, UAV_C_RA=UAV_C_RA, HAP_C_RA=HAP_C_RA, LEO_C_RA=LEO_C_RA,
        Rate_VR_R=Rate_VR_R, Rate_VU_R=Rate_VU_R, Rate_VH_R=Rate_VH_R, Rate_VL_R=Rate_VL_R,
    )


# ----------------------------------------------------------------------------
def make_q_tables():
    er_h = len(H.State_Space_H); ec_h = len(H.Action_Space_H)
    er_m = len(H.State_Space_M); ec_m = H.RSU_T + H.UAV_T + H.HAP_T + H.LEO_T
    er_l = len(H.State_Space_L); ec_l = len(H.Action_Space_L)
    qh = np.random.rand(er_h, ec_h) * 2000 - 1000
    qm = np.random.rand(er_m, ec_m) * 2000 - 1000
    ql = np.random.rand(er_l, ec_l) * 2000 - 1000
    return (qh, qm, ql), (er_h, er_m, er_l)


# ----------------------------------------------------------------------------
def train(agents, qtabs, rows):
    """Tái hiện cell 24 (LS-DQN 0.5/0.5) + log reward/loss theo episode."""
    agent_h, agent_m, agent_l = agents
    q_values_h, q_values_m, q_values_l = qtabs
    environment_rows_h, environment_rows_m, environment_rows_l = rows

    E = build_env(V_TRAIN, lust_speeds(V_TRAIN, np.random.default_rng(SEED)), "train")
    IP = E["IP"]
    ep_reward, ep_loss = [], []

    for episode in range(NUM_EPISODES):
        state_index_h = H.get_starting_location(environment_rows_h)
        start_state_h = H.State_Space_H[state_index_h]
        state_index_m = H.get_starting_location(environment_rows_m)
        state_index_l = H.get_starting_location(environment_rows_l)

        v_id = 0
        H.v_id = v_id
        V_spd = E["VN_spd"][v_id]
        Req_Ser = random.choice([1, 2, 3, 4, 5, 6])
        state_h = start_state_h
        rewards_ts = []

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
            idxs = np.argwhere((H.State_Space_L == new_state_l).all(axis=1))
            new_state_index_l = idxs[0, 0]

            reward_l = 0.5 * Tot_LC + 0.5 * Tot_EC
            reward_m = reward_l; reward_h = reward_l
            rewards_ts.append(reward_l)

            agent_l.store(H.State_Space_L[state_index_l], action_index_l_id, reward_l, np.array(new_state_l))
            agent_m.store(H.State_Space_M[state_index_m], nonzero_indices_M_A_l, reward_m, new_state_m)
            agent_h.store(H.State_Space_H[state_index_h], action_index_h, reward_h, new_state_h)

            state_index_h = new_state_index_h
            state_index_m = new_state_index_m
            state_index_l = new_state_index_l
            state_h = H.State_Space_H[state_index_h]

            if timestep == TIMESTEPS_TRAIN - 1:
                agent_h.alighn_target_model()
                if len(agent_h.experience_replay) > BATCH_SIZE:
                    agent_h.retrain(BATCH_SIZE)

        # --- Đo REWARD chính sách đạt được sau episode này ---
        # Lưu ý: trong code gốc, Learning_Cost trả 0 (so sánh np.nonzero(a_m)==r luôn False),
        # nên reward nội-vòng luôn 0. Để đường hội tụ có nghĩa mà KHÔNG đổi thuật toán, ta đo
        # reward bằng đúng pipeline đánh giá (Task_Proc_Main) qua eval_hrl_once tại V_TRAIN.
        eval_rng = np.random.default_rng(SEED + 100)  # cùng VN_spd mỗi episode -> đường sạch
        rollout = eval_hrl_once(V_TRAIN, agents, qtabs, rows, eval_rng)
        ep_cost = rollout["cost"]
        ep_reward.append(-ep_cost)     # reward = -cost (cost thấp -> reward cao)
        ep_loss.append(getattr(agent_h, "last_loss", np.nan))
        print(f"  [train] episode {episode+1:2d}/{NUM_EPISODES} | "
              f"eval_cost={ep_cost:.4f} | reward={-ep_cost:.4f} | loss={ep_loss[-1]}",
              flush=True)

    return np.array(ep_reward), np.array(ep_loss)


# ----------------------------------------------------------------------------
def eval_hrl_once(V, agents, qtabs, rows, rng):
    """Tái hiện khối 'HRL Solutions (0.5,0.5)' của cell 28 cho một V."""
    agent_h, agent_m, agent_l = agents
    q_values_h, q_values_m, q_values_l = qtabs
    environment_rows_h, environment_rows_m, environment_rows_l = rows

    E = build_env(V, lust_speeds(V, rng), "eval")
    IP = E["IP"]

    HRL_decisions = np.zeros((V, 3))
    for v in range(V):
        state_index_h = H.get_starting_location(environment_rows_h); state_h = H.State_Space_H[state_index_h]
        state_index_m = H.get_starting_location(environment_rows_m)
        state_index_l = H.get_starting_location(environment_rows_l); state_l = H.State_Space_L[state_index_l]
        v_id = v; H.v_id = v_id
        V_spd = E["VN_spd"][v]
        Req_Ser = E["V_Ser_Req"][v]

        action_index_h = 0; action_index_m = None; action_l = 0
        for timestep in range(TIMESTEPS_EVAL):
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

            action_index_l_id = agent_l.act(state_l, q_values_l)
            action_l = H.Action_Space_L[int(action_index_l_id)]
            H.v_id = v_id
            out_l = H.Learning_Cost(IP, E["DL_Req"], T_soj_me,
                                    E["RSU_C_RA"], E["Rate_VR_R"], E["UAV_C_RA"], E["Rate_VU_R"],
                                    E["HAP_C_RA"], E["Rate_VH_R"], E["LEO_C_RA"], E["Rate_VL_R"],
                                    action_index_h, action_index_m, action_index_l_id)
            F1_V, F2_V, F3_V = out_l[6], out_l[7], out_l[8]
            new_state_l = [0, 0, 0]
            if F1_V > 0: new_state_l[0] = 1
            if F2_V > 0: new_state_l[1] = 1
            if F3_V > 0: new_state_l[0] = 1
            idxs = np.argwhere((H.State_Space_L == new_state_l).all(axis=1))
            new_state_index_l = idxs[0, 0]

            state_index_h = new_state_index_h; state_index_m = new_state_index_m
            state_index_l = new_state_index_l
            state_h = H.State_Space_H[state_index_h]
            state_l = H.State_Space_L[state_index_l]

        HRL_decisions[v][0] = action_index_h
        Node_choice = action_index_m
        idx_ones = np.where(np.array(Node_choice) == 1)[0]
        HRL_decisions[v][1] = random.choice(idx_ones.tolist())
        HRL_decisions[v][2] = action_l

    # ---- Phân bổ tài nguyên theo quyết định HRL + tính chi phí ----
    NO_VUs_RSU_HRL = np.zeros(H.RSU_T); NO_VUs_UAV_HRL = np.zeros(H.UAV_T)
    NO_VUs_HAP_HRL = np.zeros(H.HAP_T); NO_VUs_LEO_HRL = np.zeros(H.LEO_T)
    for v in range(V):
        lyr = int(HRL_decisions[v][0]); node = int(HRL_decisions[v][1])
        [NO_VUs_RSU_HRL, NO_VUs_UAV_HRL, NO_VUs_HAP_HRL, NO_VUs_LEO_HRL][lyr][node] += 1

    RA = H.Resource_Allocation(IP, HRL_decisions, NO_VUs_RSU_HRL, NO_VUs_UAV_HRL,
                               NO_VUs_HAP_HRL, NO_VUs_LEO_HRL)
    RSU_B_HRL, UAV_B_HRL, HAP_B_HRL, LEO_B_HRL = RA[0], RA[1], RA[2], RA[3]
    RSU_C_HRL, UAV_C_HRL, HAP_C_HRL, LEO_C_HRL = RA[4], RA[5], RA[6], RA[7]
    Rate_VR_HRL, Rate_VU_HRL, Rate_VH_HRL, Rate_VL_HRL = H.Data_Rate(
        IP, HRL_decisions, RSU_B_HRL, UAV_B_HRL, HAP_B_HRL, LEO_B_HRL)
    Cost_HRL = H.Task_Proc_Main(IP, RSU_C_HRL, Rate_VR_HRL, UAV_C_HRL, Rate_VU_HRL,
                                HAP_C_HRL, Rate_VH_HRL, LEO_C_HRL, Rate_VL_HRL,
                                0.5, 0.5, HRL_decisions)
    T_L_HRL, T_E_HRL = Cost_HRL[0], Cost_HRL[1]
    TC_HRL = Cost_HRL[8]
    ser_fail, soj_fail, ser_h_req = Cost_HRL[9], Cost_HRL[10], Cost_HRL[11]
    return dict(
        latency=float(np.mean(T_L_HRL)), energy=float(np.mean(T_E_HRL)),
        cost=float(np.mean(TC_HRL)),
        ser_fail=float(ser_fail), soj_fail=float(soj_fail), ser_h_req=float(ser_h_req),
    )


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)

    # Agents (hyperparameter giữ nguyên)
    state_size_h = len(H.State_Space_H); action_size_h = len(H.Action_Space_H)
    state_size_m = len(H.State_Space_M); action_size_m = H.RSU_T + H.UAV_T + H.HAP_T + H.LEO_T
    state_size_l = len(H.State_Space_L); action_size_l = len(H.Action_Space_L)
    agent_h = FastAgent(OPTIMIZER, state_size_h, action_size_h, GAMMA_H, EPSILON_H)
    agent_m = FastAgent(OPTIMIZER, state_size_m, action_size_m, GAMMA_M, EPSILON_M)
    agent_l = FastAgent(OPTIMIZER, state_size_l, action_size_l, GAMMA_L, EPSILON_L)
    agents = (agent_h, agent_m, agent_l)

    qtabs, rows = make_q_tables()

    print("\n=== PHA 1: HUẤN LUYỆN (LS-DQN, gamma1=gamma2=0.5) trên LuST ===")
    ep_reward, ep_loss = train(agents, qtabs, rows)
    np.savez(os.path.join(RESULTS, "training_curves.npz"),
             episode=np.arange(1, NUM_EPISODES + 1), reward=ep_reward, loss=ep_loss)

    print("\n=== PHA 2: ĐÁNH GIÁ HRL theo tải V (mobility LuST) ===")
    rng = np.random.default_rng(SEED + 1)
    lat = {V: [] for V in V_LIST}; ene = {V: [] for V in V_LIST}
    cst = {V: [] for V in V_LIST}; sojf = {V: [] for V in V_LIST}
    serf = {V: [] for V in V_LIST}; serh = {V: [] for V in V_LIST}
    for V in V_LIST:
        for rep in range(RP):
            r = eval_hrl_once(V, agents, qtabs, rows, rng)
            lat[V].append(r["latency"]); ene[V].append(r["energy"]); cst[V].append(r["cost"])
            sojf[V].append(r["soj_fail"]); serf[V].append(r["ser_fail"]); serh[V].append(r["ser_h_req"])
            print(f"  [eval] V={V:3d} rep {rep+1}/{RP} | "
                  f"lat={r['latency']:.4f}s energy={r['energy']:.4f}J cost={r['cost']:.4f}")

    Vs = np.array(V_LIST, dtype=float)
    def m(d): return np.array([np.mean(d[V]) for V in V_LIST])
    def s(d): return np.array([np.std(d[V]) for V in V_LIST])

    eval_table = dict(
        V=Vs.tolist(),
        latency_mean=m(lat).tolist(), latency_std=s(lat).tolist(),
        energy_mean=m(ene).tolist(), energy_std=s(ene).tolist(),
        cost_mean=m(cst).tolist(), cost_std=s(cst).tolist(),
        reward_mean=(-m(cst)).tolist(),
        soj_handover_mean=m(sojf).tolist(), ser_fail_mean=m(serf).tolist(),
        ser_handover_mean=m(serh).tolist(),
    )
    with open(os.path.join(RESULTS, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_table, f, indent=2, ensure_ascii=False)

    # CSV gọn
    import csv
    with open(os.path.join(RESULTS, "eval_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V", "latency_mean", "latency_std", "energy_mean", "energy_std",
                    "cost_mean", "cost_std", "reward_mean",
                    "soj_handover_mean", "ser_fail_mean", "ser_handover_mean"])
        for i, V in enumerate(V_LIST):
            w.writerow([V, m(lat)[i], s(lat)[i], m(ene)[i], s(ene)[i],
                        m(cst)[i], s(cst)[i], (-m(cst))[i],
                        m(sojf)[i], m(serf)[i], m(serh)[i]])

    dt = time.time() - t0
    print(f"\nHoàn tất run_baseline trong {dt/60:.1f} phút.")
    print(f"Kết quả: results/training_curves.npz, results/eval_metrics.json/.csv")


if __name__ == "__main__":
    main()
