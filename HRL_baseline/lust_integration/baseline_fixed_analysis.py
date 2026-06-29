"""
baseline_fixed_analysis.py
=========================
PHÂN TÍCH RIÊNG (KHÔNG ảnh hưởng kết quả chính) để giải thích vì sao
`training_loss ≈ 0` trong baseline gốc.

Nguyên nhân nghi ngờ: trong `Learning_Cost`, các nhánh tính chi phí được canh bằng
    if (np.nonzero(a_m) == r):
mà `np.nonzero(a_m)` trả về một TUPLE -> so sánh tuple với số nguyên `r` LUÔN False
=> không nhánh nào chạy => T_L = T_E = 0 => reward nội-vòng = 0 => target DQN = 0
=> loss = 0.

Script này:
  1. Chứng minh ở mức biểu thức: tuple==int là False, còn [0][0]==int mới đúng.
  2. Tạo bản `Learning_Cost_fixed` (sửa đúng phép so sánh) bằng cách thay chuỗi trên
     source gốc — KHÔNG đụng tới hrl_core/run_baseline dùng cho kết quả chính.
  3. So sánh chi phí gốc vs đã sửa trên cùng input -> cho thấy gốc = 0, sửa ≠ 0.
  4. Ghi figure_baseline/results/loss_diagnostics.md.
"""
import os
import sys
import inspect
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "figure_baseline", "results")
sys.path.insert(0, HERE)

import run_baseline as RB
import hrl_core as H


def build_fixed_learning_cost():
    """Tạo Learning_Cost_fixed bằng cách sửa phép so sánh tuple==int -> index==int."""
    src = inspect.getsource(H.Learning_Cost)
    fixed_src = src.replace("np.nonzero(a_m)==", "np.nonzero(a_m)[0][0]==")
    n_fix = src.count("np.nonzero(a_m)==")
    ns = dict(H.__dict__)  # cùng global (np, math, IP indexing, v_id...)
    exec(fixed_src, ns)
    return ns["Learning_Cost"], n_fix


def main():
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("# Loss Diagnostics — vì sao `training_loss ≈ 0`")
    P()
    P("## 1. Mức biểu thức (root cause)")
    a_m = np.zeros(60); a_m[7] = 1  # one-hot tại node 7
    expr_bug = (np.nonzero(a_m) == 7)
    expr_fix = (np.nonzero(a_m)[0][0] == 7)
    P(f"- `a_m` là vector one-hot (node = 7).")
    P(f"- Biểu thức GỐC `np.nonzero(a_m) == 7`  -> **{expr_bug}**  "
      f"(so sánh tuple `{np.nonzero(a_m)}` với số nguyên → luôn False).")
    P(f"- Biểu thức ĐÚNG `np.nonzero(a_m)[0][0] == 7` -> **{expr_fix}**.")
    P("- Vì điều kiện luôn False, các nhánh cộng dồn chi phí trong `Learning_Cost` "
      "không bao giờ chạy ⇒ `T_L = T_E = 0`.")
    P()

    # --- 2. Mức hàm: so sánh trên cùng input ---
    Learning_Cost_fixed, n_fix = build_fixed_learning_cost()
    P("## 2. Mức hàm `Learning_Cost` (gốc vs đã sửa)")
    P(f"- Số vị trí so sánh lỗi đã thay trong bản sửa: **{n_fix}**.")

    rng = np.random.default_rng(0)
    speeds = RB.SPEED_POOL if RB.SPEED_POOL.size else np.full(100, 12.0)
    V = 100
    vn = rng.choice(speeds, size=V, replace=True).astype(float)
    E = RB.build_env(V, vn, "eval")
    IP = E["IP"]

    orig_nonzero = 0
    fixed_nonzero = 0
    fixed_errors = 0
    n_eval = 0
    orig_vals, fixed_vals = [], []
    # chọn các (v, node) mà RA đã cấp phát (RSU_C_RA[v][r] > 0) để có capacity+rate hợp lệ
    RSU_C = E["RSU_C_RA"]
    for v in range(V):
        nz = np.where(RSU_C[v] > 0)[0]
        if nz.size == 0:
            continue
        r_star = int(nz[0])
        a_m = np.zeros(H.RSU_T); a_m[r_star] = 1
        H.v_id = v
        Learning_Cost_fixed.__globals__["v_id"] = v   # bản sửa dùng namespace riêng
        o = H.Learning_Cost(IP, E["DL_Req"], 1.0, E["RSU_C_RA"], E["Rate_VR_R"],
                            E["UAV_C_RA"], E["Rate_VU_R"], E["HAP_C_RA"], E["Rate_VH_R"],
                            E["LEO_C_RA"], E["Rate_VL_R"], 0, a_m, 0.5)
        reward_o = 0.5 * o[0] + 0.5 * o[1]
        orig_vals.append(reward_o)
        orig_nonzero += int(reward_o != 0)
        n_eval += 1
        try:
            f = Learning_Cost_fixed(IP, E["DL_Req"], 1.0, E["RSU_C_RA"], E["Rate_VR_R"],
                                    E["UAV_C_RA"], E["Rate_VU_R"], E["HAP_C_RA"], E["Rate_VH_R"],
                                    E["LEO_C_RA"], E["Rate_VL_R"], 0, a_m, 0.5)
            reward_f = float(np.asarray(0.5 * f[0] + 0.5 * f[1]).ravel()[0])
            fixed_vals.append(reward_f)
            fixed_nonzero += int(reward_f != 0)
        except Exception:
            # Sửa phép so sánh sẽ KÍCH HOẠT các nhánh code "chết" chưa từng chạy/được test
            # -> đôi khi lỗi shape (max() trên array). Đây cũng là bằng chứng các nhánh
            #    chi phí KHÔNG bao giờ được thực thi trong baseline gốc.
            fixed_errors += 1

    P(f"- Số mẫu (v, node hợp lệ) đánh giá: **{n_eval}**.")
    P(f"- Reward GỐC khác 0: **{orig_nonzero}/{n_eval}** | trung bình = "
      f"**{np.mean(orig_vals) if orig_vals else 0:.4f}** ⇒ baseline gốc luôn cho reward = 0.")
    P(f"- Bản ĐÃ SỬA: reward khác 0 ở **{fixed_nonzero}/{n_eval}** mẫu "
      f"(trung bình {np.mean(fixed_vals) if fixed_vals else 0:.4f}); "
      f"**{fixed_errors}/{n_eval}** mẫu phát sinh lỗi shape do kích hoạt nhánh code chưa từng chạy.")
    P("  → Cả hai (reward≠0 hoặc lỗi shape) đều xác nhận: trong baseline gốc, các nhánh tính "
      "chi phí của `Learning_Cost` **không bao giờ được thực thi**.")
    P()
    P("## 3. Hệ quả tới training-loss")
    P("- `agent_h.retrain` đặt `target[0][action] = reward + gamma * max(target_net(next))`. "
      "Với baseline gốc, `reward = 0` và mạng khởi tạo trọng số 0 (dummy) ⇒ `target = 0`; "
      "`q_network` dự đoán ≈ 0 ⇒ **loss (MSE) ≈ 0**.")
    P("- Nói cách khác, nếu các nhánh chi phí thực sự chạy (reward ≠ 0) thì target ≠ 0 và "
      "loss > 0; nhưng trong baseline GỐC điều đó không xảy ra ⇒ loss giữ ≈ 0.")
    P()
    P("## 4. Quyết định")
    P("- **GIỮ NGUYÊN baseline gốc** cho mọi kết quả chính (không 'sửa bug' để loss đẹp hơn) — "
      "yêu cầu không được thay đổi thuật toán.")
    P("- Bản sửa chỉ tồn tại trong script phân tích này để KIỂM CHỨNG nguyên nhân, "
      "không dùng cho `run_baseline_full.py`.")
    P("- Vì vậy `full_lust_training_loss.png` ≈ 0 là **đúng theo baseline gốc**, không phải lỗi tích hợp.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "loss_diagnostics.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"\nĐã ghi {os.path.join(RESULTS, 'loss_diagnostics.md')}")


if __name__ == "__main__":
    main()
