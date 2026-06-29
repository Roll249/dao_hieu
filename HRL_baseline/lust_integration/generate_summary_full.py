"""
generate_summary_full.py
=======================
Sinh figure_baseline/experiment_summary_full_lust.md từ kết quả full-LuST.
"""
import os
import json
import numpy as np
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(REPO, "figure_baseline")
RESULTS = os.path.join(FIG, "results")


def J(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    mob = J("mobility_full_summary.json")
    pw = J("per_window_mobility.json")
    raw = J("raw_results_full_lust.json")

    cfg = raw["config"]; last = raw["final_last_episode"]; m10 = raw["final_mean_last10"]
    n_active = sum(1 for r in pw if r["n_vehicles_departing"] > 0)
    cls_count = {}
    for r in pw:
        cls_count[r.get("traffic_class", "none")] = cls_count.get(r.get("traffic_class", "none"), 0) + 1

    # bảng mobility theo giờ
    win_rows = "\n".join(
        f"| {r['window']:02d} | {r['clock']} | {r.get('traffic_class','-')} | "
        f"{r['n_vehicles_departing']} | {r.get('concurrent_mean') or 0:.0f} | "
        f"{r.get('concurrent_max') or 0} | "
        f"{(r.get('speed_mean') if r.get('speed_mean') is not None else float('nan')):.2f} |"
        for r in pw)

    md = f"""# Experiment Summary — Baseline HRL trên TOÀN BỘ LuST 24h

*Sinh tự động: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## 1. Tổng quan
Chạy lại baseline HRL (`HRL_baseline/Main_Simulation.ipynb` — offloading tác vụ phân cấp
RSU/UAV/HAP/LEO) trên **toàn bộ mobility LuST 24 giờ**, tối ưu để **không nổ RAM/CPU**.
Mobility LuST đưa vào baseline qua **một điểm duy nhất**: tốc độ xe `VN_spd` lấy mẫu từ
phân phối tốc độ THẬT của LuST 24h (FCD). **Thuật toán/hyperparameter giữ NGUYÊN.**

## 2. Đã dùng file LuST nào
- **Net:** `Lust_dataset/scenario/lust.net.xml` (đèn tín hiệu actuated — mặc định).
- **Route (mobility):** `DUARoutes/local.0.rou.xml`, `local.1`, `local.2`, `transit.rou.xml`,
  `buslines.rou.xml`.
- **Phụ trợ:** `vtypes.add.xml`, `busstops.add.xml`.
- Bộ tuyến: **{mob['route_set']}**.

## 3. Có chạy SUMO không?
**Có.** LuST không có FCD trace sẵn nên phải sinh bằng SUMO (`eclipse-sumo`, qua pip).
Để không nổ RAM/disk:
- Chia 24h thành **{cfg['num_episodes'] and mob['n_windows']} cửa sổ 1 giờ** ([0, {mob['sim_span_s'][1]}]s).
- **Một lượt `iterparse`/route file** phân loại xe vào cửa sổ (chỉ giữ counter trong RAM —
  KHÔNG nạp cả file 40 MB).
- Chạy SUMO **từng cửa sổ riêng** rồi **xóa FCD ngay** sau khi trích (disk luôn thấp).

## 4. Tổng thời gian mô phỏng & số xe
- **Thời gian mô phỏng (SUMO):** 24 giờ = **{mob['sim_span_s'][1]} s** (0:00–24:00), chia {mob['n_windows']} cửa sổ.
- **Tổng số xe (toàn ngày):** **{mob['total_vehicles']:,}**.
- **Số cửa sổ có lưu lượng:** {n_active}/{mob['n_windows']}.
- Phân loại cửa sổ: {", ".join(f"{k}={v}" for k,v in cls_count.items())}.

## 5. Số xe đồng thời (toàn mạng)
- **Trung bình theo ngày:** **{mob['concurrent_mean_over_day']:.0f}** xe.
- **Cực đại (giờ cao điểm):** **{mob['concurrent_max_over_day']:,}** xe.

## 6. Cách sampling FCD
- Bật FCD với **`--device.fcd.period 60`** → ghi vị trí + tốc độ mỗi **60 giây** (không ghi
  mỗi bước để tránh file khổng lồ).
- Trích bằng **`iterparse` streaming**, gộp tốc độ tức thời > 0.5 m/s (loại xe dừng đèn để
  tránh sojourn-time = ∞), **subsample ≤ 20.000 mẫu/cửa sổ** → pool 24h.
- **Pool tốc độ 24h:** {mob['speed_pool_size']:,} mẫu | mean **{mob['speed_mean_mps']:.2f}**,
  median **{mob['speed_median_mps']:.2f}**, p90 **{mob['speed_p90_mps']:.2f}**, max **{mob['speed_max_mps']:.2f}** m/s.

## 7. Cách đưa mobility vào baseline
- `VN_spd[v] ←` lấy mẫu ngẫu nhiên (có hoàn lại) từ pool tốc độ LuST 24h, thay cho
  `truncnorm(8,14)` tổng hợp của baseline.
- Mọi phần còn lại của môi trường/thuật toán **giữ nguyên 100%** (chỉ thêm logging/instrumentation).
- Cấu hình train (giữ nguyên paper): episodes=**{cfg['num_episodes']}**,
  timesteps/episode=**{cfg['timesteps_per_episode']}**, V_train=**{cfg['V_train']}**,
  γ={cfg['gamma']}, ε={cfg['epsilon']}, optimizer={cfg['optimizer']}, batch={cfg['batch_size']}, seed={cfg['seed']}.

## 8. Kết quả cuối cùng (episode cuối / trung bình 10 episode cuối)
| Chỉ số | Episode cuối | TB 10 ep cuối |
|---|---|---|
| Reward (= −cost) | {last['reward']:.4f} | {m10['reward']:.4f} |
| Latency [s] | {last['latency']:.4f} | {m10['latency']:.4f} |
| Energy [J] | {last['energy']:.4f} | {m10['energy']:.4f} |
| Cost | {last['cost']:.4f} | {m10['cost']:.4f} |
| avg_q_value | {last['avg_q_value']:.4f} | {m10['avg_q_value']:.4f} |

Tổng wall-clock huấn luyện: **{raw['total_wall_clock_s']/60:.1f} phút**.

## 9. Vấn đề `training_loss ≈ 0` (giải thích rõ)
`training_loss ≈ 0` là **hành vi nguyên bản của baseline gốc**, KHÔNG phải lỗi tích hợp:
trong hàm `Learning_Cost`, điều kiện `if(np.nonzero(a_m)==r)` so sánh **tuple với số nguyên**
nên **luôn False** ⇒ mọi nhánh chi phí bị bỏ qua ⇒ reward nội-vòng = 0 ⇒ target DQN = 0 ⇒
loss (MSE) ≈ 0. Chi tiết kiểm chứng (mức biểu thức + mức hàm + ảnh hưởng loss) ở
**`results/loss_diagnostics.md`** (sinh bởi `baseline_fixed_analysis.py`).
- Reward hội tụ trong báo cáo được đo bằng **pipeline đánh giá thật** (`Task_Proc_Main`),
  nên có giá trị có nghĩa dù loss nội-vòng = 0.
- **Không "sửa bug" trong đường chạy chính.** Bản sửa chỉ tồn tại trong file phân tích riêng
  `baseline_fixed_analysis.py` để kiểm chứng nguyên nhân.

## 10. Xác nhận KHÔNG thay đổi thuật toán baseline
- Reward function: **giữ nguyên** (Learning_Cost / Task_Proc_Main nguyên văn từ notebook).
- Update rule (`retrain`): **giữ nguyên** (predict→fit y hệt; chỉ gọi model trực tiếp thay
  `.predict()` cho nhanh — đã kiểm chứng sai khác = 0.0).
- Kiến trúc DQN: **giữ nguyên** (Embedding→Dense(50)→Dense(50)→Dense linear, MSE, Adam).
- Hyperparameter: **giữ nguyên** (mục 7).
- Chỉ thêm: logging/instrumentation, export số liệu, visualization.

## 11. Sản phẩm
- `figure_baseline/results/`: `raw_results_full_lust.json`, `training_log_full_lust.csv`,
  `summary_full_lust.csv` (mobility theo từng cửa sổ — low/medium/high), `loss_diagnostics.md`,
  `per_window_mobility.json`, `mobility_full_summary.json`, `lust_speed_pool_full.npy`.
- `figure_baseline/`: `full_lust_reward_convergence.png`, `full_lust_training_loss.png`,
  `full_lust_latency_curve.png`, `full_lust_energy_curve.png`, `full_lust_cost_curve.png`,
  `full_lust_wall_clock_time.png`, `full_lust_load_speed_distribution.png`.

## 12. Mobility LuST theo từng giờ (24 cửa sổ)
| Giờ | Khung | Lớp | Số xe | Đồng thời TB | Đồng thời max | Tốc độ TB [m/s] |
|---|---|---|---|---|---|---|
{win_rows}
"""
    out = os.path.join(FIG, "experiment_summary_full_lust.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
