# Experiment Summary — Baseline HRL trên TOÀN BỘ LuST 24h

*Sinh tự động: 2026-06-28 22:13*

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
- Bộ tuyến: **DUA local.0/1/2 + transit + buslines; net actuated TLs**.

## 3. Có chạy SUMO không?
**Có.** LuST không có FCD trace sẵn nên phải sinh bằng SUMO (`eclipse-sumo`, qua pip).
Để không nổ RAM/disk:
- Chia 24h thành **24 cửa sổ 1 giờ** ([0, 86400]s).
- **Một lượt `iterparse`/route file** phân loại xe vào cửa sổ (chỉ giữ counter trong RAM —
  KHÔNG nạp cả file 40 MB).
- Chạy SUMO **từng cửa sổ riêng** rồi **xóa FCD ngay** sau khi trích (disk luôn thấp).

## 4. Tổng thời gian mô phỏng & số xe
- **Thời gian mô phỏng (SUMO):** 24 giờ = **86400 s** (0:00–24:00), chia 24 cửa sổ.
- **Tổng số xe (toàn ngày):** **288,250**.
- **Số cửa sổ có lưu lượng:** 24/24.
- Phân loại cửa sổ: low=9, medium=8, high=7.

## 5. Số xe đồng thời (toàn mạng)
- **Trung bình theo ngày:** **2485** xe.
- **Cực đại (giờ cao điểm):** **10,506** xe.

## 6. Cách sampling FCD
- Bật FCD với **`--device.fcd.period 60`** → ghi vị trí + tốc độ mỗi **60 giây** (không ghi
  mỗi bước để tránh file khổng lồ).
- Trích bằng **`iterparse` streaming**, gộp tốc độ tức thời > 0.5 m/s (loại xe dừng đèn để
  tránh sojourn-time = ∞), **subsample ≤ 20.000 mẫu/cửa sổ** → pool 24h.
- **Pool tốc độ 24h:** 442,557 mẫu | mean **16.65**,
  median **13.68**, p90 **33.21**, max **48.66** m/s.

## 7. Cách đưa mobility vào baseline
- `VN_spd[v] ←` lấy mẫu ngẫu nhiên (có hoàn lại) từ pool tốc độ LuST 24h, thay cho
  `truncnorm(8,14)` tổng hợp của baseline.
- Mọi phần còn lại của môi trường/thuật toán **giữ nguyên 100%** (chỉ thêm logging/instrumentation).
- Cấu hình train (giữ nguyên paper): episodes=**50**,
  timesteps/episode=**1000**, V_train=**100**,
  γ={'H': 0.7, 'M': 0.05, 'L': 0.05}, ε={'H': 0.1, 'M': 0.7, 'L': 0.7}, optimizer=adam, batch=2, seed=2024.

## 8. Kết quả cuối cùng (episode cuối / trung bình 10 episode cuối)
| Chỉ số | Episode cuối | TB 10 ep cuối |
|---|---|---|
| Reward (= −cost) | -5.0735 | -5.3089 |
| Latency [s] | 6.6297 | 6.9102 |
| Energy [J] | 3.5174 | 3.7076 |
| Cost | 5.0735 | 5.3089 |
| avg_q_value | 0.0000 | 0.0000 |

Tổng wall-clock huấn luyện: **3.9 phút**.

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
| 00 | 00:00-01:00 | low | 878 | 121 | 147 | 20.91 |
| 01 | 01:00-02:00 | low | 1351 | 175 | 222 | 24.45 |
| 02 | 02:00-03:00 | low | 1845 | 225 | 290 | 25.32 |
| 03 | 03:00-04:00 | low | 2343 | 284 | 344 | 25.85 |
| 04 | 04:00-05:00 | low | 3679 | 517 | 756 | 22.28 |
| 05 | 05:00-06:00 | low | 7051 | 1148 | 1802 | 17.77 |
| 06 | 06:00-07:00 | medium | 16559 | 3095 | 5190 | 16.37 |
| 07 | 07:00-08:00 | high | 24156 | 5390 | 9165 | 14.91 |
| 08 | 08:00-09:00 | high | 25510 | 5972 | 9525 | 14.66 |
| 09 | 09:00-10:00 | high | 16886 | 3701 | 4601 | 14.19 |
| 10 | 10:00-11:00 | medium | 10020 | 1920 | 2411 | 13.67 |
| 11 | 11:00-12:00 | medium | 9501 | 1613 | 1891 | 15.99 |
| 12 | 12:00-13:00 | medium | 17125 | 3398 | 5044 | 15.96 |
| 13 | 13:00-14:00 | high | 17942 | 3715 | 5037 | 15.83 |
| 14 | 14:00-15:00 | medium | 12627 | 2364 | 2861 | 16.34 |
| 15 | 15:00-16:00 | medium | 9610 | 1517 | 2187 | 16.82 |
| 16 | 16:00-17:00 | medium | 13971 | 2884 | 4995 | 13.47 |
| 17 | 17:00-18:00 | high | 20551 | 5016 | 8997 | 13.01 |
| 18 | 18:00-19:00 | high | 23570 | 6216 | 10506 | 13.26 |
| 19 | 19:00-20:00 | high | 20103 | 4905 | 7118 | 14.46 |
| 20 | 20:00-21:00 | medium | 12486 | 2230 | 2622 | 16.99 |
| 21 | 21:00-22:00 | low | 8311 | 1288 | 1553 | 17.91 |
| 22 | 22:00-23:00 | low | 6911 | 1078 | 1251 | 17.25 |
| 23 | 23:00-00:00 | low | 5264 | 861 | 1077 | 14.73 |
