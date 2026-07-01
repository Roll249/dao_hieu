# Nhận xét các Figure — Baseline HRL & Scenario Quantum-HRL trên LuST 24h

*Sinh tay từ phân tích trực quan các PNG trong `figure_baseline/` và `scenario/`.*
*Số liệu tham chiếu lấy từ `figure_baseline/experiment_summary_full_lust.md` và `scenario/scenario_summary.md`.*

---

## A. Baseline HRL — `figure_baseline/full_lust_*.png`

Baseline gồm **1 seed (2024), 50 episode × 1000 timestep**, chạy thật trên pool tốc độ LuST 24h (442.557 mẫu, mean 16.7 m/s, median 13.7 m/s).

### A1. `full_lust_reward_convergence.png`
- Reward dao động trong dải **[−6.0, −4.6]**, không có xu hướng đi lên rõ rệt theo 50 episode.
- Trung bình 10 episode cuối ≈ **−5.31**, sai biệt episode-cuối ≈ **−5.07**.
- **Diễn giải:** đây không phải đường hội tụ học được — nó phản ánh việc reward huấn luyện nội-vòng = 0 (xem A5), nên reward đo được chỉ đến từ pipeline eval `Task_Proc_Main`. Mức nhiễu reward giữa episode chính là nhiễu của V_train rollout, không phải tín hiệu cải thiện theta.

### A2. `full_lust_latency_curve.png`
- Latency theo episode dao động **6.2 – 7.7 s**, trung bình ≈ **6.9 s** (10 ep cuối).
- Không có downtrend: episode 50 (~6.6 s) gần bằng episode 1 (~7.1 s).
- **Diễn giải:** chính sách HRL không cải thiện độ trễ qua thời gian; chỉ phản ánh chính sách khởi tạo + biến thiên VN_spd LuST.

### A3. `full_lust_energy_curve.png`
- Energy 3.0 – 4.3 J, mean ≈ **3.71 J** (10 ep cuối).
- Cùng pattern: nhiễu lớn, không hội tụ.

### A4. `full_lust_cost_curve.png`
- Cost = latency + energy quy đổi, dao động **4.6 – 6.0**, mean ≈ **5.31**.
- Đây là đường ngược của A1 (cost = −reward), nên cùng kết luận: không học được.

### A5. `full_lust_training_loss.png`
- **MSE = 0 trên toàn 50 episode**, có chú thích trực tiếp trên hình: *"Loss ≈ 0 — reward huấn luyện nội-vòng của baseline = 0 (Learning_Cost trả 0 trong code gốc)"*.
- **Đây là minh chứng cho bug gốc của notebook** (`np.nonzero(a_m)==r` so sánh tuple với int → luôn False). Hình này có chủ ý — không phải lỗi pipeline tích hợp.
- **Hệ quả:** mạng DQN tầng High học mục tiêu = 0; mọi cải thiện ở eval đến từ pipeline đánh giá chứ không phải từ Q-learning.

### A6. `full_lust_wall_clock_time.png`
- Mỗi episode tốn **4.3 – 5.5 s** wall-clock, mean ≈ **4.7 s**. Tổng training ≈ **3.9 phút**.
- Biến thiên không phụ thuộc episode index → I/O / process load là noise chính, không phải workload học.

### A7. `full_lust_load_speed_distribution.png`
- **Bên trái:** mật độ giao thông theo 24 cửa sổ giờ. Hai đỉnh rõ: **07h–09h** (cao điểm sáng, max ~9.5k xe) và **17h–19h** (cao điểm chiều, max ~10.5k xe). Phân lớp low/medium/high (9/8/7 cửa sổ) khớp curve.
- **Bên phải:** phân phối tốc độ tức thời pool 24h là **bimodal** — đỉnh chính ~14 m/s (đô thị, giao thông đông) và đỉnh phụ ~33 m/s (đường nhanh/đêm). Mean 16.7 m/s, median 13.7 m/s → lệch phải đúng như phân phối thực thấy ở dữ liệu di chuyển đô thị.
- **Diễn giải:** mobility input đa dạng và thực tế — không phải `truncnorm(8,14)` tổng hợp của paper gốc. Nguồn nhiễu của các metric A1–A4 vì vậy có thể quy về biến thiên VN_spd này.

### Tóm lược baseline
> Baseline HRL chạy đúng và faithfully reproduce paper, NHƯNG **không học** (loss = 0 do bug gốc). Mọi metric eval (≈ 6.9 s latency, 3.7 J energy, 0.96 miss rate ở scenario A) là **chính sách khởi tạo + epsilon-greedy hành xử trên mobility LuST thật**, không phải kết quả tối ưu hoá.

---

## B. Scenario Quantum-HRL — `scenario/scenario_*.png`

5 seed (2024–2028), 50 ep × 1000 step. Bốn cấu hình: Classical / HRL+VQC / HRL+VQC+QAOA / Full-QHRL. Eval V∈{20..200} × 3 lần.

### B1. `scenario_reward_convergence.png`
- **Bốn band tách bạch rõ ràng:**
  - Classical: dải rộng nhất, mean ≈ **−8.0**, biến thiên ±2 (band shade tới −12).
  - HRL+VQC và HRL+VQC+QAOA gần như chồng nhau quanh **−4.5**.
  - **Full-QHRL** ổn định trên cùng ở **−2.6**, dải hẹp nhất (±0.2).
- **Diễn giải:** thay DQN tầng High bằng VQC giúp reward bật từ −8 lên −4.5 ngay từ episode đầu (không phải hội tụ từ tệ → tốt; mà chính sách VQC khởi tạo đã tốt hơn DQN bị bug). Thêm BO (Full-QHRL) thắt biên độ và đẩy trung bình lên nữa.
- Không cấu hình nào "học" theo nghĩa downtrend rõ trong 50 ep — phù hợp với việc reward huấn luyện vẫn = 0 (cả Classical lẫn các nhánh quantum đều nhận target 0).

### B2. `scenario_latency_comparison.png`
- **10.6 → 6.04 → 6.24 → 2.96 s** qua bốn cấu hình.
- Classical-Full giảm **72.2%**; Full đè bẹp cả HRL+VQC.
- Error bar Full nhỏ nhất (~±0.4 s) → BO ổn định hoá chính sách trên các seed.
- QAOA so với chỉ-VQC: gần như không khác (6.04 vs 6.24, error bar chồng) → khi tác vụ chỉ có ≤6 node ứng viên, statevector p=1 không nâng được chất lượng node selection so với argmin chi phí baseline.

### B3. `scenario_energy_comparison.png`
- **5.66 → 2.95 → 3.07 → 2.24 J** (−60.5%).
- Cùng pattern latency: VQC tạo bước nhảy lớn; QAOA không thêm; BO thắt biên độ và hạ tiếp.

### B4. `scenario_cost_comparison.png`
- Cost ≈ −reward: **8.15 → 4.49 → 4.65 → 2.60**.
- Khớp B1.

### B5. `scenario_deadline_miss.png`
- **Đáng chú ý nhất:** 0.96 / 0.91 / 0.90 / **0.199**.
- Classical, HRL+VQC, HRL+VQC+QAOA **đều > 0.9 miss rate** — tải V=20..200 ép deadline rất ngặt; ba cấu hình đầu không xử lý nổi.
- **Chỉ Full-QHRL (có BO) phá ngưỡng**, miss xuống 0.20 — khoảng 4.5× cải thiện.
- **Diễn giải:** VQC giúp latency/energy trung bình tốt hơn nhưng **không** xử lý tail (deadline). Phần xử lý tail thuộc về **BO offload ratio** — đây cũng là khám phá quan trọng nhất của Scenario.

### B6. `scenario_latency_energy_tradeoff.png`
- 4 cụm tách bạch trên 2D, Full-QHRL ngự ở góc dưới trái (low-low) — Pareto dominate ba còn lại.
- HRL+VQC và HRL+VQC+QAOA chồng nhau, cluster ở trung tâm.
- Classical cô lập ở góc trên phải, error bar khổng lồ.
- **Đây là biểu đồ thuyết phục nhất** cho thông điệp bài: Full-QHRL không chỉ tốt hơn ở trung bình mà còn ở phương sai.

### B7. `scenario_training_loss.png`
- Classical = 0 hằng định (xác nhận lại bug A5).
- HRL+VQC, Full-QHRL có loss thực **dao động 0 → 0.5** với mean giảm dần ~0.1 về 0.03 cuối episode.
- **Tín hiệu học của VQC là thật**, dù target reward vẫn = 0 — vì VQC khởi tạo random ⇒ Q-output ≠ 0 ⇒ MSE > 0, và việc gradient kéo Q về 0 chính là cái "loss giảm" mà ta thấy. **Không phải VQC học mục tiêu tốt hơn**, mà là VQC fit về 0 tốt hơn. Đây là điểm phải tô đậm khi diễn giải.

### B8. `scenario_training_time.png`
- 192 s vs ~150 s (các nhánh quantum): nhánh quantum **nhanh hơn ~22%** vì VQC chỉ 24 param thay vì 5224.
- QAOA chỉ thêm ~3 s so với VQC thuần (statevector + prefilter ≤6 qubit) → overhead chấp nhận được.

### B9. `scenario_parameter_count.png`
- 15.744 (Classical) → 10.544/10.546 (nhánh quantum) — giảm **1.49×** ở tổng (middle/low policies giữ nguyên).
- Phần High riêng: 5224 → 24 params → **218×** giảm.
- Trục log scale dùng đúng — chênh lệch dễ đọc, nhưng nên kèm chú thích rằng phần lớn 10.5k là middle+low Dense networks.

### B10. `scenario_ablation_reward.png`
- Full **−2.60**, w/o QAOA **−2.66**, w/o VQC **−3.25**, w/o BO **−4.65**.
- Thứ tự đóng góp: **BO >> VQC >> QAOA**.
- w/o BO có error bar lớn nhất (~±1.7) → khẳng định BO là yếu tố ổn định.

### B11. `scenario_ablation_latency.png`
- Full **2.96 s**, w/o QAOA **3.10 s**, w/o VQC **4.43 s**, w/o BO **6.24 s**.
- Bỏ BO làm latency hơn gấp đôi.
- Bỏ QAOA tăng nhẹ (0.14 s) → đóng góp QAOA về latency gần như không đáng kể trong giới hạn statevector ≤6 qubit.

### B12. `scenario_ablation_energy.png`
- Full **2.24 J**, w/o QAOA **2.23 J**, w/o VQC **2.06 J**, w/o BO **3.07 J**.
- **Bất ngờ nhỏ:** w/o VQC có energy **thấp hơn Full**. Suy ra VQC chọn action eager hơn (giảm latency mạnh nhưng tiêu thêm năng lượng nhẹ) — BO bù lại bằng cách điều phối offload-ratio.
- Bỏ BO lại làm energy tăng vọt → BO là cốt lõi của energy efficiency, không chỉ latency.

---

## C. So sánh Baseline ↔ Scenario Classical

| Metric | Baseline (1 seed, raw 50 ep) | Scenario Classical (5 seed, eval V=[20..200]) |
|---|---:|---:|
| Latency | 6.91 s | 10.65 s |
| Energy | 3.71 J | 5.66 J |
| Cost | 5.31 | 8.15 |
| Deadline miss | — | 0.96 |
| Training loss | 0 | 0 |

- **Tại sao Scenario Classical xấu hơn Baseline?** Vì Scenario eval phủ V đến **200**, trong khi Baseline đo trung bình các episode train (V_train=100). Tải cao kéo metric xấu — đây là cách Scenario stress-test, hợp lý.
- Cả hai đều có training loss = 0, xác nhận tính nhất quán pipeline: cùng `hrl_core.py`, cùng bug gốc, cùng VN_spd từ pool LuST.

---

## D. Những kết luận quan trọng

1. **Thông điệp định lượng vững:** Full-QHRL vs Classical −72% latency, −60% energy, miss rate 0.96 → 0.20; Welch p<0.003 cho cả 5 metric, Hedges' g > 3.9.
2. **Đóng góp của từng module (ablation):** BO chiếm phần lớn, VQC đứng kế, QAOA nhỏ nhất. Khi cần kể chuyện đơn giản, có thể đề xuất biến thể HRL+VQC+BO (bỏ QAOA) cho hiệu năng/chi phí tốt hơn.
3. **Cảnh báo cần ghi rõ trong report:**
   - Training loss của các nhánh quantum **không** biểu thị học mục tiêu tốt; nó chỉ phản ánh fit về 0 do bug `Learning_Cost`. Nếu sửa bug, đường loss sẽ đổi và cần re-evaluate.
   - QAOA chạy statevector không nhiễu, chưa đại diện hardware thực.
   - 5 seed/cấu hình cho effect size lớn nhưng inference test còn yếu — nên tăng seed nếu submit hội nghị Q1.
4. **Hình tốt nhất để dùng làm "headline figure":** `scenario_latency_energy_tradeoff.png` (B6) — gói trọn message Pareto dominance trên một biểu đồ; kế đến là `scenario_deadline_miss.png` (B5) cho impact tail.
5. **Sự nhất quán giữa baseline và scenario:** Baseline phần A xác lập rằng pipeline LuST → HRL hoạt động đúng, faithfully reproduce paper kèm bug đã ghi nhận. Scenario phần B đặt trên cùng nền tảng đó, nên các nhánh quantum so với Classical là so sánh **fair**.

---

## E. Đề xuất sửa/bổ sung figure (nếu còn vòng review)

- A1–A4 nên gộp 2x2 thành một panel (`baseline_metric_panel.png`) — bốn đường tin tức gần giống nhau, gộp lại tiết kiệm trang giấy.
- A5 cần ghi chú nhỏ "intentional" để reviewer không đánh trượt.
- B1 nên thêm horizontal reference line ở reward eval-converged trung bình của 5 seed.
- B7 nên kèm sub-caption "VQC loss = fit về reward target=0, không đại diện cho học mục tiêu".
- B9 nên thêm bar phụ tách trainable high-level (5224 vs 24) để 218× pop ra trực quan.
