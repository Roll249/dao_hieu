# Scenario Experiments — Quantum-HRL trên LuST 24 giờ

*Sinh tự động lúc 2026-06-29 15:38.*

## Trạng thái hoàn thành

| Scenario | Trạng thái | Thành phần mới |
|---|---|---|
| Scenario 1 — Classical HRL | Hoàn thành; chạy trực tiếp baseline hiện tại | Không |
| Scenario 2 — HRL + VQC | Hoàn thành | `quantum/vqc_policy.py` |
| Scenario 3 — HRL + VQC + QAOA | Hoàn thành | `quantum/qaoa_node.py` |
| Scenario 4 — Full Quantum-HRL | Hoàn thành | `quantum/bo_offload.py` |
| Ablation | Hoàn thành đủ Full, w/o VQC, w/o QAOA, w/o BO | Runner cấu hình thành phần |

## Thiết lập và giả định

- Dữ liệu là pool tốc độ `lust_speed_pool_full.npy` sinh bởi pipeline LuST 24 giờ hiện tại; preprocessing không bị thay đổi.
- Classical HRL dùng nguyên `hrl_core.py`, hyperparameter gốc: 50 episode × 1000 timestep, batch size 2, cùng gamma/epsilon.
- Mỗi cấu hình chạy 5 seed độc lập: `[2024, 2025, 2026, 2027, 2028]`. Kết quả là mean ± sample SD qua seed.
- Đánh giá cuối quét `V=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]`, 3 lần lặp mỗi mức tải, giống pipeline baseline.
- VQC gồm 4 qubit, angle encoding và 2 lớp entangling; epsilon giữ nguyên nhưng được hiểu theo epsilon-greedy chuẩn vì VQC thay hoàn toàn high-level DQN.
- QAOA dùng one-hot QUBO `Σ cᵢxᵢ + P(Σxᵢ−1)²`, ánh xạ `x=(1−Z)/2`, depth `p=1`; hai góc được tối ưu trên lưới xác định cho từng QUBO. Bài toán trên 6 node dùng prefilter theo chi phí để giới hạn statevector ở 6 qubit.
- Bayesian Optimization dùng Gaussian Process RBF và expected improvement, 8 lần đánh giá tỷ lệ. Các xe dùng lịch candidate chung nhưng posterior/giá trị mục tiêu riêng để sweep LuST khả thi trên CPU.
- Bug reward bằng 0 trong `Learning_Cost` của baseline được giữ nguyên. VQC nhận cùng reward và target-update schedule; không dùng cost đánh giá để âm thầm sửa reward huấn luyện.
- `Number of trainable parameters` tính online policies một lần, không nhân đôi target network. Hai góc QAOA được tính là variational parameters; BO không có trainable parameter.

## Kết quả bốn scenario

| Scenario | Reward | Latency (s) | Energy (J) | Cost | Deadline miss rate | Average Q-value | Trainable params | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | -8.1511 ± 1.7222 | 10.6459 ± 2.3409 | 5.6562 ± 1.1060 | 8.1511 ± 1.7222 | 0.9599 ± 0.0142 | 0.0000 ± 0.0000 | 15744 | 191.8 ± 17.9 |
| HRL+VQC | -4.4948 ± 0.6460 | 6.0377 ± 0.8715 | 2.9519 ± 0.4235 | 4.4948 ± 0.6460 | 0.9066 ± 0.0521 | 0.2904 ± 0.0633 | 10544 | 150.3 ± 3.2 |
| HRL+VQC+QAOA | -4.6540 ± 1.6939 | 6.2410 ± 2.2108 | 3.0670 ± 1.1782 | 4.6540 ± 1.6939 | 0.8999 ± 0.0207 | 0.2767 ± 0.0303 | 10546 | 153.1 ± 1.4 |
| Full-QHRL | -2.5996 ± 0.1980 | 2.9642 ± 0.4436 | 2.2351 ± 0.0532 | 2.5996 ± 0.1980 | 0.1992 ± 0.1578 | 0.2767 ± 0.0303 | 10546 | 153.5 ± 1.6 |

## Ablation study

| Cấu hình | Reward | Latency (s) | Energy (J) | Deadline miss rate |
|---|---:|---:|---:|---:|
| Full Quantum-HRL | -2.5996 ± 0.1980 | 2.9642 ± 0.4436 | 2.2351 ± 0.0532 | 0.1992 ± 0.1578 |
| w/o VQC | -3.2471 ± 0.0278 | 4.4304 ± 0.0674 | 2.0638 ± 0.0197 | 0.7693 ± 0.0269 |
| w/o QAOA | -2.6628 ± 0.1234 | 3.0953 ± 0.2847 | 2.2303 ± 0.0838 | 0.2400 ± 0.1111 |
| w/o Bayesian Optimization | -4.6540 ± 1.6939 | 6.2410 ± 2.2108 | 3.0670 ± 1.1782 | 0.8999 ± 0.0207 |

## Welch's t-test: Classical HRL so với Full Quantum-HRL

| Metric | t | df | p-value | Hedges' g | Ý nghĩa ở α=0.05 |
|---|---:|---:|---:|---:|---|
| reward | -7.1608 | 4.11 | 0.001822 | -4.0906 | Có |
| latency | 7.2092 | 4.29 | 0.001499 | 4.1183 | Có |
| energy | 6.9088 | 4.02 | 0.002264 | 3.9467 | Có |
| cost | 7.1608 | 4.11 | 0.001822 | 4.0906 | Có |
| deadline_miss | 10.7346 | 4.06 | 0.0003922 | 6.1322 | Có |

Các kiểm tra Shapiro–Wilk theo nhóm và Levene (median-centered) được lưu đầy đủ trong `statistical_tests.json`. Với chỉ 5 seed mỗi nhóm, các kiểm tra giả định có công suất thấp; vì vậy effect size và độ lớn thực tế cần được đọc cùng p-value.

## Kết quả nổi bật

- Full Quantum-HRL thay đổi latency -72.16% và energy -60.48% so với Classical HRL; chênh lệch reward là +5.5514 (reward cao hơn là tốt hơn).
- VQC giảm tham số high-level 217.67×; tổng số tham số online giảm 1.49× sau khi vẫn tính middle/low policies không bị thay thế.
- Ý nghĩa của từng thành phần nên đọc trực tiếp từ ba hàng ablation; báo cáo không tuyên bố đóng góp dương nếu số liệu không hỗ trợ.

## Những vấn đề còn tồn tại

- Baseline có reward huấn luyện bằng 0 do lỗi gốc đã được ghi nhận; theo yêu cầu, lỗi này không được sửa. Điều đó làm giới hạn khả năng diễn giải về convergence và Average Q-value.
- QAOA và VQC chạy trên statevector không nhiễu, chưa đại diện latency/noise của quantum hardware thật.
- QAOA phải prefilter khi số node khả thi vượt 6; đây là giới hạn mô phỏng cổ điển, không phải thay đổi preprocessing LuST.
- Cỡ mẫu 5 seed đủ cho yêu cầu tối thiểu nhưng còn nhỏ cho suy luận thống kê mạnh.

## Khả năng tái lập

- Runner: `HRL_baseline/lust_integration/scenario_runner.py`.
- Checkpoint schema v2 lưu sau từng seed trong `scenario/results/`.
- Kết quả nguồn, bảng tổng hợp, kiểm định và parameter count nằm trong `scenario/results/`.
- Mười hai hình PNG dùng nhãn tiếng Anh, error bar là sample SD, palette dịu/colorblind-friendly và 600 DPI.
