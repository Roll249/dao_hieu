# Urgent Need To Fix

Mục tiêu của file này: gom các lỗi bắt buộc phải xử lý trước khi tiếp tục sửa paper hoặc chạy lại thí nghiệm. Thứ tự dưới đây là thứ tự ưu tiên: lỗi càng trên càng có khả năng làm paper bị reject hoặc làm kết quả không còn đáng tin.

## 1. Blocker: Paper nói dùng LuST, nhưng main quantum code chưa dùng LuST

**Lỗi:** Section Dataset/Experimental Design đang mô tả mobility layer lấy từ LuST trajectory: vị trí, vận tốc, sojourn, channel theo từng slot. Nhưng `simulation/tntn_environment.py` hiện reset xe bằng random position/speed và `_move_vehicle()` bằng random walk synthetic. `simulation/` không đọc FCD/route/SUMO/LuST.

**Vì sao phải fix:** Đây là lỗi integrity/reproducibility rất nặng. Nếu reviewer mở code, họ sẽ thấy dataset claim không khớp artifact. Khi đó toàn bộ kết quả chính bị nghi ngờ, dù model quantum đúng.

**Fix tối thiểu hợp lệ:** Chọn một trong hai:

- Nếu chưa tích hợp LuST thật: sửa paper nói rõ main experiment là synthetic mobility, LuST chỉ là motivation hoặc separate baseline study.
- Nếu muốn giữ claim LuST: phải nối LuST FCD/trajectory vào `simulation/` và rerun toàn bộ main comparison/ablation.

## 2. Blocker: Không có thí nghiệm hợp lệ nào hiện hỗ trợ đầy đủ claim trung tâm

**Lỗi:** Trong repo có hai nguồn kết quả:

- `simulation/`: baseline classical có học và kiến trúc full-replacement 24 params khớp paper, nhưng mobility là synthetic; latency improvement ở mức seed không rõ đủ significant, Quantum-HRL tốn energy hơn Classical HRL.
- `scenario/`: có LuST 24h và số đẹp hơn, nhưng baseline classical bị hỏng/không học đúng, nên so sánh Classical vs Quantum không công bằng; hơn nữa kiến trúc không còn là full-replacement 24 params.

**Vì sao phải fix:** Không thể vừa claim "LuST 24h", vừa claim "24 params", vừa claim "Quantum-HRL thắng Classical HRL có ý nghĩa thống kê" nếu không có một run duy nhất thỏa cả ba điều kiện.

**Fix tối thiểu hợp lệ:** Đóng băng một pipeline duy nhất rồi chỉ dùng số từ pipeline đó. Nếu pipeline là `simulation/`, paper phải trung thực là synthetic. Nếu pipeline là LuST, phải sửa baseline và quantum integration rồi rerun.

## 3. Blocker: Baseline EC_HRL gốc có lỗi học nghiêm trọng

**Lỗi:** Trong `EC_HRL-main/Main_Simulation.ipynb`, `Learning_Cost` dùng điều kiện kiểu `np.nonzero(a_m)==r`, so sánh tuple/array với scalar. Điều này làm nhiều nhánh cost không chạy đúng, reward nội vòng có thể về 0. Ngoài ra epsilon-greedy bị đảo nghĩa: epsilon cao lại thiên về argmax/exploit, không phải explore.

**Vì sao phải fix:** Nếu dùng baseline gốc để so sánh, reviewer có thể nói Quantum-HRL thắng vì baseline bị liệt, không phải vì phương pháp tốt hơn.

**Fix tối thiểu hợp lệ:** Không dùng notebook gốc làm baseline chính nếu chưa sửa. Dùng một reimplementation DQN sạch, có unit test reward/action, hoặc sửa notebook rồi chứng minh loss/reward/action thực sự thay đổi hợp lý.

## 4. Critical: QAOA/BO trong paper không khớp implementation

**Lỗi:** Paper mô tả BO là outer-loop chính tối ưu QAOA angles. Nhưng trong `simulation/qaoa_solver.py`, `solve()` chạy gradient descent mỗi lần chọn node; BO trong `QuantumHRLAgent.update_qaoa()` chỉ refine định kỳ trên cached Ising instance.

**Vì sao phải fix:** Claim "BO đem lại robustness/noise stability" chưa được chứng minh bởi code hiện tại. Reviewer sẽ bắt mismatch algorithm description vs implementation.

**Fix tối thiểu hợp lệ:** Sửa paper mô tả đúng: QAOA local gradient optimization per decision + periodic BO refinement. Nếu vẫn muốn claim BO quan trọng, thêm ablation `w/o BO` và noise table/figure.

## 5. Critical: QAOA extraction có thể fallback sang classical heuristic

**Lỗi:** QAOA device dùng `shots=None`, nhưng `_extract_solution()` gọi `qml.sample()`. Nếu fail, code fallback về `argmin(abs(h))`.

**Vì sao phải fix:** Kết quả "QAOA node selection" có thể thực chất là fallback classical heuristic mà không được log. Điều này làm sai claim về đóng góp QAOA.

**Fix tối thiểu hợp lệ:** Dùng finite-shot device khi sample, hoặc decode từ statevector/probabilities rõ ràng. Log tỷ lệ fallback. Nếu fallback >0, phải report.

## 6. Critical: Mệnh đề QAOA convergence trong paper quá mạnh

**Lỗi:** Paper viết finite-depth QAOA có xác suất đo optimal bitstring tăng đơn điệu theo depth `p`. Đây không phải định lý tổng quát.

**Vì sao phải fix:** Đây là lỗi lý thuyết dễ bị reviewer quantum bắt ngay.

**Fix tối thiểu hợp lệ:** Viết lại thận trọng: tăng `p` mở rộng ansatz/expressivity và có thể cải thiện nghiệm empirically, nhưng không đảm bảo monotonic cho mọi instance/optimizer/noise.

## 7. Major: Thông số network trong paper khác code

**Lỗi:** Paper nói LAP/HAP/LEO altitude khoảng 1/100/2000 km và coverage 50 m/200 m/1000 m. Code dùng 0.3/20/600 km và coverage 0.3/1.5/50/500 km.

**Vì sao phải fix:** Latency, channel gain, sojourn, deadline miss phụ thuộc trực tiếp vào các tham số này. Nếu paper và code khác nhau, kết quả không tái lập được.

**Fix tối thiểu hợp lệ:** Tạo một config duy nhất cho topology/channel/coverage, dùng cả trong code và table paper. Rerun sau khi chốt.

## 8. Major: Workload unit Mbit/Mbyte không nhất quán

**Lỗi:** Paper ghi `d_k ~ U(0.5,5) Mbytes`, code sample `0.5..5.0` rồi nhân `1e6` bits, tức Mbit.

**Vì sao phải fix:** Mbyte vs Mbit lệch 8 lần payload, làm thay đổi transmission latency/energy và ranking policy.

**Fix tối thiểu hợp lệ:** Chọn một đơn vị. Nếu giữ code hiện tại, sửa paper thành Mbit. Nếu muốn Mbyte, sửa code và rerun.

## 9. Major: Reward weights và objective không khớp

**Lỗi:** Paper ghi `beta1=beta2=0.5`, code reward/QUBO dùng `lat + energy` với beta mặc định 1.0/1.0. Penalty cũng hard-code 50 thay vì calibrated 99th percentile như paper mô tả.

**Vì sao phải fix:** Objective P1 là trung tâm của paper. Nếu reward thực tế khác paper, kết quả không chứng minh đúng bài toán đã viết.

**Fix tối thiểu hợp lệ:** Centralize config cho beta/penalty, log vào output JSON, và paper trích đúng config đó.

## 10. Major: Baseline parameter counting đang mâu thuẫn

**Lỗi:** Text nói mỗi DQN có online + target nên `W_HRL = 2 sum W_DQN`, nhưng số 20,224 chỉ là tổng online heads, không nhân 2.

**Vì sao phải fix:** Claim 843x compression là headline. Nếu cách đếm không rõ, reviewer sẽ nghi ngờ toàn bộ parameter-efficiency argument.

**Fix tối thiểu hợp lệ:** Báo cáo hai số: trainable online parameters và stored online+target parameters. Nói rõ compression dùng số nào.

## 11. Major: Baseline architecture paper không khớp EC_HRL notebook

**Lỗi:** Paper mô tả baseline MLP `n=20,h=256`. EC_HRL notebook dùng `Embedding -> Dense(50) -> Dense(50) -> output`.

**Vì sao phải fix:** Nếu nói "released implementation" nhưng dùng architecture khác, so sánh có thể bị coi là reimplementation không faithful.

**Fix tối thiểu hợp lệ:** Tách rõ:

- "Original EC_HRL notebook" để phân tích/citation.
- "Clean Classical-HRL reimplementation" dùng trong main comparison.

## 12. Major: Greedy baseline không phải nearest RSU

**Lỗi:** Paper mô tả Greedy chọn nearest RSU với alpha=1. Code `GreedyAgent` luôn trả `(0,0,1.0)`.

**Vì sao phải fix:** Baseline heuristic bị mô tả sai và có thể unfair.

**Fix tối thiểu hợp lệ:** Implement nearest RSU thật hoặc đổi tên thành fixed first-RSU full-offload.

## 13. Major: Service subset / cloud fallback được viết trong paper nhưng code chưa có

**Lỗi:** Paper có service type `xi_k`, service subset `Xi_e`, cloud relay nếu node không hỗ trợ service. Code generate `xi_k` nhưng latency/energy không dùng service availability hoặc cloud fallback.

**Vì sao phải fix:** System model trong paper phức tạp hơn simulator. Nếu reviewer hỏi, không có bằng chứng cho phần cloud/service constraints.

**Fix tối thiểu hợp lệ:** Hoặc implement, hoặc ghi rõ reported experiments omit service fallback and cloud relay.

## 14. Major: Node load/queueing gần như không hoạt động

**Lỗi:** Node CPU load update nhân thêm `1e-9`, khiến load gần như không tăng. Waiting time/queueing cũng set bằng 0 trong latency model.

**Vì sao phải fix:** Paper nói load-aware/offloading under congestion, nhưng simulator không tạo congestion động đủ mạnh.

**Fix tối thiểu hợp lệ:** Sửa load update theo đơn vị đúng, thêm queue/waiting nếu claim congestion. Nếu không, hạ claim xuống light-load/no-queue setting.

## 15. Major: Task feature bị stale khi policy chọn action

**Lỗi:** Trong training loop, task được generate rồi agent chọn action từ `state` cũ; task features chỉ được ghi vào state trong `env.step()` sau khi action đã chọn.

**Vì sao phải fix:** VQC tier/ratio policy không thật sự điều kiện hóa trên task hiện tại, dù paper nói state gồm task descriptor.

**Fix tối thiểu hợp lệ:** Fill task features vào observation trước `select_action()`.

## 16. Major: Statistical validation có nguy cơ pseudo-replication

**Lỗi:** Paper dùng pooled per-task latencies (`n=2400`) để test Quantum vs Classical. Nhưng training seeds mới là independent runs; per-task samples trong cùng policy/seed không độc lập hoàn toàn.

**Vì sao phải fix:** P-value rất nhỏ có thể là artifact của pooling. Reviewer ML/networking hay bắt lỗi này.

**Fix tối thiểu hợp lệ:** Báo cáo seed-level test trên per-seed means; pooled task test chỉ là supplementary.

## 17. Major: Noise robustness claim thiếu bảng/figure

**Lỗi:** Paper nói BO giữ stability dưới NISQ noise, nhưng Results chưa có bảng/figure noise sweep rõ ràng, cũng chưa có fixed-angle comparator.

**Vì sao phải fix:** Claim noise robustness là claim quantum/NISQ quan trọng, không thể chỉ nêu bằng lời.

**Fix tối thiểu hợp lệ:** Thêm experiment `sigma_n` sweep + w/o BO/fixed QAOA; hoặc chuyển claim này sang future work.

## 18. Moderate: `w/o QAOA` label không nhất quán giữa script

**Lỗi:** `run_experiments.py` từng label `use_quantum=False` là "w/o QAOA random node", nhưng behavior là classical greedy node selection. `paper_results.py` dùng `node_random=True` đúng hơn.

**Vì sao phải fix:** Có thể sinh nhầm ablation table tùy script được chạy.

**Fix tối thiểu hợp lệ:** Chỉ giữ một experiment driver chính; xóa hoặc sửa label script cũ.

## 19. Moderate: `bo_vqc` được tạo nhưng không dùng

**Lỗi:** `QuantumHRLAgent` tạo `self.bo_vqc`, nhưng VQC update thực tế bằng REINFORCE/PSR.

**Vì sao phải fix:** Làm người đọc code tưởng VQC có BO, trong khi paper nói PSR policy gradient.

**Fix tối thiểu hợp lệ:** Xóa `bo_vqc` hoặc implement rõ mode VQC BO, nhưng không claim nếu không dùng.

## 20. Moderate: Max node per tier trong paper ghi `<=6`, config là 5

**Lỗi:** Paper nói node-selection search space `M_l <= 6`, nhưng config `[5,3,2,2]`, max là 5.

**Vì sao phải fix:** Lỗi nhỏ nhưng làm reviewer mất niềm tin vào table/setup.

**Fix tối thiểu hợp lệ:** Sửa về `<=5` hoặc giải thích candidate thứ 6 là cloud/fallback nếu có.

## 21. Moderate: LuST scenario hiện mới có speed-pool full 24h, chưa có full trajectory lưu lại

**Lỗi:** `figure_baseline` có speed pool 24h và một FCD window mẫu. Full 24h FCD đã bị xóa sau extraction để tiết kiệm disk.

**Vì sao phải fix:** Nếu muốn paper claim trajectory-level LuST cho main quantum results, speed pool alone chưa đủ. Nó chỉ chứng minh speed distribution, không chứng minh position/sojourn/channel theo trajectory.

**Fix tối thiểu hợp lệ:** Lưu/stream FCD windows vào quantum evaluation hoặc hạ claim thành "LuST-derived speed distribution".

## 22. Moderate: Deadline miss 0% trong synthetic setup có thể không bind

**Lỗi:** Synthetic latency khoảng 0.1-0.2s, deadline `0.5-2s`, nên 0% miss có thể là trivial.

**Vì sao phải fix:** Paper dùng zero deadline violation như reliability claim; nếu constraint không bind, claim yếu.

**Fix tối thiểu hợp lệ:** Thêm stress deadline hoặc LuST/heavy-load setting nơi miss rate khác biệt có ý nghĩa.

## 23. Moderate: Table metrics định nghĩa nhiều nhưng Results báo cáo ít

**Lỗi:** Metrics table có sojourn violation, energy-suboptimality, plateau, reward variance, circuit evaluations, wall-clock, nhưng main Results chủ yếu báo latency/energy/miss/params.

**Vì sao phải fix:** Reviewer có thể hỏi vì sao định nghĩa metric mà không report.

**Fix tối thiểu hợp lệ:** Report tất cả metric đã định nghĩa hoặc bỏ khỏi table.

## 24. Moderate: Hardware/software config có nguy cơ không traceable

**Lỗi:** Paper ghi platform cụ thể, nhưng result file/script không gắn command, seeds, env, git status.

**Vì sao phải fix:** Reproducibility của bảng số hiện tại yếu.

**Fix tối thiểu hợp lệ:** Tạo `results_manifest.json` chứa command, seeds, mobility source, config hash, output files.

## Recommended fix order

1. Chọn pipeline thật: synthetic trung thực hoặc LuST thật.
2. Chốt config duy nhất: units, topology, beta, penalties, node counts.
3. Sửa environment/agent bug tối thiểu: task state stale, greedy nearest, QAOA extraction/fallback logging.
4. Rerun main comparison bằng một script duy nhất.
5. Tính stats seed-level.
6. Viết lại paper theo đúng kết quả mới, không giữ claim cũ nếu số không ủng hộ.

## Current safest paper framing if không rerun ngay

Nếu chưa chạy lại, cách an toàn nhất là:

- Không claim main results dùng LuST trajectory.
- Ghi rõ main results là synthetic T-NTN simulator.
- Trình bày LuST như planned/auxiliary mobility validation, không phải source của bảng chính.
- Hạ claim statistical significance nếu seed-level test không ủng hộ.
- Giữ parameter-efficiency claim, nhưng tách khỏi runtime/speedup và khỏi claim quantum advantage.
