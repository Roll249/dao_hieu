# Tóm tắt Paper: Quantum-Enhanced Hierarchical Reinforcement Learning for Task Offloading in Multi-Layer Non-Terrestrial Vehicular Edge Computing

> **Tên ngắn:** Quantum-HRL cho T-NTN Task Offloading
> **Tác giả:** Đào Ngọc Hiếu — Khoa CNTT, Đại học Kinh tế Quốc dân, Hà Nội
> **File chính:** `quantum_hrl_paper.tex` · **Code:** `simulation/` · **Kết quả:** `simulation/paper_results.json`
> **Cập nhật:** 12/06/2026 (sau khi điền toàn bộ số liệu thực nghiệm thật, 5 seeds)

---

## 1. Bài toán

Xe tự hành / xe kết nối sinh ra các tác vụ nặng và nhạy trễ (object detection, path planning, V2X...) vượt quá năng lực tính toán trên xe. Mạng 6G giải quyết bằng kiến trúc **T-NTN (Terrestrial–Non-Terrestrial Network)** gồm 4 tầng:

| Tầng | Độ cao | Đặc điểm |
|---|---|---|
| **RSU** (Road-Side Unit) | mặt đất | gần, băng thông hẹp, compute vừa |
| **LAP** (UAV) | ~1 km | linh hoạt, phủ sóng trung bình |
| **HAP** (khinh khí cầu tầng bình lưu) | ~100 km | phủ rộng, compute mạnh |
| **LEO** (vệ tinh quỹ đạo thấp) | ~2000 km | phủ cực rộng, trễ truyền lớn |

Tại mỗi time slot, bộ điều khiển offloading phải trả lời **3 câu hỏi gắn kết nhau**:
1. **Tầng nào?** (layer selection, 4 lựa chọn)
2. **Tỷ lệ bao nhiêu?** (offloading ratio α ∈ [0,1] — phần còn lại tính tại xe)
3. **Node nào trong tầng?** (combinatorial node selection)

Mục tiêu: tối thiểu hoá **trễ + năng lượng** có trọng số, dưới 3 ràng buộc cứng (C2: deadline, C3: sojourn time — phải xong trước khi xe rời vùng phủ, C4: năng lượng không tệ hơn tính local). Bài toán là **mixed-integer non-convex, NP-hard**.

### Hạn chế của giải pháp cũ (Classical HRL — Shinde & Tarchi)
Dùng **3 DQN nối tiếp** (tier → node → ratio):
- **Bùng nổ tham số:** O(3·n·h) trọng số — hơn 20.000 tham số ngay cả cấu hình nhỏ nhất
- **Node selection bị xử lý như bảng Q phẳng** — bỏ qua cấu trúc đại số của bài toán gán
- **Phối hợp 3 mạng nối tiếp** gây bất ổn định và hội tụ chậm

---

## 2. Giải pháp đề xuất: Quantum-HRL

Thay pipeline 3-DQN bằng **2 module lượng tử + 1 vòng tối ưu cổ điển**:

```
state s_t (n=20 chiều)
   │  Amplitude Encoding → chỉ cần q = ⌈log₂ 20⌉ = 5 qubits
   ▼
┌─────────────────────────┐
│  VQC (L=4 lớp, 20 góc)  │ → high-level: chọn tầng l* + tỷ lệ α
└─────────────────────────┘     (softmax trên ⟨Z⟩, sigmoid trên tổng ⟨Z⟩)
   ▼
┌─────────────────────────┐
│  QAOA (p=2, 4 góc)      │ → low-level: chọn node n* trong tầng l*
└─────────────────────────┘     (node selection → QUBO → Ising Hamiltonian)
   ▼
hành động (l*, n*, α) → môi trường → reward R₁ (train VQC), R₂ (train QAOA)
```

### Ba trụ cột kỹ thuật

1. **VQC + Amplitude Encoding** — nén state n-chiều vào ⌈log₂n⌉ qubits; policy chỉ cần **L·q = 20 tham số** thay vì hàng chục nghìn. VQC được huấn luyện bằng **advantage-weighted policy gradient (REINFORCE)** với gradient **chính xác** từ Parameter-Shift Rule (không cần backprop qua mạch lượng tử).

2. **QAOA cho node selection** — bài toán chọn node được viết thành **QUBO** với ràng buộc one-hot (chọn đúng 1 node), ánh xạ sang **Ising Hamiltonian**; QAOA độ sâu p=2 chỉ cần **2p = 4 tham số**, độc lập với số node. QAOA học "node nào thật sự tốt" từ phản hồi tích luỹ R₂.

3. **Bayesian Optimization** — tinh chỉnh góc QAOA (γ, β) bằng Gaussian Process + Expected Improvement: phù hợp NISQ vì chịu nhiễu đo tốt và cần rất ít lần chạy mạch.

### Điểm mới (novelty)
- **Công trình đầu tiên** kết hợp VQC-based hierarchical policy + QAOA-based combinatorial node selection cho task offloading trong T-NTN đa tầng
- Giảm tham số từ **O(n·h)** xuống **O(L·log₂n + p)** — thay đổi *bậc scaling*, không chỉ hằng số

> ⚠️ Paper **không claim quantum speedup**. Lợi ích được claim là *representational / parameter-efficiency* (compact representation, logarithmic qubit requirement) — đúng chuẩn mực mà reviewer lượng tử yêu cầu.

---

## 3. Kết quả thực nghiệm (số liệu THẬT, 5 seeds độc lập)

### 3.1 So sánh chính (mean ± std, 12 episodes đánh giá × 40 tasks/seed)

| Phương pháp | Latency (s) ↓ | Energy (J) ↓ | Miss rate | # Params |
|---|---|---|---|---|
| Random | 1.937 ± 0.029 | 1.796 ± 0.012 | 22.6% | – |
| Greedy (RSU gần nhất, α=1) | 0.164 ± 0.007 | **0.277 ± 0.004** | 0.0% | – |
| Single DQN (flat) | 0.154 ± 0.018 | 0.911 ± 0.094 | 0.0% | 17.9K |
| Classical HRL (baseline) | 0.147 ± 0.016 | 0.347 ± 0.095 | 0.0% | 20.2K |
| **Quantum-HRL (ours)** | **0.117 ± 0.038** | 0.971 ± 0.322 | **0.0%** | **24** |

**Ba con số headline:**
- ⚡ **Latency giảm 20.3%** so với Classical HRL (23.5% so với Single DQN)
- 📦 **Tham số giảm 99.9%** — 24 vs 20,224 (**843×**; lên tới ~17,000× nếu so với baseline 4 hidden layer gốc)
- ✅ **0% vi phạm** cả 3 ràng buộc (deadline, sojourn, energy budget)

### 3.2 Kiểm định thống kê
Pooled per-task latency, n = 2,400 tasks/method:
- **Welch's t-test:** t = −15.26, **p < 10⁻³**
- **Mann–Whitney U** (không cần giả định phân phối chuẩn): **p < 10⁻³**
- Hướng cải thiện nhất quán trên **cả 5 seeds**

### 3.3 Ablation study (3 seeds)

| Cấu hình | Latency (s) | Δ |
|---|---|---|
| Full Quantum-HRL | 0.108 ± 0.021 | – |
| w/o QAOA (random node) | 0.222 ± 0.156 | **+105.5%** |
| w/o VQC (random tier/ratio) | 2.009 ± 0.094 | **+1,763%** |

→ Cả hai module đều cần thiết; **VQC đóng góp lớn nhất** (quyết định tier/ratio mang nhiều giá trị nhất), QAOA giảm cả mean lẫn variance (tránh node quá tải/xa).

### 3.4 Trade-off latency–energy (báo cáo trung thực)
Quantum-HRL tiêu tốn **nhiều năng lượng hơn** baseline (0.97 vs 0.35 J/task). Cơ chế: VQC hội tụ về chính sách offload mạnh (α ≈ 0.8–0.97) → tối thiểu latency qua xử lý song song max(T_off, T_loc), nhưng trả phí truyền + compute tại edge. Hai phương pháp là **2 điểm Pareto khác nhau** của cùng một objective; có thể kéo Quantum-HRL về phía tiết kiệm năng lượng bằng cách tăng trọng số β₂ trong reward. Paper báo cáo minh bạch thay vì giấu metric bất lợi.

---

## 4. Cấu trúc paper

| Section | Nội dung |
|---|---|
| 1. Introduction | Bối cảnh, 3 hạn chế của classical HRL, đóng góp |
| 2. Notation | Bảng ký hiệu đầy đủ (đã đồng bộ với phương pháp mới) |
| 3. Related Work | 2 nhóm: HRL-for-VEC, Quantum-for-RL/optimization + research gap |
| 4. Quantum Preliminaries | Qubit, gates, Amplitude Encoding, VQC + PSR, QUBO→Ising, QAOA, BO |
| 5. System Model | Kênh truyền, mô hình latency/energy, bài toán (P1), QUBO, MDP |
| 6. Framework + Experiments | Kiến trúc, thuật toán, training (policy gradient), setup, scenarios, **kết quả + thống kê + ablation + discussion** |
| 7. Complexity Analysis | Đếm tham số chi tiết: 20,224 vs 24 → **ρ = 843×** |
| 8. Conclusion | Tổng kết với số liệu cụ thể + hướng tương lai |

---

## 5. Những gì đã sửa trong lần hoàn thiện này

### 5.1 Sửa 3 bug thật trong simulation (`simulation/`)
1. **`ClassicalHRLAgent` không hề học** — không có bước cập nhật trọng số nào → baseline tệ hơn cả Random. Đã thêm TD/DQN update đúng chuẩn (backprop numpy, target network, gradient clipping). Baseline giờ mạnh và **hợp lệ để so sánh**.
2. **VQC huấn luyện sai gradient** — code cũ dùng gradient của ⟨Z⟩ trung bình thay vì gradient của loss; hơn nữa Q-target (−0.5 … −150) không thể biểu diễn bằng output Pauli-Z bị chặn [−1,1] → VQC thực chất không học. Đã thay bằng **REINFORCE với normalized advantage** + per-observable parameter-shift gradients — đúng toán, scale-free.
3. **Ratio policy kẹt ở α≈0.5 và sụp đổ bimodal** — observable đơn qubit có gradient yếu. Đã chuyển sang **collective-sum readout** Σ⟨Z_w⟩ + **offloading prior bias** (hằng số, không phải tham số học) → α hội tụ 0.8–0.97 ổn định trên cả 5 seeds.

Ngoài ra: thêm baseline `SingleDQNAgent` (flat joint action space), ablation flags sạch (`node_random`, `vqc_random`), chuyển VQC sang analytic statevector cho tốc độ (finite-shot/noise nghiên cứu riêng ở NISQ scenario).

### 5.2 Sửa trong paper (`quantum_hrl_paper.tex`)
- **Abstract:** điền số thật thay [X%][Y%][Z%]; claim energy đổi thành trade-off trung thực
- **Bảng kết quả chính + ablation:** toàn bộ số thật mean±std
- **Đoạn Statistical validation mới:** Welch t + Mann–Whitney U
- **Đoạn Discussion mới:** "The latency–energy trade-off" với phân tích cơ chế
- **Section Complexity:** sửa số sai (144K/6005×) → đúng theo code (20,224/843×)
- **Training procedure (Step 1, Step 4, Algorithm 1, setup table):** viết lại khớp 100% với implementation — stochastic policy sampling + advantage-weighted PSR policy gradient (phương trình mới `eq:advantage`, `eq:vqc_pg`); BO chỉ dùng cho QAOA
- **Conclusion:** điền số liệu cụ thể
- Đã xác minh: mọi `\ref` có `\label`, mọi `\cite` có trong bib (**36 references** — đạt target 30–50), 3 figures được tham chiếu đều tồn tại

---

## 6. Hạn chế & việc còn lại

| Việc | Trạng thái |
|---|---|
| Compile PDF | ⚠️ Máy không có TeX engine — cần compile trên Overleaf / cài `texlive` để kiểm tra lần cuối |
| Acknowledgment | `[collaborators / funding bodies]` — tác giả tự điền |
| Scenarios Sc/HL/N (scalability, heavy-load, NISQ-noise) | Mô tả + bảng đã có trong paper; số liệu chi tiết các scenario phụ dùng figure từ run trước — nếu muốn chặt chẽ hơn nên chạy lại bằng code mới |
| Figure depth-sweep (L, p) | Figure tồn tại từ run cũ; nên regenerate với agent đã sửa nếu có thời gian |
| Quantum Preliminaries hơi dài (TODO #5) | Chưa cắt xuống appendix — mức độ ưu tiên thấp, có thể làm khi format theo template hội nghị cụ thể |

### Chạy lại thực nghiệm
```bash
cd simulation
python paper_results.py --seeds 5 --train 35 --eval 12   # ~55 phút, không cần GPU
# kết quả ghi vào simulation/paper_results.json
```

---

## 7. Thông điệp chính của paper (elevator pitch)

> **Quantum-HRL chứng minh rằng một policy lượng tử 24 tham số có thể thay thế pipeline DQN 20.000+ tham số cho bài toán offloading T-NTN — vừa giảm 20% latency với ý nghĩa thống kê mạnh (p < 0.001), vừa đổi bậc scaling của mô hình từ O(n·h) xuống O(log n), trong khi báo cáo minh bạch trade-off năng lượng và không claim quá đà về quantum advantage.**
