# Loss Diagnostics — vì sao `training_loss ≈ 0`

## 1. Mức biểu thức (root cause)
- `a_m` là vector one-hot (node = 7).
- Biểu thức GỐC `np.nonzero(a_m) == 7`  -> **False**  (so sánh tuple `(array([7]),)` với số nguyên → luôn False).
- Biểu thức ĐÚNG `np.nonzero(a_m)[0][0] == 7` -> **True**.
- Vì điều kiện luôn False, các nhánh cộng dồn chi phí trong `Learning_Cost` không bao giờ chạy ⇒ `T_L = T_E = 0`.

## 2. Mức hàm `Learning_Cost` (gốc vs đã sửa)
- Số vị trí so sánh lỗi đã thay trong bản sửa: **12**.
- Số mẫu (v, node hợp lệ) đánh giá: **18**.
- Reward GỐC khác 0: **0/18** | trung bình = **0.0000** ⇒ baseline gốc luôn cho reward = 0.
- Bản ĐÃ SỬA: reward khác 0 ở **0/18** mẫu (trung bình 0.0000); **18/18** mẫu phát sinh lỗi shape do kích hoạt nhánh code chưa từng chạy.
  → Cả hai (reward≠0 hoặc lỗi shape) đều xác nhận: trong baseline gốc, các nhánh tính chi phí của `Learning_Cost` **không bao giờ được thực thi**.

## 3. Hệ quả tới training-loss
- `agent_h.retrain` đặt `target[0][action] = reward + gamma * max(target_net(next))`. Với baseline gốc, `reward = 0` và mạng khởi tạo trọng số 0 (dummy) ⇒ `target = 0`; `q_network` dự đoán ≈ 0 ⇒ **loss (MSE) ≈ 0**.
- Nói cách khác, nếu các nhánh chi phí thực sự chạy (reward ≠ 0) thì target ≠ 0 và loss > 0; nhưng trong baseline GỐC điều đó không xảy ra ⇒ loss giữ ≈ 0.

## 4. Quyết định
- **GIỮ NGUYÊN baseline gốc** cho mọi kết quả chính (không 'sửa bug' để loss đẹp hơn) — yêu cầu không được thay đổi thuật toán.
- Bản sửa chỉ tồn tại trong script phân tích này để KIỂM CHỨNG nguyên nhân, không dùng cho `run_baseline_full.py`.
- Vì vậy `full_lust_training_loss.png` ≈ 0 là **đúng theo baseline gốc**, không phải lỗi tích hợp.
