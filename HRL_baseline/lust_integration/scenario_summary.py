"""Generate the Vietnamese scenario experiment report from validated outputs."""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCENARIO = REPO / "scenario"
RESULTS = SCENARIO / "results"
sys.path.insert(0, str(HERE / "quantum"))

import bo_offload
import qaoa_node

SCENARIOS = ["Classical", "HRL+VQC", "HRL+VQC+QAOA", "Full-QHRL"]
ABLATIONS = [("Full-QHRL", "Full Quantum-HRL"), ("w/o VQC", "w/o VQC"),
             ("w/o QAOA", "w/o QAOA"), ("HRL+VQC+QAOA", "w/o Bayesian Optimization")]


def pct_change(reference, value):
    return 100.0 * (value - reference) / abs(reference)


def main():
    raw = json.loads((RESULTS / "raw_results.json").read_text())
    summary = raw["summary"]
    tests = json.loads((RESULTS / "statistical_tests.json").read_text())["tests"]
    params = {}
    with (RESULTS / "parameter_count.csv").open() as f:
        for row in csv.DictReader(f):
            params[row["scenario"]] = row

    def scenario_row(name):
        s = summary[name]
        return (f"| {name} | {s['reward_mean']:.4f} ± {s['reward_std']:.4f} | "
                f"{s['latency_mean']:.4f} ± {s['latency_std']:.4f} | "
                f"{s['energy_mean']:.4f} ± {s['energy_std']:.4f} | "
                f"{s['cost_mean']:.4f} ± {s['cost_std']:.4f} | "
                f"{s['deadline_miss_mean']:.4f} ± {s['deadline_miss_std']:.4f} | "
                f"{s['avg_q_mean']:.4f} ± {s['avg_q_std']:.4f} | "
                f"{s['n_params']} | {s['training_time_s_mean']:.1f} ± {s['training_time_s_std']:.1f} |")

    scenario_rows = "\n".join(scenario_row(name) for name in SCENARIOS)
    ablation_rows = "\n".join(
        f"| {label} | {summary[name]['reward_mean']:.4f} ± {summary[name]['reward_std']:.4f} | "
        f"{summary[name]['latency_mean']:.4f} ± {summary[name]['latency_std']:.4f} | "
        f"{summary[name]['energy_mean']:.4f} ± {summary[name]['energy_std']:.4f} | "
        f"{summary[name]['deadline_miss_mean']:.4f} ± {summary[name]['deadline_miss_std']:.4f} |"
        for name, label in ABLATIONS)
    test_rows = "\n".join(
        f"| {key} | {value['t_stat']:.4f} | {value['df']:.2f} | {value['p_value']:.4g} | "
        f"{value['hedges_g']:.4f} | {'Có' if value['significant_0p05'] else 'Không'} |"
        for key, value in tests.items())

    classical, full = summary["Classical"], summary["Full-QHRL"]
    latency_delta = pct_change(classical["latency_mean"], full["latency_mean"])
    energy_delta = pct_change(classical["energy_mean"], full["energy_mean"])
    reward_delta = full["reward_mean"] - classical["reward_mean"]
    high_reduction = int(params["Classical"]["high_level"]) / int(params["Full-QHRL"]["high_level"])
    total_reduction = int(params["Classical"]["total_trainable"]) / int(params["Full-QHRL"]["total_trainable"])

    report = f"""# Scenario Experiments — Quantum-HRL trên LuST 24 giờ

*Sinh tự động lúc {datetime.now().strftime('%Y-%m-%d %H:%M')}.*

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
- Classical HRL dùng nguyên `hrl_core.py`, hyperparameter gốc: {raw['config']['episodes']} episode × {raw['config']['timesteps_train']} timestep, batch size 2, cùng gamma/epsilon.
- Mỗi cấu hình chạy {len(raw['config']['seeds'])} seed độc lập: `{raw['config']['seeds']}`. Kết quả là mean ± sample SD qua seed.
- Đánh giá cuối quét `V={raw['config']['V_eval']}`, {raw['config']['eval_repeats']} lần lặp mỗi mức tải, giống pipeline baseline.
- VQC gồm 4 qubit, angle encoding và 2 lớp entangling; epsilon giữ nguyên nhưng được hiểu theo epsilon-greedy chuẩn vì VQC thay hoàn toàn high-level DQN.
- QAOA dùng one-hot QUBO `Σ cᵢxᵢ + P(Σxᵢ−1)²`, ánh xạ `x=(1−Z)/2`, depth `p={qaoa_node.P}`; hai góc được tối ưu trên lưới xác định cho từng QUBO. Bài toán trên 6 node dùng prefilter theo chi phí để giới hạn statevector ở {qaoa_node.MAX_QUBITS} qubit.
- Bayesian Optimization dùng Gaussian Process RBF và expected improvement, {bo_offload.BO_CALLS} lần đánh giá tỷ lệ. Các xe dùng lịch candidate chung nhưng posterior/giá trị mục tiêu riêng để sweep LuST khả thi trên CPU.
- Bug reward bằng 0 trong `Learning_Cost` của baseline được giữ nguyên. VQC nhận cùng reward và target-update schedule; không dùng cost đánh giá để âm thầm sửa reward huấn luyện.
- `Number of trainable parameters` tính online policies một lần, không nhân đôi target network. Hai góc QAOA được tính là variational parameters; BO không có trainable parameter.

## Kết quả bốn scenario

| Scenario | Reward | Latency (s) | Energy (J) | Cost | Deadline miss rate | Average Q-value | Trainable params | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{scenario_rows}

## Ablation study

| Cấu hình | Reward | Latency (s) | Energy (J) | Deadline miss rate |
|---|---:|---:|---:|---:|
{ablation_rows}

## Welch's t-test: Classical HRL so với Full Quantum-HRL

| Metric | t | df | p-value | Hedges' g | Ý nghĩa ở α=0.05 |
|---|---:|---:|---:|---:|---|
{test_rows}

Các kiểm tra Shapiro–Wilk theo nhóm và Levene (median-centered) được lưu đầy đủ trong `statistical_tests.json`. Với chỉ 5 seed mỗi nhóm, các kiểm tra giả định có công suất thấp; vì vậy effect size và độ lớn thực tế cần được đọc cùng p-value.

## Kết quả nổi bật

- Full Quantum-HRL thay đổi latency {latency_delta:+.2f}% và energy {energy_delta:+.2f}% so với Classical HRL; chênh lệch reward là {reward_delta:+.4f} (reward cao hơn là tốt hơn).
- VQC giảm tham số high-level {high_reduction:.2f}×; tổng số tham số online giảm {total_reduction:.2f}× sau khi vẫn tính middle/low policies không bị thay thế.
- Ý nghĩa của từng thành phần nên đọc trực tiếp từ ba hàng ablation; báo cáo không tuyên bố đóng góp dương nếu số liệu không hỗ trợ.

## Những vấn đề còn tồn tại

- Baseline có reward huấn luyện bằng 0 do lỗi gốc đã được ghi nhận; theo yêu cầu, lỗi này không được sửa. Điều đó làm giới hạn khả năng diễn giải về convergence và Average Q-value.
- QAOA và VQC chạy trên statevector không nhiễu, chưa đại diện latency/noise của quantum hardware thật.
- QAOA phải prefilter khi số node khả thi vượt {qaoa_node.MAX_QUBITS}; đây là giới hạn mô phỏng cổ điển, không phải thay đổi preprocessing LuST.
- Cỡ mẫu 5 seed đủ cho yêu cầu tối thiểu nhưng còn nhỏ cho suy luận thống kê mạnh.

## Khả năng tái lập

- Runner: `HRL_baseline/lust_integration/scenario_runner.py`.
- Checkpoint schema v{raw['schema_version']} lưu sau từng seed trong `scenario/results/`.
- Kết quả nguồn, bảng tổng hợp, kiểm định và parameter count nằm trong `scenario/results/`.
- Mười hai hình PNG dùng nhãn tiếng Anh, error bar là sample SD, palette dịu/colorblind-friendly và 600 DPI.
"""
    output = SCENARIO / "scenario_summary.md"
    output.write_text(report, encoding="utf-8")
    print(f"Đã ghi {output}")


if __name__ == "__main__":
    main()
