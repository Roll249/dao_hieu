"""
make_figures_full.py
====================
Vẽ figure cho thí nghiệm baseline HRL trên TOÀN BỘ LuST 24h.
Đọc từ results/, lưu PNG vào figure_baseline/.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(REPO, "figure_baseline")
RESULTS = os.path.join(FIG, "results")


def load_log():
    rows = []
    with open(os.path.join(RESULTS, "training_log_full_lust.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    ep = np.array([r["episode"] for r in rows])
    return ep, {k: np.array([r[k] for r in rows]) for k in rows[0]}


def line(ep, y, ylabel, title, fname, color, marker="o", logy=False, note=None):
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(ep, y, color=color, marker=marker, markersize=4, linewidth=1.4)
    plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.4); plt.xlim([ep[0], ep[-1]])
    if logy and np.min(y) > 0:
        plt.yscale("log")
    if note:
        plt.annotate(note, xy=(0.5, 0.5), xycoords="axes fraction",
                     ha="center", va="center", fontsize=9, color="gray")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fname), dpi=300); plt.close()
    print(f"  + {fname}")


def main():
    ep, d = load_log()
    line(ep, d["reward"], "Reward (= −cost mỗi episode)",
         "Hội tụ reward — HRL baseline trên TOÀN BỘ LuST 24h",
         "full_lust_reward_convergence.png", "green", "*")
    note = None
    if np.max(d["loss"]) <= 0:
        note = "Loss ≈ 0 — reward huấn luyện nội-vòng của baseline = 0\n(Learning_Cost trả 0 trong code gốc). Xem loss_diagnostics.md."
    line(ep, d["loss"], "Training loss (MSE) — agent_h",
         "Training loss — DQN tầng High (HRL / LuST 24h)",
         "full_lust_training_loss.png", "crimson", "o", logy=True, note=note)
    line(ep, d["latency"], "Latency trung bình [s]",
         "Latency theo episode — HRL / LuST 24h",
         "full_lust_latency_curve.png", "navy", "s")
    line(ep, d["energy"], "Energy trung bình [J]",
         "Energy theo episode — HRL / LuST 24h",
         "full_lust_energy_curve.png", "darkorange", "^")
    line(ep, d["cost"], "Cost trung bình",
         "Cost theo episode — HRL / LuST 24h",
         "full_lust_cost_curve.png", "purple", "d")
    if "wall_clock_s" in d:
        line(ep, d["wall_clock_s"], "Wall-clock mỗi episode [s]",
             "Thời gian chạy theo episode — HRL / LuST 24h",
             "full_lust_wall_clock_time.png", "teal", "o")

    # --- load & speed distribution toàn 24h ---
    pool = np.load(os.path.join(RESULTS, "lust_speed_pool_full.npy"))
    pw = json.load(open(os.path.join(RESULTS, "per_window_mobility.json"), encoding="utf-8"))
    hours = [r["window"] for r in pw]
    cmean = [r.get("concurrent_mean") or 0 for r in pw]
    cmax = [r.get("concurrent_max") or 0 for r in pw]
    cls = [r.get("traffic_class", "none") for r in pw]
    colmap = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728", "none": "#999999"}
    bar_colors = [colmap[c] for c in cls]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].bar(hours, cmean, color=bar_colors, alpha=0.9, label="mean")
    ax[0].plot(hours, cmax, color="black", marker=".", linewidth=1, label="max")
    ax[0].set_xlabel("Giờ trong ngày (cửa sổ 1h)")
    ax[0].set_ylabel("Số xe đồng thời (toàn mạng)")
    ax[0].set_title("Tải LuST 24h theo giờ (màu = low/medium/high)")
    ax[0].set_xticks(range(0, 24, 2)); ax[0].grid(True, axis="y", alpha=0.4); ax[0].legend()

    ax[1].hist(pool, bins=70, color="darkorange", alpha=0.85)
    ax[1].axvline(np.mean(pool), color="black", linestyle="--", label=f"mean={np.mean(pool):.1f} m/s")
    ax[1].axvline(np.median(pool), color="blue", linestyle=":", label=f"median={np.median(pool):.1f} m/s")
    ax[1].set_xlabel("Tốc độ tức thời [m/s]"); ax[1].set_ylabel("Số mẫu")
    ax[1].set_title("Phân phối tốc độ THẬT LuST 24h")
    ax[1].legend(); ax[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "full_lust_load_speed_distribution.png"), dpi=300); plt.close()
    print("  + full_lust_load_speed_distribution.png")


if __name__ == "__main__":
    print("Vẽ figure full-LuST -> figure_baseline/")
    main()
    print("Xong.")
