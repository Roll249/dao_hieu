"""
make_figures.py
===============
Vẽ các hình từ kết quả baseline HRL trên LuST và lưu vào figure_baseline/.

Hình tạo:
  - reward_convergence.png : reward theo episode (pha huấn luyện)
  - training_loss.png      : training-loss của agent_h theo episode
  - latency_curve.png      : độ trễ trung bình theo số xe V
  - energy_curve.png       : năng lượng trung bình theo số xe V
  - summary_bar.png        : so sánh latency / energy / reward
  - lust_load_speed.png    : (bổ sung) tải xe & phân phối tốc độ LuST 17:00-18:00
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(REPO, "figure_baseline")
RESULTS = os.path.join(FIG, "results")


def _load_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def fig_training():
    d = np.load(os.path.join(RESULTS, "training_curves.npz"))
    ep, reward, loss = d["episode"], d["reward"], d["loss"]

    # reward convergence
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, reward, color="green", marker="*", linewidth=1.5, label="HRL (γ₁=γ₂=0.5)")
    plt.xlabel("Episode"); plt.ylabel("Reward (= −cost trung bình mỗi episode)")
    plt.title("Hội tụ reward — HRL baseline trên LuST (17:00–18:00)")
    plt.grid(True, alpha=0.4); plt.legend()
    plt.xlim([ep[0], ep[-1]]); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "reward_convergence.png"), dpi=300); plt.close()

    # training loss
    valid = ~np.isnan(loss)
    if valid.any():
        lv = loss[valid]
        plt.figure(figsize=(7, 4.5))
        plt.plot(ep[valid], lv, color="crimson", marker="o", linewidth=1.5)
        plt.xlabel("Episode"); plt.ylabel("Training loss (MSE) — agent_h")
        plt.title("Training loss — DQN tầng High (HRL baseline / LuST)")
        plt.grid(True, alpha=0.4)
        if lv.min() > 0:
            plt.yscale("log")
        else:
            # loss ≈ 0 (do reward huấn luyện nội-vòng của baseline = 0): dùng linear
            plt.annotate("Loss ≈ 0 — hệ quả reward huấn luyện nội-vòng của baseline = 0\n"
                         "(Learning_Cost trả 0 trong code gốc).",
                         xy=(0.5, 0.5), xycoords="axes fraction",
                         ha="center", va="center", fontsize=9, color="gray")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "training_loss.png"), dpi=300); plt.close()
        print("  + training_loss.png")
    else:
        print("  (bỏ qua training_loss.png — không có giá trị loss)")
    print("  + reward_convergence.png")


def fig_curves():
    m = _load_json("eval_metrics.json")
    V = np.array(m["V"])
    lat, lat_sd = np.array(m["latency_mean"]), np.array(m["latency_std"])
    ene, ene_sd = np.array(m["energy_mean"]), np.array(m["energy_std"])

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(V, lat, yerr=lat_sd, color="green", marker="*", capsize=3, label="HRL")
    plt.xlabel("Số phương tiện V (đồng thời trong vùng phủ)")
    plt.ylabel("Độ trễ trung bình [s]")
    plt.title("Độ trễ vs tải — HRL baseline trên LuST")
    plt.grid(True, alpha=0.4); plt.legend(); plt.xlim([V[0], V[-1]])
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "latency_curve.png"), dpi=300); plt.close()
    print("  + latency_curve.png")

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(V, ene, yerr=ene_sd, color="navy", marker="s", capsize=3, label="HRL")
    plt.xlabel("Số phương tiện V (đồng thời trong vùng phủ)")
    plt.ylabel("Năng lượng trung bình [J]")
    plt.title("Năng lượng vs tải — HRL baseline trên LuST")
    plt.grid(True, alpha=0.4); plt.legend(); plt.xlim([V[0], V[-1]])
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "energy_curve.png"), dpi=300); plt.close()
    print("  + energy_curve.png")


def fig_summary_bar():
    m = _load_json("eval_metrics.json")
    V = np.array(m["V"])
    lat = np.array(m["latency_mean"]); ene = np.array(m["energy_mean"])
    rew = np.array(m["reward_mean"]); cost = np.array(m["cost_mean"])

    # chọn điểm tải đại diện = V gần 100 nhất
    i = int(np.argmin(np.abs(V - 100)))
    labels = ["Latency [s]", "Energy [J]", "Cost", "Reward (=−cost)"]
    vals = [lat[i], ene[i], cost[i], rew[i]]
    colors = ["green", "navy", "gray", "orange"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Giá trị")
    plt.title(f"Tổng hợp chỉ số HRL @ V={int(V[i])} (LuST 17:00–18:00)")
    plt.grid(True, axis="y", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "summary_bar.png"), dpi=300); plt.close()
    print("  + summary_bar.png")


def fig_lust_context():
    mob = _load_json("lust_mobility.json")
    t = np.array(mob["concurrent_vehicles"]["series_time_s"])
    c = np.array(mob["concurrent_vehicles"]["series_count"])
    pool = np.load(os.path.join(RESULTS, "lust_speed_pool.npy"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot((t - t[0]) / 60.0, c, color="teal")
    ax[0].set_xlabel("Phút kể từ 17:00"); ax[0].set_ylabel("Số xe đồng thời (toàn mạng)")
    ax[0].set_title("Tải LuST theo thời gian (17:00–18:00)")
    ax[0].grid(True, alpha=0.4)

    ax[1].hist(pool, bins=60, color="darkorange", alpha=0.85)
    ax[1].axvline(np.mean(pool), color="black", linestyle="--",
                  label=f"mean={np.mean(pool):.1f} m/s")
    ax[1].set_xlabel("Tốc độ tức thời [m/s]"); ax[1].set_ylabel("Số mẫu")
    ax[1].set_title("Phân phối tốc độ THẬT của LuST")
    ax[1].legend(); ax[1].grid(True, alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "lust_load_speed.png"), dpi=300); plt.close()
    print("  + lust_load_speed.png")


def main():
    print("Vẽ hình -> figure_baseline/")
    fig_training()
    fig_curves()
    fig_summary_bar()
    fig_lust_context()
    print("Xong.")


if __name__ == "__main__":
    main()
