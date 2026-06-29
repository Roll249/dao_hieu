"""
scenario_figures.py — Publication-quality figures (English labels) -> scenario/
Reads scenario/results/raw_results.json.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCN = os.path.join(REPO, "scenario")
RESULTS = os.path.join(SCN, "results")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 600, "axes.grid": True, "grid.alpha": 0.22,
    "axes.spines.top": False, "axes.spines.right": False,
})

SCEN_ORDER = ["Classical", "HRL+VQC", "HRL+VQC+QAOA", "Full-QHRL"]
ABL_ORDER = ["Full-QHRL", "w/o VQC", "w/o QAOA", "HRL+VQC+QAOA"]
ABL_LABEL = {"Full-QHRL": "Full", "w/o VQC": "w/o VQC", "w/o QAOA": "w/o QAOA",
             "HRL+VQC+QAOA": "w/o BO"}
# muted, colorblind-friendly palette
PALETTE = {"Classical": "#4C72B0", "HRL+VQC": "#55A868", "HRL+VQC+QAOA": "#C44E52",
           "Full-QHRL": "#8172B3", "w/o VQC": "#937860", "w/o QAOA": "#DA8BC3"}


def load():
    raw = json.load(open(os.path.join(RESULTS, "raw_results.json")))
    runs = raw["runs"]; summary = raw["summary"]
    by = {}
    for r in runs:
        by.setdefault(r["name"], []).append(r)
    return raw, by, summary


def agg_episode(by, name, key):
    rs = by[name]
    M = np.array([[e[key] for e in r["training_log"]] for r in rs])  # (seeds, episodes)
    return M.mean(axis=0), M.std(axis=0, ddof=1)


def bar(names, summary, key, ylabel, title, fname, labels=None, logy=False):
    labels = labels or names
    means = [summary[n][key + "_mean"] for n in names]
    stds = [summary[n][key + "_std"] for n in names]
    colors = [PALETTE.get(n, "#777777") for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(names))
    hatches = ["", "//", "..", "xx", "\\\\", "++"]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.86,
                  edgecolor="black", linewidth=0.6)
    for rect, hatch in zip(bars, hatches):
        rect.set_hatch(hatch)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + (s if not logy else 0), f"{m:.3g}", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", hatch=h, label=l, alpha=0.86)
                       for c, h, l in zip(colors, hatches, labels)],
              frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(SCN, fname)); plt.close(fig)
    print(f"  + {fname}")


def main():
    raw, by, summary = load()

    # 1. reward convergence
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for n in SCEN_ORDER:
        mu, sd = agg_episode(by, n, "reward")
        ep = np.arange(1, len(mu) + 1)
        styles = {"Classical": "-", "HRL+VQC": "--", "HRL+VQC+QAOA": "-.", "Full-QHRL": ":"}
        ax.plot(ep, mu, marker="o", markevery=5, ms=3, linestyle=styles[n],
                color=PALETTE[n], label=n)
        ax.fill_between(ep, mu - sd, mu + sd, color=PALETTE[n], alpha=0.15)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward (= -cost)")
    ax.set_title("Reward Convergence across Scenarios (LuST)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(SCN, "scenario_reward_convergence.png")); plt.close(fig)
    print("  + scenario_reward_convergence.png")

    # 2. training loss
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for n in SCEN_ORDER:
        mu, sd = agg_episode(by, n, "loss")
        ep = np.arange(1, len(mu) + 1)
        ax.plot(ep, mu, marker="s", markevery=5, ms=3, color=PALETTE[n], label=n)
        ax.fill_between(ep, np.maximum(0.0, mu - sd), mu + sd,
                        color=PALETTE[n], alpha=0.12)
    ax.set_xlabel("Episode"); ax.set_ylabel("Training loss (MSE)")
    ax.set_title("Training Loss across Scenarios (VQC high-level)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(SCN, "scenario_training_loss.png")); plt.close(fig)
    print("  + scenario_training_loss.png")

    # 3-6 comparisons
    bar(SCEN_ORDER, summary, "latency", "Latency [s]", "Latency Comparison",
        "scenario_latency_comparison.png")
    bar(SCEN_ORDER, summary, "energy", "Energy [J]", "Energy Comparison",
        "scenario_energy_comparison.png")
    bar(SCEN_ORDER, summary, "cost", "Cost", "Cost Comparison",
        "scenario_cost_comparison.png")
    bar(SCEN_ORDER, summary, "deadline_miss", "Deadline miss rate", "Deadline Miss Rate",
        "scenario_deadline_miss.png")

    # 7. parameter count (log)
    pc = {}
    with open(os.path.join(RESULTS, "parameter_count.csv")) as f:
        import csv as _csv
        for row in _csv.DictReader(f):
            pc[row["scenario"]] = int(row["total_trainable"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(SCEN_ORDER))
    vals = [pc[n] for n in SCEN_ORDER]
    bars = ax.bar(x, vals, color=[PALETTE[n] for n in SCEN_ORDER], alpha=0.86,
                  edgecolor="black", linewidth=0.6)
    for rect, hatch in zip(bars, ["", "//", "..", "xx"]): rect.set_hatch(hatch)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(SCEN_ORDER, rotation=12)
    ax.set_ylabel("Total trainable parameters"); ax.set_title("Trainable Parameter Count")
    for xi, v in enumerate(vals):
        ax.text(xi, v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[Patch(facecolor=PALETTE[n], edgecolor="black", label=n)
                       for n in SCEN_ORDER], frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(SCN, "scenario_parameter_count.png")); plt.close(fig)
    print("  + scenario_parameter_count.png")

    # 8. training time
    bar(SCEN_ORDER, summary, "training_time_s", "Wall-clock training time [s]",
        "Total Training Time", "scenario_training_time.png")

    # 9. latency-energy tradeoff
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for n in SCEN_ORDER:
        lx = summary[n]["latency_mean"]; ly = summary[n]["energy_mean"]
        lxs = summary[n]["latency_std"]; lys = summary[n]["energy_std"]
        ax.errorbar(lx, ly, xerr=lxs, yerr=lys, fmt="o", ms=9, capsize=4,
                    color=PALETTE[n], label=n)
        ax.annotate(n, (lx, ly), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Latency [s]"); ax.set_ylabel("Energy [J]")
    ax.set_title("Latency–Energy Trade-off")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(SCN, "scenario_latency_energy_tradeoff.png")); plt.close(fig)
    print("  + scenario_latency_energy_tradeoff.png")

    # 10-12 ablation
    labels = [ABL_LABEL[n] for n in ABL_ORDER]
    bar(ABL_ORDER, summary, "reward", "Reward (= -cost)", "Ablation: Reward",
        "scenario_ablation_reward.png", labels=labels)
    bar(ABL_ORDER, summary, "latency", "Latency [s]", "Ablation: Latency",
        "scenario_ablation_latency.png", labels=labels)
    bar(ABL_ORDER, summary, "energy", "Energy [J]", "Ablation: Energy",
        "scenario_ablation_energy.png", labels=labels)


if __name__ == "__main__":
    print("Rendering scenario figures -> scenario/")
    main()
    print("Done.")
