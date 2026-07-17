"""Render fig3_depth_sweep from the measured depth_sweep.json.

Replaces the previous fig3_depth_sweep asset, which was rendered from
hard-coded latency values (visualize_results.plot_depth_sensitivity) rather
than from any experiment. This script plots ONLY what depth_sweep.py measured:
per-seed mean latency at each (L, p) point, shown as the across-seed median
with the min-max range over seeds, plus the Classical-HRL reference measured on
the same seeds.

Run:   .venv_hrl/bin/python simulation/make_depth_fig.py
Writes fig3_depth_sweep.{pdf,png} to simulation/figures/ and overleaf/figures/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
C_Q = "#d62728"
C_C = "#1f77b4"


def load():
    with open(os.path.join(HERE, "depth_sweep.json")) as f:
        return json.load(f)


def slice_points(points, key, fixed_key, fixed_val):
    """Points along `key` with the other depth held at fixed_val, sorted."""
    sel = [pt for pt in points if pt[fixed_key] == fixed_val]
    return sorted(sel, key=lambda pt: pt[key])


def draw(ax, pts, key, xlabel, classical, title):
    x = [pt[key] for pt in pts]
    seed_lat = [np.asarray(pt["seed_latency"], dtype=float) for pt in pts]
    med = np.array([np.median(s) for s in seed_lat])
    lo = np.array([s.min() for s in seed_lat])
    hi = np.array([s.max() for s in seed_lat])

    ax.plot(x, med, "o-", color=C_Q, lw=2.0, ms=6, label="Quantum-HRL (median)")
    ax.fill_between(x, lo, hi, color=C_Q, alpha=0.18, linewidth=0,
                    label="Quantum-HRL (min--max over seeds)")
    ax.axhline(np.median(classical), color=C_C, ls="--", lw=1.5,
               label="Classical HRL (median, same seeds)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg task latency (s)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    return med


def main():
    d = load()
    pts = d["points"]
    g = d["grid"]
    classical = np.asarray(d["classical_hrl_reference"]["seed_latency"], dtype=float)

    L_pts = slice_points(pts, "L", "p", g["p_fixed"])
    p_pts = slice_points(pts, "p", "L", g["L_fixed"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    med_L = draw(axes[0], L_pts, "L", "VQC depth $L$  (at $p=%d$)" % g["p_fixed"],
                 classical, "(a) VQC depth sensitivity")
    med_p = draw(axes[1], p_pts, "p", "QAOA depth $p$  (at $L=%d$)" % g["L_fixed"],
                 classical, "(b) QAOA depth sensitivity")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.tight_layout()

    for out in [os.path.join(HERE, "figures"), os.path.join(REPO, "overleaf", "figures")]:
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "fig3_depth_sweep.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out, "fig3_depth_sweep.png"), dpi=150, bbox_inches="tight")
        print("saved ->", out)

    # Report the numbers the paper's prose must quote.
    n_seeds = len(d["seeds"])
    print(f"\nseeds={d['seeds']}  hash={d['config_hash']}  "
          f"wallclock={d['wallclock_s']:.0f}s")
    print(f"Classical-HRL reference: median={np.median(classical):.4f} "
          f"mean={classical.mean():.4f}+-{classical.std():.4f}")
    print(f"\n{'L':>3} {'p':>3} {'params':>7} {'median':>8} {'mean':>8} {'std':>7} {'miss':>7}")
    for pt in sorted(pts, key=lambda z: (z["L"], z["p"])):
        s = np.asarray(pt["seed_latency"], dtype=float)
        print(f"{pt['L']:>3} {pt['p']:>3} {pt['n_params']:>7} {np.median(s):>8.4f} "
              f"{pt['latency_mean']:>8.4f} {pt['latency_std']:>7.4f} "
              f"{pt['miss_mean']*100:>6.1f}%")
    print(f"\n(n_seeds={n_seeds}; median is the robust statistic quoted in the paper)")
    print("L-slice medians:", np.round(med_L, 4), "at L =", [pt["L"] for pt in L_pts])
    print("p-slice medians:", np.round(med_p, 4), "at p =", [pt["p"] for pt in p_pts])


if __name__ == "__main__":
    main()
