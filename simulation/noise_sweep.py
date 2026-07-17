"""NISQ-noise robustness study for the QAOA outer loop (urgent_need_to_fixed.md #17).

Question: when the QAOA cost expectation is estimated under readout noise, does the
Bayesian-Optimisation angle tuner (GP surrogate averages noise) retain node-selection
quality better than plain gradient-descent tuning as the noise level grows?

For each noise std sigma_n we draw many random node-selection QUBO instances, tune the
2p QAOA angles two ways using *noisy* energy evaluations (BO vs finite-difference GD),
decode the selected node from the exact statevector, and report how often each method
recovers the true min-cost node. Output: noise_sweep.csv + noise_sweep.json.
"""
import os, sys, json, time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from qaoa_solver import (build_qaoa_cost_hamiltonian, build_qaoa_mixer,
                         build_qubo_matrix, qubo_to_ising)
from utils import BayesianOptimizer

DEPTH = 2
M = 5                      # nodes in the selection problem
N_INSTANCES = 60
BO_BUDGET = 20
GD_ITERS = 20
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]


def _ising(costs):
    cmin, cmax = costs.min(), costs.max()
    cnorm = (costs - cmin) / (cmax - cmin) if cmax - cmin > 1e-12 else np.zeros_like(costs)
    Q = build_qubo_matrix(cnorm, C.QAOA_ONEHOT_PENALTY)
    h, J, _ = qubo_to_ising(Q, cnorm)
    return h, J


def _energy_qnode(h, J, dev):
    Hc = build_qaoa_cost_hamiltonian(h, J); Hm = build_qaoa_mixer(M)

    @qml.qnode(dev, interface="autograd")
    def qn(params):
        for w in range(M):
            qml.Hadamard(w)
        for q in range(DEPTH):
            qml.qaoa.cost_layer(params[q], Hc)
            qml.qaoa.mixer_layer(params[DEPTH + q], Hm)
        return qml.expval(Hc)
    return qn


def _decode(h, J, angles, dev):
    Hc = build_qaoa_cost_hamiltonian(h, J); Hm = build_qaoa_mixer(M)

    @qml.qnode(dev)
    def pr(params):
        for w in range(M):
            qml.Hadamard(w)
        for q in range(DEPTH):
            qml.qaoa.cost_layer(params[q], Hc)
            qml.qaoa.mixer_layer(params[DEPTH + q], Hm)
        return qml.probs(wires=range(M))
    p = np.asarray(pr(angles), float).ravel()
    idx = np.arange(2 ** M)
    marg = np.array([p[((idx >> (M - 1 - n)) & 1) == 1].sum() for n in range(M)])
    return int(np.argmax(marg))


def _bo_tune(qn, rng, sigma):
    bo = BayesianOptimizer(param_bounds=[(0, 2 * np.pi)] * (2 * DEPTH),
                           n_initial=5, n_iterations=BO_BUDGET, noise_std=max(sigma, 0.05))
    for _ in range(BO_BUDGET):
        a = bo.suggest()
        e = float(qn(pnp.array(a))) + rng.normal(0, sigma)
        bo.report(a, -e)
    best, _ = bo.get_best()
    return best if best is not None else np.zeros(2 * DEPTH)


def _gd_tune(qn, rng, sigma, a0):
    a = np.array(a0, float); lr = 0.1; s = np.pi / 2
    for _ in range(GD_ITERS):
        g = np.zeros_like(a)
        for k in range(len(a)):
            ap = a.copy(); ap[k] += s; am = a.copy(); am[k] -= s
            ep = float(qn(pnp.array(ap))) + rng.normal(0, sigma)
            em = float(qn(pnp.array(am))) + rng.normal(0, sigma)
            g[k] = 0.5 * (ep - em)
        a = np.clip(a - lr * g, 0, 2 * np.pi)
    return a


def main():
    dev = qml.device("default.qubit", wires=M, shots=None)
    rng = np.random.RandomState(0)
    rows = []
    t0 = time.time()
    for sigma in SIGMAS:
        acc_bo = acc_gd = acc_fix = 0
        for _ in range(N_INSTANCES):
            costs = rng.uniform(0.1, 1.0, size=M)
            true_node = int(np.argmin(costs))
            h, J = _ising(costs)
            qn = _energy_qnode(h, J, dev)
            a0 = rng.uniform(0, 2 * np.pi, 2 * DEPTH)
            acc_fix += (_decode(h, J, a0, dev) == true_node)
            acc_gd += (_decode(h, J, _gd_tune(qn, rng, sigma, a0), dev) == true_node)
            acc_bo += (_decode(h, J, _bo_tune(qn, rng, sigma), dev) == true_node)
        n = N_INSTANCES
        row = {"sigma_n": sigma, "acc_BO": acc_bo / n,
               "acc_GD": acc_gd / n, "acc_fixed": acc_fix / n}
        rows.append(row)
        print(f"  sigma={sigma:.2f}  BO={row['acc_BO']:.2f}  "
              f"GD={row['acc_GD']:.2f}  fixed={row['acc_fixed']:.2f}", flush=True)
    out = {"depth": DEPTH, "m_nodes": M, "n_instances": N_INSTANCES,
           "bo_budget": BO_BUDGET, "gd_iters": GD_ITERS, "rows": rows,
           "wallclock_s": time.time() - t0}
    with open(os.path.join(os.path.dirname(__file__), "noise_sweep.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(os.path.dirname(__file__), "noise_sweep.csv"), "w") as f:
        f.write("sigma_n,acc_BO,acc_GD,acc_fixed\n")
        for r in rows:
            f.write(f"{r['sigma_n']},{r['acc_BO']},{r['acc_GD']},{r['acc_fixed']}\n")
    print(f"Saved noise_sweep.json/csv ({out['wallclock_s']:.0f}s)")


if __name__ == "__main__":
    main()
