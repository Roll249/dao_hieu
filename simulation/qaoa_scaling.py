"""QAOA node-selection quality vs problem size (lever 1, honest scaling picture).

Classical greedy node selection (argmin of the directly-computable per-node cost) is
EXACT, so QAOA cannot beat it on solution quality; QAOA's advantage is a constant
2p-angle / logarithmic-parameter footprint versus a DQN node head that grows with the
node count. This study quantifies the flip side honestly: at fixed depth p, QAOA's
recovery of the true min-cost node degrades as the node count m grows (more falls back
to the classical heuristic); increasing depth p recovers quality at still-constant
parameter scaling. Output: qaoa_scaling.csv/json.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from qaoa_solver import QAOASolver

MS = [3, 5, 8, 10, 12]
DEPTHS = [2, 4]
N_INSTANCES = 60


def main():
    rng = np.random.RandomState(0)
    rows = []
    t0 = time.time()
    for m in MS:
        for depth in DEPTHS:
            acc = fb = 0
            for i in range(N_INSTANCES):
                costs = rng.uniform(0.1, 1.0, size=m)
                solver = QAOASolver(n_nodes=m, depth=depth, seed=1000 + i)
                node, _, info = solver.solve(costs, penalty=C.QAOA_ONEHOT_PENALTY,
                                             n_iterations=8)
                acc += (node == int(np.argmin(costs)))
                fb += info['fell_back']
            rows.append({"m": m, "depth": depth, "qaoa_params": 2 * depth,
                         "dqn_node_head_params": 20 * 256 + 256 * m,   # grows with m
                         "acc": acc / N_INSTANCES, "fallback": fb / N_INSTANCES})
            print(f"  m={m:2d} p={depth}  acc={acc/N_INSTANCES:.2f}  "
                  f"fallback={fb/N_INSTANCES:.2f}  (QAOA {2*depth} params vs "
                  f"DQN-head {20*256+256*m})", flush=True)
    out = {"n_instances": N_INSTANCES, "rows": rows, "wallclock_s": time.time() - t0}
    with open(os.path.join(os.path.dirname(__file__), "qaoa_scaling.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(os.path.dirname(__file__), "qaoa_scaling.csv"), "w") as f:
        f.write("m,depth,qaoa_params,dqn_node_head_params,acc,fallback\n")
        for r in rows:
            f.write(f"{r['m']},{r['depth']},{r['qaoa_params']},"
                    f"{r['dqn_node_head_params']},{r['acc']},{r['fallback']}\n")
    print(f"Saved qaoa_scaling.json/csv ({out['wallclock_s']:.0f}s)")


if __name__ == "__main__":
    main()
