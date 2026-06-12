"""Clean experiment driver producing the numbers reported in the paper.

For each method and seed:
  1. Train (where applicable) for N_TRAIN episodes.
  2. Evaluate on a held-out environment seed for N_EVAL episodes with
     exploration disabled, logging per-task latency, energy and the three
     constraint-violation flags.

Outputs mean +/- std across seeds and a paired significance test
(Quantum-HRL vs. Classical-HRL). Results are written to paper_results.json.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tntn_environment import TNTNEnvironment
from quantum_hrl import (QuantumHRLAgent, ClassicalHRLAgent, SingleDQNAgent,
                         RandomAgent, GreedyAgent)
from utils import STATE_DIM, M_TIERS, vqc_params, qaoa_params, classical_hrl_params

try:
    from scipy import stats as sstats
except Exception:
    sstats = None


N_STEPS = 40  # steps (tasks) per episode; many tasks still average per episode


def evaluate(agent, eval_seed, n_eval):
    """Run policy with exploration off; return per-task arrays."""
    env = TNTNEnvironment(seed=eval_seed, n_steps_per_episode=N_STEPS)
    lat, en, f1, f2, f3 = [], [], [], [], []
    for _ in range(n_eval):
        state = env.reset()
        done = False
        while not done:
            task = env._generate_task()
            l, n, a = agent.select_action(state, task, env, explore=False)
            s_next, r, done, info = env.step(l, n, a, task)
            lat.append(info['latency']); en.append(info['energy'])
            fl = info['constraint_flags']
            f1.append(fl['F1']); f2.append(fl['F2']); f3.append(fl['F3'])
            state = s_next
    return (np.array(lat), np.array(en),
            np.array(f1, float), np.array(f2, float), np.array(f3, float))


def make_agent(name, seed, n_train):
    if name == 'Random':
        return RandomAgent(seed=seed), False
    if name == 'Greedy':
        return GreedyAgent(seed=seed), False
    if name == 'Single-DQN':
        return SingleDQNAgent(state_dim=STATE_DIM, hidden_dim=128,
                              n_episodes=n_train, seed=seed), True
    if name == 'Classical-HRL':
        return ClassicalHRLAgent(state_dim=STATE_DIM, hidden_dim=256,
                                 n_episodes=n_train, seed=seed), True
    if name == 'Quantum-HRL':
        return QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=n_train,
                               batch_size=16, bo_budget=20, vqc_layers=4,
                               qaoa_depth=2, seed=seed, use_quantum=True,
                               verbose=False), True
    raise ValueError(name)


def run_method(name, seeds, n_train, n_eval):
    per_seed = {'latency': [], 'energy': [], 'f1': [], 'f2': [], 'f3': []}
    eval_latency_pool = []
    for seed in seeds:
        agent, trains = make_agent(name, seed, n_train)
        t0 = time.time()
        if trains:
            agent.train(TNTNEnvironment(seed=seed, n_steps_per_episode=N_STEPS))
        lat, en, f1, f2, f3 = evaluate(agent, seed + 10000, n_eval)
        per_seed['latency'].append(float(lat.mean()))
        per_seed['energy'].append(float(en.mean()))
        per_seed['f1'].append(float(f1.mean()))
        per_seed['f2'].append(float(f2.mean()))
        per_seed['f3'].append(float(f3.mean()))
        eval_latency_pool.append(lat)
        print(f"    [{name}] seed={seed} lat={lat.mean():.4f} en={en.mean():.4f} "
              f"miss={f2.mean()*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    summ = {k: [float(np.mean(v)), float(np.std(v))] for k, v in per_seed.items()}
    summ['_lat_pool'] = np.concatenate(eval_latency_pool)
    return summ


def run_ablation(seeds, n_train, n_eval):
    configs = {
        'Full Quantum-HRL':            dict(use_quantum=True),
        'w/o QAOA (random node)':      dict(use_quantum=True, node_random=True),
        'w/o VQC (random tier/ratio)': dict(use_quantum=True, vqc_random=True),
    }
    out = {}
    for cname, cfg in configs.items():
        lats = []
        for seed in seeds:
            agent = QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=n_train,
                                    batch_size=16, bo_budget=20, vqc_layers=4,
                                    qaoa_depth=2, seed=seed,
                                    verbose=False, **cfg)
            agent.train(TNTNEnvironment(seed=seed, n_steps_per_episode=N_STEPS))
            lat, *_ = evaluate(agent, seed + 10000, n_eval)
            lats.append(lat.mean())
            print(f"    [abl:{cname}] seed={seed} lat={lat.mean():.4f}", flush=True)
        out[cname] = [float(np.mean(lats)), float(np.std(lats))]
    return out


def theoretical_params():
    n, h, L, p = STATE_DIM, 256, 4, 2
    q = int(np.ceil(np.log2(n)))
    w1 = classical_hrl_params(n, h, [4], n_layers=2)
    w2 = classical_hrl_params(n, h, [int(max(M_TIERS))], n_layers=2)
    w3 = classical_hrl_params(n, h, [10], n_layers=2)
    W_HRL = w1 + w2 + w3
    W_Q = vqc_params(L, q) + qaoa_params(p)
    # Flat single-DQN: one hidden layer (h=128) over the joint action space
    h_sdqn = 128
    n_actions_joint = int(M_TIERS.sum()) * 10  # nodes x ratio bins
    W_SDQN = n * h_sdqn + h_sdqn * n_actions_joint
    return {'W_DQN_tier': w1, 'W_DQN_node': w2, 'W_DQN_ratio': w3,
            'W_HRL': W_HRL, 'W_SingleDQN': W_SDQN, 'W_VQC': vqc_params(L, q),
            'W_QAOA': qaoa_params(p), 'W_QHRL': W_Q,
            'reduction': W_HRL / W_Q,
            'reduction_vs_sdqn': W_SDQN / W_Q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--train', type=int, default=40)
    ap.add_argument('--eval', type=int, default=15)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.train, args.eval = 1, 3, 3
    seeds = [42 + 137 * i for i in range(args.seeds)]

    print(f"Config: seeds={seeds} train={args.train} eval={args.eval}")
    methods = ['Random', 'Greedy', 'Single-DQN', 'Classical-HRL', 'Quantum-HRL']
    results = {'config': vars(args), 'seeds': seeds, 'methods': {}}

    t0 = time.time()
    for m in methods:
        print(f"\n== {m} ==", flush=True)
        results['methods'][m] = run_method(m, seeds, args.train, args.eval)

    # Significance: Quantum-HRL vs Classical-HRL on pooled eval latencies
    qpool = results['methods']['Quantum-HRL'].pop('_lat_pool')
    cpool = results['methods']['Classical-HRL'].pop('_lat_pool')
    for m in ['Random', 'Greedy', 'Single-DQN']:
        results['methods'][m].pop('_lat_pool', None)
    stats_out = {}
    if sstats is not None:
        n = min(len(qpool), len(cpool))
        tt = sstats.ttest_ind(qpool, cpool, equal_var=False)
        wx = sstats.mannwhitneyu(qpool, cpool, alternative='two-sided')
        stats_out = {'welch_t': float(tt.statistic), 'welch_p': float(tt.pvalue),
                     'mannwhitney_U': float(wx.statistic), 'mannwhitney_p': float(wx.pvalue),
                     'q_mean': float(qpool.mean()), 'c_mean': float(cpool.mean()),
                     'n_q': int(len(qpool)), 'n_c': int(len(cpool))}
    results['significance'] = stats_out

    print("\n== Ablation ==", flush=True)
    # Ablation reports representative values; restrict to a subset of seeds to
    # keep the (quantum-heavy) ablation tractable while the main comparison
    # above uses all seeds.
    results['ablation'] = run_ablation(seeds[:3], args.train, args.eval)
    results['params'] = theoretical_params()
    results['wallclock_s'] = time.time() - t0

    out = os.path.join(os.path.dirname(__file__), 'paper_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}  (total {results['wallclock_s']:.0f}s)")


if __name__ == '__main__':
    main()
