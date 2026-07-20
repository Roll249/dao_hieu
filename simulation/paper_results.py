"""Clean experiment driver producing the numbers reported in the paper.

Single canonical driver (run_experiments.py is deprecated). For each method and
seed it:
  1. Trains (where applicable) for N_TRAIN episodes on the LuST mobility env.
  2. Evaluates on a held-out env seed for N_EVAL episodes with exploration off,
     logging per-task latency, energy and the three constraint-violation flags.

Statistics (urgent_need_to_fixed.md #16):
  * PRIMARY  : seed-level Welch t-test on per-seed means (n = #seeds), with
               Hedges' g effect size. Seeds are the independent replicates.
  * SUPPLEMENTARY: pooled per-task test (flagged as pseudo-replicated).

Provenance (urgent_need_to_fixed.md #24) is written to results_manifest.json:
config + hash, git commit/dirty, package versions, seeds, mobility source,
per-method QAOA fallback rate, wall-clock and output files.
"""
import os, sys, json, time, argparse, subprocess, platform, warnings
import numpy as np

# BO's GP surrogate routinely hits the noise-kernel lower bound; the resulting
# sklearn ConvergenceWarning is expected and would otherwise flood the log.
warnings.filterwarnings("ignore", message=".*close to the specified lower bound.*")
warnings.filterwarnings("ignore", message=".*no trainable parameters.*")

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from tntn_environment import TNTNEnvironment
from quantum_hrl import (QuantumHRLAgent, ClassicalHRLAgent, SingleDQNAgent,
                         RandomAgent, GreedyAgent)
from classical_pg_baseline import TinyClassicalPGAgent
from utils import STATE_DIM, M_TIERS, vqc_params, qaoa_params, classical_hrl_params

try:
    from scipy import stats as sstats
except Exception:
    sstats = None

N_STEPS = 40         # decision slots (tasks) per episode
MOBILITY = "lust"


def _make_env(seed):
    return TNTNEnvironment(seed=seed, n_steps_per_episode=N_STEPS, mobility=MOBILITY)


def evaluate(agent, eval_seed, n_eval):
    """Run policy with exploration off; return per-task arrays."""
    env = _make_env(eval_seed)
    lat, en, f1, f2, f3, tiers = [], [], [], [], [], []
    for _ in range(n_eval):
        env.reset()
        done = False
        while not done:
            task = env._generate_task()
            state = env.observe(task)          # condition on current task (#15)
            l, n, a = agent.select_action(state, task, env, explore=False)
            s_next, r, done, info = env.step(l, n, a, task)
            lat.append(info['latency']); en.append(info['energy'])
            fl = info['constraint_flags']
            f1.append(fl['F1']); f2.append(fl['F2']); f3.append(fl['F3'])
            tiers.append(l)
    return (np.array(lat), np.array(en), np.array(f1, float),
            np.array(f2, float), np.array(f3, float), np.array(tiers))


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
    if name == 'TinyClassical-PG':
        # Parameter-matched classical control (Reviewer M1): same REINFORCE and
        # readouts as the VQC, low-dimensional classical MLP policy (~83 params).
        return TinyClassicalPGAgent(state_dim=STATE_DIM, hidden=3,
                                    n_episodes=n_train, batch_size=16,
                                    seed=seed), True
    if name == 'Quantum-HRL':
        return QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=n_train,
                               batch_size=16, bo_budget=20, vqc_layers=4,
                               qaoa_depth=2, seed=seed, use_quantum=True,
                               verbose=False), True
    raise ValueError(name)


def run_method(name, seeds, n_train, n_eval):
    per_seed = {'latency': [], 'energy': [], 'f1': [], 'f2': [], 'f3': []}
    lat_pool, en_pool = [], []
    fb_num = fb_den = 0
    tier_hist = np.zeros(4)
    for seed in seeds:
        agent, trains = make_agent(name, seed, n_train)
        t0 = time.time()
        if trains:
            agent.train(_make_env(seed))
        lat, en, f1, f2, f3, tiers = evaluate(agent, seed + 10000, n_eval)
        per_seed['latency'].append(float(lat.mean()))
        per_seed['energy'].append(float(en.mean()))
        per_seed['f1'].append(float(f1.mean()))
        per_seed['f2'].append(float(f2.mean()))
        per_seed['f3'].append(float(f3.mean()))
        lat_pool.append(lat); en_pool.append(en)
        for t in range(4):
            tier_hist[t] += (tiers == t).sum()
        if hasattr(agent, 'qaoa_fallback_stats'):
            fb, ex, _ = agent.qaoa_fallback_stats()
            fb_num += fb; fb_den += ex
        print(f"    [{name}] seed={seed} lat={lat.mean():.4f} en={en.mean():.4f} "
              f"miss={f2.mean()*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    summ = {k: [float(np.mean(v)), float(np.std(v))] for k, v in per_seed.items()}
    summ['_seed_latency'] = [float(x) for x in per_seed['latency']]
    summ['_seed_energy'] = [float(x) for x in per_seed['energy']]
    summ['_lat_pool'] = np.concatenate(lat_pool)
    summ['_en_pool'] = np.concatenate(en_pool)
    summ['tier_fraction'] = (tier_hist / max(tier_hist.sum(), 1)).tolist()
    if fb_den:
        summ['qaoa_fallback_rate'] = fb_num / fb_den
    return summ


def _hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = np.sqrt(sp2) if sp2 > 0 else 1e-12
    d = (a.mean() - b.mean()) / sp
    J = 1.0 - 3.0 / (4 * (na + nb) - 9)     # small-sample correction
    return float(d * J)


def seed_level_test(q_seed, c_seed, label):
    """Welch t-test on per-seed means (independent replicates) + Hedges' g."""
    q, c = np.asarray(q_seed, float), np.asarray(c_seed, float)
    out = {'q_mean': float(q.mean()), 'q_std': float(q.std(ddof=1)) if len(q) > 1 else 0.0,
           'c_mean': float(c.mean()), 'c_std': float(c.std(ddof=1)) if len(c) > 1 else 0.0,
           'n_seeds': int(len(q)), 'hedges_g': _hedges_g(q, c)}
    if sstats is not None and len(q) > 1:
        tt = sstats.ttest_ind(q, c, equal_var=False)
        out['welch_t'] = float(tt.statistic); out['welch_p'] = float(tt.pvalue)
        try:
            out['levene_p'] = float(sstats.levene(q, c).pvalue)
        except Exception:
            pass
    return out


def pooled_test(qpool, cpool):
    """SUPPLEMENTARY pooled per-task test (pseudo-replicated — flagged)."""
    if sstats is None:
        return {}
    tt = sstats.ttest_ind(qpool, cpool, equal_var=False)
    wx = sstats.mannwhitneyu(qpool, cpool, alternative='two-sided')
    return {'welch_t': float(tt.statistic), 'welch_p': float(tt.pvalue),
            'mannwhitney_U': float(wx.statistic), 'mannwhitney_p': float(wx.pvalue),
            'n_q': int(len(qpool)), 'n_c': int(len(cpool)),
            'note': 'pseudo-replicated: per-task samples within a seed are not independent'}


def run_ablation(seeds, n_train, n_eval):
    configs = {
        'Full Quantum-HRL':            dict(use_quantum=True),
        'w/o VQC (random tier/ratio)': dict(use_quantum=True, vqc_random=True),
        'w/o QAOA (random node)':      dict(use_quantum=True, node_random=True),
        'w/o BO (fixed QAOA refine)':  dict(use_quantum=True, use_bo=False),
    }
    out = {}
    for cname, cfg in configs.items():
        lats, ens, miss = [], [], []
        for seed in seeds:
            agent = QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=n_train,
                                    batch_size=16, bo_budget=20, vqc_layers=4,
                                    qaoa_depth=2, seed=seed, verbose=False, **cfg)
            agent.train(_make_env(seed))
            lat, en, f1, f2, f3, _ = evaluate(agent, seed + 10000, n_eval)
            lats.append(lat.mean()); ens.append(en.mean()); miss.append(f2.mean())
            print(f"    [abl:{cname}] seed={seed} lat={lat.mean():.4f} "
                  f"en={en.mean():.4f} miss={f2.mean()*100:.1f}%", flush=True)
        out[cname] = {'latency': [float(np.mean(lats)), float(np.std(lats))],
                      'energy': [float(np.mean(ens)), float(np.std(ens))],
                      'miss': [float(np.mean(miss)), float(np.std(miss))]}
    return out


def theoretical_params():
    n, h, L, p = STATE_DIM, 256, 4, 2
    q = int(np.ceil(np.log2(n)))
    w1 = classical_hrl_params(n, h, [4], n_layers=2)
    w2 = classical_hrl_params(n, h, [int(max(M_TIERS))], n_layers=2)
    w3 = classical_hrl_params(n, h, [10], n_layers=2)
    W_HRL = w1 + w2 + w3                 # trainable ONLINE parameters
    W_HRL_stored = 2 * W_HRL            # online + frozen target networks (#10)
    W_Q = vqc_params(L, q) + qaoa_params(p)
    h_sdqn = 128
    n_actions_joint = int(M_TIERS.sum()) * 10
    W_SDQN = n * h_sdqn + h_sdqn * n_actions_joint
    h_tiny = 3   # TinyClassical-PG hidden width (M1 control)
    W_TinyPG = n * h_tiny + h_tiny + h_tiny * 5 + 5   # state->h(tanh)->5 outputs
    return {'W_DQN_tier': w1, 'W_DQN_node': w2, 'W_DQN_ratio': w3,
            'W_HRL_online': W_HRL, 'W_HRL_stored': W_HRL_stored,
            'W_SingleDQN': W_SDQN, 'W_VQC': vqc_params(L, q),
            'W_QAOA': qaoa_params(p), 'W_QHRL': W_Q, 'W_TinyPG': W_TinyPG,
            'reduction_online': W_HRL / W_Q,
            'reduction_stored': W_HRL_stored / W_Q,
            'reduction_vs_sdqn': W_SDQN / W_Q,
            'note': 'reduction_online uses trainable online params; reduction_stored '
                    'counts online+target. Headline uses reduction_online.'}


def _git(*args):
    try:
        return subprocess.check_output(['git', *args],
                                       cwd=os.path.dirname(os.path.dirname(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _versions():
    v = {'python': platform.python_version()}
    for mod in ('numpy', 'scipy', 'sklearn', 'pennylane'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def write_manifest(args, seeds, wallclock, fallback, out_files, tag=''):
    from lust_mobility import LuSTMobilityProvider
    prov = LuSTMobilityProvider()
    manifest = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': 'python paper_results.py ' + ' '.join(sys.argv[1:]),
        'seeds': seeds,
        'config': C.CONFIG,
        'config_hash': C.config_hash(),
        'mobility': {'source': 'LuST SUMO FCD', 'window': C.LUST_WINDOW_CLOCK,
                     'fcd': C.LUST_FCD, 'roi_x_m': C.LUST_ROI_X, 'roi_y_m': C.LUST_ROI_Y,
                     'n_trajectories': prov.n_traj(), 'area_km': prov.area_km},
        'n_steps_per_episode': N_STEPS,
        'train_episodes': args.train, 'eval_episodes': args.eval,
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'versions': _versions(),
        'qaoa_fallback_rate': fallback,
        'wallclock_s': wallclock,
        'output_files': out_files,
    }
    path = os.path.join(os.path.dirname(__file__), f'results_manifest{tag}.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=float)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--train', type=int, default=30)
    ap.add_argument('--eval', type=int, default=12)
    ap.add_argument('--abl-seeds', type=int, default=3)   # ablation over a subset to bound runtime
    ap.add_argument('--tag', type=str, default='')        # output suffix, e.g. _heavy
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.train, args.eval, args.abl_seeds = 2, 4, 4, 2
    seeds = [42 + 137 * i for i in range(args.seeds)]

    print(f"Config: seeds={seeds} train={args.train} eval={args.eval} "
          f"mobility={MOBILITY} hash={C.config_hash()}")
    methods = ['Random', 'Greedy', 'Single-DQN', 'Classical-HRL',
               'TinyClassical-PG', 'Quantum-HRL']
    results = {'config': vars(args), 'seeds': seeds, 'mobility': MOBILITY,
               'config_hash': C.config_hash(), 'methods': {}}

    t0 = time.time()
    fallback = {}
    for m in methods:
        print(f"\n== {m} ==", flush=True)
        summ = run_method(m, seeds, args.train, args.eval)
        if 'qaoa_fallback_rate' in summ:
            fallback[m] = summ['qaoa_fallback_rate']
        results['methods'][m] = summ

    q = results['methods']['Quantum-HRL']
    c = results['methods']['Classical-HRL']
    results['significance'] = {
        'latency_seed_level': seed_level_test(q['_seed_latency'], c['_seed_latency'], 'latency'),
        'energy_seed_level': seed_level_test(q['_seed_energy'], c['_seed_energy'], 'energy'),
        'latency_pooled_supplementary': pooled_test(q['_lat_pool'], c['_lat_pool']),
    }
    # Reviewer M1: is the quantum policy distinguishable from a parameter-matched
    # classical policy-gradient control (and from the tri-DQN)?
    if 'TinyClassical-PG' in results['methods']:
        t = results['methods']['TinyClassical-PG']
        results['significance']['latency_quantum_vs_tinyPG'] = \
            seed_level_test(q['_seed_latency'], t['_seed_latency'], 'latency')
        results['significance']['energy_quantum_vs_tinyPG'] = \
            seed_level_test(q['_seed_energy'], t['_seed_energy'], 'energy')
        results['significance']['latency_tinyPG_vs_classical'] = \
            seed_level_test(t['_seed_latency'], c['_seed_latency'], 'latency')
    # strip bulky pooled arrays before serialising
    for mm in results['methods'].values():
        for k in ('_lat_pool', '_en_pool'):
            mm.pop(k, None)

    print("\n== Ablation ==", flush=True)
    abl_seeds = seeds[:max(1, args.abl_seeds)]
    results['ablation_seeds'] = abl_seeds
    results['ablation'] = run_ablation(abl_seeds, args.train, args.eval)
    results['params'] = theoretical_params()
    results['qaoa_fallback_rate'] = fallback
    results['wallclock_s'] = time.time() - t0

    tag = args.tag
    out = os.path.join(os.path.dirname(__file__), f'paper_results{tag}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    manifest = write_manifest(args, seeds, results['wallclock_s'], fallback,
                              [f'paper_results{tag}.json'], tag=tag)
    print(f"\nSaved {out} and {manifest}  (total {results['wallclock_s']:.0f}s)")


if __name__ == '__main__':
    main()
