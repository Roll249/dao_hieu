"""Circuit-depth sweep for the paper's Fig. 3 (VQC depth L, QAOA depth p).

Replaces the previously hard-coded/synthetic depth-sensitivity figure with a
measurement on the real LuST base regime, using exactly the same environment,
seed convention, train/eval budget and agent construction as paper_results.py.

Grid (the two 1-D slices the paper reports, sharing the L=4/p=2 operating point):
  * VQC depth  : L in {1, 2, 4, 6} at fixed p = 2
  * QAOA depth : p in {1, 2, 3}    at fixed L = 4

Classical-HRL is run on the same seeds as a reference line, so the figure's
baseline is a measured number rather than a constant carried over from an
older configuration.

The full grid takes hours, so every completed (L, p) point is checkpointed to
depth_sweep_points.json as soon as it finishes. Re-running resumes from that
file and only recomputes the missing points; a checkpoint is reused only when
the config hash, seeds and episode budget all match the current run.

Run:  ../.venv_hrl/bin/python -u depth_sweep.py            (~3 h, 3 seeds)
      ../.venv_hrl/bin/python -u depth_sweep.py --quick    (smoke test)
      ../.venv_hrl/bin/python -u depth_sweep.py --fresh    (ignore checkpoint)
"""
import os, sys, json, time, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*close to the specified lower bound.*")
warnings.filterwarnings("ignore", message=".*no trainable parameters.*")

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from quantum_hrl import QuantumHRLAgent, ClassicalHRLAgent
from utils import STATE_DIM, vqc_params, qaoa_params
from paper_results import _make_env, evaluate, _git, _versions, N_STEPS, MOBILITY

# (L, p) points: the L-slice at p=2 and the p-slice at L=4.
L_VALUES = [1, 2, 4, 6]
P_VALUES = [1, 2, 3]
L_FIXED, P_FIXED = 4, 2


CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'depth_sweep_points.json')


def grid():
    pts = [(L, P_FIXED) for L in L_VALUES]
    pts += [(L_FIXED, p) for p in P_VALUES if (L_FIXED, p) not in pts]
    return pts


def _run_key(args, seeds):
    """Identity of a run: reusing a checkpoint across a config change is unsafe."""
    return {'config_hash': C.config_hash(), 'seeds': list(seeds),
            'train': args.train, 'eval': args.eval}


def load_ckpt(key, fresh):
    """Return {(L,p): point} plus any classical reference already computed."""
    if fresh or not os.path.exists(CKPT):
        return {}, None
    try:
        with open(CKPT) as f:
            blob = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ckpt] unreadable ({e}); starting fresh", flush=True)
        return {}, None
    if blob.get('key') != key:
        print("  [ckpt] config/seeds/budget changed since checkpoint; starting fresh",
              flush=True)
        return {}, None
    pts = {(p['L'], p['p']): p for p in blob.get('points', [])}
    if pts:
        print(f"  [ckpt] resuming: {sorted(pts)} already done", flush=True)
    return pts, blob.get('classical_hrl_reference')


def save_ckpt(key, done, classical):
    tmp = CKPT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'key': key,
                   'points': [done[k] for k in sorted(done)],
                   'classical_hrl_reference': classical}, f, indent=2, default=float)
    os.replace(tmp, CKPT)          # atomic: a crash cannot truncate the checkpoint


def run_point(L, p, seeds, n_train, n_eval):
    """Train+eval Quantum-HRL at (vqc_layers=L, qaoa_depth=p) over seeds."""
    lat, en, miss, fb_num, fb_den = [], [], [], 0, 0
    for seed in seeds:
        agent = QuantumHRLAgent(state_dim=STATE_DIM, n_episodes=n_train,
                                batch_size=16, bo_budget=20, vqc_layers=L,
                                qaoa_depth=p, seed=seed, use_quantum=True,
                                verbose=False)
        t0 = time.time()
        agent.train(_make_env(seed))
        la, e, _f1, f2, _f3, _t = evaluate(agent, seed + 10000, n_eval)
        lat.append(float(la.mean())); en.append(float(e.mean()))
        miss.append(float(f2.mean()))
        if hasattr(agent, 'qaoa_fallback_stats'):
            f, ex, _ = agent.qaoa_fallback_stats()
            fb_num += f; fb_den += ex
        print(f"    [L={L} p={p}] seed={seed} lat={la.mean():.4f} "
              f"en={e.mean():.4f} miss={f2.mean()*100:.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
    return {'L': L, 'p': p,
            'n_params': int(vqc_params(L, int(np.ceil(np.log2(STATE_DIM))))
                            + qaoa_params(p)),
            'latency_mean': float(np.mean(lat)), 'latency_std': float(np.std(lat)),
            'energy_mean': float(np.mean(en)), 'miss_mean': float(np.mean(miss)),
            'seed_latency': lat, 'seed_energy': en, 'seed_miss': miss,
            'qaoa_fallback_rate': (fb_num / fb_den) if fb_den else None}


def run_classical(seeds, n_train, n_eval):
    lat = []
    for seed in seeds:
        agent = ClassicalHRLAgent(state_dim=STATE_DIM, hidden_dim=256,
                                  n_episodes=n_train, seed=seed)
        t0 = time.time()
        agent.train(_make_env(seed))
        la, _e, _f1, f2, _f3, _t = evaluate(agent, seed + 10000, n_eval)
        lat.append(float(la.mean()))
        print(f"    [Classical-HRL] seed={seed} lat={la.mean():.4f} "
              f"miss={f2.mean()*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return {'latency_mean': float(np.mean(lat)), 'latency_std': float(np.std(lat)),
            'seed_latency': lat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--train', type=int, default=30)
    ap.add_argument('--eval', type=int, default=12)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='ignore any existing checkpoint and recompute every point')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.train, args.eval = 2, 3, 2
    seeds = [42 + 137 * i for i in range(args.seeds)]

    print(f"Depth sweep: seeds={seeds} train={args.train} eval={args.eval} "
          f"mobility={MOBILITY} hash={C.config_hash()}", flush=True)

    t0 = time.time()
    key = _run_key(args, seeds)
    done, classical = load_ckpt(key, args.fresh)

    for (L, p) in grid():
        if (L, p) in done:
            print(f"\n== VQC L={L}, QAOA p={p} -- cached, skipping ==", flush=True)
            continue
        print(f"\n== VQC L={L}, QAOA p={p} ==", flush=True)
        done[(L, p)] = run_point(L, p, seeds, args.train, args.eval)
        save_ckpt(key, done, classical)

    if classical is None:
        print("\n== Classical-HRL reference ==", flush=True)
        classical = run_classical(seeds, args.train, args.eval)
        save_ckpt(key, done, classical)
    else:
        print("\n== Classical-HRL reference -- cached, skipping ==", flush=True)

    points = [done[k] for k in sorted(done)]

    out = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'command': 'python depth_sweep.py ' + ' '.join(sys.argv[1:]),
        'seeds': seeds, 'mobility': MOBILITY,
        'config_hash': C.config_hash(),
        'n_steps_per_episode': N_STEPS,
        'train_episodes': args.train, 'eval_episodes': args.eval,
        'grid': {'L_values': L_VALUES, 'p_fixed': P_FIXED,
                 'P_values': P_VALUES, 'L_fixed': L_FIXED},
        'points': points,
        'classical_hrl_reference': classical,
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'versions': _versions(),
        'wallclock_s': time.time() - t0,
    }
    path = os.path.join(os.path.dirname(__file__), 'depth_sweep.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {path}  (total {out['wallclock_s']:.0f}s)")


if __name__ == '__main__':
    main()
