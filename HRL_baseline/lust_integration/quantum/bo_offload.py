"""Batch Bayesian optimization for the low-level offloading ratio."""
import numpy as np
from scipy.special import ndtr

BO_CALLS = 8
BO_INIT = np.array([0.0, 0.5, 1.0])
GRID = np.linspace(0.0, 1.0, 101)
LENGTH_SCALE = 0.2


def _objectives(ratios, t_offl, t_loc, e_offl, e_loc, g1, g2):
    r = np.asarray(ratios, dtype=float).reshape(1, -1)
    to = np.asarray(t_offl, dtype=float).reshape(-1, 1)
    tl = np.asarray(t_loc, dtype=float).reshape(-1, 1)
    eo = np.asarray(e_offl, dtype=float).reshape(-1, 1)
    el = np.asarray(e_loc, dtype=float).reshape(-1, 1)
    latency = np.maximum(r * to, (1.0 - r) * tl)
    energy = r * eo + (1.0 - r) * el
    return g1 * latency + g2 * energy


def _kernel(a, b):
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return np.exp(-0.5 * ((a - b) / LENGTH_SCALE) ** 2)


def optimal_ratios_bo_batch(t_offl, t_loc, e_offl, e_loc, g1, g2, seed=0):
    """Optimize all vehicles jointly with a GP/EI candidate schedule.

    Each vehicle has its own GP response values.  A shared next candidate is
    selected by mean expected improvement, which makes the large LuST sweep
    tractable while retaining a genuine sequential BO loop.
    """
    del seed  # deterministic candidate grid is required for reproducibility
    X = list(BO_INIT)
    Y = _objectives(X, t_offl, t_loc, e_offl, e_loc, g1, g2)
    while len(X) < BO_CALLS:
        K = _kernel(X, X) + 1e-8 * np.eye(len(X))
        Ks = _kernel(X, GRID)
        Kinv = np.linalg.inv(K)
        mu = Y @ Kinv @ Ks
        variance = np.maximum(1.0 - np.sum(Ks * (Kinv @ Ks), axis=0), 1e-12)
        sigma = np.sqrt(variance)[None, :]
        incumbent = Y.min(axis=1, keepdims=True)
        improvement = incumbent - mu
        z = improvement / sigma
        ei = improvement * ndtr(z) + sigma * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        ei[:, np.isin(GRID, np.asarray(X))] = -np.inf
        next_x = float(GRID[int(np.argmax(np.mean(ei, axis=0)))])
        X.append(next_x)
        Y = _objectives(X, t_offl, t_loc, e_offl, e_loc, g1, g2)
    best = np.argmin(Y, axis=1)
    return np.asarray(X)[best]


def optimal_ratio_bo(t_offl, t_loc, e_offl, e_loc, g1, g2, seed=0):
    return float(optimal_ratios_bo_batch([t_offl], [t_loc], [e_offl], [e_loc],
                                         g1, g2, seed=seed)[0])


def n_trainable_params():
    return 0
