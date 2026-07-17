"""LuST trajectory-level mobility provider.

Parses a SUMO FCD export (real LuST 17:00-18:00 window) into per-vehicle
trajectories restricted to a dense urban region of interest, then replays a
sampled vehicle's *recorded* path across the decision slots of an episode.

Unlike the earlier "speed-pool only" injection (urgent_need_to_fixed.md #21),
this drives the ego vehicle's POSITION (hence node distances, channel gains and
sojourn time) from real SUMO FCD coordinates, not just a speed distribution.

Cache layout (npz): flat concatenated trajectories + offsets, so loading is
cheap and every env instance shares the same reproducible trajectory set.
"""
from __future__ import annotations
import os
import numpy as np
import xml.etree.ElementTree as ET

import config as C

# Repo root = parent of this file's directory (simulation/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def build_cache(fcd_path: str | None = None, cache_path: str | None = None,
                verbose: bool = True) -> str:
    """Parse the FCD once, extract in-ROI per-vehicle trajectories, cache them.

    Returns the cache path. Keeps only vehicles seen >= LUST_MIN_POINTS times
    inside the ROI so every retained trajectory is long enough to replay.
    """
    fcd_path = _abspath(fcd_path or C.LUST_FCD)
    cache_path = _abspath(cache_path or C.LUST_CACHE)
    x0, x1 = C.LUST_ROI_X
    y0, y1 = C.LUST_ROI_Y

    veh: dict[str, list] = {}
    cur_t = None
    ctx = ET.iterparse(fcd_path, events=("start", "end"))
    for ev, el in ctx:
        if ev == "start" and el.tag == "timestep":
            cur_t = float(el.get("time"))
        elif ev == "end" and el.tag == "vehicle":
            x = float(el.get("x")); y = float(el.get("y"))
            if x0 <= x <= x1 and y0 <= y <= y1:
                veh.setdefault(el.get("id"), []).append(
                    (cur_t, x, y, float(el.get("speed"))))
            el.clear()
        elif ev == "end" and el.tag == "timestep":
            el.clear()

    # Filter + convert to local km frame; sort each trajectory by time.
    ts, xs, ys, ss, offs = [], [], [], [], [0]
    kept = 0
    mean_speeds, travels = [], []
    for vid, rows in veh.items():
        if len(rows) < C.LUST_MIN_POINTS:
            continue
        rows.sort(key=lambda r: r[0])
        rx = np.array([(r[1] - x0) / 1000.0 for r in rows])   # km
        ry = np.array([(r[2] - y0) / 1000.0 for r in rows])
        rs = np.array([r[3] for r in rows])                    # m/s
        travel_m = float(np.hypot(np.diff(rx), np.diff(ry)).sum() * 1000.0)
        if travel_m < C.LUST_MIN_TRAVEL_M:      # drop queued/parked vehicles
            continue
        for i in range(len(rows)):
            ts.append(rows[i][0]); xs.append(rx[i]); ys.append(ry[i]); ss.append(rs[i])
        offs.append(len(xs))
        kept += 1
        mean_speeds.append(float(rs.mean())); travels.append(travel_m)

    if kept == 0:
        raise RuntimeError("No LuST trajectories met LUST_MIN_POINTS/TRAVEL in ROI")

    if verbose:
        ms = np.array(mean_speeds); tv = np.array(travels)
        print(f"[lust_mobility] kept mean-speed m/s: "
              f"p10 {np.percentile(ms,10):.2f} med {np.median(ms):.2f} "
              f"p90 {np.percentile(ms,90):.2f}; travel m: "
              f"med {np.median(tv):.0f} p90 {np.percentile(tv,90):.0f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        t=np.asarray(ts, np.float32), x=np.asarray(xs, np.float32),
        y=np.asarray(ys, np.float32), spd=np.asarray(ss, np.float32),
        off=np.asarray(offs, np.int64), area_km=np.float32(C.AREA_KM),
        n_traj=np.int64(kept),
    )
    if verbose:
        print(f"[lust_mobility] cached {kept} trajectories "
              f"({len(xs)} points) -> {cache_path}")
    return cache_path


class LuSTMobilityProvider:
    """Loads the trajectory cache and replays sampled vehicle paths."""

    def __init__(self, cache_path: str | None = None, auto_build: bool = True):
        cache_path = _abspath(cache_path or C.LUST_CACHE)
        if not os.path.exists(cache_path) and auto_build:
            build_cache(cache_path=cache_path)
        d = np.load(cache_path)
        self.t = d["t"]; self.x = d["x"]; self.y = d["y"]
        self.spd = d["spd"]; self.off = d["off"]
        self.area_km = float(d["area_km"])
        self.n = int(self.off.shape[0] - 1)

    def n_traj(self) -> int:
        return self.n

    def sample_episode(self, rng: np.random.RandomState, n_steps: int):
        """Return (X, Y, SPEED) arrays of length n_steps+1 for one episode.

        A random recorded vehicle trajectory is resampled onto n_steps+1 evenly
        spaced fractional indices (linear interpolation between FCD snapshots),
        so each decision slot advances the ego along its real LuST path.

        SPEED is the *effective* ground speed = LuST displacement over the
        slot's real elapsed time (span / n_steps), which is more faithful than
        the sparsely-sampled instantaneous FCD field (a vehicle stopped at the
        60 s sample instant may still have traversed a block since the last
        sample). X, Y are ROI-local km; SPEED is m/s.
        """
        vi = rng.randint(self.n)
        a, b = int(self.off[vi]), int(self.off[vi + 1])
        tx = self.x[a:b].astype(float)
        ty = self.y[a:b].astype(float)
        tt = self.t[a:b].astype(float)
        S = len(tx)
        u = np.linspace(0.0, S - 1, n_steps + 1)
        i0 = np.clip(np.floor(u).astype(int), 0, S - 2)
        fr = u - i0
        X = tx[i0] * (1 - fr) + tx[i0 + 1] * fr
        Y = ty[i0] * (1 - fr) + ty[i0 + 1] * fr
        span_s = max(float(tt[-1] - tt[0]), 1.0)
        dt_real = span_s / n_steps
        disp_m = np.hypot(np.diff(X), np.diff(Y)) * 1000.0
        eff = disp_m / dt_real                      # m/s per slot
        SP = np.concatenate([eff[:1], eff])          # length n_steps+1
        SP = np.clip(SP, 0.1, None)                  # floor for sojourn model
        return X, Y, SP


if __name__ == "__main__":
    import sys
    path = build_cache(verbose=True)
    p = LuSTMobilityProvider(path)
    print(f"[lust_mobility] {p.n_traj()} trajectories, area_km={p.area_km}")
    rng = np.random.RandomState(0)
    X, Y, SP = p.sample_episode(rng, 40)
    dxy = np.hypot(np.diff(X), np.diff(Y)) * 1000.0  # m per slot
    print(f"  sample traj: x[{X.min():.2f},{X.max():.2f}]km "
          f"y[{Y.min():.2f},{Y.max():.2f}]km speed mean {SP.mean():.2f} m/s "
          f"move/slot mean {dxy.mean():.1f} m")
