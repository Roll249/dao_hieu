"""Build a FULL-24h LuST trajectory cache for the generalization experiment.

Reuses the route-splitting + per-window SUMO approach of the original
mobility_24h.py, but instead of keeping only speeds it extracts per-vehicle
ROI TRAJECTORIES (positions) from each hour's FCD and pools all 24 windows into
one trajectory cache (same npz format as lust_mobility.build_cache). Disk stays
low: each hour's FCD is parsed then deleted.

Run:   ../.venv_hrl/bin/python build_lust_24h.py
Subset: MOB_WINDOWS="2,8,13,17,22" ../.venv_hrl/bin/python build_lust_24h.py
Output: simulation/lust_roi_trajectories_24h.npz
Then compare with:  QHRL_LUST_CACHE=simulation/lust_roi_trajectories_24h.npz \
                    ../.venv_hrl/bin/python paper_results.py --tag _24h
"""
import os, sys, time, subprocess
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCEN = os.path.join(REPO, "Lust_dataset", "scenario")
WORK = os.path.join(REPO, "figure_baseline", "work_24h")
os.makedirs(WORK, exist_ok=True)

WINDOW = 3600
N_WINDOWS = 24
DAY_END = WINDOW * N_WINDOWS
OUT = os.path.join(HERE, "lust_roi_trajectories_24h.npz")

ROI_X0, ROI_X1 = C.LUST_ROI_X
ROI_Y0, ROI_Y1 = C.LUST_ROI_Y

SOURCES = [
    ("buslines", os.path.join(SCEN, "buslines.rou.xml")),
    ("dua0",     os.path.join(SCEN, "DUARoutes", "local.0.rou.xml")),
    ("dua1",     os.path.join(SCEN, "DUARoutes", "local.1.rou.xml")),
    ("dua2",     os.path.join(SCEN, "DUARoutes", "local.2.rou.xml")),
    ("transit",  os.path.join(SCEN, "transit.rou.xml")),
]
ROUTES_HEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n<routes '
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')


def win_path(w, label):
    return os.path.join(WORK, f"w{w:02d}__{label}.rou.xml")


def bin_routes(windows):
    """One streaming pass per source: bin <vehicle> into 24 window route files."""
    win_has = [dict() for _ in range(N_WINDOWS)]
    for label, src in SOURCES:
        if not os.path.exists(src):
            print(f"  [skip missing] {src}"); continue
        handles = {w: open(win_path(w, label), "w", encoding="utf-8") for w in windows}
        for h in handles.values():
            h.write(ROUTES_HEADER)
        local = Counter()
        ctx = ET.iterparse(src, events=("end",))
        for _, elem in ctx:
            if elem.tag != "vehicle":
                continue
            try:
                w = int(float(elem.get("depart")) // WINDOW)
            except (TypeError, ValueError):
                elem.clear(); continue
            if w in handles:
                handles[w].write("    " + ET.tostring(elem, encoding="unicode").strip() + "\n")
                local[w] += 1
            elem.clear()
        for w, h in handles.items():
            h.write("</routes>\n"); h.close()
            if local[w] > 0:
                win_has[w][label] = win_path(w, label)
            else:
                os.remove(win_path(w, label))
        print(f"  source {label:8s}: {sum(local.values()):7d} vehicles binned", flush=True)
    return win_has


def write_cfg(w, route_files):
    cfg = os.path.join(WORK, f"w{w:02d}.sumocfg")
    fcd = os.path.join(WORK, f"w{w:02d}.fcd.xml")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.join(SCEN, 'lust.net.xml')}"/>
        <route-files value="{','.join(route_files)}"/>
        <additional-files value="{os.path.join(SCEN, 'vtypes.add.xml')},{os.path.join(SCEN, 'busstops.add.xml')}"/>
    </input>
    <output><fcd-output value="{fcd}"/></output>
    <fcd_device><device.fcd.period value="60"/></fcd_device>
    <time><begin value="{w*WINDOW}"/><end value="{(w+1)*WINDOW}"/><step-length value="1"/></time>
    <processing><ignore-junction-blocker value="20"/><time-to-teleport value="600"/><max-depart-delay value="600"/></processing>
    <report><no-step-log value="true"/><no-warnings value="true"/><verbose value="false"/></report>
</configuration>
''')
    return cfg, fcd


def extract_roi_trajectories(fcd, w, ts, xs, ys, ss, offs):
    """Stream FCD; append in-ROI per-vehicle trajectories (>=MIN_POINTS, travel filter)."""
    veh = {}
    cur_t = None
    ctx = ET.iterparse(fcd, events=("start", "end"))
    for ev, el in ctx:
        if ev == "start" and el.tag == "timestep":
            cur_t = float(el.get("time"))
        elif ev == "end" and el.tag == "vehicle":
            x = float(el.get("x")); y = float(el.get("y"))
            if ROI_X0 <= x <= ROI_X1 and ROI_Y0 <= y <= ROI_Y1:
                veh.setdefault(el.get("id"), []).append((cur_t, x, y, float(el.get("speed"))))
            el.clear()
        elif ev == "end" and el.tag == "timestep":
            el.clear()
    kept = 0
    for rows in veh.values():
        if len(rows) < C.LUST_MIN_POINTS:
            continue
        rows.sort(key=lambda r: r[0])
        rx = np.array([(r[1] - ROI_X0) / 1000.0 for r in rows])
        ry = np.array([(r[2] - ROI_Y0) / 1000.0 for r in rows])
        if float(np.hypot(np.diff(rx), np.diff(ry)).sum() * 1000.0) < C.LUST_MIN_TRAVEL_M:
            continue
        for i, r in enumerate(rows):
            ts.append(r[0]); xs.append(rx[i]); ys.append(ry[i]); ss.append(r[3])
        offs.append(len(xs)); kept += 1
    return kept


def main():
    import sumolib
    sumo_bin = sumolib.checkBinary("sumo")
    only = os.environ.get("MOB_WINDOWS")
    windows = [int(x) for x in only.split(",")] if only else list(range(N_WINDOWS))

    print(f"=== 24h LuST trajectory build: windows {windows} ===", flush=True)
    t0 = time.time()
    win_has = bin_routes(windows)
    print(f"  route binning done in {time.time()-t0:.0f}s\n", flush=True)

    ts, xs, ys, ss, offs = [], [], [], [], [0]
    total_kept = 0
    for w in windows:
        if not win_has[w]:
            print(f"  window {w:02d}: no vehicles", flush=True); continue
        cfg, fcd = write_cfg(w, list(win_has[w].values()))
        tw = time.time()
        with open(cfg.replace('.sumocfg', '.sumo.log'), 'w') as lf:
            subprocess.run([sumo_bin, "-c", cfg], stdout=lf, stderr=lf, check=True)
        kept = extract_roi_trajectories(fcd, w, ts, xs, ys, ss, offs)
        total_kept += kept
        for f in [fcd] + list(win_has[w].values()):
            try: os.remove(f)
            except OSError: pass
        print(f"  window {w:02d} ({w:02d}:00): +{kept} ROI trajectories "
              f"(total {total_kept}) [{time.time()-tw:.0f}s]", flush=True)

    if total_kept == 0:
        raise RuntimeError("No 24h ROI trajectories extracted")
    np.savez_compressed(
        OUT, t=np.asarray(ts, np.float32), x=np.asarray(xs, np.float32),
        y=np.asarray(ys, np.float32), spd=np.asarray(ss, np.float32),
        off=np.asarray(offs, np.int64), area_km=np.float32(C.AREA_KM),
        n_traj=np.int64(total_kept))
    print(f"\nSaved {OUT}: {total_kept} trajectories, {len(xs)} points "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
