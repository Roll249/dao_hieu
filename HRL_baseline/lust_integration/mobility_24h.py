"""
mobility_24h.py
===============
Sinh mobility trace cho TOÀN BỘ 24 giờ LuST theo cách TỐI ƯU RAM/disk:

  * Chia 24h thành 24 cửa sổ 1 giờ ([0, 86400]).
  * MỘT lượt iterparse mỗi route file -> phân loại <vehicle> vào cửa sổ theo `depart`
    (mỗi nguồn ghi 1 file/cửa sổ, giữ nguyên thứ tự để SUMO merge được). Chỉ giữ
    counter trong RAM, KHÔNG nạp cả file 40 MB.
  * Mỗi cửa sổ: chạy SUMO (FCD period=60) -> iterparse FCD trích tốc độ + tải đồng thời
    -> subsample -> XÓA FCD ngay (giữ disk thấp).
  * Lưu thống kê từng cửa sổ (để phân loại low/med/high) + pool tốc độ 24h gộp.

Chạy:  .venv_hrl/bin/python HRL_baseline/lust_integration/mobility_24h.py
Env override (smoke test):  MOB_WINDOWS="16,17,18"  -> chỉ chạy vài cửa sổ.
"""
import os
import sys
import json
import time
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCEN = os.path.join(REPO, "Lust_dataset", "scenario")
WORK = os.path.join(REPO, "figure_baseline", "work_24h")
RESULTS = os.path.join(REPO, "figure_baseline", "results")
os.makedirs(WORK, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

WINDOW = 3600
N_WINDOWS = 24
DAY_END = WINDOW * N_WINDOWS               # 86400
SPEED_FLOOR = 0.5                          # m/s, loại xe dừng (tránh sojourn=inf)
SUBSAMPLE_PER_WINDOW = 20000               # giới hạn mẫu tốc độ giữ lại mỗi cửa sổ

SOURCES = [
    ("buslines",   os.path.join(SCEN, "buslines.rou.xml")),
    ("dua0",       os.path.join(SCEN, "DUARoutes", "local.0.rou.xml")),
    ("dua1",       os.path.join(SCEN, "DUARoutes", "local.1.rou.xml")),
    ("dua2",       os.path.join(SCEN, "DUARoutes", "local.2.rou.xml")),
    ("transit",    os.path.join(SCEN, "transit.rou.xml")),
]

ROUTES_HEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n<routes '
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')


def win_path(w, label):
    return os.path.join(WORK, f"w{w:02d}__{label}.rou.xml")


def bin_routes():
    """Một lượt/nguồn: phân loại xe vào 24 cửa sổ. Trả về count[w], vtypes[w], có_file[w][label]."""
    win_count = [0] * N_WINDOWS
    win_vtypes = [Counter() for _ in range(N_WINDOWS)]
    win_has = [dict() for _ in range(N_WINDOWS)]
    skipped = 0
    for label, src in SOURCES:
        if not os.path.exists(src):
            print(f"  [BỎ QUA] {src}")
            continue
        handles = [open(win_path(w, label), "w", encoding="utf-8") for w in range(N_WINDOWS)]
        for h in handles:
            h.write(ROUTES_HEADER)
        local_count = [0] * N_WINDOWS
        ctx = ET.iterparse(src, events=("end",))
        for _, elem in ctx:
            if elem.tag != "vehicle":
                continue
            try:
                dep = float(elem.get("depart"))
            except (TypeError, ValueError):
                elem.clear(); continue
            w = int(dep // WINDOW)
            if 0 <= w < N_WINDOWS:
                handles[w].write("    " + ET.tostring(elem, encoding="unicode").strip() + "\n")
                local_count[w] += 1
                win_count[w] += 1
                win_vtypes[w][elem.get("type", "?")] += 1
            else:
                skipped += 1
            elem.clear()
        for w, h in enumerate(handles):
            h.write("</routes>\n"); h.close()
            if local_count[w] > 0:
                win_has[w][label] = win_path(w, label)
            else:
                os.remove(win_path(w, label))  # xóa file rỗng
        print(f"  nguồn {label:8s}: {sum(local_count):7d} xe phân vào cửa sổ")
    print(f"  (bỏ qua {skipped} xe có depart ngoài [0,{DAY_END}))")
    return win_count, win_vtypes, win_has


def write_cfg(w, route_files):
    cfg = os.path.join(WORK, f"w{w:02d}.sumocfg")
    routes = ",".join(route_files)
    fcd = f"w{w:02d}.fcd.xml"
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.join(SCEN, 'lust.net.xml')}"/>
        <route-files value="{routes}"/>
        <additional-files value="{os.path.join(SCEN, 'vtypes.add.xml')},{os.path.join(SCEN, 'busstops.add.xml')}"/>
    </input>
    <output><fcd-output value="{fcd}"/></output>
    <fcd_device><device.fcd.period value="60"/></fcd_device>
    <time><begin value="{w*WINDOW}"/><end value="{(w+1)*WINDOW}"/><step-length value="1"/></time>
    <processing>
        <ignore-junction-blocker value="20"/>
        <time-to-teleport value="600"/>
        <max-depart-delay value="600"/>
    </processing>
    <report><no-step-log value="true"/><no-warnings value="true"/><verbose value="false"/></report>
</configuration>
''')
    return cfg, os.path.join(WORK, fcd)


def run_sumo(cfg, sumo_bin):
    log = cfg.replace(".sumocfg", ".sumo.log")
    with open(log, "w") as lf:
        subprocess.run([sumo_bin, "-c", cfg], stdout=lf, stderr=lf, check=True)


def extract_fcd(fcd, rng):
    """Streaming iterparse FCD -> (speed_samples_subsampled, concurrent_series, n_distinct)."""
    speeds = []
    seen = set()
    times, counts = [], []
    cur_count = 0
    ctx = ET.iterparse(fcd, events=("start", "end"))
    for event, elem in ctx:
        if event == "start" and elem.tag == "timestep":
            cur_count = 0
        elif event == "end":
            if elem.tag == "vehicle":
                cur_count += 1
                seen.add(elem.get("id"))
                spd = float(elem.get("speed", "0"))
                if spd > SPEED_FLOOR:
                    speeds.append(spd)
                elem.clear()
            elif elem.tag == "timestep":
                times.append(float(elem.get("time"))); counts.append(cur_count)
                elem.clear()
    speeds = np.asarray(speeds, dtype=np.float64)
    if speeds.size > SUBSAMPLE_PER_WINDOW:
        idx = rng.choice(speeds.size, SUBSAMPLE_PER_WINDOW, replace=False)
        speeds = speeds[idx]
    return speeds, np.asarray(counts, dtype=int), len(seen)


def main():
    import sumolib
    sumo_bin = sumolib.checkBinary("sumo")
    rng = np.random.default_rng(2024)

    only = os.environ.get("MOB_WINDOWS")
    only_set = set(int(x) for x in only.split(",")) if only else None

    print("=== B1: phân loại route theo 24 cửa sổ (một lượt/nguồn, streaming) ===")
    t0 = time.time()
    win_count, win_vtypes, win_has = bin_routes()
    print(f"  xong B1 trong {time.time()-t0:.1f}s\n")

    print("=== B2: chạy SUMO từng cửa sổ + trích FCD (xóa FCD sau khi trích) ===")
    pool = []
    per_window = []
    for w in range(N_WINDOWS):
        if only_set is not None and w not in only_set:
            continue
        rec = {"window": w, "t0_s": w * WINDOW, "t1_s": (w + 1) * WINDOW,
               "clock": f"{w:02d}:00-{(w+1)%24:02d}:00",
               "n_vehicles_departing": int(win_count[w]),
               "vtypes": dict(win_vtypes[w])}
        if win_count[w] == 0:
            rec.update(dict(n_distinct_fcd=0, concurrent_mean=0.0, concurrent_max=0,
                            speed_mean=None, speed_median=None, n_speed_samples=0))
            per_window.append(rec)
            print(f"  cửa sổ {w:02d}: 0 xe -> bỏ qua SUMO")
            continue
        route_files = list(win_has[w].values())
        cfg, fcd = write_cfg(w, route_files)
        tw = time.time()
        run_sumo(cfg, sumo_bin)
        speeds, counts, ndist = extract_fcd(fcd, rng)
        try:
            os.remove(fcd)                       # XÓA FCD ngay (giữ disk thấp)
        except OSError:
            pass
        for f in route_files:                    # xóa route files của cửa sổ này
            try: os.remove(f)
            except OSError: pass
        cmean = float(counts.mean()) if counts.size else 0.0
        cmax = int(counts.max()) if counts.size else 0
        rec.update(dict(
            n_distinct_fcd=int(ndist),
            concurrent_mean=cmean, concurrent_max=cmax,
            speed_mean=(float(speeds.mean()) if speeds.size else None),
            speed_median=(float(np.median(speeds)) if speeds.size else None),
            n_speed_samples=int(speeds.size),
            sim_wall_s=round(time.time() - tw, 1)))
        per_window.append(rec)
        pool.append(speeds)
        print(f"  cửa sổ {w:02d} ({rec['clock']}): {win_count[w]:6d} xe | "
              f"đồng thời mean={cmean:6.0f} max={cmax:6d} | "
              f"spd mean={rec['speed_mean'] or 0:.2f} | {rec['sim_wall_s']}s", flush=True)

    pool = np.concatenate(pool) if pool else np.array([])
    np.save(os.path.join(RESULTS, "lust_speed_pool_full.npy"), pool)

    # phân loại low/med/high theo concurrent_mean (tertile, chỉ cửa sổ có xe)
    active = [r for r in per_window if r["n_vehicles_departing"] > 0 and r.get("concurrent_mean")]
    if active:
        cms = sorted(r["concurrent_mean"] for r in active)
        q1 = cms[len(cms)//3]; q2 = cms[2*len(cms)//3]
        for r in per_window:
            cm = r.get("concurrent_mean") or 0
            if r["n_vehicles_departing"] == 0:
                r["traffic_class"] = "none"
            elif cm <= q1:
                r["traffic_class"] = "low"
            elif cm <= q2:
                r["traffic_class"] = "medium"
            else:
                r["traffic_class"] = "high"

    total_veh = sum(r["n_vehicles_departing"] for r in per_window)
    all_cmean = [r["concurrent_mean"] for r in per_window if r.get("concurrent_mean")]
    all_cmax = [r["concurrent_max"] for r in per_window if r.get("concurrent_max")]
    summary = dict(
        n_windows=N_WINDOWS, window_seconds=WINDOW, sim_span_s=[0, DAY_END],
        total_vehicles=int(total_veh),
        concurrent_mean_over_day=(float(np.mean(all_cmean)) if all_cmean else 0.0),
        concurrent_max_over_day=(int(max(all_cmax)) if all_cmax else 0),
        speed_pool_size=int(pool.size),
        speed_mean_mps=(float(pool.mean()) if pool.size else None),
        speed_median_mps=(float(np.median(pool)) if pool.size else None),
        speed_p90_mps=(float(np.percentile(pool, 90)) if pool.size else None),
        speed_max_mps=(float(pool.max()) if pool.size else None),
        fcd_sampling="--device.fcd.period 60 (mỗi 60s); FCD xóa ngay sau khi trích",
        route_set="DUA local.0/1/2 + transit + buslines; net actuated TLs",
    )
    with open(os.path.join(RESULTS, "per_window_mobility.json"), "w", encoding="utf-8") as f:
        json.dump(per_window, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS, "mobility_full_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== TÓM TẮT MOBILITY 24h ===")
    print(f"  Tổng xe: {total_veh} | đồng thời mean(ngày)={summary['concurrent_mean_over_day']:.0f} "
          f"max={summary['concurrent_max_over_day']}")
    print(f"  Pool tốc độ: {pool.size} mẫu, mean={summary['speed_mean_mps']:.2f} m/s")
    print(f"  Lưu: results/per_window_mobility.json, mobility_full_summary.json, lust_speed_pool_full.npy")


if __name__ == "__main__":
    main()
