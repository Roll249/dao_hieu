"""
prepare_window.py
=================
Tiền xử lý LuST: LỌC TRỰC TIẾP các phương tiện có thời điểm khởi hành
`depart` nằm trong cửa sổ 17:00–18:00 (61200s–64800s) từ các route file của LuST,
ghi ra các route file rút gọn + một file cấu hình SUMO chỉ chạy cửa sổ này.

Mục tiêu: KHÔNG parse/mô phỏng toàn bộ 24 giờ. Dùng iterparse (streaming) để
tiết kiệm RAM — chỉ giữ từng phần tử <vehicle> rồi giải phóng ngay.

Chạy:
    python prepare_window.py
"""
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

# --- Khoảng thời gian mục tiêu: 17:00 - 18:00 ---
T0 = 61200.0
T1 = 64800.0

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCEN = os.path.join(REPO, "Lust_dataset", "scenario")
WORK = os.path.join(REPO, "figure_baseline", "work")
os.makedirs(WORK, exist_ok=True)

# Bộ route DUA (shortest-path + rerouting) — actuated TLs là mặc định của lust.net.xml
INPUT_ROUTES = [
    ("buslines.rou.xml",        os.path.join(SCEN, "buslines.rou.xml")),
    ("DUARoutes/local.0.rou.xml", os.path.join(SCEN, "DUARoutes", "local.0.rou.xml")),
    ("DUARoutes/local.1.rou.xml", os.path.join(SCEN, "DUARoutes", "local.1.rou.xml")),
    ("DUARoutes/local.2.rou.xml", os.path.join(SCEN, "DUARoutes", "local.2.rou.xml")),
    ("transit.rou.xml",         os.path.join(SCEN, "transit.rou.xml")),
]


def filter_one(src_path, out_path, t0, t1):
    """Stream-parse 1 route file, giữ lại các <vehicle> có depart in [t0, t1]."""
    kept = 0
    vtypes = Counter()
    # Ghi thủ công để giữ định dạng nhẹ
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fout.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                   'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        context = ET.iterparse(src_path, events=("end",))
        for event, elem in context:
            if elem.tag == "vehicle":
                dep = elem.get("depart")
                try:
                    dval = float(dep)
                except (TypeError, ValueError):
                    elem.clear()
                    continue
                if t0 <= dval <= t1:
                    fout.write("    " + ET.tostring(elem, encoding="unicode").strip() + "\n")
                    kept += 1
                    vtypes[elem.get("type", "unknown")] += 1
                elem.clear()
        fout.write("</routes>\n")
    return kept, vtypes


def main():
    print(f"Lọc phương tiện có depart in [{T0:.0f}, {T1:.0f}] (17:00-18:00)\n")
    total = 0
    all_vtypes = Counter()
    out_route_files = []
    per_file = {}
    for label, src in INPUT_ROUTES:
        if not os.path.exists(src):
            print(f"  [BỎ QUA] không thấy {src}")
            continue
        base = label.replace("/", "_").replace(".rou.xml", "")
        out_path = os.path.join(WORK, f"window_{base}.rou.xml")
        kept, vtypes = filter_one(src, out_path, T0, T1)
        total += kept
        all_vtypes.update(vtypes)
        per_file[label] = kept
        out_route_files.append(os.path.basename(out_path))
        print(f"  {label:32s} -> {kept:7d} xe  (ghi {os.path.basename(out_path)})")

    print(f"\nTổng số phương tiện khởi hành trong cửa sổ: {total}")
    print("Phân bố theo vType:")
    for k, v in sorted(all_vtypes.items(), key=lambda x: -x[1]):
        print(f"    {k:14s}: {v}")

    # --- Ghi file cấu hình SUMO chỉ cho cửa sổ này ---
    # additional: vtypes (định nghĩa loại xe) + busstops (cho bus)
    route_csv = ",".join(out_route_files)
    sumocfg = os.path.join(WORK, "window.sumocfg")
    fcd_out = "window.fcd.xml"
    with open(sumocfg, "w", encoding="utf-8") as f:
        f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- LuST 17:00-18:00 window (chỉ cửa sổ này, KHÔNG chạy 24h) -->
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.join(SCEN, 'lust.net.xml')}"/>
        <route-files value="{route_csv}"/>
        <additional-files value="{os.path.join(SCEN, 'vtypes.add.xml')},{os.path.join(SCEN, 'busstops.add.xml')}"/>
    </input>
    <output>
        <fcd-output value="{fcd_out}"/>
    </output>
    <fcd_device>
        <device.fcd.period value="60"/>
    </fcd_device>
    <time>
        <begin value="{int(T0)}"/>
        <end value="{int(T1)}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-junction-blocker value="20"/>
        <time-to-teleport value="600"/>
        <max-depart-delay value="600"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <verbose value="false"/>
    </report>
</configuration>
''')
    print(f"\nĐã ghi cấu hình SUMO: {sumocfg}")

    # Lưu thống kê tiền xử lý ra results/
    import json
    res = {
        "window_seconds": [T0, T1],
        "window_clock": "17:00-18:00",
        "total_vehicles_departing_in_window": total,
        "per_file_counts": per_file,
        "vtype_histogram": dict(all_vtypes),
        "route_set": "DUA (shortest-path) + transit + buslines, actuated TLs",
        "filtered_route_files": out_route_files,
        "sumocfg": os.path.basename(sumocfg),
    }
    res_dir = os.path.join(REPO, "figure_baseline", "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "lust_preprocess.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Đã ghi thống kê: figure_baseline/results/lust_preprocess.json")


if __name__ == "__main__":
    main()
