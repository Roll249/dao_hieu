"""
extract_speeds.py
=================
Trích xuất từ FCD (window.fcd.xml do SUMO sinh cho 17:00-18:00):
  1. Phân phối tốc độ THẬT của phương tiện (pool tốc độ tức thời, lọc > 0.5 m/s
     để tránh chia 0 trong tính sojourn-time của mô hình HRL).
  2. Thống kê tốc độ theo từng phương tiện (để báo cáo).
  3. Chuỗi số phương tiện ĐỒNG THỜI theo thời gian (concurrent load).

Dùng iterparse để tiết kiệm RAM (FCD có thể lớn).
Kết quả lưu vào figure_baseline/results/:
  - lust_speed_pool.npy   : mảng tốc độ tức thời (m/s) để driver lấy mẫu VN_spd
  - lust_mobility.json    : thống kê tốc độ + chuỗi tải + metadata
"""
import os
import json
import numpy as np
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(REPO, "figure_baseline", "work")
RESULTS = os.path.join(REPO, "figure_baseline", "results")
FCD = os.path.join(WORK, "window.fcd.xml")
SPEED_FLOOR = 0.5  # m/s, loại trạng thái đứng yên để tránh sojourn = inf


def main():
    if not os.path.exists(FCD):
        raise SystemExit(f"Không thấy FCD: {FCD}. Hãy chạy SUMO (prepare_window + window.sumocfg) trước.")

    speed_pool = []                 # tốc độ tức thời (moving)
    per_veh_sum = {}                # id -> tổng speed
    per_veh_cnt = {}                # id -> số mẫu
    times = []                      # mốc thời gian snapshot
    concurrent = []                 # số xe mỗi snapshot

    context = ET.iterparse(FCD, events=("start", "end"))
    cur_time = None
    cur_count = 0
    for event, elem in context:
        if event == "start" and elem.tag == "timestep":
            cur_time = float(elem.get("time"))
            cur_count = 0
        elif event == "end":
            if elem.tag == "vehicle":
                cur_count += 1
                spd = float(elem.get("speed", "0"))
                vid = elem.get("id")
                per_veh_sum[vid] = per_veh_sum.get(vid, 0.0) + spd
                per_veh_cnt[vid] = per_veh_cnt.get(vid, 0) + 1
                if spd > SPEED_FLOOR:
                    speed_pool.append(spd)
                elem.clear()
            elif elem.tag == "timestep":
                times.append(cur_time)
                concurrent.append(cur_count)
                elem.clear()

    speed_pool = np.asarray(speed_pool, dtype=np.float64)
    per_veh_mean = np.asarray(
        [per_veh_sum[v] / per_veh_cnt[v] for v in per_veh_sum], dtype=np.float64
    )
    per_veh_mean_moving = per_veh_mean[per_veh_mean > SPEED_FLOOR]

    concurrent = np.asarray(concurrent, dtype=int)
    times = np.asarray(times, dtype=float)

    os.makedirs(RESULTS, exist_ok=True)
    np.save(os.path.join(RESULTS, "lust_speed_pool.npy"), speed_pool)

    stats = {
        "fcd_file": os.path.basename(FCD),
        "n_snapshots": int(len(times)),
        "snapshot_period_s": 60,
        "n_distinct_vehicles_seen": int(len(per_veh_mean)),
        "n_speed_samples_moving": int(speed_pool.size),
        "speed_pool_mps": {
            "min": float(speed_pool.min()) if speed_pool.size else None,
            "mean": float(speed_pool.mean()) if speed_pool.size else None,
            "median": float(np.median(speed_pool)) if speed_pool.size else None,
            "p90": float(np.percentile(speed_pool, 90)) if speed_pool.size else None,
            "max": float(speed_pool.max()) if speed_pool.size else None,
            "std": float(speed_pool.std()) if speed_pool.size else None,
        },
        "per_vehicle_mean_speed_mps": {
            "mean": float(per_veh_mean_moving.mean()) if per_veh_mean_moving.size else None,
            "median": float(np.median(per_veh_mean_moving)) if per_veh_mean_moving.size else None,
        },
        "concurrent_vehicles": {
            "peak": int(concurrent.max()) if concurrent.size else 0,
            "mean": float(concurrent.mean()) if concurrent.size else 0.0,
            "at_end_of_warmup": int(concurrent[len(concurrent)//6]) if concurrent.size else 0,
            "series_time_s": times.tolist(),
            "series_count": concurrent.tolist(),
        },
        "note": (
            "Tốc độ lấy từ FCD SUMO (sampling 60s) trong cửa sổ 17:00-18:00. "
            "speed_pool gồm tốc độ tức thời > 0.5 m/s (loại xe đang dừng đèn). "
            "Mạng bắt đầu rỗng tại 61200s nên vài phút đầu là warmup."
        ),
    }
    with open(os.path.join(RESULTS, "lust_mobility.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Đã trích tốc độ LuST:")
    print(f"  - Snapshots: {stats['n_snapshots']} (mỗi 60s)")
    print(f"  - Phương tiện riêng biệt thấy trong cửa sổ: {stats['n_distinct_vehicles_seen']}")
    print(f"  - Mẫu tốc độ (moving): {stats['n_speed_samples_moving']}")
    sp = stats["speed_pool_mps"]
    print(f"  - Tốc độ (m/s): mean={sp['mean']:.2f}, median={sp['median']:.2f}, "
          f"p90={sp['p90']:.2f}, max={sp['max']:.2f}")
    cc = stats["concurrent_vehicles"]
    print(f"  - Xe đồng thời: peak={cc['peak']}, mean={cc['mean']:.1f}")
    print(f"  - Lưu: results/lust_speed_pool.npy, results/lust_mobility.json")


if __name__ == "__main__":
    main()
