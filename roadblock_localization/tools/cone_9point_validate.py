#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
9组锥桶定位验证脚本

比较：
A) 单点IPM + 0.10m地面射线修正
B) 单点IPM给方向 + bbox高度给距离
C) A/B 距离融合，并自动搜索最佳 alpha

前提：
- /hobot_dnn_detection 正常发布 ai_msgs/msg/PerceptionTargets
- /roadblock_ground_array 正常发布 roadblock_interfaces/msg/RoadblockArray
- 标定/验证时画面中只保留 1 个 roadblock
- X/Y 均指“小车旋转中心(base_link) -> 锥桶底座几何中心”
"""

import argparse
import csv
import json
import math
import statistics
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from roadblock_interfaces.msg import RoadblockArray


DEFAULT_STAGES = [
    ("P01", 0.70,  0.00, "约0°"),
    ("P02", 0.70, +0.20, "约22.5°"),
    ("P03", 0.70, -0.20, "约45°"),
    ("P04", 0.85,  0.00, "约45°"),
    ("P05", 0.85, +0.20, "约0°"),
    ("P06", 0.85, -0.20, "约22.5°"),
    ("P07", 1.00,  0.00, "约22.5°"),
    ("P08", 1.00, +0.20, "约45°"),
    ("P09", 1.00, -0.20, "约0°"),
]


class Sampler(Node):
    def __init__(self, detection_topic, ground_topic, min_conf):
        super().__init__("cone_9point_validation_sampler")
        self.min_conf = float(min_conf)
        self.lock = threading.Lock()
        self.latest_det = None
        self.latest_ground = None
        self.latest_det_t = 0.0
        self.latest_ground_t = 0.0

        self.create_subscription(
            PerceptionTargets, detection_topic, self.det_cb, 10
        )
        self.create_subscription(
            RoadblockArray, ground_topic, self.ground_cb, 10
        )

    def det_cb(self, msg):
        candidates = []
        for target in msg.targets:
            if target.type != "roadblock":
                continue
            for roi in target.rois:
                conf = float(roi.confidence)
                if conf < self.min_conf:
                    continue
                xmin = float(roi.rect.x_offset)
                ymin = float(roi.rect.y_offset)
                xmax = xmin + float(roi.rect.width)
                ymax = ymin + float(roi.rect.height)
                candidates.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "bbox_width": xmax - xmin,
                    "bbox_height": ymax - ymin,
                    "confidence": conf,
                })

        sample = candidates[0] if len(candidates) == 1 else None
        with self.lock:
            self.latest_det = sample
            self.latest_det_t = time.monotonic()

    def ground_cb(self, msg):
        sample = None
        if len(msg.obstacles) == 1:
            o = msg.obstacles[0]
            sample = {
                "ipm_x": float(o.x),
                "ipm_y": float(o.y),
            }

        with self.lock:
            self.latest_ground = sample
            self.latest_ground_t = time.monotonic()

    def snapshot(self, max_age=0.30):
        now = time.monotonic()
        with self.lock:
            d = None if self.latest_det is None else dict(self.latest_det)
            g = None if self.latest_ground is None else dict(self.latest_ground)
            dt = self.latest_det_t
            gt = self.latest_ground_t

        if d is None or g is None:
            return None
        if now - dt > max_age or now - gt > max_age:
            return None

        out = {}
        out.update(d)
        out.update(g)
        return out


def median(values):
    return float(statistics.median(values))


def summarize(samples):
    keys = [
        "xmin", "ymin", "xmax", "ymax",
        "bbox_width", "bbox_height", "confidence",
        "ipm_x", "ipm_y"
    ]
    return {k: median([x[k] for x in samples]) for k in keys}


def collect(node, seconds, hz):
    samples = []
    period = 1.0 / hz
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        s = node.snapshot()
        if s is not None:
            samples.append(s)
        time.sleep(period)
    return samples


def unit_direction(qx, qy, px, py):
    dx = px - qx
    dy = py - qy
    n = math.hypot(dx, dy)
    if n < 1e-9:
        raise ValueError("IPM点与相机地面投影点重合，无法计算方向")
    return dx / n, dy / n, n


def point_error(x, y, tx, ty):
    return math.hypot(x - tx, y - ty)


def metric_summary(errors):
    return {
        "mean_m": sum(errors) / len(errors),
        "median_m": median(errors),
        "rmse_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "max_m": max(errors),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection-topic", default="/hobot_dnn_detection")
    ap.add_argument("--ground-topic", default="/roadblock_ground_array")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--min-confidence", type=float, default=0.50)

    # 已确认参数
    ap.add_argument("--camera-ground-x", type=float, default=0.05)
    ap.add_argument("--camera-ground-y", type=float, default=0.00)

    # 方案A
    ap.add_argument("--ipm-offset", type=float, default=0.10)

    # 方案B
    ap.add_argument("--height-a", type=float, default=99.488)
    ap.add_argument("--height-b", type=float, default=0.22381)

    ap.add_argument("--alpha-step", type=float, default=0.05)
    ap.add_argument(
        "--out-dir",
        default="/root/intelligent_car_ws/test_logs/cone_9point_validation"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"cone_9point_results_{stamp}.csv"
    json_path = out_dir / f"cone_9point_summary_{stamp}.json"

    rclpy.init()
    node = Sampler(
        args.detection_topic,
        args.ground_topic,
        args.min_confidence
    )
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    time.sleep(1.0)

    print()
    print("============================================================")
    print(" 9组锥桶定位验证：A(IPM+10cm) / B(IPM方向+高度) / C(融合)")
    print("============================================================")
    print("要求：画面里只保留 1 个 roadblock。")
    print("X/Y 指小车旋转中心 -> 锥桶底座几何中心。")
    print("角度只需大概，不用量角器。")
    print()

    rows = []

    try:
        for idx, (name, tx, ty, yaw_note) in enumerate(DEFAULT_STAGES, 1):
            print("------------------------------------------------------------")
            print(
                f"[{idx}/9] {name}: 请摆锥桶中心到 "
                f"X={tx:.2f} m, Y={ty:+.2f} m，朝向 {yaw_note}"
            )
            input("摆好并静止后按 Enter 开始采 3 秒...")

            samples = collect(node, args.seconds, args.sample_hz)
            if len(samples) < 5:
                print(f"ERROR: 本组有效样本只有 {len(samples)} 帧。")
                print("检查 YOLO、/roadblock_ground_array，以及画面中是否只有一个锥桶。")
                return 2

            s = summarize(samples)

            qx = args.camera_ground_x
            qy = args.camera_ground_y
            ux, uy, ipm_ground_dist = unit_direction(
                qx, qy, s["ipm_x"], s["ipm_y"]
            )

            # A: IPM点沿射线补0.10m
            d_a = ipm_ground_dist + args.ipm_offset
            ax = qx + d_a * ux
            ay = qy + d_a * uy
            err_a = point_error(ax, ay, tx, ty)

            # B: 高度距离
            h = s["bbox_height"]
            d_b = args.height_a / h + args.height_b
            bx = qx + d_b * ux
            by = qy + d_b * uy
            err_b = point_error(bx, by, tx, ty)

            true_d = math.hypot(tx - qx, ty - qy)

            row = {
                "name": name,
                "true_x": tx,
                "true_y": ty,
                "yaw_note": yaw_note,
                "sample_count": len(samples),
                **s,
                "dir_x": ux,
                "dir_y": uy,
                "true_ground_distance": true_d,
                "ipm_ground_distance": ipm_ground_dist,
                "A_distance": d_a,
                "A_x": ax,
                "A_y": ay,
                "A_error": err_a,
                "B_distance": d_b,
                "B_x": bx,
                "B_y": by,
                "B_error": err_b,
            }
            rows.append(row)

            print(
                f"完成 {len(samples)} 帧 | "
                f"h={h:.1f}px | IPM=({s['ipm_x']:.4f},{s['ipm_y']:.4f})"
            )
            print(
                f"A误差={err_a*100:.2f}cm | "
                f"B误差={err_b*100:.2f}cm"
            )

        # 搜索最佳 alpha
        alphas = []
        a = 0.0
        while a <= 1.0 + 1e-9:
            alphas.append(round(a, 10))
            a += args.alpha_step

        best = None
        all_alpha_results = []

        for alpha in alphas:
            errs = []
            for row in rows:
                # C = alpha*A_distance + (1-alpha)*B_distance
                d_c = alpha * row["A_distance"] + (1.0 - alpha) * row["B_distance"]
                cx = args.camera_ground_x + d_c * row["dir_x"]
                cy = args.camera_ground_y + d_c * row["dir_y"]
                e = point_error(cx, cy, row["true_x"], row["true_y"])
                errs.append(e)

            m = metric_summary(errs)
            item = {"alpha": alpha, **m}
            all_alpha_results.append(item)

            # 优先最小 RMSE；相同则更小 max
            key = (m["rmse_m"], m["max_m"])
            if best is None or key < best["key"]:
                best = {"key": key, "alpha": alpha, "metrics": m}

        best_alpha = best["alpha"]

        # 回填最佳融合每组结果
        for row in rows:
            d_c = best_alpha * row["A_distance"] + (1.0 - best_alpha) * row["B_distance"]
            cx = args.camera_ground_x + d_c * row["dir_x"]
            cy = args.camera_ground_y + d_c * row["dir_y"]
            row["C_alpha"] = best_alpha
            row["C_distance"] = d_c
            row["C_x"] = cx
            row["C_y"] = cy
            row["C_error"] = point_error(
                cx, cy, row["true_x"], row["true_y"]
            )

        A_metrics = metric_summary([r["A_error"] for r in rows])
        B_metrics = metric_summary([r["B_error"] for r in rows])
        C_metrics = metric_summary([r["C_error"] for r in rows])

        # 保存 CSV
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        summary = {
            "params": {
                "camera_ground_x_m": args.camera_ground_x,
                "camera_ground_y_m": args.camera_ground_y,
                "ipm_offset_m": args.ipm_offset,
                "height_model_a": args.height_a,
                "height_model_b": args.height_b,
                "alpha_step": args.alpha_step,
            },
            "A_ipm_plus_offset": A_metrics,
            "B_ipm_direction_plus_height": B_metrics,
            "C_fusion_best": {
                "alpha": best_alpha,
                **C_metrics,
            },
            "alpha_scan": all_alpha_results,
            "rows": rows,
        }
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print()
        print("============================================================")
        print(" 最终统计")
        print("============================================================")
        print("A = IPM + 10cm")
        print(
            f"  mean={A_metrics['mean_m']*100:.2f}cm | "
            f"median={A_metrics['median_m']*100:.2f}cm | "
            f"RMSE={A_metrics['rmse_m']*100:.2f}cm | "
            f"max={A_metrics['max_m']*100:.2f}cm"
        )
        print()
        print("B = IPM方向 + 高度距离")
        print(
            f"  mean={B_metrics['mean_m']*100:.2f}cm | "
            f"median={B_metrics['median_m']*100:.2f}cm | "
            f"RMSE={B_metrics['rmse_m']*100:.2f}cm | "
            f"max={B_metrics['max_m']*100:.2f}cm"
        )
        print()
        print("C = 融合")
        print(f"  最佳 alpha = {best_alpha:.2f}")
        print("  distance_C = alpha * distance_A + (1-alpha) * distance_B")
        print(
            f"  mean={C_metrics['mean_m']*100:.2f}cm | "
            f"median={C_metrics['median_m']*100:.2f}cm | "
            f"RMSE={C_metrics['rmse_m']*100:.2f}cm | "
            f"max={C_metrics['max_m']*100:.2f}cm"
        )

        # 自动推荐
        ranked = [
            ("A(IPM+10cm)", A_metrics),
            ("B(IPM方向+高度)", B_metrics),
            (f"C(融合 alpha={best_alpha:.2f})", C_metrics),
        ]
        ranked.sort(key=lambda x: (x[1]["rmse_m"], x[1]["max_m"]))
        print()
        print(f"推荐（按 RMSE 优先、Max 次优）：{ranked[0][0]}")
        print()
        print(f"CSV : {csv_path}")
        print(f"JSON: {json_path}")
        print("============================================================")
        return 0

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
