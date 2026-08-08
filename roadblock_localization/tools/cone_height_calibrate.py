#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


class ConeCalibNode(Node):
    def __init__(self, detection_topic, ground_topic, min_confidence):
        super().__init__('cone_height_calibration_sampler')
        self.min_confidence = float(min_confidence)
        self.lock = threading.Lock()
        self.latest_detection = None
        self.latest_ground = None
        self.latest_detection_time = 0.0
        self.latest_ground_time = 0.0

        self.create_subscription(
            PerceptionTargets, detection_topic, self._det_cb, 10
        )
        self.create_subscription(
            RoadblockArray, ground_topic, self._ground_cb, 10
        )

    def _det_cb(self, msg):
        candidates = []
        for target in msg.targets:
            if target.type != 'roadblock':
                continue
            for roi in target.rois:
                conf = float(roi.confidence)
                if conf < self.min_confidence:
                    continue

                xmin = float(roi.rect.x_offset)
                ymin = float(roi.rect.y_offset)
                xmax = xmin + float(roi.rect.width)
                ymax = ymin + float(roi.rect.height)

                candidates.append({
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'bbox_width': xmax - xmin,
                    'bbox_height': ymax - ymin,
                    'bbox_center_u': 0.5 * (xmin + xmax),
                    'bbox_bottom_u': 0.5 * (xmin + xmax),
                    'bbox_bottom_v': ymax,
                    'confidence': conf,
                })

        sample = candidates[0] if len(candidates) == 1 else None
        with self.lock:
            self.latest_detection = sample
            self.latest_detection_time = time.monotonic()

    def _ground_cb(self, msg):
        ground = None
        if len(msg.obstacles) == 1:
            obs = msg.obstacles[0]
            ground = {'ipm_x': float(obs.x), 'ipm_y': float(obs.y)}

        with self.lock:
            self.latest_ground = ground
            self.latest_ground_time = time.monotonic()

    def snapshot(self, max_age_s=0.25):
        now = time.monotonic()
        with self.lock:
            det = None if self.latest_detection is None else dict(self.latest_detection)
            grd = None if self.latest_ground is None else dict(self.latest_ground)
            det_t = self.latest_detection_time
            grd_t = self.latest_ground_time

        if det is None or grd is None:
            return None
        if now - det_t > max_age_s or now - grd_t > max_age_s:
            return None

        out = {}
        out.update(det)
        out.update(grd)
        return out


def med(values):
    return float(statistics.median(values))


def summarize(samples):
    keys = [
        'xmin', 'ymin', 'xmax', 'ymax',
        'bbox_width', 'bbox_height',
        'bbox_center_u', 'bbox_bottom_u', 'bbox_bottom_v',
        'confidence', 'ipm_x', 'ipm_y'
    ]
    return {k: med([s[k] for s in samples]) for k in keys}


def collect(node, seconds, sample_hz):
    samples = []
    end = time.monotonic() + seconds
    period = 1.0 / sample_hz
    while time.monotonic() < end:
        s = node.snapshot()
        if s is not None:
            samples.append(s)
        time.sleep(period)
    return samples


def save_csv(path, rows):
    fields = [
        'stage', 'true_x', 'true_y',
        'xmin', 'ymin', 'xmax', 'ymax',
        'bbox_width', 'bbox_height',
        'bbox_center_u', 'bbox_bottom_u', 'bbox_bottom_v',
        'confidence', 'ipm_x', 'ipm_y'
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detection-topic', default='/hobot_dnn_detection')
    p.add_argument('--ground-topic', default='/roadblock_ground_array')
    p.add_argument('--seconds', type=float, default=3.0)
    p.add_argument('--sample-hz', type=float, default=20.0)
    p.add_argument('--min-confidence', type=float, default=0.50)
    p.add_argument('--camera-ground-x', type=float, default=0.05)
    p.add_argument('--camera-ground-y', type=float, default=0.00)
    p.add_argument('--camera-height', type=float, default=0.28)
    p.add_argument('--cone-height', type=float, default=0.30)
    p.add_argument('--fy', type=float, default=418.42498779296875)
    p.add_argument('--out-dir', default='/root/intelligent_car_ws/test_logs/cone_calibration')
    args = p.parse_args()

    stages = [
        ('A_CALIB', 0.80, 0.00, '标定点'),
        ('B_NEAR',  0.60, 0.00, '近距离验证'),
        ('C_FAR',   1.00, 0.00, '远距离验证'),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    raw_csv = out_dir / f'cone_height_calib_raw_{stamp}.csv'
    summary_json = out_dir / f'cone_height_calib_summary_{stamp}.json'

    rclpy.init()
    node = ConeCalibNode(args.detection_topic, args.ground_topic, args.min_confidence)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    time.sleep(1.0)

    print('\n=== 锥形桶高度标定：1 点标定 + 2 点验证 ===')
    print('要求：画面中只保留 1 个 roadblock。')
    print('X/Y 指小车旋转中心 -> 锥桶底座几何中心。\n')

    all_rows = []
    summaries = {}

    try:
        for name, true_x, true_y, note in stages:
            print(f'[{name}] {note}: 请摆到 X={true_x:.2f} m, Y={true_y:.2f} m')
            input('摆好并静止后按 Enter 开始采集...')
            samples = collect(node, args.seconds, args.sample_hz)

            if len(samples) < 5:
                print(f'ERROR: 有效样本只有 {len(samples)} 帧。')
                print('检查 YOLO、/roadblock_ground_array，以及画面中是否只有一个锥桶。')
                return 2

            s = summarize(samples)
            summaries[name] = {
                'true_x': true_x,
                'true_y': true_y,
                'sample_count': len(samples),
                **s,
            }

            for row in samples:
                all_rows.append({
                    'stage': name,
                    'true_x': true_x,
                    'true_y': true_y,
                    **row,
                })

            print(
                f'完成 {len(samples)} 帧 | '
                f'h_med={s["bbox_height"]:.2f}px | '
                f'w_med={s["bbox_width"]:.2f}px | '
                f'IPM=({s["ipm_x"]:.4f},{s["ipm_y"]:.4f})\n'
            )

        a = summaries['A_CALIB']
        d0 = math.hypot(
            a['true_x'] - args.camera_ground_x,
            a['true_y'] - args.camera_ground_y
        )
        h0 = a['bbox_height']
        k = d0 * h0

        theoretical_k = args.fy * args.cone_height
        diff_pct = 100.0 * (k - theoretical_k) / theoretical_k

        validations = {}
        for name in ('B_NEAR', 'C_FAR'):
            s = summaries[name]
            true_d = math.hypot(
                s['true_x'] - args.camera_ground_x,
                s['true_y'] - args.camera_ground_y
            )
            pred_d = k / s['bbox_height']
            err = pred_d - true_d
            validations[name] = {
                'true_distance_m': true_d,
                'pred_distance_m': pred_d,
                'signed_error_m': err,
                'abs_error_m': abs(err),
            }

        max_err = max(v['abs_error_m'] for v in validations.values())

        result = {
            'camera': {
                'ground_x_m': args.camera_ground_x,
                'ground_y_m': args.camera_ground_y,
                'height_m': args.camera_height,
                'fy_px': args.fy,
            },
            'cone': {
                'height_m': args.cone_height,
                'base_width_m': 0.20,
                'base_length_m': 0.20,
                'safe_radius_m': math.sqrt(0.1**2 + 0.1**2),
            },
            'model': {
                'equation': 'distance_m = k / bbox_height_px',
                'k': k,
                'theoretical_fy_times_H': theoretical_k,
                'k_vs_theory_percent': diff_pct,
            },
            'stages': summaries,
            'validation': validations,
            'max_abs_validation_error_m': max_err,
        }

        save_csv(raw_csv, all_rows)
        summary_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        print('=== 标定结果 ===')
        print(f'单点模型: distance_m = {k:.6f} / bbox_height_px')
        print(f'理论 fy*H = {theoretical_k:.6f}')
        print(f'实测 k 与理论值差异 = {diff_pct:+.2f}%')

        for name, label in [('B_NEAR', 'X=0.60'), ('C_FAR', 'X=1.00')]:
            v = validations[name]
            print(
                f'{label}: true={v["true_distance_m"]:.4f} m, '
                f'pred={v["pred_distance_m"]:.4f} m, '
                f'abs_err={v["abs_error_m"]*100:.2f} cm'
            )

        if max_err <= 0.05:
            print('PASS建议：两个验证点最大绝对误差 <= 5 cm，可先保留单点模型。')
        else:
            print('建议升级：至少一个验证点误差 > 5 cm，改为两点修正 d=a/h+b。')

        print(f'原始数据: {raw_csv}')
        print(f'汇总结果: {summary_json}')
        return 0

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
