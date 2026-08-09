#!/usr/bin/env python3
"""Reproducible stop-level fit for the roadblock Adaptive IPM model."""

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np


MOTION_CONES = {
    "L2": (1.60, 0.60),
    "L1": (1.60, 0.30),
    "C": (1.60, 0.00),
    "R1": (1.60, -0.30),
    "R2": (1.60, -0.60),
}
NEAR_CONES = {
    "L1": (1.00, 0.30),
    "C": (1.00, 0.00),
    "R1": (1.00, -0.30),
}
MOTION_REMAP = {
    "POSE45_027": {"L2": "L1", "L1": "C", "C": "R1"},
    "POSE45_030": {"L2": "L1", "L1": "C"},
    "POSE45_031": {"L2": "L1", "L1": "C"},
    "POSE45_033": {"L2": "L1", "L1": "C", "C": "R1", "R1": "R2"},
}
DISTANCE_BINS = (
    ("<0.70", -math.inf, 0.70),
    ("0.70~0.90", 0.70, 0.90),
    ("0.90~1.20", 0.90, 1.20),
    (">1.20", 1.20, math.inf),
)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def snapshot_scalar(path, key):
    prefix = key + ":"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return float(stripped.split(":", 1)[1].strip())
    raise ValueError(f"{key} not found in {path}")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_common(row, edge_margin, image_width):
    if row.get("capture_status") != "PASS" or not finite(row.get("n")):
        return False
    if float(row["n"]) <= 0.0:
        return False
    required = (
        "bbox_center_u_mean",
        "bbox_width_mean",
        "bbox_height_mean",
        "ipm_raw_distance_mean",
    )
    if not all(finite(row.get(name)) for name in required):
        return False
    width = float(row["bbox_width_mean"])
    height = float(row["bbox_height_mean"])
    center_u = float(row["bbox_center_u_mean"])
    if width <= 0.0 or height <= 0.0:
        return False
    xmin = center_u - 0.5 * width
    xmax = center_u + 0.5 * width
    return xmin >= edge_margin and xmax <= image_width - edge_margin


def normalize_row(row, session, cone_name, truth, image_width):
    r_raw = float(row["ipm_raw_distance_mean"])
    width = float(row["bbox_width_mean"])
    height = float(row["bbox_height_mean"])
    center_u = float(row["bbox_center_u_mean"])
    return {
        "session": session,
        "stop_id": f"{session}:{row['stop_id']}",
        "raw_stop_id": row["stop_id"],
        "pose": str(int(round(float(row["cone_pose_deg"])))) + "deg",
        "cone": cone_name,
        "truth": float(truth),
        "r_raw": r_raw,
        "width": width,
        "height": height,
        "center_u": center_u,
        "image_width": float(image_width),
    }


def load_motion(directory, edge_margin, image_width):
    snapshot = directory / "parameters_snapshot.yaml"
    camera_x = snapshot_scalar(snapshot, "camera_ground_x_m")
    camera_y = snapshot_scalar(snapshot, "camera_ground_y_m")
    clean = []
    remap_counts = {}
    rejected = {"excluded_stop": 0, "invalid_or_edge": 0, "unknown_cone": 0}
    for row in read_csv(directory / "stops_summary.csv"):
        stop = row["stop_id"]
        if stop == "POSE45_019":
            rejected["excluded_stop"] += 1
            continue
        if not valid_common(row, edge_margin, image_width):
            rejected["invalid_or_edge"] += 1
            continue
        logger_name = row["cone_name"]
        actual_name = MOTION_REMAP.get(stop, {}).get(logger_name, logger_name)
        if actual_name != logger_name:
            remap_counts[stop] = remap_counts.get(stop, 0) + 1
        if actual_name not in MOTION_CONES:
            rejected["unknown_cone"] += 1
            continue
        cone_x, cone_y = MOTION_CONES[actual_name]
        progress = float(row["robot_progress_m"])
        lateral = float(row["robot_lateral_m"])
        truth = math.hypot(
            cone_x - progress - camera_x,
            cone_y - lateral - camera_y,
        )
        clean.append(normalize_row(row, "motion", actual_name, truth, image_width))
    return clean, rejected, remap_counts, (camera_x, camera_y)


def load_near(directory, edge_margin, image_width):
    snapshot = directory / "parameters_snapshot.yaml"
    camera_x = snapshot_scalar(snapshot, "camera_ground_x_m")
    camera_y = snapshot_scalar(snapshot, "camera_ground_y_m")
    clean = []
    crosscheck = []
    rejected = {"excluded_stop": 0, "invalid_or_edge": 0, "unknown_cone": 0}
    for row in read_csv(directory / "stops_summary.csv"):
        stop = row["stop_id"]
        if stop == "POSE45_041":
            rejected["excluded_stop"] += 1
            continue
        if not valid_common(row, edge_margin, image_width):
            rejected["invalid_or_edge"] += 1
            continue
        cone_name = row["cone_name"]
        if cone_name not in NEAR_CONES or not finite(row.get("truth_distance_mean_m")):
            rejected["unknown_cone"] += 1
            continue
        cone_x, cone_y = NEAR_CONES[cone_name]
        expected = math.hypot(
            cone_x - float(row["robot_dx_m"]) - camera_x,
            cone_y - float(row["robot_dy_m"]) - camera_y,
        )
        truth = float(row["truth_distance_mean_m"])
        crosscheck.append(abs(expected - truth))
        clean.append(normalize_row(row, "near", cone_name, truth, image_width))
    return clean, rejected, crosscheck, (camera_x, camera_y)


def design_matrix(rows):
    values = []
    for row in rows:
        r_raw = row["r_raw"]
        width = row["width"]
        aspect = width / row["height"]
        u_norm = (row["center_u"] - row["image_width"] / 2.0) / (
            row["image_width"] / 2.0
        )
        values.append(
            [
                1.0,
                r_raw - 1.0,
                (width * r_raw - 100.0) / 50.0,
                (aspect - 0.78) / 0.15,
                u_norm * u_norm,
            ]
        )
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite model feature")
    return matrix


def truth_array(rows):
    return np.asarray([row["truth"] for row in rows], dtype=float)


def baseline_prediction(rows):
    return np.asarray(
        [
            0.70 * (row["r_raw"] + 0.10)
            + 0.30 * (99.488 / row["height"] + 0.22381)
            for row in rows
        ],
        dtype=float,
    )


def fit_coefficients(rows):
    target_offset = truth_array(rows) - np.asarray(
        [row["r_raw"] for row in rows], dtype=float
    )
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix(rows), target_offset, rcond=None)
    return coefficients


def adaptive_prediction(rows, coefficients):
    raw = np.asarray([row["r_raw"] for row in rows], dtype=float)
    return raw + design_matrix(rows) @ coefficients


def metrics(rows, prediction):
    truth = truth_array(rows)
    error = np.asarray(prediction, dtype=float) - truth
    return {
        "n": int(len(error)),
        "bias_m": float(np.mean(error)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "max_abs_m": float(np.max(np.abs(error))),
        "max_positive_m": float(np.max(error)),
        "max_negative_m": float(np.min(error)),
        "within_4cm_fraction": float(np.mean(np.abs(error) <= 0.04)),
    }


def make_group_folds(rows, n_splits=5):
    group_indices = {}
    for index, row in enumerate(rows):
        group_indices.setdefault(row["stop_id"], []).append(index)
    if len(group_indices) < n_splits:
        raise ValueError("not enough unique stops for GroupKFold")
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    ordered = sorted(group_indices.items(), key=lambda item: (-len(item[1]), item[0]))
    for group, indices in ordered:
        fold = min(range(n_splits), key=lambda number: (fold_sizes[number], number))
        fold_groups[fold].append(group)
        fold_sizes[fold] += len(indices)
    folds = []
    all_indices = set(range(len(rows)))
    for groups in fold_groups:
        validation = sorted(index for group in groups for index in group_indices[group])
        training = sorted(all_indices.difference(validation))
        folds.append((training, validation, groups))
    return folds


def group_oof(rows):
    prediction = np.empty(len(rows), dtype=float)
    fold_reports = []
    folds = make_group_folds(rows)
    for number, (training_indices, validation_indices, groups) in enumerate(folds, 1):
        training = [rows[index] for index in training_indices]
        validation = [rows[index] for index in validation_indices]
        coefficients = fit_coefficients(training)
        fold_prediction = adaptive_prediction(validation, coefficients)
        prediction[validation_indices] = fold_prediction
        fold_reports.append(
            {
                "fold": number,
                "train_stops": sorted({row["stop_id"] for row in training}),
                "validation_stops": sorted(groups),
                "coefficients": [float(value) for value in coefficients],
                "metrics": metrics(validation, fold_prediction),
            }
        )
    return prediction, fold_reports


def subset_report(rows, prediction, selector):
    indices = [index for index, row in enumerate(rows) if selector(row)]
    if not indices:
        return {"n": 0}
    selected_rows = [rows[index] for index in indices]
    return metrics(selected_rows, np.asarray(prediction)[indices])


def grouped_reports(rows, prediction):
    distance = {}
    for label, lower, upper in DISTANCE_BINS:
        distance[label] = subset_report(
            rows,
            prediction,
            lambda row, lo=lower, hi=upper: row["truth"] >= lo and row["truth"] < hi,
        )
    cones = {
        cone: subset_report(rows, prediction, lambda row, value=cone: row["cone"] == value)
        for cone in ("L2", "L1", "C", "R1", "R2")
    }
    poses = {
        pose: subset_report(rows, prediction, lambda row, value=pose: row["pose"] == value)
        for pose in ("0deg", "45deg")
    }
    return {"distance": distance, "cone": cones, "pose": poses}


def print_metrics(label, value):
    print(
        f"{label}: n={value['n']} bias={value['bias_m']*100:.4f}cm "
        f"MAE={value['mae_m']*100:.4f}cm RMSE={value['rmse_m']*100:.4f}cm "
        f"MaxAbs={value['max_abs_m']*100:.4f}cm"
    )


def evaluate_margin(motion_dir, near_dir, margin, image_width):
    motion, motion_rejected, remaps, motion_camera = load_motion(
        motion_dir, margin, image_width
    )
    near, near_rejected, crosscheck, near_camera = load_near(near_dir, margin, image_width)
    if motion_camera != near_camera:
        raise ValueError(f"camera ground offsets differ: {motion_camera} vs {near_camera}")
    rows = motion + near
    baseline = baseline_prediction(rows)
    oof, folds = group_oof(rows)
    final_coefficients = fit_coefficients(rows)
    final_prediction = adaptive_prediction(rows, final_coefficients)
    offsets = final_prediction - np.asarray([row["r_raw"] for row in rows])
    percentiles = np.percentile(offsets, [1, 5, 50, 95, 99])
    clamp_min = math.floor((float(np.min(offsets)) - 0.005) * 1000.0) / 1000.0
    clamp_max = math.ceil((float(np.max(offsets)) + 0.005) * 1000.0) / 1000.0

    motion_coefficients = fit_coefficients(motion)
    external_prediction = adaptive_prediction(near, motion_coefficients)
    report = {
        "edge_margin_px": margin,
        "camera_ground_x_m": motion_camera[0],
        "camera_ground_y_m": motion_camera[1],
        "motion": {
            "rows": len(motion),
            "stops": len({row["stop_id"] for row in motion}),
            "rejected": motion_rejected,
            "remap_counts": remaps,
        },
        "near": {
            "rows": len(near),
            "stops": len({row["stop_id"] for row in near}),
            "rejected": near_rejected,
            "truth_crosscheck_max_abs_m": max(crosscheck) if crosscheck else None,
        },
        "merged": {"rows": len(rows), "stops": len({row["stop_id"] for row in rows})},
        "baseline_a": metrics(rows, baseline),
        "adaptive_c_oof": metrics(rows, oof),
        "adaptive_c_oof_grouped": grouped_reports(rows, oof),
        "truth_lt_0_90": {
            "baseline_a": subset_report(rows, baseline, lambda row: row["truth"] < 0.90),
            "adaptive_c_oof": subset_report(rows, oof, lambda row: row["truth"] < 0.90),
        },
        "folds": folds,
        "final_fit": {
            "coefficients": {
                name: float(value)
                for name, value in zip(("c0", "cr", "cw", "ca", "cu"), final_coefficients)
            },
            "training_metrics": metrics(rows, final_prediction),
            "offset_distribution_m": {
                "min": float(np.min(offsets)),
                "p1": float(percentiles[0]),
                "p5": float(percentiles[1]),
                "median": float(percentiles[2]),
                "p95": float(percentiles[3]),
                "p99": float(percentiles[4]),
                "max": float(np.max(offsets)),
            },
            "recommended_offset_min_m": clamp_min,
            "recommended_offset_max_m": clamp_max,
            "data_ranges": {
                key: [float(min(row[key] for row in rows)), float(max(row[key] for row in rows))]
                for key in ("truth", "r_raw", "width", "height", "center_u")
            },
        },
        "motion_train_near_test": {
            "coefficients": [float(value) for value in motion_coefficients],
            "baseline_a_metrics": metrics(near, baseline_prediction(near)),
            "adaptive_c_metrics": metrics(near, external_prediction),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--near-dir", type=Path, required=True)
    parser.add_argument("--edge-margin", type=float, default=8.0)
    parser.add_argument("--image-width", type=float, default=640.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    margins = []
    for margin in (args.edge_margin, 20.0, 40.0):
        if margin not in margins:
            margins.append(margin)
    reports = {
        str(margin): evaluate_margin(args.motion_dir, args.near_dir, margin, args.image_width)
        for margin in margins
    }
    primary = reports[str(args.edge_margin)]
    print("EXCLUSIONS: motion:POSE45_019, near:POSE45_041")
    print("REMAPS: POSE45_027, POSE45_030, POSE45_031, POSE45_033")
    print(
        f"PRIMARY {args.edge_margin:g}px: motion={primary['motion']['rows']} rows/"
        f"{primary['motion']['stops']} stops near={primary['near']['rows']} rows/"
        f"{primary['near']['stops']} stops merged={primary['merged']['rows']} rows/"
        f"{primary['merged']['stops']} stops"
    )
    print(f"near truth cross-check max={primary['near']['truth_crosscheck_max_abs_m']:.12g}m")
    print_metrics("A OOF-equivalent", primary["baseline_a"])
    print_metrics("C 5-fold OOF", primary["adaptive_c_oof"])
    print_metrics("A truth<0.90", primary["truth_lt_0_90"]["baseline_a"])
    print_metrics("C truth<0.90", primary["truth_lt_0_90"]["adaptive_c_oof"])
    for label, value in primary["adaptive_c_oof_grouped"]["distance"].items():
        if value["n"]:
            print_metrics(f"C distance {label}", value)
    for label, value in primary["adaptive_c_oof_grouped"]["pose"].items():
        if value["n"]:
            print_metrics(f"C pose {label}", value)
    for label, value in primary["adaptive_c_oof_grouped"]["cone"].items():
        if value["n"]:
            print_metrics(f"C cone {label}", value)
    print_metrics(
        "motion-train -> near-test A",
        primary["motion_train_near_test"]["baseline_a_metrics"],
    )
    print_metrics(
        "motion-train -> near-test C",
        primary["motion_train_near_test"]["adaptive_c_metrics"],
    )
    coefficients = primary["final_fit"]["coefficients"]
    print("FINAL COEFFICIENTS")
    for name in ("c0", "cr", "cw", "ca", "cu"):
        print(f"  {name}={coefficients[name]:.10f}")
    print("OFFSET DISTRIBUTION", primary["final_fit"]["offset_distribution_m"])
    print(
        "RECOMMENDED CLAMP",
        primary["final_fit"]["recommended_offset_min_m"],
        primary["final_fit"]["recommended_offset_max_m"],
    )
    print("SENSITIVITY")
    for margin, report in reports.items():
        a = report["baseline_a"]
        c = report["adaptive_c_oof"]
        print(
            f"  {margin}px rows={report['merged']['rows']} "
            f"A_MAE={a['mae_m']*100:.4f}cm A_RMSE={a['rmse_m']*100:.4f}cm "
            f"C_MAE={c['mae_m']*100:.4f}cm C_RMSE={c['rmse_m']*100:.4f}cm"
        )

    output_dir = args.output_dir
    if output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("/root/intelligent_car_ws/test_logs/roadblock_adaptive_ipm_fit") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adaptive_ipm_fit_report.json"
    payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "motion_dir": str(args.motion_dir),
        "near_dir": str(args.near_dir),
        "primary_edge_margin_px": args.edge_margin,
        "reports": reports,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"REPORT {output_path}")


if __name__ == "__main__":
    main()
