#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed safety gate for the yellow-lane closed-loop system.

Only this file converts the controller's JSON command into ``/cmd_vel``.
The default ``motion_enabled=false`` is safe for zero-output smoke tests.
Use ``--self-test`` for ROS-free state-machine tests, ``--report`` to build
summary.json/csv from feedback.log, and ``--zero`` for a bounded zero burst.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SELF_TEST = "--self-test" in sys.argv
REPORT_MODE = "--report" in sys.argv

NORMAL_MODES = frozenset({"green_dual_inner_edge"})
DEGRADED_MODES = frozenset(
    {"green_yellow_hybrid", "single_green_width_offset"}
)
ALLOWED_MODES = NORMAL_MODES | DEGRADED_MODES
FORBIDDEN_MODES = frozenset({"single_boundary_normal_offset", "invalid"})


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass
class GateConfig:
    ready_frames: int = 4
    controller_timeout_sec: float = 0.40
    pipeline_timeout_sec: float = 0.75
    feedback_timeout_sec: float = 0.20
    path_timeout_sec: float = 0.40
    max_linear_normal: float = 0.03
    max_linear_recovery: float = 0.018
    max_angular: float = 0.15
    max_angular_slew_rad_s2: float = 0.30
    max_feedback_linear_speed: float = 0.15
    max_feedback_angular_speed: float = 0.60
    publish_rate_hz: float = 20.0
    motion_enabled: bool = False

    def validate(self) -> None:
        self.ready_frames = max(1, int(self.ready_frames))
        for name in (
            "controller_timeout_sec",
            "pipeline_timeout_sec",
            "feedback_timeout_sec",
            "path_timeout_sec",
            "max_linear_normal",
            "max_linear_recovery",
            "max_angular",
            "max_angular_slew_rad_s2",
            "max_feedback_linear_speed",
            "max_feedback_angular_speed",
            "publish_rate_hz",
        ):
            value = finite(getattr(self, name), -1.0)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if self.max_linear_normal > 0.03:
            raise ValueError("max_linear_normal cannot exceed 0.03")
        if not 0.015 <= self.max_linear_recovery <= 0.02:
            raise ValueError("max_linear_recovery must be in [0.015, 0.02]")
        if self.max_angular > 0.15:
            raise ValueError("max_angular cannot exceed 0.15")


@dataclass
class InputSnapshot:
    document: Optional[Dict[str, Any]] = None
    received_at: float = 0.0
    parse_error: str = ""
    sequence: int = 0


@dataclass
class FeedbackSnapshot:
    received_at: float = 0.0
    linear_x: float = 0.0
    angular_z: float = 0.0
    sequence: int = 0


@dataclass
class GateDecision:
    timestamp: str
    gate_state: str
    gate_reason: str
    gate_v: float = 0.0
    gate_w: float = 0.0
    ready_count: int = 0
    controller_age_ms: float = -1.0
    pipeline_age_ms: float = -1.0
    feedback_age_ms: float = -1.0


class LaneControlGateCore:
    """ROS-independent gate state machine used by the node and self-tests."""

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()
        self.config.validate()
        self.controller = InputSnapshot()
        self.pipeline = InputSnapshot()
        self.feedback = FeedbackSnapshot()
        self.state = "WAIT_READY"
        self.reason = "waiting_for_inputs"
        self.ready_count = 0
        self.last_evaluated_controller_sequence = 0
        self.last_output_w = 0.0
        self.last_output_time = 0.0
        self.estop_latched = False
        self.manual_stop_latched = False

    def update_controller(
        self,
        document: Optional[Mapping[str, Any]],
        now: float,
        parse_error: str = "",
    ) -> None:
        self.controller = InputSnapshot(
            dict(document) if document is not None else None,
            float(now),
            parse_error,
            self.controller.sequence + 1,
        )

    def update_pipeline(
        self,
        document: Optional[Mapping[str, Any]],
        now: float,
        parse_error: str = "",
    ) -> None:
        self.pipeline = InputSnapshot(
            dict(document) if document is not None else None,
            float(now),
            parse_error,
            self.pipeline.sequence + 1,
        )

    def update_feedback(
        self, linear_x: float, angular_z: float, now: float
    ) -> None:
        self.feedback = FeedbackSnapshot(
            float(now),
            finite(linear_x),
            finite(angular_z),
            self.feedback.sequence + 1,
        )

    def request_stop(self) -> None:
        self.manual_stop_latched = True

    def request_estop(self) -> None:
        self.estop_latched = True

    @staticmethod
    def _age(now: float, received_at: float) -> float:
        return now - received_at if received_at > 0.0 else -1.0

    def _failure_reason(self, now: float) -> str:
        if self.estop_latched:
            return "estop_latched"
        if self.manual_stop_latched:
            return "manual_stop"
        if not self.config.motion_enabled:
            return "motion_disabled"
        controller_age = self._age(now, self.controller.received_at)
        pipeline_age = self._age(now, self.pipeline.received_at)
        feedback_age = self._age(now, self.feedback.received_at)
        if controller_age < 0.0:
            return "controller_not_received"
        if controller_age > self.config.controller_timeout_sec:
            return "controller_timeout"
        if self.controller.parse_error or self.controller.document is None:
            return "controller_invalid"
        command = self.controller.document
        if not bool(command.get("ready", False)):
            return str(command.get("reason") or "controller_blocked")
        if not bool(command.get("path_valid", False)):
            return "path_invalid"
        path_age_ms = finite(command.get("path_age_ms"), math.inf)
        if path_age_ms > self.config.path_timeout_sec * 1000.0:
            return "path_timeout"
        controller_pipeline_age_ms = finite(
            command.get("pipeline_age_ms"), math.inf
        )
        if (
            controller_pipeline_age_ms
            > self.config.pipeline_timeout_sec * 1000.0
        ):
            return "controller_pipeline_timeout"
        if pipeline_age < 0.0:
            return "pipeline_not_received"
        if pipeline_age > self.config.pipeline_timeout_sec:
            return "pipeline_timeout"
        if self.pipeline.parse_error or self.pipeline.document is None:
            return "pipeline_invalid"
        pipeline = self.pipeline.document
        if not bool(pipeline.get("centerline_valid", False)):
            return "pipeline_invalid"
        mode = str(command.get("mode", ""))
        pipeline_mode = str(pipeline.get("centerline_mode", ""))
        if mode in FORBIDDEN_MODES or pipeline_mode in FORBIDDEN_MODES:
            return "mode_forbidden"
        if mode not in ALLOWED_MODES or pipeline_mode not in ALLOWED_MODES:
            return "mode_not_allowed"
        if mode != pipeline_mode:
            return "mode_mismatch"
        if feedback_age < 0.0:
            return "feedback_not_received"
        if feedback_age > self.config.feedback_timeout_sec:
            return "feedback_timeout"
        if abs(self.feedback.linear_x) > self.config.max_feedback_linear_speed:
            return "feedback_linear_overspeed"
        if abs(self.feedback.angular_z) > self.config.max_feedback_angular_speed:
            return "feedback_angular_overspeed"
        requested_v = finite(command.get("linear_x"), math.nan)
        requested_w = finite(command.get("angular_z"), math.nan)
        if not math.isfinite(requested_v) or not math.isfinite(requested_w):
            return "controller_command_nonfinite"
        if requested_v < 0.0:
            return "reverse_forbidden"
        return ""

    def _zero_decision(self, now: float, state: str, reason: str) -> GateDecision:
        self.state = state
        self.reason = reason
        self.ready_count = 0
        self.last_output_w = 0.0
        self.last_output_time = now
        return self._decision(now, 0.0, 0.0)

    def _decision(self, now: float, linear_x: float, angular_z: float) -> GateDecision:
        return GateDecision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            gate_state=self.state,
            gate_reason=self.reason,
            gate_v=float(linear_x),
            gate_w=float(angular_z),
            ready_count=self.ready_count,
            controller_age_ms=self._age(now, self.controller.received_at) * 1000.0,
            pipeline_age_ms=self._age(now, self.pipeline.received_at) * 1000.0,
            feedback_age_ms=self._age(now, self.feedback.received_at) * 1000.0,
        )

    def _shape_command(self, now: float, degraded: bool) -> Tuple[float, float]:
        command = self.controller.document or {}
        cap = (
            self.config.max_linear_recovery
            if degraded
            else self.config.max_linear_normal
        )
        requested_v = clamp(finite(command.get("linear_x")), 0.0, cap)
        requested_w = clamp(
            finite(command.get("angular_z")),
            -self.config.max_angular,
            self.config.max_angular,
        )
        lateral = abs(finite(command.get("lateral_error")))
        heading = abs(finite(command.get("heading_error")))
        confidence = clamp(
            finite(command.get("controller_confidence"), 0.0), 0.0, 1.0
        )
        requested_v *= clamp(1.0 - lateral / 0.25, 0.35, 1.0)
        requested_v *= clamp(1.0 - heading / math.radians(40.0), 0.35, 1.0)
        requested_v *= clamp(1.0 - abs(requested_w) / 0.30, 0.50, 1.0)
        requested_v *= clamp(0.55 + 0.45 * confidence, 0.55, 1.0)
        dt = (
            now - self.last_output_time
            if self.last_output_time > 0.0
            else 1.0 / self.config.publish_rate_hz
        )
        max_delta = self.config.max_angular_slew_rad_s2 * max(0.0, dt)
        output_w = self.last_output_w + clamp(
            requested_w - self.last_output_w, -max_delta, max_delta
        )
        self.last_output_w = output_w
        self.last_output_time = now
        return max(0.0, requested_v), output_w

    def evaluate(self, now: Optional[float] = None) -> GateDecision:
        instant = time.monotonic() if now is None else float(now)
        reason = self._failure_reason(instant)
        if reason:
            state = "ESTOP" if reason == "estop_latched" else "STOPPED"
            if reason in {
                "controller_not_received",
                "pipeline_not_received",
                "feedback_not_received",
            } and self.config.motion_enabled:
                state = "WAIT_READY"
            return self._zero_decision(instant, state, reason)

        if self.controller.sequence != self.last_evaluated_controller_sequence:
            self.last_evaluated_controller_sequence = self.controller.sequence
            self.ready_count += 1
        if self.ready_count < self.config.ready_frames:
            self.state = "WAIT_READY"
            self.reason = "ready_stabilizing"
            self.last_output_w = 0.0
            self.last_output_time = instant
            return self._decision(instant, 0.0, 0.0)

        command = self.controller.document or {}
        quality = str(command.get("quality", ""))
        mode = str(command.get("mode", ""))
        degraded = mode in DEGRADED_MODES or quality in {"degraded", "recovery"}
        self.state = "DEGRADED" if degraded else "RUNNING"
        self.reason = "degraded_allowed" if degraded else "normal_allowed"
        linear_x, angular_z = self._shape_command(instant, degraded)
        return self._decision(instant, linear_x, angular_z)


def _distribution(values: Sequence[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"min": None, "median": None, "max": None, "p95": None}
    index = min(len(clean) - 1, max(0, math.ceil(0.95 * len(clean)) - 1))
    return {
        "min": clean[0],
        "median": statistics.median(clean),
        "max": clean[-1],
        "p95": clean[index],
    }


def generate_report(log_path: Path, output_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    timestamps = [finite(row.get("monotonic_sec"), math.nan) for row in rows]
    finite_times = [value for value in timestamps if math.isfinite(value)]
    duration = max(finite_times) - min(finite_times) if len(finite_times) > 1 else 0.0
    states = [str(row.get("gate_state", "")) for row in rows]
    ready = [bool(row.get("controller_ready", False)) for row in rows]
    reasons = [str(row.get("gate_reason", "")) for row in rows]
    summary = {
        "sample_count": len(rows),
        "runtime_sec": round(duration, 3),
        "mode_distribution": _distribution(
            [str(row.get("mode", "")) for row in rows]
        ),
        "controller_ready_ratio": sum(ready) / len(ready) if ready else 0.0,
        "controller_blocked_ratio": 1.0 - (sum(ready) / len(ready)) if ready else 0.0,
        "gate_state_distribution": _distribution(states),
        "gate_state_ratio": {
            state: states.count(state) / len(states) for state in sorted(set(states))
        } if states else {},
        "stop_reason_distribution": _distribution(
            [reason for state, reason in zip(states, reasons) if state in {"STOPPED", "ESTOP"}]
        ),
        "gate_v": _stats([finite(row.get("gate_v"), math.nan) for row in rows]),
        "gate_w": _stats([finite(row.get("gate_w"), math.nan) for row in rows]),
        "lateral_error": _stats([finite(row.get("lateral_error"), math.nan) for row in rows]),
        "heading_error": _stats([finite(row.get("heading_error"), math.nan) for row in rows]),
        "path_point_count": _stats([finite(row.get("path_point_count"), math.nan) for row in rows]),
        "path_span": _stats([finite(row.get("path_span"), math.nan) for row in rows]),
        "dp_ms": _stats([finite(row.get("dp_ms"), math.nan) for row in rows]),
        "process_fps": _stats([finite(row.get("process_fps"), math.nan) for row in rows]),
        "feedback_hz": _stats([finite(row.get("feedback_hz"), math.nan) for row in rows]),
        "timeout_count": sum("timeout" in reason for reason in reasons),
        "exception_count": sum("exception" in reason for reason in reasons),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, json.dumps(value, ensure_ascii=False)))
    return summary


def _controller(mode: str = "green_dual_inner_edge", ready: bool = True) -> Dict[str, Any]:
    return {
        "ready": ready,
        "reason": "" if ready else "controller_blocked",
        "path_valid": True,
        "path_age_ms": 20.0,
        "pipeline_age_ms": 20.0,
        "linear_x": 0.10,
        "angular_z": 0.10,
        "quality": "recovery" if mode == "single_green_width_offset" else "normal",
        "mode": mode,
        "lateral_error": 0.01,
        "heading_error": 0.02,
        "controller_confidence": 0.9,
        "path_point_count": 12,
        "path_span_m": 0.6,
    }


def _pipeline(mode: str = "green_dual_inner_edge", valid: bool = True) -> Dict[str, Any]:
    return {
        "centerline_valid": valid,
        "centerline_mode": mode,
        "final_centerline_point_count": 12,
        "centerline_forward_span_m": 0.6,
    }


def _feed_ready(core: LaneControlGateCore, start: float = 1.0, frames: int = 4) -> GateDecision:
    decision = core.evaluate(start)
    for index in range(frames):
        now = start + 0.1 * index
        core.update_pipeline(_pipeline(), now)
        core.update_feedback(0.0, 0.0, now)
        core.update_controller(_controller(), now)
        decision = core.evaluate(now + 0.01)
    return decision


def run_self_test() -> None:
    tests: List[Tuple[str, Any]] = []

    def test(name: str):
        def register(function: Any) -> Any:
            tests.append((name, function))
            return function
        return register

    @test("WAIT_READY never moves")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        assert core.evaluate(1.0).gate_v == 0.0

    @test("four ready frames enter RUNNING")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        assert _feed_ready(core).gate_state == "RUNNING"

    @test("normal mode is allowed")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        decision = _feed_ready(core)
        assert 0.0 < decision.gate_v <= 0.03

    @test("recovery enters DEGRADED")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        for index in range(4):
            now = 1.0 + index * 0.1
            core.update_pipeline(_pipeline("single_green_width_offset"), now)
            core.update_feedback(0.0, 0.0, now)
            core.update_controller(_controller("single_green_width_offset"), now)
            decision = core.evaluate(now + 0.01)
        assert decision.gate_state == "DEGRADED"
        assert 0.0 < decision.gate_v <= 0.02

    @test("green-yellow hybrid enters DEGRADED")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        for index in range(4):
            now = 1.0 + index * 0.1
            core.update_pipeline(_pipeline("green_yellow_hybrid"), now)
            core.update_feedback(0.0, 0.0, now)
            command = _controller("green_yellow_hybrid")
            command["quality"] = "degraded"
            core.update_controller(command, now)
            decision = core.evaluate(now + 0.01)
        assert decision.gate_state == "DEGRADED"

    @test("controller BLOCKED immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.update_controller(_controller(ready=False), 1.5)
        assert core.evaluate(1.51).gate_v == 0.0

    @test("invalid pipeline immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.update_pipeline(_pipeline(valid=False), 1.5)
        assert core.evaluate(1.51).gate_reason == "pipeline_invalid"

    @test("single boundary normal offset is forbidden")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        core.update_controller(_controller("single_boundary_normal_offset"), 1.0)
        core.update_pipeline(_pipeline("single_boundary_normal_offset"), 1.0)
        core.update_feedback(0.0, 0.0, 1.0)
        assert core.evaluate(1.01).gate_v == 0.0

    @test("controller timeout immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        assert core.evaluate(2.0).gate_reason == "controller_timeout"

    @test("pipeline timeout immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.update_controller(_controller(), 2.09)
        core.update_feedback(0.0, 0.0, 2.09)
        assert core.evaluate(2.10).gate_reason == "pipeline_timeout"

    @test("feedback timeout immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.update_controller(_controller(), 1.7)
        core.update_pipeline(_pipeline(), 1.7)
        assert core.evaluate(1.71).gate_reason == "feedback_timeout"

    @test("feedback overspeed immediately zeros")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.update_controller(_controller(), 1.5)
        core.update_pipeline(_pipeline(), 1.5)
        core.update_feedback(0.16, 0.0, 1.5)
        assert core.evaluate(1.51).gate_reason == "feedback_linear_overspeed"

    @test("linear and angular limits hold")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        decision = _feed_ready(core)
        assert 0.0 <= decision.gate_v <= 0.03
        assert abs(decision.gate_w) <= 0.15

    @test("reverse is forbidden")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        command = _controller()
        command["linear_x"] = -0.01
        core.update_controller(command, 1.0)
        core.update_pipeline(_pipeline(), 1.0)
        core.update_feedback(0.0, 0.0, 1.0)
        assert core.evaluate(1.01).gate_reason == "reverse_forbidden"

    @test("angular direction changes through slew")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        before = core.last_output_w
        command = _controller()
        command["angular_z"] = -0.15
        core.update_controller(command, 1.4)
        core.update_feedback(0.0, 0.0, 1.4)
        core.update_pipeline(_pipeline(), 1.4)
        after = core.evaluate(1.41).gate_w
        assert after > -0.15 and abs(after - before) <= 0.031

    @test("safety stop bypasses slew")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        assert _feed_ready(core).gate_w != 0.0
        core.update_pipeline(_pipeline(valid=False), 1.5)
        decision = core.evaluate(1.51)
        assert decision.gate_w == 0.0

    @test("historical speed is never reused")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        assert _feed_ready(core).gate_v > 0.0
        core.update_controller(None, 1.5, "bad_json")
        assert core.evaluate(1.51).gate_v == 0.0

    @test("manual stop and estop remain zero")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(core)
        core.request_stop()
        assert core.evaluate(1.5).gate_v == 0.0
        other = LaneControlGateCore(GateConfig(motion_enabled=True))
        _feed_ready(other)
        other.request_estop()
        assert other.evaluate(1.5).gate_state == "ESTOP"

    @test("motion-disabled zero smoke")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=False))
        for index in range(20):
            now = 1.0 + index * 0.05
            core.update_controller(_controller(), now)
            core.update_pipeline(_pipeline(), now)
            core.update_feedback(0.0, 0.0, now)
            decision = core.evaluate(now + 0.01)
            assert decision.gate_v == 0.0 and decision.gate_w == 0.0

    @test("telemetry JSON is serializable")
    def _() -> None:
        core = LaneControlGateCore(GateConfig(motion_enabled=True))
        json.dumps(asdict(_feed_ready(core)))

    @test("Ctrl-C cleanup has a bounded zero burst")
    def _() -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        assert "except KeyboardInterrupt" in source
        assert "node.publish_zero_burst(2.0)" in source

    @test("stop and estop scripts have independent zero bursts")
    def _() -> None:
        script = (
            Path(__file__).resolve().parent / "tools" / "lane_closed_loop.sh"
        ).read_text(encoding="utf-8")
        assert "zero_burst 2.0" in script
        assert "zero_burst 5.0" in script

    @test("report JSON and CSV are generated")
    def _() -> None:
        root = Path.cwd() / f".lane_gate_report_selftest_{os.getpid()}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            log_path = root / "feedback.log"
            log_path.write_text(
                json.dumps(
                    {
                        "monotonic_sec": 1.0,
                        "gate_state": "STOPPED",
                        "gate_reason": "feedback_timeout",
                        "gate_v": 0.0,
                        "gate_w": 0.0,
                        "controller_ready": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = generate_report(log_path, root)
            assert summary["timeout_count"] == 1
            assert (root / "summary.json").is_file()
            assert (root / "summary.csv").is_file()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    failures: List[str] = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {failures[-1]}")
    print(f"self-test: {len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if not SELF_TEST and not REPORT_MODE:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from origincar_msg.msg import Data
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String


    class LaneControlGateNode(Node):
        def __init__(self) -> None:
            super().__init__("lane_control_gate")
            defaults = GateConfig()
            for name, value in asdict(defaults).items():
                self.declare_parameter(name, value)
            self.declare_parameter("control_cmd_topic", "/gxb_test/lane_control_cmd")
            self.declare_parameter("pipeline_status_topic", "/gxb_test/pipeline/status")
            self.declare_parameter("gate_status_topic", "/gxb_test/lane_control_gate/status")
            self.declare_parameter("gate_control_topic", "/gxb_test/lane_control_gate/control")
            self.declare_parameter("cmd_vel_topic", "/cmd_vel")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("telemetry_log_path", "")
            values = {name: self.get_parameter(name).value for name in asdict(defaults)}
            self.config = GateConfig(**values)
            self.config.validate()
            self.core = LaneControlGateCore(self.config)
            telemetry_text = str(
                self.get_parameter("telemetry_log_path").value
            ).strip()
            self.telemetry_path = Path(telemetry_text) if telemetry_text else None
            if self.telemetry_path is not None:
                self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            self.cmd_publisher = self.create_publisher(
                Twist, str(self.get_parameter("cmd_vel_topic").value), 10
            )
            self.status_publisher = self.create_publisher(
                String, str(self.get_parameter("gate_status_topic").value), 10
            )
            self.create_subscription(
                String,
                str(self.get_parameter("control_cmd_topic").value),
                self._controller_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("pipeline_status_topic").value),
                self._pipeline_callback,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odom_topic").value),
                self._odom_callback,
                50,
            )
            self.create_subscription(Data, "/robotvel", self._robotvel_callback, 50)
            self.create_subscription(
                String,
                str(self.get_parameter("gate_control_topic").value),
                self._control_callback,
                10,
            )
            self.feedback_times: List[float] = []
            self.robotvel_received_at = 0.0
            self.robotvel_x = 0.0
            self.robotvel_y = 0.0
            self.robotvel_z = 0.0
            self.last_log_time = 0.0
            self.last_logged_state = ""
            self.create_timer(1.0 / self.config.publish_rate_hz, self._tick)
            self.get_logger().info(
                "lane control gate started: motion_enabled="
                f"{self.config.motion_enabled} max_v_normal="
                f"{self.config.max_linear_normal:.3f} max_v_recovery="
                f"{self.config.max_linear_recovery:.3f} "
                f"max_w={self.config.max_angular:.3f}"
            )

        @staticmethod
        def _decode(text: str) -> Dict[str, Any]:
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("JSON root must be object")
            return value

        def _controller_callback(self, message: String) -> None:
            now = time.monotonic()
            try:
                self.core.update_controller(self._decode(message.data), now)
            except Exception as exc:
                self.core.update_controller(None, now, type(exc).__name__)

        def _pipeline_callback(self, message: String) -> None:
            now = time.monotonic()
            try:
                self.core.update_pipeline(self._decode(message.data), now)
            except Exception as exc:
                self.core.update_pipeline(None, now, type(exc).__name__)

        def _odom_callback(self, message: Odometry) -> None:
            now = time.monotonic()
            self.core.update_feedback(
                message.twist.twist.linear.x,
                message.twist.twist.angular.z,
                now,
            )
            self.feedback_times.append(now)
            cutoff = now - 2.0
            while self.feedback_times and self.feedback_times[0] < cutoff:
                self.feedback_times.pop(0)

        def _robotvel_callback(self, message: Data) -> None:
            self.robotvel_received_at = time.monotonic()
            self.robotvel_x = finite(message.x)
            self.robotvel_y = finite(message.y)
            self.robotvel_z = finite(message.z)

        def _control_callback(self, message: String) -> None:
            try:
                command = str(self._decode(message.data).get("command", ""))
            except Exception:
                return
            if command == "estop":
                self.core.request_estop()
            elif command == "stop":
                self.core.request_stop()

        def _feedback_hz(self) -> float:
            if len(self.feedback_times) < 2:
                return 0.0
            return (len(self.feedback_times) - 1) / (
                self.feedback_times[-1] - self.feedback_times[0]
            )

        @staticmethod
        def _metric(document: Mapping[str, Any], *keys: str) -> float:
            for key in keys:
                if key in document:
                    return finite(document.get(key))
            return 0.0

        def _telemetry(self, decision: GateDecision, now: float) -> Dict[str, Any]:
            controller = self.core.controller.document or {}
            pipeline = self.core.pipeline.document or {}
            record = asdict(decision)
            record.update(
                {
                    "monotonic_sec": now,
                    "mode": str(controller.get("mode", "")),
                    "quality": str(controller.get("quality", "")),
                    "path_valid": bool(controller.get("path_valid", False)),
                    "pipeline_valid": bool(pipeline.get("centerline_valid", False)),
                    "controller_ready": bool(controller.get("ready", False)),
                    "controller_reason": str(controller.get("reason", "")),
                    "lateral_error": finite(controller.get("lateral_error")),
                    "heading_error": finite(controller.get("heading_error")),
                    "controller_v": finite(controller.get("linear_x")),
                    "controller_w": finite(controller.get("angular_z")),
                    "path_point_count": int(finite(controller.get("path_point_count"))),
                    "path_span": finite(controller.get("path_span_m")),
                    "process_fps": self._metric(pipeline, "process_fps"),
                    "dp_ms": self._metric(
                        pipeline,
                        "single_green_dp_ms",
                        "corridor_dp_ms",
                        "dynamic_programming_ms",
                    ),
                    "processing_time_ms": self._metric(pipeline, "processing_time_ms"),
                    "odom_linear_x": self.core.feedback.linear_x,
                    "odom_angular_z": self.core.feedback.angular_z,
                    "robotvel_x": self.robotvel_x,
                    "robotvel_y": self.robotvel_y,
                    "robotvel_z": self.robotvel_z,
                    "robotvel_age_ms": (
                        (now - self.robotvel_received_at) * 1000.0
                        if self.robotvel_received_at > 0.0
                        else -1.0
                    ),
                    "feedback_hz": self._feedback_hz(),
                }
            )
            return record

        def _publish_zero(self) -> None:
            self.cmd_publisher.publish(Twist())

        def _tick(self) -> None:
            now = time.monotonic()
            try:
                decision = self.core.evaluate(now)
                command = Twist()
                command.linear.x = decision.gate_v
                command.angular.z = decision.gate_w
                self.cmd_publisher.publish(command)
                record = self._telemetry(decision, now)
                status = String()
                status.data = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                self.status_publisher.publish(status)
                if self.telemetry_path is not None:
                    with self.telemetry_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (
                    decision.gate_state != self.last_logged_state
                    or now - self.last_log_time >= 0.5
                ):
                    self.last_logged_state = decision.gate_state
                    self.last_log_time = now
                    self.get_logger().info(
                        f"{decision.gate_state} reason={decision.gate_reason} "
                        f"v={decision.gate_v:.3f} w={decision.gate_w:+.3f} "
                        f"ready={decision.ready_count}/{self.config.ready_frames}"
                    )
            except Exception as exc:
                self.core.request_estop()
                self._publish_zero()
                self.get_logger().error(f"gate exception: {type(exc).__name__}: {exc}")

        def publish_zero_burst(self, seconds: float) -> None:
            end = time.monotonic() + max(0.0, seconds)
            while rclpy.ok() and time.monotonic() < end:
                self._publish_zero()
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(0.05)


def run_zero(seconds: float, topic: str) -> None:
    rclpy.init()
    node = Node("lane_closed_loop_zero_guard")
    publisher = node.create_publisher(Twist, topic, 10)
    end = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < end:
        publisher.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.05)
    for _ in range(10):
        publisher.publish(Twist())
        time.sleep(0.02)
    node.destroy_node()
    rclpy.shutdown()


def wait_odom(timeout: float, topic: str) -> int:
    rclpy.init()
    node = Node("lane_closed_loop_feedback_probe")
    count = 0

    def callback(_message: Odometry) -> None:
        nonlocal count
        count += 1

    subscription = node.create_subscription(Odometry, topic, callback, 50)
    _ = subscription
    start = time.monotonic()
    while time.monotonic() - start < timeout and count < 20:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print(f"odom_count={count}")
    return 0 if count >= 20 else 1


def main() -> None:
    if SELF_TEST:
        run_self_test()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-log", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--wait-odom", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    args, ros_args = parser.parse_known_args()
    if args.report:
        if not args.report_log or not args.output_dir:
            raise SystemExit("--report-log and --output-dir are required")
        print(json.dumps(generate_report(Path(args.report_log), Path(args.output_dir)), indent=2))
        return
    if args.zero:
        run_zero(args.seconds, args.topic)
        return
    if args.wait_odom:
        raise SystemExit(wait_odom(args.timeout, "/odom"))
    rclpy.init(args=ros_args)
    node: Optional[LaneControlGateNode] = None
    try:
        node = LaneControlGateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        if node is not None:
            node.core.request_estop()
            node.publish_zero_burst(2.0)
        raise
    finally:
        if node is not None:
            node.core.request_estop()
            node.publish_zero_burst(2.0)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
