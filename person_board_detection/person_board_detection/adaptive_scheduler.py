"""Inference throttling with EMA, hysteresis, and confirmed transitions."""

import time
from typing import Dict, Optional


class AdaptiveScheduler:
    def __init__(
        self,
        frequencies: Dict[str, float],
        ema_alpha: float,
        confirm_count: int,
        middle_enter: float,
        middle_exit: float,
        near_enter: float,
        near_exit: float,
    ) -> None:
        self.frequencies = frequencies
        self.ema_alpha = ema_alpha
        self.confirm_count = confirm_count
        self.middle_enter = middle_enter
        self.middle_exit = middle_exit
        self.near_enter = near_enter
        self.near_exit = near_exit
        self.state = "SEARCH"
        self.smoothed_short_side = 0.0
        self.next_inference_time = 0.0
        self._candidate: Optional[str] = None
        self._candidate_count = 0

    def reset(self) -> None:
        self.state = "SEARCH"
        self.smoothed_short_side = 0.0
        self.next_inference_time = 0.0
        self._candidate = None
        self._candidate_count = 0

    def should_infer(self, now: Optional[float] = None) -> bool:
        return (time.monotonic() if now is None else now) >= self.next_inference_time

    def mark_inference(
        self, state: Optional[str] = None, now: Optional[float] = None
    ) -> None:
        current_state = self.state if state is None else state
        hz = self.frequencies[current_state]
        self.next_inference_time = (time.monotonic() if now is None else now) + 1.0 / hz

    def current_hz(self, state: Optional[str] = None) -> float:
        return self.frequencies[self.state if state is None else state]

    def target_lost(self) -> str:
        self.state = "SEARCH"
        self.smoothed_short_side = 0.0
        self._candidate = None
        self._candidate_count = 0
        return self.state

    def update(self, measured_short_side: float) -> str:
        if self.smoothed_short_side <= 0.0:
            self.smoothed_short_side = measured_short_side
        else:
            self.smoothed_short_side = (
                self.ema_alpha * measured_short_side
                + (1.0 - self.ema_alpha) * self.smoothed_short_side
            )
        desired = self._desired_state(self.smoothed_short_side)
        if desired == self.state:
            self._candidate = None
            self._candidate_count = 0
        else:
            if desired != self._candidate:
                self._candidate = desired
                self._candidate_count = 1
            else:
                self._candidate_count += 1
            if self._candidate_count >= self.confirm_count:
                self.state = desired
                self._candidate = None
                self._candidate_count = 0
        return self.state

    def _desired_state(self, size: float) -> str:
        if self.state == "TRACK_NEAR":
            if size >= self.near_exit:
                return "TRACK_NEAR"
            return "TRACK_MIDDLE" if size >= self.middle_exit else "TRACK_FAR"
        if self.state == "TRACK_MIDDLE":
            if size >= self.near_enter:
                return "TRACK_NEAR"
            if size < self.middle_exit:
                return "TRACK_FAR"
            return "TRACK_MIDDLE"
        if size >= self.near_enter:
            return "TRACK_NEAR"
        if size >= self.middle_enter:
            return "TRACK_MIDDLE"
        return "TRACK_FAR"
