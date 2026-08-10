"""Pure-Python static roadblock tracking and robust position filtering."""

from dataclasses import dataclass, field
import math
import statistics
from typing import List, Optional, Sequence, Tuple


TENTATIVE = 'TENTATIVE'
CONFIRMED = 'CONFIRMED'


@dataclass(frozen=True)
class Measurement:
    raw_id: int
    x: float
    y: float


@dataclass
class Track:
    track_id: int
    state: str
    stable_x: float
    stable_y: float
    history: List[Tuple[float, float]] = field(default_factory=list)
    tentative_hits: int = 1
    tentative_age_frames: int = 1
    last_measurement_stamp: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class DecisionEvent:
    decision: str
    measurement: Optional[Measurement]
    matched_track_id: Optional[int]
    track_state_before: str = ''
    stable_x_before: Optional[float] = None
    stable_y_before: Optional[float] = None
    distance_m: Optional[float] = None
    stable_x_after: Optional[float] = None
    stable_y_after: Optional[float] = None
    track_state_after: str = ''


@dataclass(frozen=True)
class FrameResult:
    duplicate_frame: bool
    events: Tuple[DecisionEvent, ...]


class RoadblockMapFilterCore:
    """Track static obstacles without vehicle pose, TTL, or motion models."""

    def __init__(
        self,
        association_gate_m: float = 0.15,
        update_gate_m: float = 0.07,
        candidate_confirm_gate_m: float = 0.08,
        new_track_suppression_gate_m: float = 0.18,
        reacquire_gate_m: float = 0.25,
        history_size: int = 5,
        tentative_window_frames: int = 5,
        tentative_required_hits: int = 3,
    ):
        self.association_gate_m = float(association_gate_m)
        self.update_gate_m = float(update_gate_m)
        self.candidate_confirm_gate_m = float(candidate_confirm_gate_m)
        self.new_track_suppression_gate_m = float(
            new_track_suppression_gate_m
        )
        self.reacquire_gate_m = float(reacquire_gate_m)
        self.history_size = int(history_size)
        self.tentative_window_frames = int(tentative_window_frames)
        self.tentative_required_hits = int(tentative_required_hits)
        self._validate_parameters()
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.last_processed_stamp: Optional[Tuple[int, int]] = None

    def _validate_parameters(self) -> None:
        if self.update_gate_m < 0.0:
            raise ValueError('update_gate_m must be non-negative')
        if self.candidate_confirm_gate_m < 0.0:
            raise ValueError('candidate_confirm_gate_m must be non-negative')
        if self.new_track_suppression_gate_m < 0.0:
            raise ValueError(
                'new_track_suppression_gate_m must be non-negative'
            )
        if self.reacquire_gate_m < 0.0:
            raise ValueError('reacquire_gate_m must be non-negative')
        if self.association_gate_m < max(
            self.update_gate_m, self.candidate_confirm_gate_m
        ):
            raise ValueError('association_gate_m must cover both update gates')
        if self.association_gate_m > self.reacquire_gate_m:
            raise ValueError(
                'reacquire_gate_m must cover association_gate_m'
            )
        if self.history_size < 1:
            raise ValueError('history_size must be at least 1')
        if self.tentative_window_frames < 1:
            raise ValueError('tentative_window_frames must be at least 1')
        if not 1 <= self.tentative_required_hits <= self.tentative_window_frames:
            raise ValueError('tentative_required_hits must fit the frame window')

    @staticmethod
    def _distance(measurement: Measurement, track: Track) -> float:
        return math.hypot(
            measurement.x - track.stable_x,
            measurement.y - track.stable_y,
        )

    @staticmethod
    def _finite(measurement: Measurement) -> bool:
        return math.isfinite(measurement.x) and math.isfinite(measurement.y)

    def _append_history(self, track: Track, measurement: Measurement) -> None:
        track.history.append((measurement.x, measurement.y))
        if len(track.history) > self.history_size:
            del track.history[:-self.history_size]
        track.stable_x = float(statistics.median(x for x, _ in track.history))
        track.stable_y = float(statistics.median(y for _, y in track.history))

    def _new_tentative(
        self,
        measurement: Measurement,
        stamp: Optional[Tuple[int, int]],
    ) -> Track:
        track = Track(
            track_id=self.next_track_id,
            state=TENTATIVE,
            stable_x=measurement.x,
            stable_y=measurement.y,
            history=[(measurement.x, measurement.y)],
            tentative_hits=1,
            tentative_age_frames=1,
            last_measurement_stamp=stamp,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def process_frame(
        self,
        measurements: Sequence[Measurement],
        stamp: Optional[Tuple[int, int]] = None,
    ) -> FrameResult:
        """Process one independent measurement frame.

        A ``None`` stamp represents an invalid/zero upstream timestamp and is
        deliberately never deduplicated.
        """
        if stamp is not None and stamp == self.last_processed_stamp:
            events = tuple(
                DecisionEvent('DUPLICATE_FRAME', measurement, None)
                for measurement in measurements
            )
            return FrameResult(True, events)
        if stamp is not None:
            self.last_processed_stamp = stamp

        valid_measurements: List[Measurement] = []
        events: List[DecisionEvent] = []
        for measurement in measurements:
            if self._finite(measurement):
                valid_measurements.append(measurement)
            else:
                events.append(DecisionEvent(
                    'INVALID_MEASUREMENT', measurement, None
                ))

        tentative_ids_at_frame_start = {
            track.track_id for track in self.tracks if track.state == TENTATIVE
        }
        confirmed_candidates = []
        for measurement_index, measurement in enumerate(valid_measurements):
            for track in self.tracks:
                if track.state != CONFIRMED:
                    continue
                distance = self._distance(measurement, track)
                if distance <= self.association_gate_m:
                    confirmed_candidates.append(
                        (distance, track.track_id, measurement_index, track)
                    )
        confirmed_candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )

        used_measurements = set()
        used_tracks = set()
        confirmed_matches = []
        for distance, track_id, measurement_index, track in confirmed_candidates:
            if measurement_index in used_measurements or track_id in used_tracks:
                continue
            used_measurements.add(measurement_index)
            used_tracks.add(track_id)
            confirmed_matches.append((measurement_index, track, distance))

        for measurement_index, track, distance in confirmed_matches:
            measurement = valid_measurements[measurement_index]
            stable_before = (track.stable_x, track.stable_y)
            if distance <= self.update_gate_m:
                self._append_history(track, measurement)
                track.last_measurement_stamp = stamp
                decision = 'ACCEPT'
            else:
                decision = 'SUSPECT'
            events.append(DecisionEvent(
                decision=decision,
                measurement=measurement,
                matched_track_id=track.track_id,
                track_state_before=CONFIRMED,
                stable_x_before=stable_before[0],
                stable_y_before=stable_before[1],
                distance_m=distance,
                stable_x_after=track.stable_x,
                stable_y_after=track.stable_y,
                track_state_after=track.state,
            ))

        unmatched_confirmed_tracks = [
            track for track in self.tracks
            if track.state == CONFIRMED and track.track_id not in used_tracks
        ]
        reacquire_candidates = []
        for measurement_index, measurement in enumerate(valid_measurements):
            if measurement_index in used_measurements:
                continue
            for track in unmatched_confirmed_tracks:
                distance = self._distance(measurement, track)
                if (
                    self.association_gate_m < distance
                    <= self.reacquire_gate_m
                ):
                    reacquire_candidates.append(
                        (distance, measurement_index, track.track_id, track)
                    )
        reacquire_candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )

        for distance, measurement_index, track_id, track in reacquire_candidates:
            if measurement_index in used_measurements or track_id in used_tracks:
                continue
            used_measurements.add(measurement_index)
            used_tracks.add(track_id)
            measurement = valid_measurements[measurement_index]
            events.append(DecisionEvent(
                decision='REACQUIRE_SUSPECT',
                measurement=measurement,
                matched_track_id=track.track_id,
                track_state_before=CONFIRMED,
                stable_x_before=track.stable_x,
                stable_y_before=track.stable_y,
                distance_m=distance,
                stable_x_after=track.stable_x,
                stable_y_after=track.stable_y,
                track_state_after=CONFIRMED,
            ))

        tentative_candidates = []
        for measurement_index, measurement in enumerate(valid_measurements):
            if measurement_index in used_measurements:
                continue
            for track in self.tracks:
                if track.state != TENTATIVE:
                    continue
                distance = self._distance(measurement, track)
                if distance <= self.association_gate_m:
                    tentative_candidates.append(
                        (distance, track.track_id, measurement_index, track)
                    )
        tentative_candidates.sort(
            key=lambda item: (item[0], item[1], item[2])
        )

        tentative_matches = []
        for distance, track_id, measurement_index, track in tentative_candidates:
            if measurement_index in used_measurements or track_id in used_tracks:
                continue
            used_measurements.add(measurement_index)
            used_tracks.add(track_id)
            tentative_matches.append((measurement_index, track, distance))

        for measurement_index, track, distance in tentative_matches:
            measurement = valid_measurements[measurement_index]
            stable_before = (track.stable_x, track.stable_y)
            if distance <= self.candidate_confirm_gate_m:
                track.tentative_hits += 1
                self._append_history(track, measurement)
                track.last_measurement_stamp = stamp
                if track.tentative_hits >= self.tentative_required_hits:
                    track.state = CONFIRMED
                    decision = 'CONFIRM'
                else:
                    decision = 'TENTATIVE_HIT'
            else:
                decision = 'SUSPECT'
            events.append(DecisionEvent(
                decision=decision,
                measurement=measurement,
                matched_track_id=track.track_id,
                track_state_before=TENTATIVE,
                stable_x_before=stable_before[0],
                stable_y_before=stable_before[1],
                distance_m=distance,
                stable_x_after=track.stable_x,
                stable_y_after=track.stable_y,
                track_state_after=track.state,
            ))

        for measurement_index, measurement in enumerate(valid_measurements):
            if measurement_index in used_measurements:
                continue
            confirmed_tracks = (
                track for track in self.tracks if track.state == CONFIRMED
            )
            nearest_confirmed = min(
                (
                    (self._distance(measurement, track), track.track_id, track)
                    for track in confirmed_tracks
                ),
                default=None,
                key=lambda item: (item[0], item[1]),
            )
            if (
                nearest_confirmed is not None
                and nearest_confirmed[0]
                <= self.new_track_suppression_gate_m
            ):
                distance, _, track = nearest_confirmed
                events.append(DecisionEvent(
                    decision='SUPPRESS_NEAR_CONFIRMED',
                    measurement=measurement,
                    matched_track_id=track.track_id,
                    track_state_before=CONFIRMED,
                    stable_x_before=track.stable_x,
                    stable_y_before=track.stable_y,
                    distance_m=distance,
                    stable_x_after=track.stable_x,
                    stable_y_after=track.stable_y,
                    track_state_after=CONFIRMED,
                ))
                continue
            track = self._new_tentative(measurement, stamp)
            events.append(DecisionEvent(
                decision='NEW_TENTATIVE',
                measurement=measurement,
                matched_track_id=track.track_id,
                track_state_after=track.state,
                stable_x_after=track.stable_x,
                stable_y_after=track.stable_y,
            ))

        for track in list(self.tracks):
            if track.track_id in tentative_ids_at_frame_start and track.state == TENTATIVE:
                track.tentative_age_frames += 1
            if track.state == CONFIRMED and track.track_id not in used_tracks:
                events.append(DecisionEvent(
                    decision='UNMATCHED_CONFIRMED',
                    measurement=None,
                    matched_track_id=track.track_id,
                    track_state_before=CONFIRMED,
                    stable_x_before=track.stable_x,
                    stable_y_before=track.stable_y,
                    stable_x_after=track.stable_x,
                    stable_y_after=track.stable_y,
                    track_state_after=CONFIRMED,
                ))

        retained_tracks = []
        for track in self.tracks:
            if (
                track.state == TENTATIVE
                and track.tentative_age_frames >= self.tentative_window_frames
                and track.tentative_hits < self.tentative_required_hits
            ):
                events.append(DecisionEvent(
                    decision='TENTATIVE_EXPIRE',
                    measurement=None,
                    matched_track_id=track.track_id,
                    track_state_before=TENTATIVE,
                    stable_x_before=track.stable_x,
                    stable_y_before=track.stable_y,
                ))
                continue
            retained_tracks.append(track)
        self.tracks = retained_tracks
        return FrameResult(False, tuple(events))

    def confirmed_tracks(self) -> Tuple[Track, ...]:
        return tuple(sorted(
            (track for track in self.tracks if track.state == CONFIRMED),
            key=lambda track: track.track_id,
        ))

    def reset(self) -> int:
        count = len(self.tracks)
        self.tracks.clear()
        self.next_track_id = 1
        self.last_processed_stamp = None
        return count
