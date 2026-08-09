"""Short-term base_link association memory for current visual measurements."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Track:
    id: int
    x: float
    y: float
    last_seen_time: float


class RoadblockTracker:
    """Assign stable local IDs without publishing unobserved historical tracks."""

    def __init__(self, association_max_distance_m: float, track_ttl_sec: float):
        if association_max_distance_m <= 0.0 or track_ttl_sec <= 0.0:
            raise ValueError("association gate and track TTL must be positive")
        self.association_max_distance_m = float(association_max_distance_m)
        self.track_ttl_sec = float(track_ttl_sec)
        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> Sequence[Track]:
        return tuple(self._tracks)

    def _prune(self, now_sec: float) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if now_sec - track.last_seen_time <= self.track_ttl_sec
        ]

    def associate_current_measurements(
        self,
        measurements: Iterable[Tuple[float, float]],
        now_sec: float,
    ) -> List[dict]:
        """Associate and return only measurements observed in the current frame."""
        current = [(float(x), float(y)) for x, y in measurements]
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in current):
            raise ValueError("measurements must be finite")
        self._prune(now_sec)

        candidates = []
        for measurement_index, measurement in enumerate(current):
            for track_index, track in enumerate(self._tracks):
                distance = math.hypot(
                    measurement[0] - track.x, measurement[1] - track.y
                )
                if distance <= self.association_max_distance_m:
                    candidates.append((distance, measurement_index, track_index))
        candidates.sort()

        matched_measurements = set()
        matched_tracks = set()
        measurement_ids = {}
        for _, measurement_index, track_index in candidates:
            if measurement_index in matched_measurements or track_index in matched_tracks:
                continue
            matched_measurements.add(measurement_index)
            matched_tracks.add(track_index)
            track = self._tracks[track_index]
            track.x, track.y = current[measurement_index]
            track.last_seen_time = now_sec
            measurement_ids[measurement_index] = track.id

        for measurement_index, measurement in enumerate(current):
            if measurement_index in matched_measurements:
                continue
            track = Track(
                id=self._next_id,
                x=measurement[0],
                y=measurement[1],
                last_seen_time=now_sec,
            )
            self._tracks.append(track)
            measurement_ids[measurement_index] = track.id
            self._next_id += 1

        outputs = [
            {"id": measurement_ids[index], "x": measurement[0], "y": measurement[1]}
            for index, measurement in enumerate(current)
        ]
        return sorted(outputs, key=lambda item: item["id"])
