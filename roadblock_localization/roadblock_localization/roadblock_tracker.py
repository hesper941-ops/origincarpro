"""Small odom-frame geometric track pool with stable, non-reused IDs."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class OdomPose:
    x: float
    y: float
    yaw: float


@dataclass
class Track:
    id: int
    odom_x: float
    odom_y: float
    last_valid_time: float
    last_seen_time: float


def base_to_odom(x: float, y: float, pose: OdomPose) -> Tuple[float, float]:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return (
        pose.x + cosine * x - sine * y,
        pose.y + sine * x + cosine * y,
    )


def odom_to_base(x: float, y: float, pose: OdomPose) -> Tuple[float, float]:
    dx = x - pose.x
    dy = y - pose.y
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


class RoadblockTracker:
    def __init__(
        self,
        association_max_distance_m: float,
        track_ttl_sec: float,
        track_min_x_m: float,
        track_max_distance_m: float,
    ):
        if association_max_distance_m <= 0.0 or track_ttl_sec <= 0.0:
            raise ValueError("association gate and track TTL must be positive")
        if track_max_distance_m <= 0.0:
            raise ValueError("track_max_distance_m must be positive")
        self.association_max_distance_m = float(association_max_distance_m)
        self.track_ttl_sec = float(track_ttl_sec)
        self.track_min_x_m = float(track_min_x_m)
        self.track_max_distance_m = float(track_max_distance_m)
        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> Sequence[Track]:
        return tuple(self._tracks)

    def _prune(self, pose: OdomPose, now_sec: float) -> None:
        kept = []
        for track in self._tracks:
            x, y = odom_to_base(track.odom_x, track.odom_y, pose)
            if now_sec - track.last_seen_time > self.track_ttl_sec:
                continue
            if x < self.track_min_x_m or math.hypot(x, y) > self.track_max_distance_m:
                continue
            kept.append(track)
        self._tracks = kept

    def update(
        self,
        reliable_detections_base: Iterable[Tuple[float, float]],
        pose: OdomPose,
        now_sec: float,
    ) -> None:
        detections = [(float(x), float(y)) for x, y in reliable_detections_base]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in detections):
            raise ValueError("detections must be finite")
        self._prune(pose, now_sec)

        predicted = [odom_to_base(track.odom_x, track.odom_y, pose) for track in self._tracks]
        candidates = []
        for detection_index, detection in enumerate(detections):
            for track_index, track_position in enumerate(predicted):
                distance = math.hypot(
                    detection[0] - track_position[0], detection[1] - track_position[1]
                )
                if distance <= self.association_max_distance_m:
                    candidates.append((distance, detection_index, track_index))
        candidates.sort()

        matched_detections = set()
        matched_tracks = set()
        for _, detection_index, track_index in candidates:
            if detection_index in matched_detections or track_index in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_index)
            track = self._tracks[track_index]
            track.odom_x, track.odom_y = base_to_odom(*detections[detection_index], pose)
            track.last_valid_time = now_sec
            track.last_seen_time = now_sec

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            odom_x, odom_y = base_to_odom(*detection, pose)
            self._tracks.append(
                Track(
                    id=self._next_id,
                    odom_x=odom_x,
                    odom_y=odom_y,
                    last_valid_time=now_sec,
                    last_seen_time=now_sec,
                )
            )
            self._next_id += 1

    def snapshot(self, pose: OdomPose, now_sec: float) -> List[dict]:
        self._prune(pose, now_sec)
        result = []
        for track in sorted(self._tracks, key=lambda item: item.id):
            x, y = odom_to_base(track.odom_x, track.odom_y, pose)
            result.append({"id": track.id, "x": x, "y": y})
        return result
