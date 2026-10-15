"""Dependency-free eye-closure logic for the webcam POC."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

Point = tuple[float, float]


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """Calculate EAR from six points ordered around an eye."""
    if len(points) != 6:
        raise ValueError("EAR requires exactly six eye landmarks")

    horizontal = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
    if horizontal == 0:
        raise ValueError("The eye's horizontal landmark distance cannot be zero")

    upper_lower_one = hypot(
        points[1][0] - points[5][0], points[1][1] - points[5][1]
    )
    upper_lower_two = hypot(
        points[2][0] - points[4][0], points[2][1] - points[4][1]
    )
    return (upper_lower_one + upper_lower_two) / (2 * horizontal)


@dataclass
class ClosureTracker:
    """Detect eye closure that remains below threshold for a set duration."""

    ear_threshold: float = 0.21
    closed_seconds: float = 1.5
    _closed_since: float | None = None

    def update(self, ear: float, now: float) -> bool:
        if ear >= self.ear_threshold:
            self._closed_since = None
            return False

        if self._closed_since is None:
            self._closed_since = now
        return now - self._closed_since >= self.closed_seconds

    def duration(self, now: float) -> float:
        if self._closed_since is None:
            return 0.0
        return max(0.0, now - self._closed_since)

    def reset(self):
        self._closed_since = None


class FaceTrackManager:
    """Associate nearby face centers with stable IDs across adjacent frames."""

    def __init__(self, max_distance: float = 0.2):
        self.max_distance = max_distance
        self._tracks: dict[int, Point] = {}
        self._next_id = 1

    def assign(self, centers: Sequence[Point]) -> list[int]:
        matches = []
        for index, center in enumerate(centers):
            for face_id, previous in self._tracks.items():
                distance = hypot(center[0] - previous[0], center[1] - previous[1])
                if distance <= self.max_distance:
                    matches.append((distance, index, face_id))

        assigned: dict[int, int] = {}
        used_ids: set[int] = set()
        for _, index, face_id in sorted(matches):
            if index not in assigned and face_id not in used_ids:
                assigned[index] = face_id
                used_ids.add(face_id)

        for index in range(len(centers)):
            if index not in assigned:
                assigned[index] = self._next_id
                self._next_id += 1

        self._tracks = {
            assigned[index]: center for index, center in enumerate(centers)
        }
        return [assigned[index] for index in range(len(centers))]