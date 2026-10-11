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