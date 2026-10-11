
import time
from typing import Protocol, Sequence

import cv2
import mediapipe as mp

from drowsiness_logic import ClosureTracker, eye_aspect_ratio


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


class Landmark(Protocol):
    x: float
    y: float


def eye_points(
    landmarks: Sequence[Landmark],
    indices: tuple[int, ...],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    return [
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in indices
    ]


class EARDetector:
    def __init__(self, threshold=0.21, closed_seconds=1):
        self.threshold = threshold
        self.tracker = ClosureTracker(threshold, closed_seconds)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def predict(self, frame, now=None):
        if now is None:
            now = time.monotonic()

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # Reset the closure timer when no face is detected.
            self.tracker.reset()
            return {
                "ear": None,
                "closure": 0.0,
                "alert": False,
                "face_detected": False,
                "landmarks": None,
            }

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(
            eye_points(landmarks, LEFT_EYE, width, height)
        )
        right_ear = eye_aspect_ratio(
            eye_points(landmarks, RIGHT_EYE, width, height)
        )

        ear = (left_ear + right_ear) / 2
        closure_ear = max(left_ear, right_ear)
        alert = self.tracker.update(closure_ear, now)

        return {
            "ear": ear,
            "closure": self.tracker.duration(now),
            "alert": alert,
            "face_detected": True,
            "landmarks": results.multi_face_landmarks[0],
        }

    def close(self):
        self.face_mesh.close()
