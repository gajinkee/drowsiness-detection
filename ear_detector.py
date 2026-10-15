
import time
from typing import Protocol, Sequence

import cv2

from drowsiness_logic import ClosureTracker, FaceTrackManager, eye_aspect_ratio
from mediapipe_tasks import (
    create_face_landmarker,
    next_video_timestamp_ms,
    to_mp_image,
)


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
        self.closed_seconds = closed_seconds
        self.tracks = FaceTrackManager()
        self.trackers = {}
        self.face_landmarker = create_face_landmarker()
        self.last_timestamp_ms = None

    def predict(self, frame, now=None):
        if now is None:
            now = time.monotonic()

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = to_mp_image(rgb)
        timestamp_ms = next_video_timestamp_ms(now, self.last_timestamp_ms)
        self.last_timestamp_ms = timestamp_ms
        results = self.face_landmarker.detect_for_video(image, timestamp_ms)

        detected_faces = results.face_landmarks
        if not detected_faces:
            self.trackers.clear()
            self.tracks.assign([])
            return {
                "ear": None,
                "closure": 0.0,
                "alert": False,
                "face_detected": False,
                "landmarks": None,
                "faces": [],
            }

        centers = []
        for landmarks in detected_faces:
            points = landmarks
            center_x = sum(point.x for point in points) / len(points)
            center_y = sum(point.y for point in points) / len(points)
            centers.append((center_x, center_y))
        face_ids = self.tracks.assign(centers)
        self.trackers = {
            face_id: self.trackers.get(
                face_id, ClosureTracker(self.threshold, self.closed_seconds)
            )
            for face_id in face_ids
        }

        faces = []
        for face_id, landmarks in zip(face_ids, detected_faces):
            left_ear = eye_aspect_ratio(
                eye_points(landmarks, LEFT_EYE, width, height)
            )
            right_ear = eye_aspect_ratio(
                eye_points(landmarks, RIGHT_EYE, width, height)
            )
            ear = (left_ear + right_ear) / 2
            tracker = self.trackers[face_id]
            alert = tracker.update(max(left_ear, right_ear), now)

            xs = [point.x for point in landmarks]
            ys = [point.y for point in landmarks]
            padding_x = (max(xs) - min(xs)) * 0.12
            padding_y = (max(ys) - min(ys)) * 0.12
            x1 = max(0, int((min(xs) - padding_x) * width))
            y1 = max(0, int((min(ys) - padding_y) * height))
            x2 = min(width, int((max(xs) + padding_x) * width))
            y2 = min(height, int((max(ys) + padding_y) * height))

            faces.append({
                "id": face_id,
                "ear": ear,
                "closure": tracker.duration(now),
                "alert": alert,
                "landmarks": landmarks,
                "box": (x1, y1, x2, y2),
            })

        return {
            "ear": sum(face["ear"] for face in faces) / len(faces),
            "closure": max(face["closure"] for face in faces),
            "alert": any(face["alert"] for face in faces),
            "face_detected": True,
            "landmarks": faces[0]["landmarks"],
            "faces": faces,
        }

    def close(self):
        self.face_landmarker.close()
