"""USB webcam student drowsiness demonstration using eye closure duration."""

from __future__ import annotations

import argparse
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a local alert when eyes appear continuously closed."
    )
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="Eye aspect ratio below which eyes count as closed (default: 0.21)",
    )
    parser.add_argument(
        "--closed-seconds",
        type=float,
        default=1.5,
        help="Seconds of continuous closure before alerting (default: 1.5)",
    )
    args = parser.parse_args()
    if args.ear_threshold <= 0:
        parser.error("--ear-threshold must be greater than zero")
    if args.closed_seconds <= 0:
        parser.error("--closed-seconds must be greater than zero")
    return args


def eye_points(
    landmarks: Sequence[Landmark], indices: tuple[int, ...], width: int, height: int
) -> list[tuple[float, float]]:
    return [
        (landmarks[index].x * width, landmarks[index].y * height)
        for index in indices
    ]


def draw_hud(frame, status: str, ear: float | None, closure: float, alert: bool) -> None:
    height, width = frame.shape[:2]
    color = (42, 48, 226) if alert else (54, 194, 128)
    cv2.rectangle(frame, (18, 18), (width - 18, 122), (18, 25, 29), -1)
    cv2.rectangle(frame, (18, 18), (width - 18, 122), color, 2)
    cv2.putText(
        frame,
        status,
        (38, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    ear_text = "--" if ear is None else f"{ear:.3f}"
    cv2.putText(
        frame,
        f"Eye openness (EAR): {ear_text}    Closed: {closure:.1f}s",
        (38, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (229, 234, 231),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Q / Esc: quit  |  Local processing only; frames are not saved",
        (24, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (229, 234, 231),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(
            f"Could not open camera {args.camera}. Check the USB connection or try --camera 1.",
            file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = ClosureTracker(args.ear_threshold, args.closed_seconds)
    face_mesh_api = mp.solutions.face_mesh
    alert_active = False

    try:
        with face_mesh_api.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera stopped returning frames.", file=sys.stderr)
                    return 1

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                ear = None
                alert = False

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_ear = eye_aspect_ratio(
                        eye_points(landmarks, LEFT_EYE, width, height)
                    )
                    right_ear = eye_aspect_ratio(
                        eye_points(landmarks, RIGHT_EYE, width, height)
                    )
                    ear = (left_ear + right_ear) / 2
                    closure_ear = max(left_ear, right_ear)
                    alert = tracker.update(closure_ear, now)
                    status = "DROWSINESS ALERT - TAKE A BREAK" if alert else "MONITORING"
                else:
                    tracker.update(args.ear_threshold, now)
                    status = "NO FACE DETECTED - FACE THE CAMERA"

                closure = tracker.duration(now)
                if alert and not alert_active:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                alert_active = alert

                draw_hud(frame, status, ear, closure, alert)
                cv2.imshow("Student Drowsiness POC", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())