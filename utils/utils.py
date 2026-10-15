import argparse
import cv2

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

def draw_face_box(frame, face_landmarks, label="Face detected", alert=False):
    h, w = frame.shape[:2]

    xs = [int(point.x * w) for point in face_landmarks]
    ys = [int(point.y * h) for point in face_landmarks]

    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))

    color = (0, 0, 255) if alert else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
    )