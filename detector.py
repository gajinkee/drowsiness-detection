"""USB webcam student drowsiness demonstration using eye closure duration."""

from __future__ import annotations

import sys
import time
import cv2

from ear_detector import EARDetector
from CNN.camera_cnn import CNNDetector
from utils.utils import draw_hud, parse_args, draw_face_box

def main() -> int:
    args = parse_args()
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        return 1

    ear_detector = None
    cnn_detector = None

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ear_detector = EARDetector(args.ear_threshold, args.closed_seconds)
        cnn_detector = CNNDetector()
        alert_active = False

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            now = time.monotonic()

            ear_result = ear_detector.predict(frame, now)
            cnn_result = cnn_detector.predict(frame, now)
            if ear_result["landmarks"] is not None:
                draw_face_box(frame, ear_result["landmarks"])

            ear_alert = ear_result["alert"]
            cnn_alert = cnn_result["alert"]
            alert = ear_alert or cnn_alert

            if alert:
                status = "DROWSINESS ALERT - TAKE A BREAK"
            elif not ear_result["face_detected"] and cnn_result["probability"] is None:
                status = "NO FACE DETECTED"
            else:
                status = "MONITORING"

            if alert and not alert_active:
                sys.stdout.write("\a")
                sys.stdout.flush()
            alert_active = alert

            draw_hud(
                frame,
                status,
                ear_result["ear"],
                ear_result["closure"],
                alert,
            )

            probability = cnn_result["probability"]
            cnn_text = (
                "CNN fatigue: --"
                if probability is None
                else f"CNN fatigue: {probability:.1%}"
            )

            cv2.putText(
                frame,
                cnn_text,
                (38, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255) if cnn_alert else (229, 234, 231),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Student Drowsiness POC", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if ear_detector is not None:
            ear_detector.close()
        if cnn_detector is not None:
            cnn_detector.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())