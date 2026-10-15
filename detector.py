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
            face_regions = [
                (face["id"], face["box"]) for face in ear_result["faces"]
            ]
            cnn_result = cnn_detector.predict(frame, now, face_regions)
            cnn_faces = {
                face["id"]: face for face in cnn_result["faces"]
            }

            face_alerts = []
            for face in ear_result["faces"]:
                cnn_face = cnn_faces[face["id"]]
                face_alert = face["alert"] or cnn_face["alert"]
                face_alerts.append(face_alert)

            alert = any(face_alerts)
            face_count = len(ear_result["faces"])

            if alert:
                status = f"DROWSINESS ALERT - {sum(face_alerts)}/{face_count} FACE"
            elif face_count == 0:
                status = "NO FACE DETECTED"
            else:
                status = "MONITORING 1 FACE"

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

            probabilities = [
                face["probability"] for face in cnn_result["faces"]
            ]
            cnn_text = (
                "CNN fatigue: --"
                if not probabilities
                else f"CNN fatigue: {max(probabilities):.1%} max"
            )

            cv2.putText(
                frame,
                cnn_text,
                (38, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255) if alert else (229, 234, 231),
                2,
                cv2.LINE_AA,
            )

            for face, face_alert in zip(ear_result["faces"], face_alerts):
                probability = cnn_faces[face["id"]]["probability"]
                label = (
                    f"P{face['id']} EAR {face['ear']:.2f} "
                    f"CNN {probability:.0%}"
                )
                if face_alert:
                    label += " ALERT"
                draw_face_box(frame, face["landmarks"], label, face_alert)

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