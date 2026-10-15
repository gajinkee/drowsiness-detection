
import cv2
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from drowsiness_logic import FaceTrackManager
from mediapipe_tasks import (
    VIDEO_MODE,
    create_face_detector,
    next_video_timestamp_ms,
    to_mp_image,
)


MODEL_PATH = "models/drowsiness_cnn_20261011_184723.pth"
THRESHOLD = 0.65
ALERT_SECONDS = 1.5


class CNNDetector:
    def __init__(
        self,
        model_path=MODEL_PATH,
        threshold=THRESHOLD,
        alert_seconds=ALERT_SECONDS,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        classes = checkpoint["classes"]
        self.class_to_idx = checkpoint["class_to_idx"]

        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            len(classes),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        self.fatigue_idx = self.class_to_idx["fatigue"]
        self.transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        self.threshold = threshold
        self.alert_seconds = alert_seconds
        self.drowsy_since = {}
        self.face_tracks = FaceTrackManager()
        self.detector = create_face_detector(VIDEO_MODE)
        self.last_timestamp_ms = None

    def predict(self, frame, now=None, face_regions=None):
        """Return predictions for face regions, or detect regions from the frame."""
        if now is None:
            now = time.monotonic()

        h, w = frame.shape[:2]
        if face_regions is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = next_video_timestamp_ms(
                now, self.last_timestamp_ms
            )
            self.last_timestamp_ms = timestamp_ms
            results = self.detector.detect_for_video(
                to_mp_image(rgb), timestamp_ms
            )
            detections = results.detections
            boxes = []
            if detections:
                detection = max(
                    detections,
                    key=lambda item: (
                        item.bounding_box.width * item.bounding_box.height
                    ),
                )
                box = detection.bounding_box
                x1 = max(0, box.origin_x)
                y1 = max(0, box.origin_y)
                x2 = min(w, box.origin_x + box.width)
                y2 = min(h, box.origin_y + box.height)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

            centers = [
                ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h))
                for x1, y1, x2, y2 in boxes
            ]
            face_ids = self.face_tracks.assign(centers)
            face_regions = list(zip(face_ids, boxes))
            rgb_for_crops = rgb
        else:
            rgb_for_crops = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        valid_regions = [
            (face_id, box)
            for face_id, box in face_regions
            if box[2] > box[0] and box[3] > box[1]
        ]
        if len(valid_regions) > 1:
            valid_regions = [
                max(
                    valid_regions,
                    key=lambda region: (
                        (region[1][2] - region[1][0])
                        * (region[1][3] - region[1][1])
                    ),
                )
            ]
        active_ids = {face_id for face_id, _ in valid_regions}
        self.drowsy_since = {
            face_id: since
            for face_id, since in self.drowsy_since.items()
            if face_id in active_ids
        }

        if not valid_regions:
            return {
                "label": "NO FACE",
                "probability": None,
                "drowsy": False,
                "alert": False,
                "box": None,
                "faces": [],
            }

        images = [
            self.transform(Image.fromarray(rgb_for_crops[y1:y2, x1:x2]))
            for _, (x1, y1, x2, y2) in valid_regions
        ]
        batch = torch.stack(images).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)

        faces = []
        for (face_id, box), probs in zip(valid_regions, probabilities):
            probability = probs[self.fatigue_idx].item()
            drowsy = probability >= self.threshold
            if drowsy:
                if face_id not in self.drowsy_since:
                    self.drowsy_since[face_id] = now
            else:
                self.drowsy_since.pop(face_id, None)

            alert = (
                face_id in self.drowsy_since
                and now - self.drowsy_since[face_id] >= self.alert_seconds
            )
            faces.append({
                "id": face_id,
                "label": "DROWSINESS ALERT" if alert else (
                    "DROWSY" if drowsy else "ACTIVE"
                ),
                "probability": probability,
                "drowsy": drowsy,
                "alert": alert,
                "box": box,
            })

        primary = max(
            faces,
            key=lambda face: (
                face["box"][2] - face["box"][0]
            ) * (face["box"][3] - face["box"][1]),
        )
        return {
            **primary,
            "alert": any(face["alert"] for face in faces),
            "faces": faces,
        }

    def close(self):
        self.detector.close()


# Keep the file usable as a standalone camera demo too.
def main():
    detector = CNNDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            result = detector.predict(frame)

            for face in result["faces"]:
                box = face["box"]
                cv2.rectangle(
                    frame, box[:2], box[2:],
                    (0, 0, 255) if face["drowsy"] else (0, 255, 0),
                    2,
                )
                probability = face["probability"]
                color = (0, 0, 255) if face["alert"] else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"P{face['id']} {face['label']} ({probability:.2f})",
                    (box[0], max(25, box[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.imshow("CNN Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
