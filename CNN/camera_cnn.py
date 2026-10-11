
import cv2
import time
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


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
        self.drowsy_since = None

        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    def predict(self, frame, now=None):
        """Return CNN prediction for a BGR camera frame."""
        if now is None:
            now = time.monotonic()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            self.drowsy_since = None
            return {
                "label": "NO FACE",
                "probability": None,
                "drowsy": False,
                "alert": False,
                "box": None,
            }

        detection = max(
            results.detections,
            key=lambda d: (
                d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
            ),
        )

        box = detection.location_data.relative_bounding_box
        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))

        if x2 <= x1 or y2 <= y1:
            self.drowsy_since = None
            return {
                "label": "NO FACE",
                "probability": None,
                "drowsy": False,
                "alert": False,
                "box": None,
            }

        face = rgb[y1:y2, x1:x2]
        image = self.transform(Image.fromarray(face))
        image = image.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probs = torch.softmax(self.model(image), dim=1)[0]
            probability = probs[self.fatigue_idx].item()

        drowsy = probability >= self.threshold

        if drowsy:
            if self.drowsy_since is None:
                self.drowsy_since = now
        else:
            self.drowsy_since = None

        alert = (
            self.drowsy_since is not None
            and now - self.drowsy_since >= self.alert_seconds
        )

        return {
            "label": "DROWSINESS ALERT" if alert else (
                "DROWSY" if drowsy else "ACTIVE"
            ),
            "probability": probability,
            "drowsy": drowsy,
            "alert": alert,
            "box": (x1, y1, x2, y2),
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

            box = result["box"]
            if box is not None:
                cv2.rectangle(
                    frame, box[:2], box[2:],
                    (0, 0, 255) if result["drowsy"] else (0, 255, 0),
                    2,
                )

            probability = result["probability"]
            prob_text = "--" if probability is None else f"{probability:.2f}"
            color = (0, 0, 255) if result["alert"] else (0, 255, 0)

            cv2.putText(
                frame, f"{result['label']} ({prob_text})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2,
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
