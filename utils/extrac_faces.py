
from pathlib import Path
import cv2
import mediapipe as mp

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("data_faces")
SPLITS = ["train", "val", "test"]
CLASSES = ["active", "fatigue"]

# Expand the detected face box to retain more facial context.
PADDING = 0.25

face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in SPLITS:
    for label in CLASSES:
        src_dir = INPUT_DIR / split / label
        dst_dir = OUTPUT_DIR / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"Missing folder: {src_dir}")
            continue

        files = [
            p for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

        saved = 0
        skipped = 0

        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                skipped += 1
                continue

            height, width = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if not results.detections:
                skipped += 1
                continue

            # If multiple faces exist, use the largest detected face.
            detection = max(
                results.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height
                )
            )

            box = detection.location_data.relative_bounding_box
            x1 = int(box.xmin * width)
            y1 = int(box.ymin * height)
            x2 = int((box.xmin + box.width) * width)
            y2 = int((box.ymin + box.height) * height)

            pad_x = int((x2 - x1) * PADDING)
            pad_y = int((y2 - y1) * PADDING)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                skipped += 1
                continue

            # Preserve the original relative folder structure and filename.
            relative_path = path.relative_to(src_dir)
            out_path = (dst_dir / relative_path).with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if cv2.imwrite(str(out_path), face):
                saved += 1
            else:
                skipped += 1

        print(f"{split}/{label}: saved {saved}, skipped {skipped}")

face_detector.close()
print("Face extraction complete.")
