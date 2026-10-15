from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_detector import (
    FaceDetector,
    FaceDetectorOptions,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)


MODEL_DIR = Path(__file__).resolve().parent / "models"
FACE_DETECTOR_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
FACE_DETECTOR_FULL_RANGE_MODEL = MODEL_DIR / "blaze_face_full_range.tflite"
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"

_FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_FULL_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
)
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

IMAGE_MODE = mp.tasks.vision.RunningMode.IMAGE
VIDEO_MODE = mp.tasks.vision.RunningMode.VIDEO


def _base_options(model_path: Path, model_url: str) -> BaseOptions:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            f"Download it from {model_url}."
        )
    return BaseOptions(model_asset_path=str(model_path))


def create_face_detector(
    running_mode: VisionTaskRunningMode,
    model_path: Path = FACE_DETECTOR_MODEL,
    model_url: str = _FACE_DETECTOR_URL,
) -> FaceDetector:
    options = FaceDetectorOptions(
        base_options=_base_options(model_path, model_url),
        running_mode=running_mode,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def create_face_landmarker() -> FaceLandmarker:
    options = FaceLandmarkerOptions(
        base_options=_base_options(
            FACE_LANDMARKER_MODEL, _FACE_LANDMARKER_URL
        ),
        running_mode=VIDEO_MODE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def to_mp_image(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def next_video_timestamp_ms(now: float, previous_ms: int | None) -> int:
    timestamp_ms = int(now * 1000)
    if previous_ms is not None and timestamp_ms <= previous_ms:
        return previous_ms + 1
    return timestamp_ms
