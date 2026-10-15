# Student Drowsiness Detection POC

A local USB-webcam prototype that estimates student drowsiness using two approaches:

- **Baseline — Eye Aspect Ratio (EAR):** Uses the MediaPipe Face Landmarker task to estimate eye openness for up to 10 faces and triggers an independent alert when both eyes remain closed for a sustained period.
- **Improvement — CNN detection:** Uses a fine-tuned MobileNetV3-Small model to classify each detected face crop as `active` or `fatigue`. A sustained high fatigue probability triggers an independent alert.

The system is intended for a supervised classroom prototype, not medical use, grading, attendance, discipline, or monitoring students without informed consent.

## Setup

Use a Python version supported by the packages in `requirements.txt`. From the project root, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the MediaPipe task models required by the detectors:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" `
  -OutFile "models\blaze_face_short_range.tflite"
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite" `
  -OutFile "models\blaze_face_full_range.tflite"
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "models\face_landmarker.task"
```

On macOS or Linux, download the same models with:

```bash
curl -fL "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" -o models/blaze_face_short_range.tflite
curl -fL "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite" -o models/blaze_face_full_range.tflite
curl -fL "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o models/face_landmarker.task
```

Run the detector from the activated environment with `python .\detector.py`.

Ensure the trained CNN checkpoint exists at:

```text
models/drowsiness_cnn_20261011_184723.pth
```

## Run the Combined Detector

Connect the USB webcam and run:

```bash
python detector.py
```

The default camera index is `0`. Select another camera with `--camera 1`. Press `q` or `Esc` to close the window.

The EAR threshold and required eye-closure duration can be adjusted:

```bash
python detector.py --closed-seconds 1.5 --ear-threshold 0.21
```

The CNN uses its own default settings:

- **Fatigue probability threshold:** `0.65`
- **Alert duration:** `1.5` seconds
- **Camera index:** `0`

These CNN settings can be changed in `CNN/camera_cnn.py`.

## Test the CNN Separately

Run the standalone CNN demo from the project root:

```bash
python CNN/camera_cnn.py
```

This tests the CNN independently of the EAR detector.

The standalone CNN pipeline works as follows:

1. OpenCV captures a webcam frame.
2. The MediaPipe Face Detector task detects all faces in the frame and extracts each face region.
3. The face crop is preprocessed using the MobileNetV3-Small input transforms.
4. The trained CNN predicts probabilities for `active` and `fatigue`.
5. If the fatigue probability stays at or above `0.65` for `1.5` seconds, a drowsiness alert is triggered.

The CNN processes frames locally and does not need to save webcam images.

## What It Does

- **EAR detector:** Evaluates up to 10 faces and requires both eyes in a face to remain below the eye-openness threshold. Each face has its own closure timer.
- **CNN detector:** Estimates fatigue independently for each face crop and triggers an alert when that face's probability threshold is sustained.
- **Combined detector:** Raises an alert if either detector triggers.
- **No face detected:** Displays a no-face status when neither detector can detect a face. This is not evidence of drowsiness.
- **Privacy:** Camera frames are processed in memory; the prototype does not save or transmit video or facial landmarks by default.

## Validate

Run the logic tests without opening the webcam:

```bash
python -m unittest -v
```

For a practical test, try normal attentive behaviour, brief blinks, prolonged eye closure, and different lighting and camera positions. Compare the EAR and CNN outputs rather than assuming either detector is always correct.

## Limitations and Responsible Use

EAR depends on face geometry, camera placement, and lighting. The CNN depends on its training data and may perform differently on people or conditions not adequately represented in that data. Neither approach can reliably establish that a student is drowsy from a single prediction.

This is a proof of concept, not a validated safety system. Obtain informed consent, offer a non-camera alternative, minimise data collection, and have a human assess any real-world concern. Do not use its outputs as the sole basis for decisions about students.