# Student Drowsiness Detection POC

A local USB-webcam demo that estimates eye openness with MediaPipe Face Mesh and shows an alert when both eyes appear continuously closed. It is intended for a supervised classroom prototype, not medical use, grading, attendance, discipline, or monitoring students without informed consent.

Baseline -> eye aspect ratio (EAR)
Improvment -> CNN detection

## Setup

Use Python 3.10-3.12. From this folder, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate` instead.

## Run

Connect the USB webcam and run:

```bash
python drowsiness_detector.py
```

The default camera index is `0`. Select another camera with `--camera 1`. Press `q` or `Esc` to close the window. The alert duration and eye-openness threshold can be adjusted:

```bash
python drowsiness_detector.py --closed-seconds 1.5 --ear-threshold 0.21
```

EAR values vary with face, camera placement, and lighting. If eyes are frequently classified as closed while open, lower `--ear-threshold`; if closure is missed, raise it slightly. Test in the actual room and seating setup before demonstrating it.

## What it does

- Requires both eye ratios to stay below threshold; the HUD shows their average EAR.
- Raises a visible warning and terminal bell after sustained closure; a brief blink does not trigger it.
- Processes camera frames in memory. It does not save or transmit video or landmarks.
- Shows `NO FACE DETECTED` if a face is not visible; that state is not evidence of drowsiness.

## Validate

The logic tests do not need webcam or MediaPipe dependencies:

```bash
python -m unittest -v
```

This threshold-based signal can miss drowsiness and can trigger incorrectly. Do not use it as a safety system or as the sole basis for decisions about a student. Obtain informed consent, provide a non-camera alternative, and confirm any real-world concern with a human.