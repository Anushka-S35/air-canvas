# Air Canvas

A real-time hand-gesture drawing application built with **Python**, **OpenCV**, **MediaPipe**, and **NumPy**.

## Features

1. Live webcam feed (mirrored for natural selfie view)
2. Real-time single-hand tracking with MediaPipe Hands
3. Index fingertip detection (landmark 8)
4. Smooth line drawing by moving the index finger
5. Two-finger gesture (index + middle) to enter color-selection mode
6. Multiple brush colors via an on-screen palette (Blue, Green, Red, Yellow, Purple, White, Eraser)
7. Open-palm gesture (four fingers raised) to clear the canvas
8. On-screen HUD: FPS, drawing mode, status, brush thickness

## Gestures

| Gesture | Action |
|---------|--------|
| Index finger only raised | **Draw** — draw smooth lines |
| Index + Middle raised | **Select** — hover over a palette button to pick a color/eraser |
| Open palm (4 fingers up) | **Clear** — wipe the canvas |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit the application |
| `c` | Clear the canvas |
| `+` / `=` | Increase brush thickness (max 100) |
| `-` | Decrease brush thickness |

## Project Structure

```
air_canvas/
├── main.py      # Application source code
├── requirements.txt  # Python dependencies
└── README.md    # This file
```

## Installation & Usage

### 1. Create a virtual environment (recommended)

```bash
cd /Users/anushkashelke/.cline/data/workspaces/chat/air_canvas
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Python 3.9 users:** `mediapipe==0.10.14` depends on `jax`, which contains Python 3.10+ syntax (`match` statements) in an experimental module. If `pip install` fails with a `SyntaxError` on `match`, install with `--no-compile` to skip bytecode compilation (the module is not used at runtime):
> ```bash
> pip install --no-compile -r requirements.txt
> ```
> Python 3.10+ does not have this issue and can use the standard `pip install -r requirements.txt`.

> **Note:** `requirements.txt` installs `opencv-contrib-python` (with GUI support). If you previously installed `opencv-python-headless`, uninstall it first:
> ```bash
> pip uninstall -y opencv-python-headless
> ```

### 3. Run the application

```bash
python3 main.py
```

### 4. Quit

Press `q` or `ESC` while the video window is focused.

## Requirements

- macOS / Windows / Linux with a working webcam
- Python 3.9+ (Python 3.10+ recommended to avoid the jax compile note above)
- Camera permissions granted to your terminal / IDE

## Troubleshooting

- **Camera not opening**: Close other apps using the webcam, and grant camera permission to your terminal/IDE in System Settings → Privacy & Security → Camera.
- **`cv2.imshow` errors**: Make sure you installed `opencv-python`, not `opencv-python-headless`.
- **Hand not detected**: Ensure good lighting and that your hand is fully visible to the camera.