# Real-Time Face Detection

A Python application that opens your webcam and performs real-time face detection using OpenCV. Supports both Haar Cascade and OpenCV DNN detectors.

## Features

- **Real-time face detection** via webcam
- **Two detector modes**: Haar Cascade (fast) or OpenCV DNN (more accurate)
- **Visual feedback**: Bounding boxes around detected faces
- **On-screen stats**: FPS and face count
- **Graceful error handling**: Camera not found, permission denied, etc.
- **Extra controls**: Switch detectors, toggle detection, resize frames for performance

## Requirements

- Python 3.8 or higher
- Webcam
- Windows / macOS / Linux

## Installation

### Step 1: Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the application

```bash
python main.py
```

## Usage

### Basic run

```bash
python main.py
```

### Command-line options

| Option        | Default | Description                          |
|---------------|---------|--------------------------------------|
| `--camera`    | 0       | Camera device index                  |
| `--detector`  | haar    | Initial detector: `haar` or `dnn`    |
| `--no-resize` | -       | Disable frame resizing               |
| `--max-width` | 1280    | Max frame width when resizing        |
| `--max-height`| 720     | Max frame height when resizing       |

### Examples

```bash
# Use DNN detector from the start
python main.py --detector dnn

# Use a different camera (e.g., external webcam)
python main.py --camera 1

# Disable resizing (full resolution, may be slower)
python main.py --no-resize

# Custom resize limits
python main.py --max-width 960 --max-height 540
```

### Keyboard controls

| Key | Action                    |
|-----|---------------------------|
| `q` | Quit application          |
| `d` | Switch detector (Haar ↔ DNN) |
| `t` | Toggle detection on/off   |

## Project structure

```
face_detection/
├── main.py          # Entry point, main loop, camera handling
├── utils.py         # Face detection, drawing, FPS, model loading
├── requirements.txt # Python dependencies
├── README.md        # This file
└── models/          # DNN model files (created on first DNN use)
```

## DNN model download

When you first use the DNN detector (via `--detector dnn` or by pressing `d`), the application will download the required Caffe model files (~10 MB) into the `models/` folder. This happens automatically and requires an internet connection.

## Troubleshooting

### Camera not opening

- Ensure no other application is using the webcam
- Try different camera indices: `python main.py --camera 1`
- On Linux, check permissions: `sudo usermod -a -G video $USER` (then log out and back in)

### Low FPS

- Use Haar Cascade: `python main.py --detector haar`
- Enable frame resizing (default): avoid `--no-resize`
- Lower resolution: `python main.py --max-width 640 --max-height 480`

### DNN download fails

- Check your internet connection
- Ensure the `models/` directory is writable
- Manually download and place in `models/`:
  - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
  - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

## License

MIT
