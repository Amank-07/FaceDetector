# Real-Time Face Detection

A Python application for face detection using OpenCV. Supports **webcam** (live feed), **image** (upload/select a photo), and **video** (upload/select a video file). Uses Haar Cascade or OpenCV DNN detectors.

## Features

- **Multiple input modes**: Webcam, image file, or video file
- **Image mode**: Upload/select a photo → see face count and bounding boxes
- **Video mode**: Upload/select a video → detect faces frame-by-frame with count
- **Webcam mode**: Real-time face detection from camera
- **Two detector modes**: Haar Cascade (fast) or OpenCV DNN (more accurate)
- **Visual feedback**: Bounding boxes around detected faces
- **On-screen stats**: FPS, face count, max faces (video)
- **Graceful error handling**: Camera not found, file not found, permission denied, etc.
- **Extra controls**: Switch detectors, toggle detection (webcam), resize frames for performance

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

### Image mode

```bash
# Open file picker to select an image
python main.py --image

# Or provide path directly
python main.py --image path/to/photo.jpg
```

### Video mode

```bash
# Open file picker to select a video
python main.py --video

# Or provide path directly
python main.py --video path/to/video.mp4
```

### Command-line options

| Option        | Default | Description                          |
|---------------|---------|--------------------------------------|
| `--camera`    | 0       | Camera device index (webcam mode)    |
| `--detector`  | haar    | Initial detector: `haar` or `dnn`    |
| `--image`     | -       | Detect faces in an image (path or picker) |
| `--video`     | -       | Detect faces in a video (path or picker)   |
| `--no-resize` | -       | Disable frame resizing               |
| `--max-width` | 1280    | Max frame width when resizing        |
| `--max-height`| 720     | Max frame height when resizing       |

### Examples

```bash
# Webcam (default)
python main.py

# Image with file picker
python main.py --image

# Image with path
python main.py --image C:\Photos\group.jpg

# Video with file picker
python main.py --video

# Video with DNN detector
python main.py --video clip.mp4 --detector dnn

# Use a different camera (e.g., external webcam)
python main.py --camera 1

# Disable resizing (full resolution, may be slower)
python main.py --no-resize

# Custom resize limits
python main.py --max-width 960 --max-height 540
```

### Keyboard controls

| Key | Action                    | Mode      |
|-----|---------------------------|-----------|
| `q` | Quit application          | All       |
| `d` | Switch detector (Haar ↔ DNN) | Webcam, video |
| `t` | Toggle detection on/off   | Webcam only |
| Any key | Close image window   | Image only |

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
