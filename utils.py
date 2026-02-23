"""
Utility functions for face detection application.
Handles face detection logic, tracking, drawing, FPS calculation, and model loading.
"""

import math
import cv2
import os
import urllib.request
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# DNN model URLs (OpenCV's pre-trained face detector)
# Prototxt from OpenCV main repo; caffemodel from OpenCV 3rdparty
DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_CAFFEMODEL_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
]

# Default model storage directory
MODELS_DIR = Path(__file__).parent / "models"


def ensure_models_dir() -> Path:
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR


def download_dnn_models() -> Tuple[str, str]:
    """
    Download DNN face detector models if not present.
    Returns paths to prototxt and caffemodel files.
    """
    ensure_models_dir()
    prototxt_path = MODELS_DIR / "deploy.prototxt"
    caffemodel_path = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    if not prototxt_path.exists():
        print("Downloading DNN prototxt...")
        try:
            urllib.request.urlretrieve(DNN_PROTOTXT_URL, str(prototxt_path))
            print("Prototxt downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download DNN prototxt: {e}") from e

    if not caffemodel_path.exists():
        print("Downloading DNN caffemodel (this may take a moment)...")
        last_error = None
        for url in DNN_CAFFEMODEL_URLS:
            try:
                urllib.request.urlretrieve(url, str(caffemodel_path))
                if caffemodel_path.exists() and caffemodel_path.stat().st_size > 1_000_000:
                    print("Caffemodel downloaded successfully.")
                    break
            except Exception as e:
                last_error = e
                if caffemodel_path.exists():
                    caffemodel_path.unlink(missing_ok=True)
        else:
            err_msg = str(last_error) if last_error else "Unknown error"
            raise RuntimeError(
                f"Failed to download DNN caffemodel. {err_msg}\n"
                "Please download manually from the README and save to models/"
            ) from last_error

    return str(prototxt_path), str(caffemodel_path)


class HaarCascadeDetector:
    """Face detector using OpenCV Haar Cascade classifier."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade classifier.")

    def detect(self, frame: cv2.Mat, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a grayscale frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
        )
        return [tuple(map(int, (x, y, w, h))) for (x, y, w, h) in faces]


class DNNDetector:
    """Face detector using OpenCV DNN module with Caffe model."""

    def __init__(self, confidence_threshold: float = 0.5):
        prototxt_path, caffemodel_path = download_dnn_models()
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame using DNN.
        Returns list of (x, y, w, h) bounding boxes.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces


class FPSCounter:
    """Simple FPS counter using frame timestamps."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self.prev_time = None

    def update(self) -> float:
        """Update and return current FPS."""
        import time
        current_time = time.perf_counter()
        if self.prev_time is not None:
            frame_time = current_time - self.prev_time
            if frame_time > 0:
                new_fps = 1.0 / frame_time
                self.fps = self.alpha * new_fps + (1 - self.alpha) * self.fps
        self.prev_time = current_time
        return self.fps

    def get(self) -> float:
        """Return current FPS value."""
        return self.fps


class FaceTracker:
    """
    Very simple centroid-based tracker that assigns a stable ID to each face
    across frames based on nearest-neighbour matching of bounding boxes.
    """

    def __init__(self, max_distance: float = 50.0, max_lost: int = 30):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id: int = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Update internal tracks with new detections.
        Returns list of (track_id, bbox) for current frame.
        """
        assigned_track_ids = set()
        results: List[Tuple[int, Tuple[int, int, int, int]]] = []

        # Associate each detection to an existing track if close enough
        for det in detections:
            cx, cy = self._center(det)
            best_id = None
            best_dist = self.max_distance

            for track_id, data in self.tracks.items():
                if track_id in assigned_track_ids:
                    continue
                tx, ty = self._center(data["bbox"])  # type: ignore[index]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None:
                # Update existing track
                self.tracks[best_id]["bbox"] = det  # type: ignore[index]
                self.tracks[best_id]["lost"] = 0  # type: ignore[index]
                assigned_track_ids.add(best_id)
                results.append((best_id, det))
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {"bbox": det, "lost": 0}
                assigned_track_ids.add(track_id)
                results.append((track_id, det))

        # Mark unassigned tracks as lost, and remove if lost for too long
        for track_id in list(self.tracks.keys()):
            if track_id not in assigned_track_ids:
                self.tracks[track_id]["lost"] = (  # type: ignore[index]
                    self.tracks[track_id].get("lost", 0) + 1  # type: ignore[index]
                )
                if self.tracks[track_id]["lost"] > self.max_lost:  # type: ignore[index]
                    del self.tracks[track_id]

        return results


def draw_detections(
    frame: cv2.Mat,
    faces: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw bounding boxes around detected faces (modifies frame in-place)."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_overlay_text(
    frame: cv2.Mat,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)


def resize_frame(frame: cv2.Mat, max_width: int = 1280, max_height: int = 720) -> cv2.Mat:
    """
    Resize frame to fit within max dimensions while preserving aspect ratio.
    Improves performance on high-resolution cameras.
    """
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
