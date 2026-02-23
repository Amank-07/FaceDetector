"""
Real-time Face Detection Application

Supports webcam, image, and video modes. Detects faces using Haar Cascade
or OpenCV DNN, and displays bounding boxes with face count.

Modes:
    Webcam - Live feed from camera
    Image  - Upload/select a photo
    Video  - Upload/select a video file

Controls (webcam/video):
    q       - Quit application
    d       - Switch detector (Haar Cascade <-> DNN)
    t       - Toggle detection on/off (webcam only)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import cv2

from utils import (
    HaarCascadeDetector,
    DNNDetector,
    FPSCounter,
    FaceTracker,
    draw_detections,
    draw_overlay_text,
    resize_frame,
)


def get_file_path(mode: str, file_types: list) -> Optional[Path]:
    """
    Open a file dialog to select an image or video.
    Returns the selected path or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title=f"Select {mode} file",
        filetypes=file_types,
    )
    root.destroy()

    return Path(file_path) if file_path else None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Face detection using OpenCV (webcam, image, or video)")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for webcam mode (default: 0)",
    )
    parser.add_argument(
        "--detector",
        choices=["haar", "dnn"],
        default="haar",
        help="Face detector: haar (Haar Cascade) or dnn (OpenCV DNN)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable frame resizing for performance (use full resolution)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max frame width when resizing (default: 1280)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Max frame height when resizing (default: 720)",
    )
    # Image and video modes
    parser.add_argument(
        "--image",
        nargs="?",
        const="",
        metavar="PATH",
        help="Detect faces in an image. Provide path or omit to open file picker.",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const="",
        metavar="PATH",
        help="Detect faces in a video. Provide path or omit to open file picker.",
    )
    return parser.parse_args()


def init_camera(camera_index: int):
    """
    Initialize and return camera capture object.
    Handles common errors gracefully.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        error_msg = (
            f"Failed to open camera (index {camera_index}).\n"
            "Possible causes:\n"
            "  - Camera not connected\n"
            "  - Camera in use by another application\n"
            "  - Permission denied\n"
            "  - Invalid camera index (try --camera 0, 1, 2...)\n"
        )
        raise RuntimeError(error_msg)

    # Set reasonable resolution for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap


def run_image_detection(args, detector, detector_name: str, image_path: Path):
    """Detect faces in a single image and display results."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not load image from {image_path}", file=sys.stderr)
        sys.exit(1)

    if not args.no_resize:
        frame = resize_frame(frame, args.max_width, args.max_height)

    faces = detector.detect(frame)

    # Assign a simple unique ID per face in this image
    id_faces = list(enumerate(faces, start=1))
    for face_id, (x, y, w, h) in id_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {face_id}",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    draw_overlay_text(frame, f"Faces detected: {len(faces)}", (10, 35), color=(0, 255, 0))
    draw_overlay_text(frame, f"Detector: {detector_name}", (10, 70), color=(255, 255, 0))

    print(f"Image: {image_path.name} | Faces detected: {len(faces)}")

    cv2.imshow("Face Detection - Image", frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_detection(args, detector, detector_name: str, video_path: Path):
    """Detect faces in a video file frame by frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps_counter = FPSCounter()
    tracker = FaceTracker()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_faces_seen = 0

    print(f"Video: {video_path.name}")
    print("Controls: q - Quit, d - Switch detector")
    print("-" * 40)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video - loop from start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            if not args.no_resize:
                frame = resize_frame(frame, args.max_width, args.max_height)

            faces = detector.detect(frame)
            tracked = tracker.update(faces)
            max_faces_seen = max(max_faces_seen, len(tracked))

            # Draw tracked faces with IDs
            for face_id, (x, y, w, h) in tracked:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {face_id}",
                    (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            fps = fps_counter.update()

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            draw_overlay_text(frame, f"FPS: {fps:.1f}", (10, 35), color=(0, 255, 255))
            draw_overlay_text(frame, f"Faces in frame: {len(tracked)}", (10, 70), color=(0, 255, 0))
            draw_overlay_text(frame, f"Max faces: {max_faces_seen}", (10, 105), color=(255, 255, 0))
            draw_overlay_text(frame, f"Detector: {detector_name}", (10, 140), color=(255, 255, 0))
            if total_frames > 0:
                draw_overlay_text(frame, f"Frame {frame_num}/{total_frames}", (10, 175), color=(200, 200, 200))

            cv2.imshow("Face Detection - Video", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                try:
                    if detector_name == "Haar Cascade":
                        detector = DNNDetector()
                        detector_name = "DNN"
                    else:
                        detector = HaarCascadeDetector()
                        detector_name = "Haar Cascade"
                    print(f"Switched to {detector_name} detector")
                except RuntimeError as e:
                    print(f"Failed to switch detector: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Video finished. Max faces in any frame: {max_faces_seen}")


def run_face_detection(args):
    """Main application loop."""
    # Initialize camera
    try:
        cap = init_camera(args.camera)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize detector
    try:
        if args.detector == "haar":
            detector = HaarCascadeDetector()
            detector_name = "Haar Cascade"
        else:
            detector = DNNDetector()
            detector_name = "DNN"
    except RuntimeError as e:
        print(f"Error initializing detector: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    fps_counter = FPSCounter()
    tracker = FaceTracker()
    detection_enabled = True
    current_detector = detector
    current_detector_name = detector_name

    print("Face Detection Started")
    print("-" * 40)
    print("Controls:")
    print("  q - Quit")
    print("  d - Switch detector (Haar <-> DNN)")
    print("  t - Toggle detection on/off")
    print("-" * 40)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.", file=sys.stderr)
                break

            # Resize for better performance (optional)
            if not args.no_resize:
                frame = resize_frame(frame, args.max_width, args.max_height)

            # Detect faces when enabled
            faces = []
            if detection_enabled:
                try:
                    faces = current_detector.detect(frame)
                except Exception as e:
                    print(f"Detection error: {e}", file=sys.stderr)

            # Track faces and draw bounding boxes with unique IDs
            tracked = []
            if detection_enabled and faces:
                tracked = tracker.update(faces)
                for face_id, (x, y, w, h) in tracked:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID {face_id}",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # Update FPS
            fps = fps_counter.update()

            # Draw overlay info
            draw_overlay_text(
                frame,
                f"FPS: {fps:.1f}",
                (10, 35),
                color=(0, 255, 255),
            )
            draw_overlay_text(
                frame,
                f"Faces: {len(tracked) if detection_enabled else 0}",
                (10, 70),
                color=(0, 255, 0),
            )
            draw_overlay_text(
                frame,
                f"Detector: {current_detector_name}",
                (10, 105),
                color=(255, 255, 0),
            )
            if not detection_enabled:
                draw_overlay_text(
                    frame,
                    "[Detection OFF - press 't' to toggle]",
                    (10, 140),
                    color=(0, 165, 255),
                )

            # Display frame
            cv2.imshow("Face Detection", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                # Switch detector
                try:
                    if current_detector_name == "Haar Cascade":
                        current_detector = DNNDetector()
                        current_detector_name = "DNN"
                    else:
                        current_detector = HaarCascadeDetector()
                        current_detector_name = "Haar Cascade"
                    print(f"Switched to {current_detector_name} detector")
                except RuntimeError as e:
                    print(f"Failed to switch detector: {e}", file=sys.stderr)
            elif key == ord("t"):
                detection_enabled = not detection_enabled
                status = "ON" if detection_enabled else "OFF"
                print(f"Detection {status}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def main():
    """Entry point."""
    args = parse_args()

    # Always try GUI launcher first; fall back to CLI if Tkinter is unavailable
    try:
        import tkinter as tk
    except ImportError:
        print("Tkinter is not available in this Python environment. Falling back to CLI.", file=sys.stderr)
    else:
        root = tk.Tk()
        root.title("Face Detection Launcher")

        detector_var = tk.StringVar(value=args.detector)

        tk.Label(root, text="Detector:", font=("Segoe UI", 10, "bold")).pack(
            padx=10, pady=(10, 0), anchor="w"
        )
        det_frame = tk.Frame(root)
        det_frame.pack(padx=10, pady=5, anchor="w")
        tk.Radiobutton(det_frame, text="Haar Cascade", variable=detector_var, value="haar").pack(side="left")
        tk.Radiobutton(det_frame, text="DNN", variable=detector_var, value="dnn").pack(side="left")

        tk.Label(root, text="Choose mode:", font=("Segoe UI", 10, "bold")).pack(
            padx=10, pady=(10, 5), anchor="w"
        )
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=10, pady=(0, 10))

        def start_webcam():
            args.detector = detector_var.get()
            args.image = None
            args.video = None
            root.destroy()

        def start_image():
            args.detector = detector_var.get()
            args.image = ""
            args.video = None
            root.destroy()

        def start_video():
            args.detector = detector_var.get()
            args.video = ""
            args.image = None
            root.destroy()

        tk.Button(btn_frame, text="Webcam", width=12, command=start_webcam).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Image", width=12, command=start_image).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Video", width=12, command=start_video).grid(row=0, column=2, padx=5)

        root.resizable(False, False)
        root.mainloop()

    # Resolve image path
    if args.image is not None:
        if args.image:
            image_path = Path(args.image)
        else:
            image_path = get_file_path(
                "image",
                [
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                    ("All files", "*.*"),
                ],
            )
        if image_path is None:
            print("No image selected. Exiting.")
            sys.exit(0)
        if not image_path.exists():
            print(f"Error: File not found: {image_path}", file=sys.stderr)
            sys.exit(1)

    # Resolve video path
    if args.video is not None:
        if args.video:
            video_path = Path(args.video)
        else:
            video_path = get_file_path(
                "video",
                [
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"),
                    ("All files", "*.*"),
                ],
            )
        if video_path is None:
            print("No video selected. Exiting.")
            sys.exit(0)
        if not video_path.exists():
            print(f"Error: File not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    # Initialize detector (shared by all modes)
    try:
        if args.detector == "haar":
            detector = HaarCascadeDetector()
            detector_name = "Haar Cascade"
        else:
            detector = DNNDetector()
            detector_name = "DNN"
    except RuntimeError as e:
        print(f"Error initializing detector: {e}", file=sys.stderr)
        sys.exit(1)

    # Run appropriate mode
    if args.image is not None:
        run_image_detection(args, detector, detector_name, image_path)
    elif args.video is not None:
        run_video_detection(args, detector, detector_name, video_path)
    else:
        run_face_detection(args)


if __name__ == "__main__":
    main()
