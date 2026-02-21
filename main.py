"""
Real-time Face Detection Application

Opens the webcam, detects faces using Haar Cascade or OpenCV DNN,
and displays bounding boxes with FPS and face count.

Controls:
    q       - Quit application
    d       - Switch detector (Haar Cascade <-> DNN)
    t       - Toggle detection on/off
"""

import sys
import argparse
import cv2

from utils import (
    HaarCascadeDetector,
    DNNDetector,
    FPSCounter,
    draw_detections,
    draw_overlay_text,
    resize_frame,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time face detection using OpenCV")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--detector",
        choices=["haar", "dnn"],
        default="haar",
        help="Initial face detector: haar (Haar Cascade) or dnn (OpenCV DNN)",
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

            # Draw bounding boxes
            draw_detections(frame, faces)

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
                f"Faces: {len(faces)}",
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
    run_face_detection(args)


if __name__ == "__main__":
    main()
