"""Video processing module for RoadGuard AI."""

import os
import sys
from pathlib import Path
import cv2

# Ensure project root is in sys.path for direct script execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.detection import load_model
from src.tracking import track_vehicles
from src.speed_estimation import SpeedEstimator
from src.analytics import TrafficAnalytics
from src.violations import ViolationDetector, annotate_frame_with_violations


def process_video(
    input_path,
    output_path,
    model=None,
    confidence_threshold=0.5,
    max_frames=None,
    speed_limit_px=1000.0,
    return_analytics=False,
):
    """Process a video frame-by-frame, track vehicles, estimate speed, detect violations, and save video.

    Args:
        input_path: Path to the input video file.
        output_path: Path where the output annotated video will be saved.
        model: Optional pre-loaded YOLO model instance. If None, loads yolo11n.pt.
        confidence_threshold: Minimum confidence score to retain detection (default: 0.5).
        max_frames: Optional maximum number of frames to process.
        speed_limit_px: Speed limit in pixels/sec above which a vehicle is marked for violation.
        return_analytics: If True, returns (processed_count, analytics_summary).

    Returns:
        int or tuple: processed_count (or (processed_count, analytics_summary)).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file does not exist: {input_path}")
        return (0, {}) if return_analytics else 0

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return (0, {}) if return_analytics else 0

    out = None
    processed_count = 0

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if width <= 0 or height <= 0 or fps <= 0:
            print(f"Error: Invalid video properties (width={width}, height={height}, fps={fps})")
            return (0, {}) if return_analytics else 0

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not open video writer for output: {output_path}")
            return (0, {}) if return_analytics else 0

        if model is None:
            model = load_model("yolo11n.pt")

        speed_estimator = SpeedEstimator(fps=fps, buffer_size=10, min_history=5)
        analytics = TrafficAnalytics()
        violation_detector = ViolationDetector(speed_limit_px=speed_limit_px)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

        while cap.isOpened():
            if max_frames is not None and processed_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # 1. ByteTrack Tracking
            tracked_vehicles = track_vehicles(
                model=model,
                frame=frame,
                confidence_threshold=confidence_threshold,
            )

            # 2. Speed Estimation
            tracked_vehicles = speed_estimator.update(
                tracked_vehicles=tracked_vehicles,
                frame_index=processed_count,
            )

            # 3. Violation Detection Engine
            tracked_vehicles = violation_detector.check_violations(
                tracked_vehicles=tracked_vehicles,
                frame_index=processed_count,
            )

            # 4. Traffic Analytics
            analytics.update(tracked_vehicles)

            # 5. Annotation (highlighting normal vs violating vehicles)
            annotated_frame = annotate_frame_with_violations(frame, tracked_vehicles)
            out.write(annotated_frame)

            processed_count += 1
            if processed_count % 30 == 0 or processed_count == total_frames:
                print(f"  Processed {processed_count}/{total_frames} frames...")

        print(f"Successfully processed {processed_count} frames.")
        print(f"Saved annotated video to: {output_path}")

        analytics_summary = analytics.get_summary()
        violations_summary = violation_detector.get_summary()
        analytics_summary["violations_count"] = violations_summary["total_violations"]
        analytics_summary["violation_logs"] = violations_summary["logs"]

        if return_analytics:
            return processed_count, analytics_summary
        return processed_count

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()


if __name__ == "__main__":
    input_video = "data/videos/highway.mp4"
    output_video = "outputs/processed_videos/annotated_highway_speed.mp4"
    process_video(input_video, output_video, confidence_threshold=0.35, max_frames=None)
