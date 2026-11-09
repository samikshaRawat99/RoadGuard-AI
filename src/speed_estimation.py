"""Vehicle speed estimation module for RoadGuard AI (pixel-based)."""

import math
import cv2


class SpeedEstimator:
    """Estimates vehicle speed in pixels per second (px/s) using tracking history."""

    def __init__(
        self,
        fps=30.0,
        buffer_size=10,
        min_history=5,
        max_gap=10,
        stale_timeout=60,
    ):
        """Initialize the speed estimator with input parameter validation.

        Args:
            fps: Frame rate of the video (must be > 0).
            buffer_size: Number of past positions to retain for smoothing (must be >= 2).
            min_history: Minimum number of frames needed before estimating speed (must be >= 2 and <= buffer_size).
            max_gap: Maximum allowed frame gap before resetting vehicle history (must be >= 1).
            stale_timeout: Frames without detection before removing a vehicle from memory (must be >= 1).
        """
        # 1. Parameter validation
        if fps <= 0:
            raise ValueError(f"fps must be a positive number, got {fps}")
        if buffer_size < 2:
            raise ValueError(f"buffer_size must be at least 2, got {buffer_size}")
        if min_history < 2:
            raise ValueError(f"min_history must be at least 2, got {min_history}")
        if min_history > buffer_size:
            raise ValueError(f"min_history ({min_history}) cannot be greater than buffer_size ({buffer_size})")
        if max_gap < 1:
            raise ValueError(f"max_gap must be at least 1, got {max_gap}")
        if stale_timeout < 1:
            raise ValueError(f"stale_timeout must be at least 1, got {stale_timeout}")

        self.fps = float(fps)
        self.buffer_size = int(buffer_size)
        self.min_history = int(min_history)
        self.max_gap = int(max_gap)
        self.stale_timeout = int(stale_timeout)

        # Dictionary mapping track_id -> list of (frame_index, cx, cy)
        self.history = {}

    def _cleanup_stale_tracks(self, current_frame):
        """Remove tracked vehicles not detected for longer than stale_timeout frames."""
        stale_ids = [
            tid for tid, points in self.history.items()
            if points and (current_frame - points[-1][0]) > self.stale_timeout
        ]
        for tid in stale_ids:
            del self.history[tid]

    def update(self, tracked_vehicles, frame_index):
        """Update positions of tracked vehicles and calculate speed in px/s.

        Args:
            tracked_vehicles: List of vehicle dicts from track_vehicles().
            frame_index: Current frame sequence number (int).

        Returns:
            The same list of vehicle dicts, each augmented with a 'speed_px' key.
        """
        # 2. Cleanup stale tracks to bound memory usage
        self._cleanup_stale_tracks(frame_index)

        for vehicle in tracked_vehicles:
            track_id = vehicle.get("track_id")

            # 3. If tracking ID is unavailable, speed cannot be estimated
            if track_id is None:
                vehicle["speed_px"] = None
                continue

            # 4. Use bottom-centre of bounding box as reference point (tire-road contact)
            x1, y1, x2, y2 = vehicle["box"]
            cx = (x1 + x2) / 2.0
            cy = float(y2)

            # 5. Check for gap in detection: if vehicle was lost for too long, reset history
            if track_id in self.history and len(self.history[track_id]) > 0:
                last_frame = self.history[track_id][-1][0]
                gap = frame_index - last_frame
                if gap > self.max_gap:
                    # Reset history on long gap to avoid teleporting speed calculation
                    self.history[track_id] = []
            else:
                self.history[track_id] = []

            # 6. Store current position in tracking history
            self.history[track_id].append((frame_index, cx, cy))

            # Keep only the last buffer_size positions
            if len(self.history[track_id]) > self.buffer_size:
                self.history[track_id] = self.history[track_id][-self.buffer_size:]

            # 7. Require minimum history frames before calculating speed
            if len(self.history[track_id]) < self.min_history:
                vehicle["speed_px"] = None
                continue

            # 8. Calculate displacement between oldest and newest position in buffer
            old_frame, old_x, old_y = self.history[track_id][0]
            new_frame, new_x, new_y = self.history[track_id][-1]

            frames_elapsed = new_frame - old_frame
            if frames_elapsed <= 0:
                vehicle["speed_px"] = None
                continue

            # Time elapsed in seconds
            time_seconds = frames_elapsed / self.fps

            # Euclidean pixel displacement: sqrt((x2 - x1)^2 + (y2 - y1)^2)
            displacement_px = math.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)

            # Speed in pixels per second
            speed_px = displacement_px / time_seconds
            vehicle["speed_px"] = round(speed_px, 1)

        return tracked_vehicles


def draw_vehicle_speeds(frame, tracked_vehicles):
    """Draw bounding boxes, IDs, reference points, and speed labels on a frame.

    Args:
        frame: Input image frame as a NumPy array (BGR format).
        tracked_vehicles: List of vehicle dictionaries from SpeedEstimator.update().

    Returns:
        A copy of the image annotated with boxes, IDs, and speed (px/s).
    """
    annotated_frame = frame.copy()

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = [int(coord) for coord in vehicle["box"]]
        track_id = vehicle.get("track_id")
        speed_px = vehicle.get("speed_px")

        # Color based on track ID
        if track_id is not None:
            color = (
                int((track_id * 67) % 256),
                int((track_id * 131) % 256),
                int((track_id * 197) % 256),
            )
            # Display speed alongside ID if available
            if speed_px is not None:
                label = f"#{track_id} {vehicle['class_name']} | {speed_px:.0f} px/s"
            else:
                label = f"#{track_id} {vehicle['class_name']}"
        else:
            color = (0, 255, 0)
            label = vehicle["class_name"]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw bottom-centre reference point (small red circle)
        cx = int((x1 + x2) / 2)
        cv2.circle(annotated_frame, (cx, y2), 5, (0, 0, 255), -1)

        # Draw label text slightly above top-left corner
        text_y = max(y1 - 10, 20)
        cv2.putText(
            annotated_frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame
