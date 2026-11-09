"""Traffic violations detection engine for RoadGuard AI."""

import cv2


class ViolationDetector:
    """Detects speeding violations, logs evidence, and draws violation badges."""

    def __init__(self, speed_limit_px=1000.0):
        """Initialize the violation detector.

        Args:
            speed_limit_px: Threshold speed in pixels/sec above which a vehicle is speeding.
        """
        self.speed_limit_px = float(speed_limit_px)
        # Store unique track IDs that violated the speed limit
        self.violating_tracks = set()
        # Evidence log: list of dicts with violation details
        self.violation_logs = []

    def check_violations(self, tracked_vehicles, frame_index):
        """Check current frame's tracked vehicles for speed violations.

        Args:
            tracked_vehicles: List of vehicle dicts with 'track_id', 'speed_px', 'box', 'class_name'.
            frame_index: Current frame number.

        Returns:
            The augmented list of vehicle dicts, each with an 'is_violation' boolean key.
        """
        for vehicle in tracked_vehicles:
            speed_px = vehicle.get("speed_px")
            track_id = vehicle.get("track_id")

            # Check if speeding
            is_speeding = False
            if speed_px is not None and speed_px > self.speed_limit_px:
                is_speeding = True

                # If this is the first time this vehicle was caught violating, record evidence
                if track_id is not None and track_id not in self.violating_tracks:
                    self.violating_tracks.add(track_id)
                    self.violation_logs.append({
                        "track_id": track_id,
                        "type": "Overspeeding",
                        "class_name": vehicle.get("class_name", "vehicle"),
                        "speed_px": speed_px,
                        "limit_px": self.speed_limit_px,
                        "frame": frame_index,
                    })

            # Tag vehicle dict
            vehicle["is_violation"] = is_speeding or (track_id in self.violating_tracks if track_id is not None else False)

        return tracked_vehicles

    def get_summary(self):
        """Return total violation count and structured evidence list."""
        return {
            "total_violations": len(self.violating_tracks),
            "logs": self.violation_logs,
        }


def annotate_frame_with_violations(frame, tracked_vehicles):
    """Draw bounding boxes with red highlight for violators and green/ID for normal vehicles.

    Args:
        frame: OpenCV BGR image frame.
        tracked_vehicles: Vehicle dictionaries with 'box', 'track_id', 'speed_px', 'is_violation'.

    Returns:
        Annotated BGR image frame.
    """
    annotated = frame.copy()

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = [int(c) for c in vehicle["box"]]
        track_id = vehicle.get("track_id")
        speed_px = vehicle.get("speed_px")
        is_violation = vehicle.get("is_violation", False)

        # Color: Bright Red for violators, unique ID-derived color or green for compliant
        if is_violation:
            color = (0, 0, 255)  # BGR Red
            badge = " [VIOLATION: SPEEDING]"
        elif track_id is not None:
            color = (
                int((track_id * 67) % 256),
                int((track_id * 131) % 256),
                int((track_id * 197) % 256),
            )
            badge = ""
        else:
            color = (0, 255, 0)
            badge = ""

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2 if not is_violation else 3)

        # Bottom-centre reference dot
        cx = int((x1 + x2) / 2)
        cv2.circle(annotated, (cx, y2), 5, (0, 0, 255) if is_violation else (255, 0, 0), -1)

        # Label
        id_str = f"#{track_id} " if track_id is not None else ""
        speed_str = f" | {speed_px:.0f} px/s" if speed_px is not None else ""
        label = f"{id_str}{vehicle.get('class_name', '')}{speed_str}{badge}"

        # Draw text with dark background for crisp readability
        text_y = max(y1 - 10, 20)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, text_y - th - 4), (x1 + tw + 4, text_y + 2), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return annotated
