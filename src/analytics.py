"""Live Traffic Analytics module for RoadGuard AI."""

from collections import Counter


class TrafficAnalytics:
    """Aggregates vehicle detections, tracking IDs, class breakdown, and speeds across frames."""

    def __init__(self):
        """Initialize counters and data structures for video session analytics."""
        # 1. Count every vehicle detection bounding box across all frames
        self.total_detections = 0

        # 2. Store all observed class occurrences for each unique track_id:
        #    track_id -> list of class names observed across frames
        self.track_classes = {}

        # 3. Store all valid speed observations (in px/s) for each unique track_id:
        #    track_id -> list of float speed_px values
        self.track_speeds = {}

    def update(self, tracked_vehicles):
        """Record observations from the current frame.

        Args:
            tracked_vehicles: List of vehicle dicts for the current frame,
                              each containing 'class_name', 'track_id', and optional 'speed_px'.
        """
        for vehicle in tracked_vehicles:
            # Increment raw total detections for every detected bounding box
            self.total_detections += 1

            track_id = vehicle.get("track_id")
            # If no track_id assigned (e.g. tracker lost or unassigned), skip per-track aggregation
            if track_id is None:
                continue

            class_name = vehicle.get("class_name")
            if class_name:
                if track_id not in self.track_classes:
                    self.track_classes[track_id] = []
                self.track_classes[track_id].append(class_name)

            speed_px = vehicle.get("speed_px")
            if speed_px is not None:
                if track_id not in self.track_speeds:
                    self.track_speeds[track_id] = []
                self.track_speeds[track_id].append(float(speed_px))

    def get_summary(self):
        """Compute aggregated statistics for the video session.

        Returns:
            dict: Summary metrics containing:
                - 'total_detections': Total vehicle bounding boxes detected across all frames.
                - 'unique_tracked_vehicles': Total distinct non-null ByteTrack IDs observed.
                - 'class_counts': Breakdown of unique tracked IDs by dominant category.
                - 'average_speed_px': Mean speed across tracked vehicles (averaged per track first).
        """
        # 1. Distinct non-null ByteTrack IDs observed
        unique_tracked_vehicles = len(self.track_classes)

        # 2. Class counts by unique tracked IDs:
        #    Determine the dominant/most frequent class observed for each track ID
        #    to handle any transient single-frame classification flicker gracefully.
        class_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        for track_id, class_list in self.track_classes.items():
            if class_list:
                # Most common class label observed for this persistent track ID
                dominant_class = Counter(class_list).most_common(1)[0][0]
                if dominant_class in class_counts:
                    class_counts[dominant_class] += 1
                else:
                    class_counts[dominant_class] = 1

        # 3. Average speed calculation:
        #    Compute the average speed per track first (so longer-observed vehicles
        #    do not disproportionately bias the overall average), then average across tracks.
        per_track_averages = []
        for track_id, speeds in self.track_speeds.items():
            if speeds:
                track_avg = sum(speeds) / len(speeds)
                per_track_averages.append(track_avg)

        if per_track_averages:
            overall_avg_speed = round(sum(per_track_averages) / len(per_track_averages), 1)
        else:
            overall_avg_speed = None

        return {
            "total_detections": self.total_detections,
            "unique_tracked_vehicles": unique_tracked_vehicles,
            "class_counts": class_counts,
            "average_speed_px": overall_avg_speed,
        }
