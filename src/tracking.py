"""Vehicle tracking and unique ID assignment module using Ultralytics ByteTrack."""

import cv2

# Target vehicle categories from COCO dataset
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]


def track_vehicles(model, frame, confidence_threshold=0.5, tracker="bytetrack.yaml"):
    """Track vehicles across consecutive video frames using ByteTrack.

    Args:
        model: Loaded YOLO model instance.
        frame: Input image frame as a NumPy array (BGR format).
        confidence_threshold: Minimum confidence score to retain detection (default: 0.5).
        tracker: Tracker configuration file (default: 'bytetrack.yaml').

    Returns:
        A list of dictionaries for each tracked vehicle:
            - 'class_name': Name of the vehicle category (str)
            - 'confidence': Model confidence score between 0.0 and 1.0 (float)
            - 'box': Bounding box coordinates [x1, y1, x2, y2] (list of float)
            - 'track_id': Unique persistent tracking ID (int or None)
    """
    # persist=True maintains Kalman filters and track history across frames
    results = model.track(
        source=frame,
        persist=True,
        tracker=tracker,
        verbose=False,
    )[0]

    tracked_vehicles = []

    # If no objects were detected in the frame, return empty list
    if results.boxes is None or len(results.boxes) == 0:
        return tracked_vehicles

    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Safely extract tracking ID (may be None in the first detection frame)
            track_id = int(box.id[0]) if box.id is not None else None

            tracked_vehicles.append({
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "track_id": track_id,
            })

    return tracked_vehicles


def draw_tracks(frame, tracked_vehicles):
    """Draw bounding boxes, unique tracking IDs, and class labels on an image frame.

    Args:
        frame: Input image as a NumPy array (BGR format).
        tracked_vehicles: List of tracked vehicle dictionaries from track_vehicles().

    Returns:
        A copy of the image with bounding boxes, tracking IDs, and labels drawn.
    """
    annotated_frame = frame.copy()

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = [int(coord) for coord in vehicle["box"]]
        track_id = vehicle["track_id"]

        # Format label with tracking ID if available
        if track_id is not None:
            label = f"#{track_id} {vehicle['class_name']} {vehicle['confidence']:.2f}"
            # Generate a distinct deterministic color for each vehicle ID
            color = (
                int((track_id * 67) % 256),
                int((track_id * 131) % 256),
                int((track_id * 197) % 256),
            )
        else:
            label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"
            color = (0, 255, 0)

        # Draw bounding box rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label text slightly above the top-left corner
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
