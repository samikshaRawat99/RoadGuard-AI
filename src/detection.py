"""Vehicle and object detection module using Ultralytics YOLO."""

import cv2
from ultralytics import YOLO

# Target vehicle categories from COCO dataset
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]


def load_model(model_path="yolo11n.pt"):
    """Load and return the pretrained YOLO model."""
    model = YOLO(model_path)
    return model


def detect_vehicles(model, frame, confidence_threshold=0.5):
    """Detect vehicles in an image frame and return filtered detections.

    Args:
        model: Loaded YOLO model instance.
        frame: Input image (numpy array from OpenCV or image path string).
        confidence_threshold: Minimum confidence score to retain detection (default: 0.5).

    Returns:
        A list of dictionaries, each containing:
            - 'class_name': Name of the vehicle category (str)
            - 'confidence': Model confidence score between 0.0 and 1.0 (float)
            - 'box': Bounding box coordinates [x1, y1, x2, y2] (list of float)
    """
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and class labels on an image frame.

    Args:
        frame: Input image as a numpy array (BGR format from OpenCV).
        detections: List of detection dictionaries from detect_vehicles().

    Returns:
        A copy of the image with bounding boxes and labels drawn.
    """
    annotated_frame = frame.copy()

    for vehicle in detections:
        x1, y1, x2, y2 = [int(coord) for coord in vehicle["box"]]
        label = f"{vehicle['class_name']} {vehicle['confidence']:.2f}"

        # Draw green bounding box rectangle around the vehicle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label text slightly above the top-left corner
        text_y = max(y1 - 10, 20)
        cv2.putText(
            annotated_frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


if __name__ == "__main__":
    # Test execution for Stage 2
    model = load_model("yolo11n.pt")
    input_path = "data/images/traffic.jpg"
    output_path = "outputs/violations/annotated_traffic.jpg"

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image at {input_path}")
    else:
        detected = detect_vehicles(model, image, confidence_threshold=0.5)
        annotated_image = draw_detections(image, detected)
        cv2.imwrite(output_path, annotated_image)
        print(f"Detected {len(detected)} vehicle(s).")
        print(f"Saved annotated image to: {output_path}")
