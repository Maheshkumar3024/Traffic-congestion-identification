from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np

# Load video file
cap = cv2.VideoCapture("../Videos/cars.mp4")  # Replace with your video path

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Traffic thresholds
LOW_THRESHOLD = 10
MODERATE_THRESHOLD = 30

prev_frame_time = 0
new_frame_time = 0

# Heatmap configuration
GRID_SIZE = 10  # Number of grid cells along one dimension

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    new_frame_time = time.time()
    results = model(img, stream=True)

    # Initialize counters and heatmap
    vehicle_counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
    height, width, _ = img.shape
    cell_height = height // GRID_SIZE
    cell_width = width // GRID_SIZE
    heatmap = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1

                # Map bounding box center to heatmap grid
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                grid_x = center_x // cell_width
                grid_y = center_y // cell_height

                # Increment heatmap count for the corresponding cell
                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    heatmap[grid_y][grid_x] += 1

            # Annotate image
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Normalize the heatmap values
    max_count = max(max(row) for row in heatmap)
    if max_count > 0:
        heatmap = [[count / max_count for count in row] for row in heatmap]

    # Generate the heatmap overlay
    heatmap_overlay = np.zeros_like(img, dtype=np.uint8)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            intensity = heatmap[y][x]
            color = cv2.applyColorMap(np.uint8([[intensity * 255]]), cv2.COLORMAP_JET)[0][0]
            # Fill the corresponding grid cell with the color
            cv2.rectangle(heatmap_overlay,
                          (x * cell_width, y * cell_height),
                          ((x + 1) * cell_width, (y + 1) * cell_height),
                          color.tolist(),
                          thickness=-1)

    # Blend the heatmap overlay with the original image
    alpha = 0.5  # Transparency factor
    blended_frame = cv2.addWeighted(heatmap_overlay, alpha, img, 1 - alpha, 0)

    # Calculate traffic density
    total_vehicles = sum(vehicle_counts.values())
    if total_vehicles <= LOW_THRESHOLD:
        congestion_level = "Low"
        color = (0, 255, 0)  # Green
    elif total_vehicles <= MODERATE_THRESHOLD:
        congestion_level = "Moderate"
        color = (0, 255, 255)  # Yellow
    else:
        congestion_level = "High"
        color = (0, 0, 255)  # Red

    # Display results
    cvzone.putTextRect(blended_frame, f'Traffic Congestion: {congestion_level}', (50, 50), scale=2, thickness=2, colorR=color)
    cvzone.putTextRect(blended_frame, f'Total Vehicles: {total_vehicles}', (50, 100), scale=2, thickness=2, colorR=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Show frame with heatmap
    cv2.imshow("Traffic Heatmap", blended_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit on 'q' key press

cap.release()
cv2.destroyAllWindows()
