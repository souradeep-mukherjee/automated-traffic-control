import time
from ultralytics import YOLO
import cv2
import glob
import os

# Load the YOLOv8 model (use a pre-trained model such as 'yolov8n', 'yolov8s', etc.)
model = YOLO('yolov8n.pt')  # Replace with a fine-tuned model if available for specific traffic datasets

# Define the object classes to count
target_classes = {"car": 2, "bus": 5}  # Class IDs based on the YOLOv8 COCO dataset

# Get all image files with specified extensions
image_extensions = ('*.jpg', '*.png', '*.jpeg')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join("dataset", ext)))

# Dictionary to store counts for each image (assumed as representing lanes)
lane_counts = {}

# Function to simulate traffic lights
def simulate_traffic_light(lane_name, is_green):
    if is_green:
        print(f"🟢 Green Light: Lane '{lane_name}'")
    else:
        print(f"🔴 Red Light: Lane '{lane_name}'")

# Process each image to count cars and buses
for image_path in image_files:
    # Load the traffic image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        continue

    # Run YOLOv8 inference
    results = model.predict(source=image, save=False, conf=0.3)  # Adjust confidence threshold as needed

    # Initialize counts for cars and buses
    car_count = 0
    bus_count = 0

    # Parse detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == target_classes["car"]:
                car_count += 1
            elif class_id == target_classes["bus"]:
                bus_count += 1

    # Store the counts
    lane_name = os.path.basename(image_path).split('.')[0]  # Use file name as lane identifier
    lane_counts[lane_name] = {"cars": car_count, "buses": bus_count, "total": car_count + bus_count}

# Priority Scheduling
# Sort lanes by total vehicle count in descending order (highest priority first)
sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1]['total'], reverse=True)

# Process lanes in batches of 4
batch_size = 4
print("\nPriority Schedule with Traffic Lights:")
for i in range(0, len(sorted_lanes), batch_size):
    batch = sorted_lanes[i:i + batch_size]
    print(f"\nProcessing Batch {i // batch_size + 1}:")
    for priority, (lane, counts) in enumerate(batch, start=1):
        is_green = priority == 1  # Highest priority lane gets the green light
        simulate_traffic_light(lane, is_green)
        print(f"  Lane '{lane}': Cars = {counts['cars']}, Buses = {counts['buses']}, Total = {counts['total']}")
    print("\nWaiting for the next batch...")
    time.sleep(10)  # 10-second interval