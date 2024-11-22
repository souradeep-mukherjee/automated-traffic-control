import time
import streamlit as st
from ultralytics import YOLO
import cv2
import glob
import os

# Load the YOLOv8 model (use a pre-trained model such as 'yolov8n', 'yolov8s', etc.)
model = YOLO('yolov8n.pt')  # Replace with a fine-tuned model if available for specific traffic datasets

# Define the object classes to count
target_classes = {"car": 2, "bus": 5}  # Class IDs based on the YOLOv8 COCO dataset

# Function to simulate traffic lights
def simulate_traffic_light(lane_name, is_green):
    if is_green:
        return f"🟢 Green Light: Lane '{lane_name}'"
    else:
        return f"🔴 Red Light: Lane '{lane_name}'"

# Streamlit app UI
st.set_page_config(layout="wide")  # Use the entire width of the screen
st.title("Automated Traffic Light Simulation")

# Apply custom CSS to stretch text and enhance appearance
st.markdown("""
    <style>
        .stTitle {
            font-size: 36px;
            font-weight: bold;
            line-height: 1.8;
            text-align: center;
        }
        .stHeader {
            font-size: 24px;
            font-weight: bold;
            line-height: 2;
            color: #333;
        }
        .stText {
            font-size: 18px;
            line-height: 2;
            font-family: 'Arial', sans-serif;
            color: #444;
        }
        .full-width {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Create a sidebar for directory input
st.sidebar.title("Image Upload Configuration")
directory_path = st.sidebar.text_input("Enter the path to the images directory:")

# Create a button to load images from the specified directory
if st.sidebar.button("Load Images from Directory"):
    if directory_path:
        # Get all image files from the specified directory
        image_extensions = ('*.jpg', '*.png', '*.jpeg')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))

        if not image_files:
            st.sidebar.error("No images found in the specified directory!")
        else:
            st.sidebar.success(f"Loaded {len(image_files)} images from '{directory_path}'")
    else:
        st.sidebar.error("Please enter a valid directory path.")

# Layout: Create two columns - left for image upload, right for results
col1, col2 = st.columns([2, 3])  # Adjusting the width ratio to give more space to the results

# Left column: Display the images (automatically update)
with col1:
    st.markdown('<p class="stHeader">Traffic Images</p>', unsafe_allow_html=True)
    if 'image_files' in locals():
        for image_path in image_files:
            image = cv2.imread(image_path)
            st.image(image, caption=f"Image from {image_path}", use_container_width=True)  # Updated line
    else:
        st.markdown('<p class="stText">No images loaded. Please specify a directory and load images.</p>', unsafe_allow_html=True)

# Right column: Display results after processing
with col2:
    st.markdown('<p class="stHeader">Traffic Light Simulation Results</p>', unsafe_allow_html=True)

    if 'image_files' in locals() and image_files:
        lane_counts = {}

        # Process uploaded images to count cars and buses
        for image_path in image_files:
            # Read the image from the path
            image = cv2.imread(image_path)
            if image is None:
                st.write(f"Could not load image: {image_path}")
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
        st.write("\n**Priority Schedule with Traffic Lights:**")
        for i in range(0, len(sorted_lanes), batch_size):
            batch = sorted_lanes[i:i + batch_size]
            st.write(f"\n**Processing Batch {i // batch_size + 1}:**")
            for priority, (lane, counts) in enumerate(batch, start=1):
                is_green = priority == 1  # Highest priority lane gets the green light
                light_status = simulate_traffic_light(lane, is_green)
                st.markdown(f"<p class='stText'>{light_status}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='stText'>  Lane '{lane}': Cars = {counts['cars']}, Buses = {counts['buses']}, Total = {counts['total']}</p>", unsafe_allow_html=True)
            st.write("\nWaiting for the next batch...")
            time.sleep(10)  # 10-second interval
    else:
        st.markdown('<p class="stText">Please specify a directory and load images to start the simulation.</p>', unsafe_allow_html=True)
