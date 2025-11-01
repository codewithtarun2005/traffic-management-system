from flask import Flask, render_template, jsonify
from ultralytics import YOLO
import cv2
import time
import os

app = Flask(__name__)

TOTAL_CYCLE_TIME = 60
YELLOW_TIME = 5
roads = ["Road 1", "Road 2", "Road 3", "Road 4"]

# Load YOLO model once globally
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = YOLO(model_path)
vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck


def get_vehicle_counts_once():
    """
    Capture traffic for all 4 roads sequentially using video or webcam.
    Automatically switches to webcam if no sample video found.
    """
    vehicle_counts = {r: 0 for r in roads}

    # Try to open webcam; if not available, use sample video
    video_source = "sample_video.mp4" if os.path.exists("sample_video.mp4") else 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("[WARNING] No camera or video file found. Skipping detection.")
        return vehicle_counts

    try:
        for i, road in enumerate(roads):
            print(f"[INFO] Capturing {road} traffic...")
            start_time = time.time()
            total_detections = 0
            frames = 0

            while time.time() - start_time < 5:  # 5 seconds per road
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                frame_vehicle_count = 0
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in vehicle_classes:
                            frame_vehicle_count += 1

                total_detections += frame_vehicle_count
                frames += 1

            avg_count = total_detections // frames if frames > 0 else 0
            vehicle_counts[road] = avg_count

    finally:
        cap.release()

    print(f"[INFO] Vehicle counts: {vehicle_counts}")
    return vehicle_counts


def calculate_signal_times(vehicle_counts):
    """
    Calculate dynamic green/yellow/red times.
    Roads 1 & 3 share same time; Roads 2 & 4 share same time.
    """
    count_13 = (vehicle_counts["Road 1"] + vehicle_counts["Road 3"]) // 2
    count_24 = (vehicle_counts["Road 2"] + vehicle_counts["Road 4"]) // 2
    total = count_13 + count_24 if (count_13 + count_24) > 0 else 1

    green_13 = int((count_13 / total) * TOTAL_CYCLE_TIME)
    green_24 = TOTAL_CYCLE_TIME - green_13
    yellow = YELLOW_TIME

    signal_times = {
        "Road 1": {"green": green_13, "yellow": yellow},
        "Road 3": {"green": green_13, "yellow": yellow},
        "Road 2": {"green": green_24, "yellow": yellow},
        "Road 4": {"green": green_24, "yellow": yellow},
    }

    for road in signal_times:
        signal_times[road]["red"] = TOTAL_CYCLE_TIME - signal_times[road]["green"] - yellow

    print(f"[INFO] Signal times: {signal_times}")
    return signal_times


@app.route("/")
def index():
    vehicle_counts = get_vehicle_counts_once()
    signal_times = calculate_signal_times(vehicle_counts)
    return render_template("index.html", vehicle_counts=vehicle_counts, signal_times=signal_times)


@app.route("/api/data")
def get_data_api():
    """API endpoint for front-end JS (optional for dynamic update)"""
    vehicle_counts = get_vehicle_counts_once()
    signal_times = calculate_signal_times(vehicle_counts)
    return jsonify({
        "vehicle_counts": vehicle_counts,
        "signal_times": signal_times
    })


if __name__ == "__main__":
    # For Render or any online platform: must bind to 0.0.0.0 and use port from environment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
