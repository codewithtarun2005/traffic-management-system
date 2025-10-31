from flask import Flask, render_template
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

TOTAL_CYCLE_TIME = 60
YELLOW_TIME = 5
roads = ["Road 1", "Road 2", "Road 3", "Road 4"]

# Load YOLO model
model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck

# Global variables to store results after one camera run
vehicle_counts_result = {}
signal_times_result = {}

def get_vehicle_counts_once():
    """
    Run webcam once to capture all 4 roads sequentially
    and count vehicles using YOLOv8.
    """
    cap = cv2.VideoCapture(0)
    vehicle_counts = {}

    try:
        for i, road in enumerate(roads):
            print(f"[INFO] Capturing {road} traffic...")
            start_time = time.time()
            count = 0
            frames = 0

            while time.time() - start_time < 5:  # 5 sec per road
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                frame_count = 0
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in vehicle_classes:
                            frame_count += 1
                count += frame_count
                frames += 1

                # Optional: show live feed
                annotated_frame = results[0].plot()
                cv2.imshow(f"{road} - Live Feed", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            avg_count = count // frames if frames > 0 else 0
            vehicle_counts[road] = avg_count
            cv2.destroyAllWindows()

    finally:
        cap.release()
        cv2.destroyAllWindows()  # Camera closed permanently

    return vehicle_counts

def calculate_signal_times(vehicle_counts):
    """
    Calculate signal times so that opposite roads have same timing.
    Road1 & Road3 same, Road2 & Road4 same.
    Total cycle = 60 sec, yellow fixed = 5 sec, red = remaining.
    """
    # Average opposite roads
    count_13 = (vehicle_counts["Road 1"] + vehicle_counts["Road 3"]) // 2
    count_24 = (vehicle_counts["Road 2"] + vehicle_counts["Road 4"]) // 2
    total = count_13 + count_24
    if total == 0:
        total = 1

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

    return signal_times

# Run camera once at start
vehicle_counts_result = get_vehicle_counts_once()
signal_times_result = calculate_signal_times(vehicle_counts_result)

@app.route("/")
def index():
    return render_template("index.html",
                           vehicle_counts=vehicle_counts_result,
                           signal_times=signal_times_result)

if __name__ == "__main__":
    app.run(debug=True)
