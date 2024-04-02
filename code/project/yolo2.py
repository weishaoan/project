import csv
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

video_path = "project\\video\\721558923.288186.mp4"
cap = cv2.VideoCapture(video_path)

# Track history and data storage
track_history = defaultdict(lambda: [])
track_data = defaultdict(list)

frame_number = 0  # Initialize frame counter

while cap.isOpened():
    success, frame = cap.read()

    if success:
        #物件追蹤
        results = model.track(frame, persist=True)

        #軌跡更新
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize results
        annotated_frame = results[0].plot()

        # Process each tracked object
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            center_y = y + h / 2
            track.append((float(x), float(center_y)))  # x, y center point
            #initialize speed and direction
            speed = 0
            direction = ""
             
            # Calculate distance speed and direction
            #確保追蹤歷史中至少有兩個點。
            if len(track) >= 2:
                dx, dy = np.diff(track[-2:], axis=0)[0]
                speed = np.sqrt(dx**2 + dy**2)  # np.sqrt 求平方根
                angle = np.degrees(np.arctan2(dy, dx))  #arctan2 計算兩點之間的角度  -π 到 π degree 此函數將角度從弧度轉換為度
                if -45 <= angle < 45:
                    direction = "East"
                elif 45 <= angle < 135:
                    direction = "North"
                elif -135 <= angle < -45:
                    direction = "South"
                else:
                    direction = "West"

                distance = np.sum(np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1)))
                print(f"ID {track_id}: coordination = ({int(x)},{int(y)}), Distance = {distance:.2f} pixels, speed={speed}, direction = {direction}")
                track_data[track_id].append((frame_number, x, center_y, distance, speed, direction))

            if len(track) > 50:
                track.pop(0)

            # 畫軌跡線
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)

        # Display frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        # Update frame counter
        frame_number += 1

        # Exit conditions
        if key == ord('q'):
            break
        elif key == ord('p'):
            while True:
                key = cv2.waitKey(1)
                if key == ord('p') or key == ord('q'):
                    break
            if key == ord('q'):
                break
    elif not success:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Write track data to CSV file
with open('track_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # 寫入標題行，包括所需的列名
    writer.writerow(['Frame Number', 'Track ID', 'Y', 'Distance Traveled', 'Speed', 'Direction'])
    for track_id, data in track_data.items():
        for frame_info in data:
            frame_number, x, y, distance, speed, direction = frame_info
            writer.writerow([track_id] + list(frame_info))