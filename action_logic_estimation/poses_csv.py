from ultralytics import YOLO
import cv2
import pandas as pd
import os

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt").to("cpu")  # Running on CPU

# Open Raspberry Pi Camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Reduce resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define CSV filename
csv_file = "poses.csv"

# Define keypoint names for better readability
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Ensure CSV file exists with proper headers
if not os.path.exists(csv_file):
    columns = ["frame", "person_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    for kp in keypoint_names:
        columns += [f"{kp}_x", f"{kp}_y"]  # Add x, y for each keypoint
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

frame_count = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Run YOLOv8 Pose inference
        results = model(frame)

        pose_data = []
        
        # Track person ID per frame
        person_id = 0
        
        for result in results:
            for box, keypoints in zip(result.boxes.xyxy, result.keypoints.xy):
                box = box.cpu().numpy().astype(int)  # Bounding box (x1, y1, x2, y2)
                keypoints = keypoints.cpu().numpy().astype(int)  # Keypoints (x, y)

                # Extract bounding box
                x1, y1, x2, y2 = box

                # Draw bounding box around detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Flatten keypoints and handle missing ones
                keypoints_flat = []
                for i in range(17):
                    if len(keypoints) > i:
                        keypoints_flat.extend(keypoints[i])  # Add x, y
                    else:
                        keypoints_flat.extend([None, None])  # Missing keypoint

                # Append data for CSV
                pose_data.append([frame_count, person_id, x1, y1, x2, y2] + keypoints_flat)

                # Increment person_id for the next detected person in the same frame
                person_id += 1

        # Save to CSV only if pose data is available
        if pose_data:
            columns = ["frame", "person_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
            for kp in keypoint_names:
                columns += [f"{kp}_x", f"{kp}_y"]
            df = pd.DataFrame(pose_data, columns=columns)
            df.to_csv(csv_file, mode='a', header=False, index=False)

        frame_count += 1

        # Show camera feed with bounding boxes (no pose overlay)
        cv2.imshow("YOLOv8 Pose Detection - Raspberry Pi", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
