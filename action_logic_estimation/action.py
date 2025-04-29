from ultralytics import YOLO
import cv2
import pandas as pd
import os
import numpy as np

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt").to("cpu")  # Running on CPU

# Open Raspberry Pi Camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Reduce resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define CSV filenames
pose_csv = "poses.csv"
actions_csv = "actions.csv"

# Define keypoint names
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Ensure CSV files exist with proper headers
if not os.path.exists(actions_csv):
    pd.DataFrame(columns=["frame", "person_id", "Teaching", "Writing", "Studying", "Listenting", "Walking", "Raising_Hand", "Typing"]).to_csv(actions_csv, index=False)

def classify_action(keypoints):
    """Classifies an action based on keypoint positions."""
    actions = {
        "Teaching": 0,
        "Writing": 0,
        "Studying": 0,
        "Listenting": 0,
        "Walking": 0,
        "Raising_Hand": 0,
        "Typing": 0
    }
    
    # Example logic (can be refined with more data)
    if keypoints[6][1] < keypoints[8][1] and keypoints[5][1] < keypoints[7][1] and keypoints[6][1] < keypoints[0][1] and keypoints[5][1] < keypoints[0][1]:  
        actions["Teaching"] = 90  

# Writing: Wrist lower than elbow and positioned in front of body  
    if keypoints[9][1] > keypoints[7][1] and abs(keypoints[9][0] - keypoints[7][0]) < 30:  
        actions["Writing"] = 85  

# Sleeping: Head significantly lower than hips  
    if keypoints[0][1] > keypoints[11][1] and keypoints[0][1] > keypoints[12][1] and abs(keypoints[11][1] - keypoints[12][1]) < 20:  # Hips are at the same level (lying down)
        actions["Listenting"] = 95  

# Walking: Legs apart, one foot in front of the other, and hip movement  
    if abs(keypoints[11][0] - keypoints[12][0]) > 50 and abs(keypoints[13][0] - keypoints[14][0]) > 20:  
        actions["Walking"] = 80  

# Raising Hand: One hand raised significantly higher than the shoulder level  
    if (keypoints[10][1] < keypoints[6][1] and keypoints[10][1] < keypoints[8][1]) and (keypoints[9][1] > keypoints[7][1] or keypoints[9][1] > keypoints[5][1]):  # The other hand remains down
        actions["Raising_Hand"] = 88  

# Typing: Both wrists near chest level, elbows bent downward  
    if keypoints[9][1] > keypoints[11][1] and keypoints[10][1] > keypoints[12][1] and abs(keypoints[9][1] - keypoints[10][1]) < 10:  # Both wrists aligned
        actions["Typing"] = 85  
    
    # Normalize and apply threshold
    for action, prob in actions.items():
        actions[action] = prob if prob >= 75 else 0
    
    return actions

frame_count = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Run YOLOv8 Pose inference
        results = model(frame)
        action_data = []
        
        # Process detected people
        for person_id, result in enumerate(results):
            for box, keypoints in zip(result.boxes.xyxy, result.keypoints.xy):
                box = box.cpu().numpy().astype(int)  # Bounding box (x1, y1, x2, y2)
                keypoints = keypoints.cpu().numpy().astype(int)  # Keypoints (x, y)

                # Ensure all 17 keypoints exist, otherwise fill with None
                keypoints_full = [keypoints[i] if i < len(keypoints) else [None, None] for i in range(17)]
                
                # Classify action
                action_probs = classify_action(keypoints_full)
                
                # Append action data
                action_data.append([frame_count, person_id] + list(action_probs.values()))

                # Draw bounding box and action label
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display detected actions on screen
                y_offset = 20
                for action, prob in action_probs.items():
                    if prob > 75:
                        cv2.putText(frame, f"{action}: {prob}%", (x1, y1 - y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        y_offset += 20
        
        # Save actions to CSV
        if action_data:
            df = pd.DataFrame(action_data, columns=["frame", "person_id", "Teaching", "Writing", "Studying", "Sleeping", "Walking", "Raising_Hand", "Typing"])
            df.to_csv(actions_csv, mode='a', header=False, index=False)

        frame_count += 1

        # Show camera feed
        cv2.imshow("YOLOv8 Pose-based Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
