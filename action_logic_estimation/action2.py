from ultralytics import YOLO
import cv2
import pandas as pd
import os
import numpy as np
from collections import deque

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt").to("cpu")  # Running on CPU

# Open Camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define CSV file
actions_csv = "actions.csv"
if not os.path.exists(actions_csv):
    pd.DataFrame(columns=["frame", "person_id", "Writing_on_Board", "Using_Phone_or_Book", 
                          "Sleeping_at_Desk", "Walking_Around", "Raising_Hand"]).to_csv(actions_csv, index=False)

# Create tracking history for temporal consistency
action_history = {}  # {person_id: {action: deque(recent_probabilities)}}
HISTORY_LENGTH = 5  # Number of frames to maintain history

# Keypoint indices for easier reference
KEYPOINTS = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16
}

def get_keypoint(name, keypoints):
    """Returns keypoint coordinates if valid, else None."""
    index = KEYPOINTS.get(name)
    if index is not None and 0 <= index < len(keypoints):
        kp = keypoints[index]
        if kp is not None and not np.array_equal(kp, [0, 0]):
            return np.array(kp)
    return None

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return float('inf')
    return np.linalg.norm(point1 - point2)

def get_angle(point1, point2, point3):
    """Calculate angle between three points (in degrees)."""
    if any(p is None for p in [point1, point2, point3]):
        return None
    
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return None
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    # Ensure cos_angle is within [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def classify_action(person_id, keypoints):
    """Classifies classroom actions based on keypoint positions with enhanced logic."""
    actions = {
        "Writing_on_Board": 0,
        "Using_Phone_or_Book": 0,
        "Sleeping_at_Desk": 0, 
        "Walking_Around": 0,
        "Raising_Hand": 0
    }
    
    # Extract keypoints
    nose = get_keypoint("nose", keypoints)
    left_eye = get_keypoint("left_eye", keypoints)
    right_eye = get_keypoint("right_eye", keypoints)
    left_ear = get_keypoint("left_ear", keypoints)
    right_ear = get_keypoint("right_ear", keypoints)
    left_shoulder = get_keypoint("left_shoulder", keypoints)
    right_shoulder = get_keypoint("right_shoulder", keypoints)
    left_elbow = get_keypoint("left_elbow", keypoints)
    right_elbow = get_keypoint("right_elbow", keypoints)
    left_wrist = get_keypoint("left_wrist", keypoints)
    right_wrist = get_keypoint("right_wrist", keypoints)
    left_hip = get_keypoint("left_hip", keypoints)
    right_hip = get_keypoint("right_hip", keypoints)
    left_knee = get_keypoint("left_knee", keypoints)
    right_knee = get_keypoint("right_knee", keypoints)
    left_ankle = get_keypoint("left_ankle", keypoints)
    right_ankle = get_keypoint("right_ankle", keypoints)
    
    # Calculate important reference measurements
    shoulder_width = euclidean_distance(left_shoulder, right_shoulder) if left_shoulder is not None and right_shoulder is not None else None
    hip_width = euclidean_distance(left_hip, right_hip) if left_hip is not None and right_hip is not None else None
    
    # Reference height (top of head to ankle)
    body_height = None
    if nose is not None and (left_ankle is not None or right_ankle is not None):
        ankle_y = min(left_ankle[1] if left_ankle is not None else float('inf'), 
                     right_ankle[1] if right_ankle is not None else float('inf'))
        body_height = ankle_y - nose[1]
    
    # Normalized measurements based on body dimensions
    norm_factor = shoulder_width if shoulder_width is not None else 50  # Default if can't calculate
    
    # Average positions
    mid_shoulder = None
    if left_shoulder is not None and right_shoulder is not None:
        mid_shoulder = (left_shoulder + right_shoulder) / 2
    
    mid_hip = None
    if left_hip is not None and right_hip is not None:
        mid_hip = (left_hip + right_hip) / 2
    
    # ---- Action: Writing on Board ----
    # Logic: One hand is raised near shoulder height or above it, and person is in standing position
    if all(x is not None for x in [left_wrist, left_shoulder, left_hip]) and left_wrist[1] <= left_shoulder[1] + 20:
        # Check if person is standing (significant distance between shoulder and hip)
        if left_shoulder[1] < left_hip[1] - norm_factor/2:
            actions["Writing_on_Board"] = 85
    
    if all(x is not None for x in [right_wrist, right_shoulder, right_hip]) and right_wrist[1] <= right_shoulder[1] + 20:
        # Check if person is standing
        if right_shoulder[1] < right_hip[1] - norm_factor/2:
            actions["Writing_on_Board"] = 85
    
    # ---- Action: Using Phone or Book ----
    # Logic: Both hands near each other in front of the body between chest and waist level
    if all(x is not None for x in [left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]):
        shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
        
        wrist_distance = euclidean_distance(left_wrist, right_wrist)
        
        # Check if wrists are between chest and waist level
        if (shoulder_midpoint_y < left_wrist[1] < hip_midpoint_y and 
            shoulder_midpoint_y < right_wrist[1] < hip_midpoint_y):
            
            # Check if wrists are close to each other (normalized by shoulder width)
            if wrist_distance < norm_factor * 0.8:
                # Check if person is looking down
                if nose is not None and mid_shoulder is not None:
                    head_tilt = nose[1] - mid_shoulder[1]
                    if head_tilt < -5:  # Head is tilted downward
                        actions["Using_Phone_or_Book"] = 92
    
    # ---- Action: Sleeping at Desk ----
    # Logic: Head position is low and eyes are not detected, or head is on desk level
    if nose is not None and left_shoulder is not None and right_shoulder is not None:
        # Calculate desk level (estimate as shoulder height + 20% of shoulder-to-hip distance)
        desk_level_y = None
        if mid_shoulder is not None and mid_hip is not None:
            shoulder_to_hip = mid_hip[1] - mid_shoulder[1]
            desk_level_y = mid_shoulder[1] + shoulder_to_hip * 0.2
            
            # Check if head is near or below desk level
            if nose[1] >= desk_level_y:
                actions["Sleeping_at_Desk"] = 90
        
        # Additional check: if eyes are not detected but nose/ears are, person might be facing down
        if nose is not None and (left_ear is not None or right_ear is not None):
            if left_eye is None and right_eye is None:
                actions["Sleeping_at_Desk"] = 85
    
    # ---- Action: Walking Around ----
    # Logic: Detect walking motion from leg positions and overall posture
    if left_hip is not None and right_hip is not None and mid_shoulder is not None:
        # Check if person is standing
        if mid_shoulder[1] < mid_hip[1] - norm_factor/2:
            # Check for leg movement (stride)
            if left_knee is not None and right_knee is not None and left_ankle is not None and right_ankle is not None:
                knee_horizontal_distance = abs(left_knee[0] - right_knee[0])
                ankle_horizontal_distance = abs(left_ankle[0] - right_ankle[0])
                
                # If knees or ankles are apart horizontally (normalized by hip width)
                if hip_width is not None and (knee_horizontal_distance > hip_width * 0.7 or 
                                             ankle_horizontal_distance > hip_width * 0.7):
                    actions["Walking_Around"] = 85
    
    # ---- Action: Raising Hand ----
    # Logic: One arm is extended upward with wrist above head level
    if nose is not None:
        head_y = nose[1]
        
        # Check if either wrist is significantly above head level
        if left_wrist is not None and left_wrist[1] < head_y - 20:
            # Check arm extension angle
            if left_elbow is not None and left_shoulder is not None:
                arm_angle = get_angle(left_shoulder, left_elbow, left_wrist)
                if arm_angle is not None and arm_angle > 120:  # Relatively straight arm
                    actions["Raising_Hand"] = 90
        
        if right_wrist is not None and right_wrist[1] < head_y - 20:
            # Check arm extension angle
            if right_elbow is not None and right_shoulder is not None:
                arm_angle = get_angle(right_shoulder, right_elbow, right_wrist)
                if arm_angle is not None and arm_angle > 120:  # Relatively straight arm
                    actions["Raising_Hand"] = 90
    
    # Apply temporal consistency using action history
    if person_id not in action_history:
        action_history[person_id] = {action: deque([0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) 
                                    for action in actions.keys()}
    
    for action, prob in actions.items():
        # Update history with current probability
        action_history[person_id][action].append(prob)
        
        # Calculate weighted average (more recent frames have higher weight)
        weights = np.linspace(0.5, 1.0, HISTORY_LENGTH)
        weighted_history = np.array(list(action_history[person_id][action])) * weights
        smoothed_prob = np.sum(weighted_history) / np.sum(weights)
        
        # Update action probability with smoothed value
        actions[action] = int(smoothed_prob)
    
    return {action: prob if prob >= 75 else 0 for action, prob in actions.items()}

frame_count = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        results = model(frame)
        action_data = []

        for person_id, result in enumerate(results):
            for box, keypoints in zip(result.boxes.xyxy, result.keypoints.xy):
                box = box.cpu().numpy().astype(int)
                keypoints = keypoints.cpu().numpy().astype(int)
                keypoints_full = [keypoints[i] if i < len(keypoints) else [0, 0] for i in range(17)]

                action_probs = classify_action(person_id, keypoints_full)
                action_data.append([frame_count, person_id] + list(action_probs.values()))

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset = 20
                for action, prob in action_probs.items():
                    if prob > 75:
                        cv2.putText(frame, f"{action}: {prob}%", (x1, y1 - y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        y_offset += 20

        if action_data:
            df = pd.DataFrame(action_data, columns=["frame", "person_id", "Writing_on_Board", 
                                                     "Using_Phone_or_Book", "Sleeping_at_Desk", 
                                                     "Walking_Around", "Raising_Hand"])
            df.to_csv(actions_csv, mode='a', header=False, index=False)

        frame_count += 1
        cv2.imshow("YOLOv8 Pose-based Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()