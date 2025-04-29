from ultralytics import YOLO
import cv2
import pandas as pd
import os
import numpy as np
from collections import deque
import datetime

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt").to("cpu")  # Running on CPU

# Open Camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Define CSV file
actions_csv = "actions_with_timestamps.csv"
if not os.path.exists(actions_csv):
    pd.DataFrame(columns=["timestamp", "frame", "person_id", "action"]).to_csv(actions_csv, index=False)

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

def x_alignment(point1, point2, threshold):
    """Check if two points are aligned on the x-axis within a threshold."""
    if point1 is None or point2 is None:
        return False
    return abs(point1[0] - point2[0]) <= threshold


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
    x_alignment_threshold = norm_factor * 0.15  # Threshold for x-alignment (15% of shoulder width)
    
    # Average positions
    mid_shoulder = None
    if left_shoulder is not None and right_shoulder is not None:
        mid_shoulder = (left_shoulder + right_shoulder) / 2
    
    mid_hip = None
    if left_hip is not None and right_hip is not None:
        mid_hip = (left_hip + right_hip) / 2
    
    # ---- Action: Writing on Board ----
    # Logic: One hand is raised near shoulder height or above AND person is standing
    # BUT the wrist and shoulder are NOT aligned on x-axis (hand is ahead/writing)
    if all(x is not None for x in [left_wrist, left_shoulder, left_hip]) and left_wrist[1] <= left_shoulder[1] + 20:
        # Check if person is standing (significant distance between shoulder and hip)
        if left_shoulder[1] < left_hip[1] - norm_factor/2:
            # Check that wrist is NOT aligned with shoulder on x-axis (writing motion)
            if not x_alignment(left_wrist, left_shoulder, x_alignment_threshold):
                # Wrist should be more forward than shoulder for writing
                if left_wrist[0] > left_shoulder[0]:
                    actions["Writing_on_Board"] = 85
    
    if all(x is not None for x in [right_wrist, right_shoulder, right_hip]) and right_wrist[1] <= right_shoulder[1] + 20:
        # Check if person is standing
        if right_shoulder[1] < right_hip[1] - norm_factor/2:
            # Check that wrist is NOT aligned with shoulder on x-axis (writing motion)
            if not x_alignment(right_wrist, right_shoulder, x_alignment_threshold):
                # Wrist should be more forward than shoulder for writing
                if right_wrist[0] < right_shoulder[0]:  # Right hand will be to the left of shoulder when writing
                    actions["Writing_on_Board"] = 85
    
    # ---- Action: Using Phone or Book ----
    # NEW LOGIC: Both wrists are close together and at chest level (between shoulder and hip)
    if all(x is not None for x in [left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip]):
        # Calculate chest level (area between shoulders and hips)
        shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
        chest_level_min = shoulder_midpoint_y
        chest_level_max = shoulder_midpoint_y + (hip_midpoint_y - shoulder_midpoint_y) * 0.6  # Upper 60% between shoulder and hip
        
        # Check if both wrists are close to each other
        wrist_distance = euclidean_distance(left_wrist, right_wrist)
        wrists_are_close = (wrist_distance < norm_factor * 0.7)
        
        # Check if both wrists are at chest level
        left_wrist_at_chest = (chest_level_min <= left_wrist[1] <= chest_level_max)
        right_wrist_at_chest = (chest_level_min <= right_wrist[1] <= chest_level_max)
        
        # Detect phone/book usage when both wrists are close and at chest level
        if wrists_are_close and left_wrist_at_chest and right_wrist_at_chest:
            actions["Using_Phone_or_Book"] = 90
        
        # Additional check for single hand near face (phone call)
        elif ((left_wrist is not None and left_ear is not None and 
              euclidean_distance(left_wrist, left_ear) < norm_factor * 0.7) or
              (right_wrist is not None and right_ear is not None and 
              euclidean_distance(right_wrist, right_ear) < norm_factor * 0.7)):
            actions["Using_Phone_or_Book"] = 85
    
    # 1. Head position-based detection - primary indicator
    head_down_detected = False
    if nose is not None:
        # If shoulders are visible, check if head is significantly lower than expected position
        if left_shoulder is not None or right_shoulder is not None:
            ref_shoulder_y = left_shoulder[1] if left_shoulder is not None else right_shoulder[1]
            # Head significantly below shoulder level indicates sleeping
            if nose[1] > ref_shoulder_y:
                head_down_detected = True
                actions["Sleeping_at_Desk"] = 85
        
        # If eyes are not detected but nose is, likely head down
        if nose is not None and left_eye is None and right_eye is None:
            head_down_detected = True
            actions["Sleeping_at_Desk"] = 80
    
    # 2. Body posture-based detection - secondary indicators
    if not head_down_detected:  # Only check if not already detected by head position
        # Horizontal body orientation check (if both shoulders and hips visible)
        if all(x is not None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
            
            # Small vertical difference between shoulders and hips suggests horizontal posture
            if abs(shoulder_midpoint_y - hip_midpoint_y) < norm_factor * 0.5:
                actions["Sleeping_at_Desk"] = 80
        
        # Check for low arm position (both arms down on desk)
        if all(x is not None for x in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
            # Both wrists significantly lower than shoulders
            if (left_wrist[1] > left_shoulder[1] + norm_factor * 0.5 and 
                right_wrist[1] > right_shoulder[1] + norm_factor * 0.5):
                
                # Wrists close to each other (arms on desk)
                wrist_distance = euclidean_distance(left_wrist, right_wrist)
                if wrist_distance < norm_factor * 1.2:  # More relaxed threshold
                    actions["Sleeping_at_Desk"] = 80
    
    # 3. Visibility-based detection - tertiary indicators
    if not head_down_detected:  # Only check if not already detected
        # Few keypoints visible suggests occlusion (possibly lying down)
        visible_keypoints = sum(1 for kp in [nose, left_eye, right_eye, left_ear, right_ear, 
                                           left_shoulder, right_shoulder] if kp is not None)
        
        # If face is partially visible but upper body keypoints are missing
        if nose is not None and visible_keypoints < 4:
            actions["Sleeping_at_Desk"] = 75  # Lower confidence
    

    
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
    # AND wrist and shoulder are aligned on x-axis (vertical raising)
    if nose is not None:
        head_y = nose[1]
        
        # Check left arm
        if all(x is not None for x in [left_wrist, left_shoulder, left_elbow]) and left_wrist[1] < head_y - 20:
            # Check arm extension angle (relatively straight arm)
            arm_angle = get_angle(left_shoulder, left_elbow, left_wrist)
            if arm_angle is not None and arm_angle > 120:
                # Check if wrist and shoulder are aligned on x-axis (vertical hand raising)
                if x_alignment(left_wrist, left_shoulder, x_alignment_threshold):
                    actions["Raising_Hand"] = 90
        
        # Check right arm
        if all(x is not None for x in [right_wrist, right_shoulder, right_elbow]) and right_wrist[1] < head_y - 20:
            # Check arm extension angle (relatively straight arm)
            arm_angle = get_angle(right_shoulder, right_elbow, right_wrist)
            if arm_angle is not None and arm_angle > 120:
                # Check if wrist and shoulder are aligned on x-axis (vertical hand raising)
                if x_alignment(right_wrist, right_shoulder, x_alignment_threshold):
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
    
    # Return only actions with confidence above threshold (75%)
    detected_actions = {action: prob for action, prob in actions.items() if prob >= 75}
    return detected_actions

frame_count = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        # Get current timestamp before processing
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Process frame with model
        try:
            results = model(frame)
            action_data = []
        except Exception as e:
            print(f"Model inference error: {e}")
            continue
            
        # Safe processing of results
        for person_id, result in enumerate(results):
            try:
                # Skip if no keypoints available
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue
                if not hasattr(result.keypoints, 'xy') or result.keypoints.xy is None:
                    continue
                    
                # Skip if no boxes available
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                if not hasattr(result.boxes, 'xyxy') or result.boxes.xyxy is None:
                    continue
                
                # Process each person detected
                for box_idx, keypoints_idx in zip(range(len(result.boxes.xyxy)), range(len(result.keypoints.xy))):
                    try:
                        box = result.boxes.xyxy[box_idx].cpu().numpy().astype(int)
                        keypoints = result.keypoints.xy[keypoints_idx].cpu().numpy()
                        
                        # Ensure keypoints are in int format and handle missing ones
                        keypoints_full = []
                        for i in range(17):  # 17 keypoints in COCO format
                            if i < len(keypoints):
                                keypoints_full.append(keypoints[i].astype(int))
                            else:
                                keypoints_full.append([0, 0])
                        
                        # Classify actions
                        detected_actions = classify_action(person_id, keypoints_full)
                        
                        # Visualization
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Display actions
                        y_offset = 20
                        for action, prob in detected_actions.items():
                            cv2.putText(frame, f"{action}: {prob}%", (x1, y1 - y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            y_offset += 20
                            
                            # Record action data
                            action_data.append([current_time, frame_count, person_id, action])
                    
                    except Exception as e:
                        print(f"Error processing person {person_id}, detection {box_idx}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error processing result {person_id}: {e}")
                continue
                
        # Save data to CSV
        if action_data:
            try:
                df = pd.DataFrame(action_data, columns=["timestamp", "frame", "person_id", "action"])
                df.to_csv(actions_csv, mode='a', header=False, index=False)
            except Exception as e:
                print(f"CSV writing error: {e}")
        
        # Display frame and check for quit
        frame_count += 1
        cv2.imshow("YOLOv8 Pose-based Action Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Critical error in main loop: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Always clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")