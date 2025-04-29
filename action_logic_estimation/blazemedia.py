import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import uuid
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Try to load YOLO model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"YOLO model loading error: {e}")
    model = None

# Define CSV files with unique identifiers for each run
actions_csv = f"actions_{uuid.uuid4().hex[:8]}.csv"
pose_csv = f"pose_landmarks_{uuid.uuid4().hex[:8]}.csv"

# Predefined list of landmarks to track
LANDMARKS_TO_TRACK = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 
    'left_pinky', 'right_pinky', 
    'left_index', 'right_index', 
    'left_thumb', 'right_thumb', 
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 
    'left_ankle', 'right_ankle', 
    'left_heel', 'right_heel', 
    'left_foot_index', 'right_foot_index'
]

# Prepare CSV files with headers
if not os.path.exists(actions_csv):
    pd.DataFrame(columns=["frame", "person_id", "Writing_on_Board", "Using_Phone_or_Book", 
                          "Sleeping_at_Desk", "Walking_Around", "Raising_Hand", 
                          "Mobile_Detected"]).to_csv(actions_csv, index=False)

# Prepare pose landmarks CSV
landmark_columns = [
    "frame", "person_id",
    *[f"{landmark}_{coord}" for landmark in LANDMARKS_TO_TRACK 
      for coord in ["x", "y", "z", "visibility"]]
]

if not os.path.exists(pose_csv):
    pd.DataFrame(columns=landmark_columns).to_csv(pose_csv, index=False)

def get_landmark(results, landmark_name):
    """Get specific landmark coordinates from MediaPipe results."""
    if not results.pose_landmarks:
        return None
    
    try:
        landmark_index = getattr(mp_pose.PoseLandmark, landmark_name.upper())
        landmark = results.pose_landmarks.landmark[landmark_index]
        
        # Only return if landmark is visible
        if landmark.visibility > 0.5:
            return np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return None
    except AttributeError:
        # Handle cases where specific landmark might not exist
        if landmark_name.upper() == 'CHEST':
            # Calculate chest as midpoint between shoulders
            left_shoulder = get_landmark(results, 'left_shoulder')
            right_shoulder = get_landmark(results, 'right_shoulder')
            
            if left_shoulder is not None and right_shoulder is not None:
                chest = (left_shoulder + right_shoulder) / 2
                return chest
        return None

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return float('inf')
    return np.linalg.norm(point1[:2] - point2[:2])

def get_angle(point1, point2, point3):
    """Calculate angle between three points (in degrees)."""
    if any(p is None for p in [point1, point2, point3]):
        return None
    
    vector1 = point1[:2] - point2[:2]
    vector2 = point3[:2] - point2[:2]
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return None
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def detect_mobile(frame):
    """Detect mobile phones using YOLO."""
    if model is None:
        return False, None
    
    try:
        results = model(frame)
        mobile_detected = any(det[5] == 77 for det in results.xyxy[0])  # 77 is the class index for cell phone
        return mobile_detected, results
    except Exception as e:
        print(f"Mobile detection error: {e}")
        return False, None

def classify_action(results, frame_height, frame_width, mobile_detected):
    """Classifies classroom actions based on landmark positions with enhanced logic."""
    actions = {
        "Writing_on_Board": 0,
        "Using_Phone_or_Book": 0,
        "Sleeping_at_Desk": 0, 
        "Walking_Around": 0,
        "Raising_Hand": 0,
        "Mobile_Detected": mobile_detected
    }
    
    # Extract landmarks with more comprehensive detection
    landmarks = {
        "nose": get_landmark(results, "nose"),
        "left_shoulder": get_landmark(results, "left_shoulder"),
        "right_shoulder": get_landmark(results, "right_shoulder"),
        "left_elbow": get_landmark(results, "left_elbow"),
        "right_elbow": get_landmark(results, "right_elbow"),
        "left_wrist": get_landmark(results, "left_wrist"),
        "right_wrist": get_landmark(results, "right_wrist"),
        "left_hip": get_landmark(results, "left_hip"),
        "right_hip": get_landmark(results, "right_hip"),
        "left_knee": get_landmark(results, "left_knee"),
        "right_knee": get_landmark(results, "right_knee"),
        "left_ankle": get_landmark(results, "left_ankle"),
        "right_ankle": get_landmark(results, "right_ankle"),
        "chest": get_landmark(results, "chest")
    }
    
    # Scale landmarks to pixel coordinates for easier processing
    def scale_landmark(landmark):
        if landmark is not None:
            return np.array([landmark[0] * frame_width, landmark[1] * frame_height])
        return None
    
    # Scale all landmarks
    scaled_landmarks = {k: scale_landmark(v) for k, v in landmarks.items()}
    
    # Compute reference measurements
    shoulder_width = euclidean_distance(
        scaled_landmarks['left_shoulder'], 
        scaled_landmarks['right_shoulder']
    ) if scaled_landmarks['left_shoulder'] is not None and scaled_landmarks['right_shoulder'] is not None else None
    
    # Midpoint calculations
    mid_shoulder = None
    if scaled_landmarks['left_shoulder'] is not None and scaled_landmarks['right_shoulder'] is not None:
        mid_shoulder = (scaled_landmarks['left_shoulder'] + scaled_landmarks['right_shoulder']) / 2
    
    mid_hip = None
    if scaled_landmarks['left_hip'] is not None and scaled_landmarks['right_hip'] is not None:
        mid_hip = (scaled_landmarks['left_hip'] + scaled_landmarks['right_hip']) / 2
    
    # Mobile Detection Takes Precedence
    if mobile_detected:
        actions["Using_Phone_or_Book"] = 95
    
    # Static Posture Detection
    if mid_shoulder is not None and mid_hip is not None:
        # Vertical position of body
        shoulder_height = mid_shoulder[1]
        hip_height = mid_hip[1]
        
        # Check for potential desk-based activities
        if abs(shoulder_height - hip_height) < 50:  # Very close vertical positions
            # Potential sitting/desk activities
            if scaled_landmarks['left_wrist'] is not None and scaled_landmarks['right_wrist'] is not None:
                # Check wrist positions relative to body
                if (mid_shoulder[1] < scaled_landmarks['left_wrist'][1] < mid_hip[1] and
                    mid_shoulder[1] < scaled_landmarks['right_wrist'][1] < mid_hip[1]):
                    # Wrists are positioned near body, likely using phone/book
                    actions["Using_Phone_or_Book"] = 85
                else:
                    # Could be writing on board or other desk activity
                    actions["Writing_on_Board"] = 75
        
        # Check for sleeping posture
        if scaled_landmarks['nose'] is not None and mid_shoulder is not None:
            if scaled_landmarks['nose'][1] > mid_shoulder[1] + 30:
                actions["Sleeping_at_Desk"] = 90
    
    # Raising Hand Detection with more lenient criteria
    if scaled_landmarks['nose'] is not None:
        head_y = scaled_landmarks['nose'][1]
        for side in ['left', 'right']:
            wrist = scaled_landmarks[f'{side}_wrist']
            elbow = scaled_landmarks[f'{side}_elbow']
            shoulder = scaled_landmarks[f'{side}_shoulder']
            
            if (wrist is not None and 
                shoulder is not None and 
                elbow is not None):
                # More lenient hand raising detection
                if wrist[1] < head_y and abs(wrist[0] - shoulder[0]) > shoulder_width * 0.5:
                    actions["Raising_Hand"] = 85
    
    # Final thresholding - only keep actions above 75% confidence
    return {action: prob if prob >= 75 else 0 for action, prob in actions.items()}

def get_bounding_box(results, frame_width, frame_height):
    """Extract bounding box coordinates for the detected person."""
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    x_coords = [lm.x * frame_width for lm in landmarks]
    y_coords = [lm.y * frame_height for lm in landmarks]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(frame_width, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(frame_height, y_max + padding)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def main():
    # Open Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Detect mobile first
            mobile_detected, yolo_results = detect_mobile(frame)
            
            # Convert frame to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find poses
            try:
                results = pose.process(image_rgb)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Error processing pose: {e}")
                continue
            
            # If a pose is detected
            action_data = []
            pose_data = []
            if results.pose_landmarks:
                # Get bounding box
                bbox = get_bounding_box(results, frame.shape[1], frame.shape[0])
                
                # Classify action (now including mobile detection)
                action_probs = classify_action(results, frame.shape[0], frame.shape[1], mobile_detected)
                
                # Convert probabilities to data row
                action_row = [frame_count, 0] + list(action_probs.values())
                action_data.append(action_row)
                
                # Prepare pose landmarks data
                landmark_row = [frame_count, 0]
                for landmark in LANDMARKS_TO_TRACK:
                    landmark_coords = get_landmark(results, landmark)
                    if landmark_coords is not None:
                        landmark_row.extend(landmark_coords)
                    else:
                        # If landmark not found, add zeros
                        landmark_row.extend([0, 0, 0, 0])
                pose_data.append(landmark_row)
                
                # Determine box color and primary action
                box_color = (0, 255, 0)  # Default green
                if action_probs["Using_Phone_or_Book"] > 75 or action_probs["Sleeping_at_Desk"] > 75:
                    box_color = (0, 0, 255)  # Red for phone or sleeping
                
                # Draw bounding box
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Determine primary action
                primary_action = max(action_probs, key=action_probs.get)
                primary_prob = action_probs[primary_action]
                
                # Display primary action above box
                if primary_prob > 75:
                    if bbox:
                        text_y = max(0, y1 - 10)
                        cv2.putText(frame, f"{primary_action}: {primary_prob}%", 
                                    (x1, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, f"{primary_action}: {primary_prob}%", 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Save action data to CSV
            if action_data:
                df = pd.DataFrame(action_data, columns=["frame", "person_id", "Writing_on_Board", 
                                                         "Using_Phone_or_Book", "Sleeping_at_Desk", 
                                                         "Walking_Around", "Raising_Hand", 
                                                         "Mobile_Detected"])
                df.to_csv(actions_csv, mode='a', header=False, index=False)
            
            # Save pose landmarks to CSV
            if pose_data:
                df_pose = pd.DataFrame(pose_data, columns=landmark_columns)
                df_pose.to_csv(pose_csv, mode='a', header=False, index=False)

            frame_count += 1
            cv2.imshow("MediaPipe Action Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    print(f"Action data saved to {actions_csv}")
    print(f"Pose landmarks saved to {pose_csv}")

if __name__ == "__main__":
    main()