from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Open Raspberry Pi camera
cap = cv2.VideoCapture(0)  # Change index if using USB camera

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Initialize person count
    person_count = 0

    # Loop through detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Get class ID
            if model.names[cls] == "person":
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display status
    text = "Alone" if person_count == 1 else "Multiple Persons" if person_count > 1 else "No Person"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Person Detection - Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
