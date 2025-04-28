from ultralytics import YOLO
import cv2
from utils.overlay import overlay_tshirt
import mediapipe as mp

# Load YOLOv8 segmentation model
model = YOLO('models/yolov8n-seg.pt') 

# Initialize MediaPipe Pose for pose landmarks
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load transparent T-shirt image
tshirt_img = cv2.imread('t_shirts/red_tshirt.png', cv2.IMREAD_UNCHANGED)
if tshirt_img is None:
    raise FileNotFoundError("T-shirt image not found at t_shirts/red_tshirt.png")

# Start the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO detection
    results = model(frame)[0]

    found = False
    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det
        class_id = int(cls)

        # Ensure the detection corresponds to the 'person' class
        if class_id == 0:  # 'person' class (full-body)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Apply pose estimation for better placement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                # Get shoulder landmarks for better alignment
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                center_x = int((left_shoulder.x + right_shoulder.x) * frame.shape[1] / 2)
                center_y = int((left_shoulder.y + right_shoulder.y) * frame.shape[0] / 2)

                # Place the T-shirt using the overlay function
                frame = overlay_tshirt(frame, (x1, y1, x2, y2), tshirt_img)

            # Draw bounding box for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            found = True
            break

    if not found:
        print("No person detected in this frame.")

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
