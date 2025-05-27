import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ultralytics import YOLO
import cv2
from utils.overlay import overlay_tshirt
import mediapipe as mp

model = YOLO('models/yolov8n-seg.pt') 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

tshirt_img = cv2.imread('t_shirts/red_tshirt.png', cv2.IMREAD_UNCHANGED)
if tshirt_img is None:
    raise FileNotFoundError("❌ T-shirt image not found at 't_shirts/red_tshirt.png'")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 

   
    results = model(frame)[0]
    found = False

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det
        class_id = int(cls)

    
        if class_id == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                left_coords = (
                    int(left_shoulder.x * frame.shape[1]),
                    int(left_shoulder.y * frame.shape[0])
                )
                right_coords = (
                    int(right_shoulder.x * frame.shape[1]),
                    int(right_shoulder.y * frame.shape[0])
                )

            
                frame = overlay_tshirt(frame, left_coords, right_coords, tshirt_img)

            found = True
            break

    if not found:
        print("👤 No person detected in this frame.")

    cv2.imshow("👕 Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
