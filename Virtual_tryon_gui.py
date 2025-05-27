import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading

from utils.overlay import overlay_tshirt
from ultralytics import YOLO

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class VirtualTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👕 Virtual Try-On")
        self.root.geometry("920x720")

        self.model = YOLO('models/yolov8n-seg.pt')
        self.cap = None
        self.tshirt_img = None
        self.running = False

        self.create_widgets()

    def create_widgets(self):
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)

        self.start_btn = ttk.Button(button_frame, text="▶ Start Webcam", bootstyle="success", command=self.start_webcam)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ttk.Button(button_frame, text="■ Stop Webcam", bootstyle="danger", command=self.stop_webcam, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.select_btn = ttk.Button(button_frame, text="🎽 Choose T-Shirt", bootstyle="info", command=self.select_tshirt)
        self.select_btn.grid(row=0, column=2, padx=10)

        self.status_label = ttk.Label(self.root, text="🔹 Load a T-shirt and start your webcam", bootstyle="info")
        self.status_label.pack(pady=10)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="❌ Cannot access webcam", bootstyle="danger")
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.update_feed, daemon=True).start()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        self.video_label.config(image='')
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="🛑 Webcam stopped", bootstyle="secondary")

    def select_tshirt(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if file_path:
            shirt = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if shirt is not None and shirt.shape[2] == 4:
                self.tshirt_img = shirt
                self.status_label.config(text="✅ T-shirt loaded", bootstyle="success")
            else:
                self.status_label.config(text="❌ Invalid PNG. Must have transparency.", bootstyle="danger")

    def update_feed(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            results = self.model(frame)[0]
            person_detected = False
            for det in results.boxes.data:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Person class
                    person_detected = True
                    break

            if person_detected and self.tshirt_img is not None:
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

                    frame = overlay_tshirt(frame, left_coords, right_coords, self.tshirt_img)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

if __name__ == "__main__":
    app = ttk.Window(themename="superhero")
    app.iconbitmap(default=None)
    VirtualTryOnApp(app)
    app.mainloop()
