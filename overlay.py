import cv2
import numpy as np

last_center_x = None
last_top_y = None
smoothing_factor = 0.2  

def smooth_transition(current, last):
    if last is None:
        return current
    return int((1 - smoothing_factor) * last + smoothing_factor * current)

def overlay_tshirt(frame, left_shoulder, right_shoulder, tshirt_img, y_offset=30, width_scale=2.2, height_scale=2.0):
    global last_center_x, last_top_y

    h, w, _ = frame.shape
    center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
    shoulder_width = int(abs(left_shoulder[0] - right_shoulder[0]))
    estimated_height = int(shoulder_width * height_scale)

    top_y = int((left_shoulder[1] + right_shoulder[1]) / 2) - y_offset

    center_x = smooth_transition(center_x, last_center_x)
    top_y = smooth_transition(top_y, last_top_y)
    last_center_x = center_x
    last_top_y = top_y

    shirt_width = int(shoulder_width * width_scale)
    shirt_height = estimated_height

    tshirt_resized = cv2.resize(tshirt_img, (shirt_width, shirt_height))
    b, g, r, a = cv2.split(tshirt_resized)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    x_start = center_x - shirt_width // 2
    y_start = top_y
    x_end = x_start + shirt_width
    y_end = y_start + shirt_height

    if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
        return frame

    roi = frame[y_start:y_end, x_start:x_end]
    img_bg = cv2.bitwise_and(roi, 255 - mask)
    img_fg = cv2.bitwise_and(overlay_color, mask)
    frame[y_start:y_end, x_start:x_end] = cv2.add(img_bg, img_fg)

    return frame
