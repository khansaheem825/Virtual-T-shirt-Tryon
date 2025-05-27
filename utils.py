import cv2
import numpy as np

last_center_x = None
last_top_y = None
smoothing_factor = 0.2

def smooth_transition(current, last):
    """ Smooth transition between current and previous positions for less jitter. """
    if last is None:
        return current
    return int((1 - smoothing_factor) * last + smoothing_factor * current)

def overlay_tshirt(frame, bbox, tshirt_img):
    """ Overlay the t-shirt on the detected person with resizing and smoothing. """
    global last_center_x, last_top_y

    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape

    torso_width = x2 - x1
    torso_height = y2 - y1

    center_x = x1 + torso_width // 2
    top_y = y1 + int(torso_height * 0.15) 

    center_x = smooth_transition(center_x, last_center_x)
    top_y = smooth_transition(top_y, last_top_y)
    last_center_x = center_x
    last_top_y = top_y

    shirt_width = int(torso_width * 1.5)  
    shirt_height = int(torso_height * 0.9) 

    tshirt_resized = cv2.resize(tshirt_img, (shirt_width, shirt_height))

    b, g, r, a = cv2.split(tshirt_resized)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    x_start = center_x - shirt_width // 2
    y_start = top_y
    x_end = x_start + shirt_width
    y_end = y_start + shirt_height

    if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
        print("T-shirt overlay out of bounds, skipping...")
        return frame

    roi = frame[y_start:y_end, x_start:x_end]

    img_bg = cv2.bitwise_and(roi, 255 - mask)  
    img_fg = cv2.bitwise_and(overlay_color, mask)  

    combined = cv2.add(img_bg, img_fg)

    frame[y_start:y_end, x_start:x_end] = combined

    return frame
