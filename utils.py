import cv2
import numpy as np

# Global for smoother fitting
last_center_x = None
last_top_y = None
smoothing_factor = 0.2  # [0.0 - 1.0], higher = smoother

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

    # Calculate torso width and height (bounding box dimensions)
    torso_width = x2 - x1
    torso_height = y2 - y1

    # Calculate the center of the torso (used for T-shirt placement)
    center_x = x1 + torso_width // 2
    top_y = y1 + int(torso_height * 0.15)  # Starting from a bit below the neck area

    # Smooth transition for center_x and top_y to reduce jittering
    center_x = smooth_transition(center_x, last_center_x)
    top_y = smooth_transition(top_y, last_top_y)
    last_center_x = center_x
    last_top_y = top_y

    # Resize the T-shirt dynamically based on torso size
    shirt_width = int(torso_width * 1.5)  # Slightly wider than torso
    shirt_height = int(torso_height * 0.9)  # Make it fit properly vertically

    tshirt_resized = cv2.resize(tshirt_img, (shirt_width, shirt_height))

    # Extract alpha channel for transparent overlay
    b, g, r, a = cv2.split(tshirt_resized)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    # Calculate position to center the T-shirt on the torso
    x_start = center_x - shirt_width // 2
    y_start = top_y
    x_end = x_start + shirt_width
    y_end = y_start + shirt_height

    # Boundary check (if the T-shirt goes out of frame)
    if x_start < 0 or y_start < 0 or x_end > w or y_end > h:
        print("T-shirt overlay out of bounds, skipping...")
        return frame

    # Extract the region of interest (ROI) where T-shirt will go
    roi = frame[y_start:y_end, x_start:x_end]

    # Apply alpha masking to blend the T-shirt with the background (ROI)
    img_bg = cv2.bitwise_and(roi, 255 - mask)  # Background
    img_fg = cv2.bitwise_and(overlay_color, mask)  # Foreground (T-shirt)

    # Combine the background and foreground
    combined = cv2.add(img_bg, img_fg)

    # Place the T-shirt on the frame
    frame[y_start:y_end, x_start:x_end] = combined

    return frame
