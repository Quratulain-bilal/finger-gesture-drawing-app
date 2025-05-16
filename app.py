#drawing_app
import cv2
import mediapipe as mp
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Setup Tkinter Window
root = Tk()
root.title("Camera Canvas")
root.geometry("1280x720")
root.configure(bg="#1a1a1a")

# Create Canvas for Drawing
canvas = Canvas(root, bg="black", bd=0, highlightthickness=0)
canvas.pack(fill=BOTH, expand=True)

# Create Video Label
video_label = Label(canvas)
video_label.place(relx=0.5, rely=0.5, anchor=CENTER)

# Color Palette
colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Eraser": (255, 255, 255)
}

current_color = colors["Red"]  # Default color
brush_size = 10

# Drawing Variables
drawing = False
last_x, last_y = -1, -1

# Overlay for persistent drawing
overlay = None

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Function to detect raised fingers
def detect_raised_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]  # Landmark IDs for finger tips
    raised_fingers = []

    # Thumb (special case)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        raised_fingers.append(1)
    else:
        raised_fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            raised_fingers.append(1)
        else:
            raised_fingers.append(0)

    return raised_fingers

# Function to detect an open palm (clear screen)
def is_palm_gesture(hand_landmarks):
    raised_fingers = detect_raised_fingers(hand_landmarks)
    if sum(raised_fingers) == 5:  # All fingers raised
        return True
    return False

# Function to detect a thumb gesture (draw circle)
def is_thumb_gesture(hand_landmarks):
    raised_fingers = detect_raised_fingers(hand_landmarks)
    if sum(raised_fingers) == 1 and raised_fingers[0] == 1:  # Only thumb raised
        return True
    return False

# Function to draw a circle
def draw_circle(img, center, radius):
    cv2.circle(img, center, radius, current_color, brush_size)

# Function to clear the canvas
def clear_canvas():
    global overlay
    overlay = np.zeros_like(overlay) if overlay is not None else None

# Create Paint Toolbar
toolbar = Frame(root, bg="#2d2d2d", bd=2)
toolbar.place(relx=0.02, rely=0.02)

# Add Color Buttons
for color_name, color_val in colors.items():
    btn = Frame(toolbar,
                bg=f"#{color_val[2]:02x}{color_val[1]:02x}{color_val[0]:02x}",
                width=40,
                height=40,
                bd=2,
                relief="ridge")
    btn.pack(side=LEFT, padx=5, pady=5)

    def set_color(c=color_val, name=color_name):
        global current_color
        current_color = c
        btn.config(relief="sunken")
        for sibling in toolbar.winfo_children():
            if sibling != btn:
                sibling.config(relief="ridge")

    btn.bind("<Button-1>", lambda e, c=set_color: c())

# Add Clear Button
clear_btn = Button(toolbar, text="Clear", bg="#4d4d4d", fg="white", command=clear_canvas)
clear_btn.pack(side=LEFT, padx=5, pady=5)

def update_frame():
    global last_x, last_y, drawing, overlay, current_color, brush_size

    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if overlay is None or overlay.shape[:2] != img.shape[:2]:
        overlay = np.zeros_like(img)

    results = hands.process(imgRGB)
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Detect palm gesture (clear screen)
            if is_palm_gesture(handLms):
                clear_canvas()  # Clear the canvas
                continue

            # Detect thumb gesture (draw circle)
            if is_thumb_gesture(handLms):
                index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                center = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))
                draw_circle(overlay, center, 50)  # Draw a circle
                continue

            # Detect raised fingers for color change
            raised_fingers = detect_raised_fingers(handLms)
            num_raised = sum(raised_fingers)
            if num_raised > 0 and num_raised <= len(colors):
                color_keys = list(colors.keys())
                current_color = colors[color_keys[num_raised - 1]]

            # Get Thumb and Index Finger Tips
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb and index finger
            thumb_x, thumb_y = int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0])
            index_x, index_y = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])
            distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Adjust brush size based on distance
            brush_size = int(distance / 10)
            brush_size = max(2, min(brush_size, 20))  # Clamp brush size between 2 and 20

            # Get Index Finger Tip Position
            x, y = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])

            # Drawing Logic
            if current_color != colors["Eraser"]:
                if not drawing:
                    drawing = True
                    last_x, last_y = x, y
                else:
                    cv2.line(overlay, (last_x, last_y), (x, y), current_color, brush_size)
                    last_x, last_y = x, y
            else:
                # Eraser mode
                cv2.circle(overlay, (x, y), brush_size * 2, (0, 0, 0), -1)

    # Update drawing state
    if hand_detected:
        drawing = True
    else:
        drawing = False

    # Combine overlay with current frame
    combined = cv2.addWeighted(img, 1, overlay, 1, 0)

    # Convert to Tkinter Image
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(combined)
    img_tk = ImageTk.PhotoImage(img_pil)

    video_label.img = img_tk
    video_label.configure(image=img_tk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
