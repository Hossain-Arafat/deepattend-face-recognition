import cv2
import numpy as np
import tensorflow as tf
import json
import time
import pandas as pd
from datetime import datetime
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model(
    "DeepAttend.keras",
    compile=False
)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7
STABILITY_TIME = 2  # seconds

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = None
running = False
attendance = {}
stable_label = None
stable_start = None
ready_to_capture = False
current_detected = None

root = Tk()
root.title("Attendance System")
root.geometry("900x700")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Idle", font=("Arial", 14))
status_label.pack(pady=10)

def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()

def update_frame():
    global stable_label, stable_start, ready_to_capture, current_detected

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected_label = None
    confidence = 0

    if len(faces) > 0:
        # take first face only
        x, y, w, h = faces[0]

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)
        confidence = np.max(preds)
        label_index = np.argmax(preds)

        if confidence >= CONFIDENCE_THRESHOLD:
            detected_label = class_names[label_index]
            color = (0, 255, 0)
            text = f"{detected_label} ({confidence:.2f})"
        else:
            detected_label = None
            color = (0, 0, 255)
            text = f"Unknown ({confidence:.2f})"

        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display_frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    current_time = time.time()

    if detected_label is not None:
        if stable_label == detected_label:
            if current_time - stable_start >= STABILITY_TIME:
                ready_to_capture = True
                current_detected = detected_label
                status_label.config(text=f"Ready: {detected_label}")
        else:
            stable_label = detected_label
            stable_start = current_time
            ready_to_capture = False
            status_label.config(text="Stabilizing...")
    else:
        stable_label = None
        ready_to_capture = False
        current_detected = None
        status_label.config(text="Face not recognized")

    img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

def capture_attendance():
    global attendance

    if not ready_to_capture or current_detected is None:
        messagebox.showerror("Error", "Face not stable or not recognized")
        return

    student_id = current_detected
    today = datetime.now().strftime("%Y-%m-%d")

    key = f"{student_id}_{today}"

    if key not in attendance:
        attendance[key] = {
            "Student_ID": student_id,
            "Date": today,
            "Time": datetime.now().strftime("%H:%M:%S")
        }
        messagebox.showinfo("Success", f"{student_id} marked present")
    else:
        messagebox.showwarning("Warning", f"{student_id} already marked today")

def save_attendance():
    if not attendance:
        messagebox.showwarning("Warning", "No attendance to save")
        return

    df = pd.DataFrame(attendance.values())
    filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    messagebox.showinfo("Saved", f"Attendance saved as {filename}")

btn_frame = Frame(root)
btn_frame.pack(pady=20)

Button(btn_frame, text="Start Camera", command=start_camera, width=15).grid(row=0, column=0, padx=10)
Button(btn_frame, text="Capture", command=capture_attendance, width=15).grid(row=0, column=1, padx=10)
Button(btn_frame, text="Save CSV", command=save_attendance, width=15).grid(row=0, column=2, padx=10)
Button(btn_frame, text="Exit", command=root.quit, width=15).grid(row=0, column=3, padx=10)

root.mainloop()
