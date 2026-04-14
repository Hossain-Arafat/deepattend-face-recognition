# DeepAttend: Face Recognition-Based Attendance System

An intelligent real-time attendance system powered by **deep learning** and **computer vision**.  
This project uses **MobileNetV2 (Transfer Learning)** to recognize faces and automate attendance marking.

---

## Features

- Real-time face detection using OpenCV  
- Face recognition using MobileNetV2 (Transfer Learning)  
- Supports **50+ classes (students)** dynamically  
- Confidence thresholding to filter unknown faces  
- Stability-based recognition (2–3 seconds) for accuracy  
- GUI-based system using Tkinter  
- Duplicate prevention (per day)  
- CSV export of attendance records  
- User-friendly feedback

---

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy / Pandas**
- **Tkinter (GUI)**
- **MobileNetV2 (Pretrained Model)**

---

## How It Works

1. Dataset is organized into folders (one folder per student).
2. The model is trained using **transfer learning (MobileNetV2)**.
3. During runtime:
   - Face is detected via OpenCV
   - Model predicts identity
   - If confidence > threshold → recognized
   - If stable for ~2 seconds → ready for capture
4. Attendance is recorded and saved as a CSV file.

---

## Setup Instructions

### Clone the repository
```bash
git clone https://github.com/Hossain-Arafat/deepattend-face-recognition.git
cd deepattend-face-recognition
pip install -r requirements.txt
python main.py
```
---
## Model Details

- Model: MobileNetV2
- Input Size: 128x128
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Training Enhancements:
  - Data Augmentation
  - Fine-tuning
  - Early Stopping

---
## Output

Attendance saved as: attendance_YYYYMMDD_HHMMSS.csv

---
## Limitations
- Sensitive to lighting conditions
- Haar Cascade may miss faces at extreme angles
- No anti-spoofing (photo/video attack possible)

---
## Future Improvements
- Face anti-spoofing detection
- Multi-face simultaneous attendance
- Database integration
- Web-based interface (Flask/React)
- Live student information display

---
## Note
This project is developed for academic and learning purposes and can be extended into a full-scale production system.
