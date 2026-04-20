# 🚗 ADAS Real-Time Object Detection & Risk Warning System

A real-time Advanced Driver Assistance System (ADAS) prototype that detects objects on the road and provides risk-level warnings based on ego-centric spatial reasoning.

---

## 📌 Overview

This project implements a real-time object detection and forward collision warning system using YOLOv8 and computer vision techniques.

Instead of estimating absolute distance, the system focuses on:
- A **forward driving region (ROI)**
- The **bottom-center point of detected objects**
- Relative spatial reasoning from the **vehicle's perspective**

This approach enables a more robust and practical risk estimation for real-world driving scenarios.

---

## 🎯 Key Features

- 🚘 Real-time object detection using YOLOv8
- 🧠 Ego-centric risk estimation (vehicle perspective)
- 📐 Trapezoidal ROI (forward driving region)
- ⚠️ Risk classification:
  - DANGER (red)
  - WARNING (yellow)
  - SAFE (green)
- 🎥 Video processing with real-time FPS display
- 📊 Clean modular architecture (production-ready structure)

---

## 🧠 System Design

### Pipeline
Input Video
↓
YOLOv8 Detection
↓
Bottom-center extraction
↓
ROI filtering (forward region)
↓
Distance estimation (pixel-based geometry)
↓
Risk classification
↓
Visualization (bbox + label + warning)

---

## 📐 Core Concept

### Ego-Centric Geometry

Instead of estimating real-world distance directly, the system computes:

- **Vertical distance** → how close the object is to the vehicle  
- **Horizontal offset** → how aligned the object is with vehicle direction  

Origin point: Bottom-center of frame (ego vehicle position)

---

## 📊 Risk Logic

| Condition | Risk Level |
|----------|--------|
| Close + centered | 🔴 DANGER |
| Medium distance | 🟡 WARNING |
| Far or off-center | 🟢 SAFE |

---

## 🗂️ Project Structure
adas-realtime-warning/
│
├── configs/
│ └── config.yaml
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│ └── best.pt
│
├── src/
│ ├── detector.py
│ ├── distance.py
│ ├── risk.py
│ ├── inference.py
│ ├── utils.py
│ └── main.py
│
├── results/
│ ├── demo.mp4
│ └── logs/
│
├── notebooks/
│ └── data_exploration.ipynb
│
└── README.md

---

## ⚙️ Configuration

All parameters are configurable via: configs/config.yaml

Example:

```yaml
risk:
  danger_vertical_thresh: 80
  warning_vertical_thresh: 180
  danger_offset_thresh: 70
  warning_offset_thresh: 140

roi:
  top_ratio: 0.45
  top_width_ratio: 0.22
  bottom_width_ratio: 0.60
  ```
  
🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the system
python src/main.py
3. Output
Real-time visualization window
Output video saved in:
results/demo.mp4

📈 Performance
Real-time processing (~15–30 FPS depending on hardware)
Lightweight model (YOLOv8n)
🔥 Future Improvements
Convert to ONNX for faster inference
Add sound alert for danger events
Multi-object risk prioritization
Bird's Eye View (IPM) transformation
Deploy API with FastAPI
Web demo with Streamlit
🧑‍💻 Tech Stack
Python
OpenCV
YOLOv8 (Ultralytics)
NumPy
