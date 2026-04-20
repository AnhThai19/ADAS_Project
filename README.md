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
```
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
```

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

## 📥 Dataset

This project uses the BDD100k dataset.

Due to size limitations, the dataset is not included in this repository.

Download:
https://bdd-data.berkeley.edu/

Expected structure:
```
data/
├── raw_video/              # Original videos for inference or demo
│   └── sample_video.mp4
├── raw/                    # Original BDD100K dataset  
│   ├── bdd100k_images/
│   │   └── 100k/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── bdd100k_labels/
│       └── 100k/
│           ├── train/
│           ├── val/
│           └── test/
└── processed/              # Preprocessed data
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## 🗂️ Project Structure
```
adas-realtime-warning/
├── configs/                # Configuration files (paths, hyperparameters)
│   └── config.yaml
│   └── data.yaml
├── data/                   
│   ├── raw/                # Original BDD100K images/labels
│   └── processed/          # Preprocessed data
│   └── raw_video/          # Raw video
├── models/                 
│   └── best.pt             # Trained model weights (YOLOv8)
├── notebooks/              # Research, EDA, and prototyping
│   └── data_exploration.ipynb
├── results/                
│   ├── demo.mp4            # Inference result videos
│   └── logs/               # Prepare dataset logs
├── src/                    # Core logic     
│   ├── detector.py          
│   ├── distance.py          
│   ├── risk.py            
│   ├── inference.py         
│   ├── utils.py            
│   └── main.py              
├── scripts/               
│   ├── convert_bdd100k_to_yolo.py
│   └── training_yolov8n.py
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation and setup guide
```

---

💡 Design Insight

- Instead of using naive distance estimation, this system prioritizes objects based on their spatial relationship to the vehicle.

- A trapezoidal ROI is used to focus on the forward driving region, improving robustness and reducing irrelevant detections.

- This approach avoids the need for camera calibration while still providing meaningful risk estimation.

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

🚀 Quick Start
```
pip install -r requirements.txt
python src/main.py
```
  
🧑‍💻 Tech Stack
```
Python
OpenCV
YOLOv8 (Ultralytics)
NumPy
```
⚠️ Note

This is a prototype ADAS system for educational purposes and is not intended for real-world deployment.
