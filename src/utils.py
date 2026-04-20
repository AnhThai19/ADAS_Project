import os 
import math 
import cv2 
import yaml 

# utils.py - Các hàm tiện ích chung cho dự án
def load_config(config_path: str = '../configs/config.yaml') -> dict:
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config 
    
# Tạo thư mục nếu chưa tồn tại
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    
# Mở video bằng OpenCV
def open_video(video_path: str) ->cv2.VideoCapture:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    
    return cap

# Tạo VideoWriter để lưu video output
def create_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None or fps <= 1 or math.isnan(fps):
        fps = 25.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer