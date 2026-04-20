import cv2
import numpy as np

from detector import Detector
from distance import compute_distance_from_origin, get_bottom_center
from risk import get_risk_level, get_box_color

class InferenceEngine:
    def __init__(
        self, 
        detector: Detector, 
        class_names: list,
        danger_vertical_thresh: float,
        warning_vertical_thresh: float,
        danger_offset_thresh: float,
        warning_offset_thresh: float,
        roi_top_ratio: float,
        roi_top_width_ratio: float,
        roi_bottom_width_ratio: float
    ):
        self.detector = detector
        self.class_names = class_names
        self.danger_vertical_thresh = danger_vertical_thresh
        self.warning_vertical_thresh = warning_vertical_thresh
        self.danger_offset_thresh = danger_offset_thresh
        self.warning_offset_thresh = warning_offset_thresh
        self.roi_top_ratio = roi_top_ratio 
        self.roi_top_width_ratio = roi_top_width_ratio
        self.roi_bottom_width_ratio = roi_bottom_width_ratio 
        
    # Xây dựng ROI polygon     
    def build_roi(self, frame_width:  int, frame_height: int):
        center_x = frame_width // 2     
        top_y = int(frame_height * self.roi_top_ratio)
        bottom_y = frame_height
        
        top_half_width = int(frame_width * self.roi_top_width_ratio / 2)
        bottom_half_width = int(frame_width * self.roi_bottom_width_ratio / 2)
        
        roi = np.array([
            [center_x - top_half_width, top_y],
            [center_x + top_half_width, top_y],
            [center_x + bottom_half_width, bottom_y],
            [center_x - bottom_half_width, bottom_y]
        ], dtype=np.int32)
    
        return roi
    
    # Vẽ label lên frame
    def draw_label(self, frame, text, x1, y1, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        y_text = max(y1 - 10, text_height + 5)
        
        cv2.rectangle(frame, (x1, y_text - text_height -8), (x1 + text_width + 8, y_text + 4), color, -1)
        cv2.putText(frame, text, (x1 + 4, y_text), font, scale, (0,0,0), thickness, cv2.LINE_AA)
    
    def process_frame(self, frame):
        '''
        Xử lý 1 frame:
        - Nhận diện đối tượng
        - Lọc object trong ROI
        - Tính distance, offset, vertical distance
        - Phân loại risk
        - Vẽ bbox + label
        '''
        
        height, width = frame.shape[:2]
        origin_x = width // 2
        origin_y = height
        
        annotated = frame.copy()
        
        # ===== ROI =====
        roi = self.build_roi(width, height)
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [roi], (0, 255, 0))
        cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, dst=annotated)
        cv2.polylines(annotated, [roi], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Vẽ điểm gốc đầu xe
        cv2.circle(annotated, (origin_x, origin_y - 1), 6, (255, 255, 255), -1)
        
        # ===== Detection =====
        result = self.detector.predict(frame)
        if result is None or result.boxes is None:
            return annotated
        
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = self.class_names.get(cls_id, f'class_{cls_id}')
            
            # ===== bottom center point =====
            obj_x, obj_y = get_bottom_center(x1, y1, x2, y2)
            
            # Chỉ xét object nằm trong ROI
            inside_roi = cv2.pointPolygonTest(roi, (obj_x, obj_y), False)
            if inside_roi < 0:
                continue
            
            # ===== Geometry-based distance =====
            distance_px, horizontal_offset, vertical_distance = compute_distance_from_origin(
                obj_x=obj_x,
                obj_y=obj_y,
                frame_width=width,
                frame_height=height
            )
            
            # ===== Risk classification =====
            risk_level = get_risk_level(
                vertical_distance=vertical_distance,
                horizontal_offset=horizontal_offset,
                danger_vertical_thresh=self.danger_vertical_thresh,
                warning_vertical_thresh=self.warning_vertical_thresh,
                danger_offset_thresh=self.danger_offset_thresh,
                warning_offset_thresh=self.warning_offset_thresh
            )
            
            color = get_box_color(risk_level)
        
            # ===== Draw box =====
            thickness = 3 if risk_level == 'DANGER' else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # ===== Draw line from ego origin to object =====
            cv2.line(annotated, (origin_x, origin_y), (obj_x, obj_y), color, thickness)
            cv2.circle(annotated, (obj_x, obj_y), 5, color, -1)
            
            # ===== Label =====
            label = f'{class_name.upper()} {conf:.2f} | {distance_px:.0f}px | {risk_level}'
        
            self.draw_label(annotated, label, x1, y1, color)
            
        return annotated
        
        
        
        
    