from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str, imgsz: int = 640, conf_threshold: float = 0.35):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

    def predict(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )
        return results[0] if results else None
        