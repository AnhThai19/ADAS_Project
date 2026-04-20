import cv2
import time
import os

from detector import Detector
from inference import InferenceEngine
from utils import load_config, open_video, create_video_writer, ensure_dir


def resize_for_display(frame, max_width=960):
    """
    Resize frame để hiển thị vừa màn hình (không ảnh hưởng output gốc)
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def main():
    # ================= LOAD CONFIG =================
    config = load_config("../configs/config.yaml")

    # ================= INIT DETECTOR =================
    detector = Detector(
        model_path=config["model"]["path"],
        imgsz=config["model"]["imgsz"],
        conf_threshold=config["model"]["conf_threshold"]
    )

    # ================= INIT INFERENCE =================
    engine = InferenceEngine(
        detector=detector,
        class_names=config["classes"],
        danger_vertical_thresh=config["risk"]["danger_vertical_thresh"],
        warning_vertical_thresh=config["risk"]["warning_vertical_thresh"],
        danger_offset_thresh=config["risk"]["danger_offset_thresh"],
        warning_offset_thresh=config["risk"]["warning_offset_thresh"],
        roi_top_ratio=config["roi"]["top_ratio"],
        roi_top_width_ratio=config["roi"]["top_width_ratio"],
        roi_bottom_width_ratio=config["roi"]["bottom_width_ratio"]
    )

    # ================= VIDEO INPUT =================
    video_path = config["input"]["video_path"]
    cap = open_video(video_path)

    # ================= OUTPUT =================
    save_dir = config["output"]["save_dir"]
    ensure_dir(save_dir)

    output_path = os.path.join(save_dir, config["output"]["video_name"])
    writer = create_video_writer(cap, output_path)

    save_video = config["output"]["save_video"]
    show_window = config["output"]["show_window"]

    # ================= WINDOW =================
    if show_window:
        cv2.namedWindow("ADAS Demo", cv2.WINDOW_NORMAL)

    # ================= LOOP =================
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

    
        # ===== Inference =====
        output = engine.process_frame(frame)

        # ===== FPS =====
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            output,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ===== Show =====
        if show_window:
            preview = resize_for_display(output, max_width=960)
            cv2.imshow("ADAS Demo", preview)

        # ===== Save =====
        if save_video:
            writer.write(output)

        # ===== Exit =====
        if show_window:
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # ================= CLEANUP =================
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Video saved at: {output_path}")


if __name__ == "__main__":
    main()