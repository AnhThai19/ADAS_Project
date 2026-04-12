import json 
import os 
import cv2
from tqdm import tqdm
from collections import defaultdict
import logging 

# ================== CONFIG ==================
IMAGES_BASE = r"..\data\raw\bdd100k_images_100k\100k"
LABELS_BASE = r"..\data\raw\bdd100k_labels\100k"
OUTPUT_BASE = r"..\data\processed\processed_bdd100k"
LOG_FILE = r"..\results\prepare_dataset_log.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger()


PHASES = ["train", "val", "test"]
MAX_FILES = {
    "train": 10000,
    "val": 2000,
    "test": 2000
}

# Chọn 5 trong 21 lớp phù hợp cho bài toán detection 
CLASS_MAP = {
    "car": 0,
    "truck" : 1,
    "bus" : 2,
    "person" : 3,
    "bike" : 4,
    "motor" : 5
}

# ================== STATS ==================
stats = {
    "total_json" : 0,
    "converted_images" : 0,
    "missing_images" : 0,
    "empty_labels" : 0,
    "object_counts" : defaultdict(int)
}

# ================== UTILS ==================
def log(msg):
    logger.info(msg)
    
def normalize_bbox(x1, y1, x2, y2, w, h):
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    box_width = (x2 - x1) / w
    box_height = (y2 - y1) / h 
    return x_center, y_center, box_width, box_height

# ================== MAIN ==================
for phase in PHASES:
    log(f"START converting phase : {phase}")
    
    img_dir = os.path.join(IMAGES_BASE, phase)
    label_dir = os.path.join(LABELS_BASE, phase)
    
    out_img_dir = os.path.join(OUTPUT_BASE, "images", phase)
    out_lbl_dir = os.path.join(OUTPUT_BASE, "labels", phase)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    json_files = sorted([
        f for f in os.listdir(label_dir) if f.endswith(".json")
    ])[:MAX_FILES[phase]]
    
    phase_converted = 0
    
    for json_file in tqdm(json_files, desc=f"Converting {phase}"):
        stats["total_json"] += 1 
        
        json_path = os.path.join(label_dir, json_file)
        img_name = json_file.replace(".json", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            stats["missing_images"] += 1 
            log(f"Missing image for {json_file}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            stats["missing_images"] += 1
            continue
        
        height, width = img.shape[:2]
        
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        yolo_lines = []
        
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")
                if category not in CLASS_MAP:
                    continue
                if "box2d" not in obj:
                    continue
                
                box = obj["box2d"]
                x_center, y_center, box_width, box_height = normalize_bbox(
                    box["x1"], box["y1"], box["x2"], box["y2"], width, height
                )
                
                class_id = CLASS_MAP[category]
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )
                stats["object_counts"][category] += 1
                
            if not yolo_lines:
                stats["empty_labels"] += 1
                continue 
            
            # save image & label
            cv2.imwrite(os.path.join(out_img_dir, img_name), img)
            with open(os.path.join(out_lbl_dir, json_file.replace(".json", ".txt")), "w") as f:
                f.write("\n".join(yolo_lines))
                
            phase_converted += 1
            stats["converted_images"] += 1
            
    log(
        f"END phase : {phase} | "
        f"json_used={len(json_files)}, "
        f"converted={phase_converted}"
    )
    
# ================== SUMMARY ==================
logger.info("\n========== CONVERT SUMMARY ==========")
logger.info(f"Total JSON files processed: {stats['total_json']}")
logger.info(f"Total images converted: {stats['converted_images']}")
logger.info(f"Total missing images: {stats['missing_images']}")
logger.info(f"Total empty labels: {stats['empty_labels']}")

logger.info("\nObject counts:")
for category, count in stats["object_counts"].items():
    logger.info(f"  {category}: {count}")
