import math
from typing import Tuple

def get_bottom_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    obj_x = (x1 + x2) // 2
    obj_y = y2
    return obj_x, obj_y

def compute_distance_from_origin(
    obj_x: int,
    obj_y: int,
    frame_width: int,
    frame_height: int
) -> Tuple[float, float, float]:
    origin_x = frame_width / 2
    origin_y = frame_height
    dx = obj_x - origin_x
    dy = origin_y - obj_y # càng nhỏ -> càng gần xe
    
    distance_px = math.sqrt(dx**2 + dy**2)
    horizontal_dist = abs(dx)
    vertical_dist = abs(dy)
    
    return distance_px, horizontal_dist, vertical_dist