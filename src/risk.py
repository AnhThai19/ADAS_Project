def get_risk_level(
    vertical_distance: float,
    horizontal_offset: float,
    danger_vertical_thresh: float,
    danger_offset_thresh: float,
    warning_vertical_thresh: float,
    warning_offset_thresh: float
) -> str:
    if vertical_distance < danger_vertical_thresh and horizontal_offset < danger_offset_thresh:
        return 'DANGER'

    if vertical_distance < warning_vertical_thresh and horizontal_offset < warning_offset_thresh:
        return 'WARNING'
    
    return 'SAFE'

# DANGER : nguy hiểm
# WARNING : cảnh báo
# SAFE : an toàn
def get_box_color(risk_level: str):
    if risk_level == 'DANGER':
        return (0, 0, 255) # đỏ
    
    if risk_level == 'WARNING':
        return (0, 255, 255) # vàng
    
    return (0, 255, 0) # xanh lá
