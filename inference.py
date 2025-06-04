import time
import numpy as np
import cv2
import logging
from ultralytics import YOLO
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Danh sách lớp mới
class_names = [
    'AppleCore', 'Battery', 'EggShell', 'OrangePeel', 'Paper',
    'PaperCup', 'BananaPeel', 'Cans', 'PlasticBottle', 'cigarette'
]

# Phân loại các loại rác
waste_categories = {
    'AppleCore': 'huu_co',
    'EggShell': 'huu_co',
    'OrangePeel': 'huu_co',
    'BananaPeel': 'huu_co',
    'cigarette': 'huu_co',
    'Battery': 'vo_co',
    'Paper': 'vo_co',
    'PaperCup': 'vo_co',
    'Cans': 'vo_co',
    'PlasticBottle': 'vo_co',
}

# Đường dẫn model YOLOv8
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')

# Cấu hình cho Raspberry Pi
CONF_THRESHOLD = 0.25  # Giảm ngưỡng confidence để phát hiện tốt hơn
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 10
IMG_SIZE = 640  # Kích thước ảnh đầu vào cho model

try:
    # Load model với cấu hình tối ưu cho Raspberry Pi
    model = YOLO(MODEL_PATH)
    logger.info(f"Loaded YOLOv8 model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {e}")
    raise

_last_time = time.time()

def predict(image, force=False):
    """
    Thực hiện dự đoán với model YOLOv8 (best.pt)
    Tối ưu cho Raspberry Pi
    """
    global _last_time
    now = time.time()
    _last_time = now
    
    try:
        # Resize ảnh về kích thước chuẩn để tăng tốc độ xử lý
        if image.shape[0] > IMG_SIZE or image.shape[1] > IMG_SIZE:
            scale = IMG_SIZE / max(image.shape[0], image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
        
        # YOLOv8 nhận ảnh BGR (cv2)
        results = model.predict(
            image,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return np.array([]), np.array([])
            
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.array([]), np.array([])
            
        # Lấy boxes, scores, class_probs
        boxes = r.boxes.xywhn.cpu().numpy()  # [x_center, y_center, w, h] (normalized)
        scores = r.boxes.conf.cpu().numpy()  # confidence
        class_ids = r.boxes.cls.cpu().numpy().astype(int)  # class index
        
        # Tạo mảng class_probs (one-hot * confidence)
        class_probs = np.zeros((len(boxes), len(class_names)), dtype=np.float32)
        for i, (cls, conf) in enumerate(zip(class_ids, scores)):
            class_probs[i, cls] = conf
            
        return boxes, class_probs
        
    except Exception as e:
        logger.error(f"YOLOv8 prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return np.array([]), np.array([])

def get_waste_category(boxes, class_probs, conf_thr=None):
    """
    Xác định loại rác dựa trên các detection
    """
    if boxes.size == 0 or class_probs.size == 0:
        logger.info("Không có phát hiện nào")
        return None, 'unknown', 0.0
        
    conf_thr = conf_thr or CONF_THRESHOLD
    best_conf = 0.0
    best_class_id = -1
    
    for i in range(len(boxes)):
        class_id = np.argmax(class_probs[i])
        confidence = float(class_probs[i][class_id])
        if confidence > best_conf:
            best_conf = confidence
            best_class_id = class_id
            
    if best_class_id >= 0 and best_conf >= conf_thr:
        class_name = class_names[best_class_id] if best_class_id < len(class_names) else 'unknown'
        waste_type = waste_categories.get(class_name, 'unknown')
        logger.info(f"Phát hiện loại rác: {waste_type} ({class_name}), confidence: {best_conf:.4f}")
        return waste_type, class_name, best_conf
    else:
        logger.info("Không phát hiện được loại rác nào có confidence đủ cao")
        return None, 'unknown', 0.0

def draw_boxes(image, boxes, class_probs, current=None, conf_thr=None):
    """
    Vẽ các bounding box lên hình ảnh
    Tối ưu cho Raspberry Pi
    """
    conf_thr = conf_thr or CONF_THRESHOLD
    result_img = image.copy()
    h, w = image.shape[:2]
    
    if current:
        text = 'HỮU CƠ' if current == 'huu_co' else 'VÔ CƠ'
        cv2.putText(result_img, f'PHÂN LOẠI: {text}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    for i, box in enumerate(boxes):
        class_id = np.argmax(class_probs[i])
        confidence = float(class_probs[i][class_id])
        
        if confidence > conf_thr:
            class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
            waste_type = waste_categories.get(class_name, 'unknown')
            color = (0, 255, 0) if waste_type == 'huu_co' else (0, 0, 255)
            
            # Chuyển đổi tọa độ normalized sang pixel
            x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Đảm bảo tọa độ nằm trong ảnh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Vẽ box và label
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({confidence:.2f})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_img
