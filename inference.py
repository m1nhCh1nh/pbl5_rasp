import time
import numpy as np
import tensorflow as tf
import cv2
import logging
from app.core.config import settings
import os

# Định nghĩa tên các lớp - chính xác như trên Raspberry Pi
labels = {
    0: 'AppleCore', 1: 'DryLeaves', 2: 'EggShell', 3: 'OrangePeel',
    4: 'Paper', 5: 'PaperCup', 6: 'PlasticBag', 7: 'BananaPeel',
    8: 'Cans', 9: 'PlasticBottle'
}

# Danh sách lớp theo thứ tự giống Raspberry Pi
class_names = [
    "AppleCore", "DryLeaves", "EggShell", "OrangePeel",
    "Paper", "PaperCup", "PlasticBag", "BananaPeel",
    "Cans", "PlasticBottle"
]

# Phân loại các loại rác
waste_categories = {
    'AppleCore': 'huu_co', 'DryLeaves': 'huu_co', 'EggShell': 'huu_co',
    'OrangePeel': 'huu_co', 'Paper': 'vo_co', 'PaperCup': 'vo_co',
    'PlasticBag': 'vo_co', 'BananaPeel': 'huu_co', 'Cans': 'vo_co', 'PlasticBottle': 'vo_co'
}

# Load TFLite model
try:
    # Kiểm tra xem file mô hình có tồn tại không
    if not os.path.exists(settings.MODEL_PATH):
        logging.error(f"Model file not found: {settings.MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
    
    logging.info(f"Loading model from: {settings.MODEL_PATH}")
    _interpreter = tf.lite.Interpreter(model_path=settings.MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    
    # Log model details
    logging.info("========== MODEL DETAILS ==========")
    logging.info(f"Input details: {len(_input_details)}")
    for i, detail in enumerate(_input_details):
        logging.info(f"Input #{i}: shape={detail['shape']}, dtype={detail['dtype']}")
    
    logging.info(f"Output details: {len(_output_details)}")
    for i, detail in enumerate(_output_details):
        logging.info(f"Output #{i}: shape={detail['shape']}, dtype={detail['dtype']}")
    
    # Kiểm tra và đảm bảo kích thước input/output phù hợp
    _image_height = _input_details[0]['shape'][1]
    _image_width = _input_details[0]['shape'][2]
    
    logging.info(f"Model input dimensions: {_image_width}x{_image_height}")
    logging.info("==================================")
    
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

_last_time = time.time()

def preprocess_image(img):
    """
    Chuẩn bị hình ảnh cho model best_float32.tflite
    """
    # Resize về kích thước đầu vào của mô hình
    img_resized = cv2.resize(img, (_image_width, _image_height))
    
    # Chuẩn hóa pixel về [0, 1]
    img_float = np.true_divide(img_resized, 255, dtype=np.float32)
    
    # Thêm batch dimension
    img_batch = np.expand_dims(img_float, axis=0)
    
    return img_batch

def predict(image, force=False):
    """
    Thực hiện dự đoán với model best_float32.tflite
    """
    global _last_time
    now = time.time()
    
    # Kiểm tra thời gian giữa các lần dự đoán
    if not force and (now - _last_time) < settings.PREDICTION_INTERVAL:
        return np.array([]), np.array([])
    
    _last_time = now
    
    try:
        # Tiền xử lý ảnh
        input_tensor = preprocess_image(image)
        
        # Đặt tensor đầu vào và chạy model
        _interpreter.set_tensor(_input_details[0]['index'], input_tensor)
        _interpreter.invoke()
        
        # Lấy kết quả đầu ra - Đảm bảo phù hợp với định dạng của model best_float32.tflite
        
        # Xác định xem output có dạng như thế nào
        has_correct_format = False
        output = None
        
        # Kiểm tra từng output để tìm output chính
        for i, output_detail in enumerate(_output_details):
            temp_output = _interpreter.get_tensor(output_detail['index'])
            logging.info(f"Output #{i} shape: {temp_output.shape}")
            
            # Nếu là output chính (thường có dạng [1, n, 14] hoặc tương tự)
            if len(temp_output.shape) == 3 and temp_output.shape[2] > 4:
                output = temp_output
                logging.info(f"Found main detection output: {output.shape}")
                has_correct_format = True
                break
        
        # Nếu không tìm thấy output phù hợp, sử dụng output đầu tiên
        if output is None:
            output = _interpreter.get_tensor(_output_details[0]['index'])
            logging.warning(f"Using default output format: {output.shape}")
        
        # Xử lý output
        # Các model YOLOv8 thường xuất ra dạng [batch, boxes, 4+num_classes] 
        # trong đó 4 là box coords và phần còn lại là class scores
        
        output = output[0]  # Lấy batch đầu tiên
        
        # Kiểm tra xem output có cần chuyển vị không (dựa vào kích thước)
        num_classes = len(labels)
        expected_output_size = 4 + num_classes
        
        if has_correct_format:
            # Đã tìm thấy output chính, tiếp tục xử lý như YOLOv8
            if output.shape[1] == expected_output_size:
                # output có dạng [num_detections, 4+num_classes]
                boxes = output[:, :4]  # x_center, y_center, width, height
                confidence = output[:, 4:]  # confidence cho mỗi class
            else:
                # output có dạng [4+num_classes, num_detections] - cần chuyển vị
                output = output.T  # Chuyển vị để đúng với định dạng cần
                boxes = output[:, :4]  # x_center, y_center, width, height
                confidence = output[:, 4:]  # confidence cho mỗi class
        else:
            # Thử đoán cách xử lý dựa vào kích thước
            logging.warning("Trying to guess output format")
            if len(output.shape) == 2:
                # output có dạng 2D
                if output.shape[1] > expected_output_size:
                    # Có vẻ như là [num_detections, 4+num_classes]
                    boxes = output[:, :4]
                    confidence = output[:, 4:4+num_classes]
                else:
                    # Có vẻ như là [4+num_classes, num_detections] - cần chuyển vị
                    output = output.T
                    boxes = output[:, :4]
                    confidence = output[:, 4:4+num_classes]
            else:
                # Format không rõ ràng, thử reshape dữ liệu
                logging.error(f"Unexpected output format: {output.shape}")
                return np.array([]), np.array([])
        
        # Tính confidence và class cho mỗi box
        scores = np.max(confidence, axis=1)  # Lấy confidence lớn nhất cho mỗi box
        classes = np.argmax(confidence, axis=1)  # Lấy class có confidence cao nhất
        
        # Logging thông tin
        logging.info(f"Processed {len(boxes)} raw detections")
        top_indices = np.argsort(scores)[-5:][::-1]  # 5 detection có điểm cao nhất
        for i in top_indices:
            if scores[i] > 0.1:  # Chỉ hiển thị những detection có confidence > 0.1
                logging.info(f"Top detection: class={class_names[classes[i]]}, score={scores[i]:.4f}")
        
        # Áp dụng ngưỡng confidence
        valid_indices = np.where(scores > settings.CONF_THRESHOLD)[0]
        
        if len(valid_indices) == 0:
            logging.info("No detections above threshold")
            return np.array([]), np.array([])
        
        # Lọc các phát hiện
        filtered_boxes = boxes[valid_indices]
        filtered_probs = confidence[valid_indices]
        
        # Áp dụng Non-Maximum Suppression để loại bỏ các box chồng lấp
        try:
            # Chuyển boxes từ [cx, cy, w, h] -> [y1, x1, y2, x2] cho NMS
            cx, cy, w, h = np.split(filtered_boxes, 4, axis=1)
            y1 = cy - h/2
            x1 = cx - w/2
            y2 = cy + h/2
            x2 = cx + w/2
            
            # Đảm bảo tọa độ nằm trong khoảng [0, 1]
            y1 = np.clip(y1, 0, 1)
            x1 = np.clip(x1, 0, 1)
            y2 = np.clip(y2, 0, 1)
            x2 = np.clip(x2, 0, 1)
            
            # Format boxes cho NMS
            nms_boxes = np.concatenate([y1, x1, y2, x2], axis=1)
            
            # Tính class scores cho mỗi box
            class_indices = np.argmax(filtered_probs, axis=1)
            class_scores = np.array([filtered_probs[i, class_indices[i]] for i in range(len(class_indices))])
            
            # Áp dụng NMS
            keep_indices = tf.image.non_max_suppression(
                boxes=nms_boxes,
                scores=class_scores,
                max_output_size=settings.MAX_DETECTIONS_PER_CLASS * len(labels),
                iou_threshold=settings.NMS_IOU_THRESHOLD
            ).numpy()
            
            if len(keep_indices) == 0:
                logging.info("No detections after NMS")
                return np.array([]), np.array([])
            
            # Lấy kết quả cuối cùng
            final_boxes = filtered_boxes[keep_indices]
            final_probs = filtered_probs[keep_indices]
            
            logging.info(f"Final detections after NMS: {len(final_boxes)}")
            return final_boxes, final_probs
            
        except Exception as e:
            logging.error(f"NMS error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return np.array([]), np.array([])
            
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return np.array([]), np.array([])

def get_waste_category(boxes, class_probs, conf_thr=None):
    """
    Xác định loại rác dựa trên các detection
    """
    if boxes.size == 0 or class_probs.size == 0:
        logging.info("Không có phát hiện nào")
        return None, 'unknown', 0.0
    
    # Sử dụng ngưỡng confidence từ tham số hoặc từ config
    conf_thr = conf_thr or settings.CONF_THRESHOLD
    
    # Tìm detection có confidence cao nhất
    best_conf = 0.0
    best_class_id = -1
    
    for i in range(len(boxes)):
        class_id = np.argmax(class_probs[i])
        confidence = float(class_probs[i][class_id])
        
        if confidence > best_conf:
            best_conf = confidence
            best_class_id = class_id
    
    # Lấy tên class và loại rác
    if best_class_id >= 0 and best_conf >= conf_thr:
        class_name = class_names[best_class_id] if best_class_id < len(class_names) else 'unknown'
        waste_type = waste_categories.get(class_name, 'unknown')
        
        logging.info(f"Phát hiện loại rác: {waste_type} ({class_name}), confidence: {best_conf:.4f}")
        return waste_type, class_name, best_conf
    else:
        logging.info("Không phát hiện được loại rác nào có confidence đủ cao")
        return None, 'unknown', 0.0

def draw_boxes(image, boxes, class_probs, current=None, conf_thr=None):
    """
    Vẽ các bounding box lên hình ảnh
    """
    # Sử dụng ngưỡng confidence từ tham số hoặc từ config
    conf_thr = conf_thr or settings.CONF_THRESHOLD
    
    # Tạo bản sao của hình ảnh để không thay đổi hình ảnh gốc
    result_img = image.copy()
    h, w = image.shape[:2]
    
    # Hiển thị loại rác đã phân loại (nếu có)
    if current:
        text = 'HỮU CƠ' if current == 'huu_co' else 'VÔ CƠ'
        cv2.putText(result_img, f'PHÂN LOẠI: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Vẽ bounding box cho từng detection
    for i, box in enumerate(boxes):
        # Lấy class có confidence cao nhất
        class_id = np.argmax(class_probs[i])
        confidence = float(class_probs[i][class_id])
        
        # Chỉ vẽ những detection có confidence vượt ngưỡng
        if confidence > conf_thr:
            # Lấy thông tin về class và loại rác
            class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
            waste_type = waste_categories.get(class_name, 'unknown')
            
            # Màu sắc cho box dựa trên loại rác
            color = (0, 255, 0) if waste_type == 'huu_co' else (0, 0, 255)
            
            # Tính toán tọa độ của bounding box
            x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Đảm bảo các tọa độ nằm trong hình ảnh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Vẽ bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Tạo nhãn với tên class và confidence
            label = f"{class_name} ({confidence:.2f})"
            
            # Vẽ nền cho text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Vẽ text
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_img