import serial
import time
import cv2
import os
import base64
import logging
from websocket import create_connection
from flask import Flask, render_template, send_from_directory
from inference import predict, get_waste_category, draw_boxes
from database import init_db, insert_log, get_logs

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình cho Raspberry Pi
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
IMAGE_DIR = 'captured_images'
WEBSOCKET_URL = "ws://192.168.110.51:8000/dashboard/ws"
CAMERA_INDEXES = [0, 1, 2]  # Thứ tự ưu tiên các camera
CAMERA_WARMUP_TIME = 0.5  # Thời gian khởi động camera
DISTANCE_MIN = 7  # Khoảng cách tối thiểu (cm)
DISTANCE_MAX = 20  # Khoảng cách tối đa (cm)
CAPTURE_COOLDOWN = 2  # Thời gian chờ giữa các lần chụp (giây)

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs(IMAGE_DIR, exist_ok=True)

# Khởi tạo serial port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    logger.info(f"Đã kết nối với Arduino qua {SERIAL_PORT}")
except Exception as e:
    logger.error(f"Lỗi kết nối Arduino: {e}")
    ser = None

app = Flask(__name__)
waste_info = {}

# Khởi tạo database
init_db()

def send_waste_detection_ws(
    waste_type, waste_class, max_conf, processing_time, num_detections, img_b64_orig, img_b64_result
):
    """Gửi dữ liệu phát hiện qua WebSocket"""
    try:
        ws = create_connection(WEBSOCKET_URL)
        data = {
            "waste_type": waste_type,
            "waste_class": waste_class,
            "max_conf": max_conf,
            "processing_time": processing_time,
            "num_detections": num_detections,
            "img_b64_orig": img_b64_orig,
            "img_b64_result": img_b64_result
        }
        import json
        ws.send(json.dumps(data))
        ws.close()
        logger.info("Đã gửi dữ liệu qua WebSocket thành công")
    except Exception as e:
        logger.error(f"Lỗi gửi WebSocket: {e}")

def capture_and_infer_image():
    """Chụp và xử lý ảnh từ camera"""
    for idx in CAMERA_INDEXES:
        try:
            cam = cv2.VideoCapture(idx)
            time.sleep(CAMERA_WARMUP_TIME)
            ret, frame = cam.read()
            if ret:
                start_time = time.time()
                
                # Lưu ảnh gốc
                filename = f"{int(time.time())}.jpg"
                filepath = os.path.join(IMAGE_DIR, filename)
                cv2.imwrite(filepath, frame)
                cam.release()
                
                # Nhận diện ảnh
                boxes, class_probs = predict(frame, force=True)
                waste_type, class_name, conf = get_waste_category(boxes, class_probs)
                
                # Vẽ bounding box
                result_img = draw_boxes(frame, boxes, class_probs, current=waste_type)
                result_filename = f"result_{filename}"
                result_filepath = os.path.join(IMAGE_DIR, result_filename)
                cv2.imwrite(result_filepath, result_img)
                
                # Lưu thông tin
                waste_info[result_filename] = {
                    'waste_type': waste_type,
                    'class_name': class_name,
                    'confidence': conf
                }
                
                # Lưu vào database
                insert_log(result_filename, waste_type, class_name, conf)
                
                # Chuẩn bị dữ liệu gửi đi
                processing_time = time.time() - start_time
                num_detections = len(boxes) if boxes is not None else 0
                
                # Encode ảnh
                with open(filepath, "rb") as f:
                    img_b64_orig = base64.b64encode(f.read()).decode('utf-8')
                with open(result_filepath, "rb") as f:
                    img_b64_result = base64.b64encode(f.read()).decode('utf-8')
                
                # Gửi dữ liệu
                send_waste_detection_ws(
                    waste_type, class_name, conf, processing_time, 
                    num_detections, img_b64_orig, img_b64_result
                )
                
                # Điều khiển servo
                if ser and waste_type:
                    if waste_type == 'vo_co':
                        ser.write(b'LEFT\n')
                    elif waste_type == 'huu_co':
                        ser.write(b'RIGHT\n')
                
                return result_filename
            cam.release()
        except Exception as e:
            logger.error(f"Lỗi camera {idx}: {e}")
            continue
            
    logger.error("Không tìm thấy camera nào hoạt động!")
    return None

def monitor_distance():
    """Giám sát khoảng cách từ cảm biến"""
    last_capture_time = 0
    while True:
        try:
            if not ser:
                logger.error("Không có kết nối Arduino")
                time.sleep(1)
                continue
                
            line = ser.readline().decode().strip()
            if line:
                try:
                    distance = float(line)
                    logger.info(f"Khoảng cách: {distance} cm")
                    now = time.time()
                    
                    if (DISTANCE_MIN <= distance <= DISTANCE_MAX and 
                        (now - last_capture_time) > CAPTURE_COOLDOWN):
                        filename = capture_and_infer_image()
                        if filename:
                            logger.info(f"Đã chụp và nhận diện ảnh: {filename}")
                            last_capture_time = now
                except ValueError:
                    pass
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Lỗi đọc cảm biến: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Trang chủ hiển thị ảnh"""
    images = [img for img in sorted(os.listdir(IMAGE_DIR), reverse=True) 
             if img.startswith('result_')]
    return render_template('index.html', images=images, waste_info=waste_info)

@app.route('/images/<filename>')
def images(filename):
    """API trả về ảnh"""
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/history')
def history():
    """Trang lịch sử"""
    logs = get_logs()
    return render_template('history.html', logs=logs)

if __name__ == '__main__':
    import threading
    # Khởi chạy thread giám sát khoảng cách
    t = threading.Thread(target=monitor_distance, daemon=True)
    t.start()
    # Khởi chạy web server
    app.run(host='0.0.0.0', port=5000)
