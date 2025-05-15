import serial
import time
import cv2
import os
import base64
from websocket import create_connection
from flask import Flask, render_template, send_from_directory
from inference import predict, get_waste_category, draw_boxes
from config import settings
from database import init_db, insert_log, get_logs

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
IMAGE_DIR = 'captured_images'
os.makedirs(IMAGE_DIR, exist_ok=True)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

app = Flask(__name__)

# Lưu thông tin loại rác cho từng ảnh
waste_info = {}

# Khởi tạo database
init_db()

# def send_image_ws(filename, waste_type, class_name, confidence):
#     try:
#         ws = create_connection("ws://192.168.110.51:8765")
#         with open(os.path.join(IMAGE_DIR, filename), "rb") as f:
#             img_bytes = f.read()
#         img_b64 = base64.b64encode(img_bytes).decode('utf-8')
#         data = {
#             "filename": filename,
#             "waste_type": waste_type,
#             "class_name": class_name,
#             "confidence": confidence,
#             "image": img_b64
#         }
#         import json
#         ws.send(json.dumps(data))
#         ws.close()
#     except Exception as e:
#         print("WebSocket send error:", e)

def send_waste_detection_ws(
    waste_type, waste_class, max_conf, processing_time, num_detections, img_b64_orig, img_b64_result
):
    try:
        # ws = create_connection("ws://192.168.110.51:8765")
        ws = create_connection("ws://192.168.110.51:8000/dashboard/ws")
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
    except Exception as e:
        print("WebSocket send error:", e)

def capture_and_infer_image():
    for idx in range(3):
        cam = cv2.VideoCapture(idx)
        time.sleep(0.5)
        ret, frame = cam.read()
        if ret:
            start_time = time.time()
            filename = f"{int(time.time())}.jpg"
            filepath = os.path.join(IMAGE_DIR, filename)
            cv2.imwrite(filepath, frame)
            cam.release()
            # Nhận diện ảnh
            boxes, class_probs = predict(frame, force=True)
            waste_type, class_name, conf = get_waste_category(boxes, class_probs)
            # Vẽ bounding box lên ảnh
            result_img = draw_boxes(frame, boxes, class_probs, current=waste_type)
            result_filename = f"result_{filename}"
            result_filepath = os.path.join(IMAGE_DIR, result_filename)
            cv2.imwrite(result_filepath, result_img)
            # Lưu thông tin loại rác
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
            # Encode ảnh gốc và ảnh kết quả sang base64
            with open(filepath, "rb") as f:
                img_b64_orig = base64.b64encode(f.read()).decode('utf-8')
            with open(result_filepath, "rb") as f:
                img_b64_result = base64.b64encode(f.read()).decode('utf-8')
            # Gửi dữ liệu qua WebSocket
            send_waste_detection_ws(
                waste_type, class_name, conf, processing_time, num_detections, img_b64_orig, img_b64_result
            )
            # Gửi lệnh về Arduino để điều khiển servo
            if waste_type == 'vo_co':
                ser.write(b'LEFT\n')   # Lệnh quay trái
            elif waste_type == 'huu_co':
                ser.write(b'RIGHT\n')  # Lệnh quay phải
            return result_filename
        cam.release()
    print("Không tìm thấy camera nào hoạt động!")
    return None

def monitor_distance():
    last_capture_time = 0
    cooldown = 2  # 2 giây chờ sau khi chụp ảnh
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                try:
                    distance = float(line)
                    print(f"Khoảng cách: {distance} cm")
                    now = time.time()
                    if 7 <= distance <= 20 and (now - last_capture_time) > cooldown:
                        filename = capture_and_infer_image()
                        print(f"Đã chụp và nhận diện ảnh: {filename}")
                        last_capture_time = time.time()
                except ValueError:
                    pass
            time.sleep(0.1)
        except Exception as e:
            print("Lỗi:", e)
            time.sleep(1)

@app.route('/')
def index():
    images = [img for img in sorted(os.listdir(IMAGE_DIR), reverse=True) if img.startswith('result_')]
    return render_template('index.html', images=images, waste_info=waste_info)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/history')
def history():
    logs = get_logs()
    return render_template('history.html', logs=logs)

if __name__ == '__main__':
    import threading
    t = threading.Thread(target=monitor_distance, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)
