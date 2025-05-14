import serial
import time
import cv2
import os
from flask import Flask, render_template, send_from_directory, jsonify, request
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

def capture_and_infer_image():
    for idx in range(3):
        cam = cv2.VideoCapture(idx)
        time.sleep(0.5)
        ret, frame = cam.read()
        if ret:
            filename_raw = f"{int(time.time())}_raw.jpg"
            filepath_raw = os.path.join(IMAGE_DIR, filename_raw)
            cv2.imwrite(filepath_raw, frame)
            cam.release()
            # Nhận diện ảnh
            boxes, class_probs = predict(frame, force=True)
            waste_type, class_name, conf = get_waste_category(boxes, class_probs)
            # Vẽ bounding box lên ảnh
            result_img = draw_boxes(frame, boxes, class_probs, current=waste_type)
            filename_result = f"result_{int(time.time())}.jpg"
            filepath_result = os.path.join(IMAGE_DIR, filename_result)
            cv2.imwrite(filepath_result, result_img)
            # Lưu thông tin loại rác
            waste_info[filename_result] = {
                'waste_type': waste_type,
                'class_name': class_name,
                'confidence': conf,
                'filename_raw': filename_raw
            }
            # Lưu vào database
            insert_log(filename_raw, filename_result, waste_type, class_name, conf)
            # Gửi lệnh về Arduino để điều khiển servo
            if waste_type == 'vo_co':
                ser.write(b'LEFT\n')   # Lệnh quay trái
            elif waste_type == 'huu_co':
                ser.write(b'RIGHT\n')  # Lệnh quay phải
            return filename_raw, filename_result, waste_type, class_name, conf
        cam.release()
    print("Không tìm thấy camera nào hoạt động!")
    return None, None, None, None, None

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
                        filename_raw, filename_result, waste_type, class_name, conf = capture_and_infer_image()
                        print(f"Đã chụp và nhận diện ảnh: {filename_raw}, {filename_result}")
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

@app.route('/capture', methods=['POST'])
def capture_api():
    filename_raw, filename_result, waste_type, class_name, conf = capture_and_infer_image()
    if filename_result:
        return jsonify({
            'filename_raw': filename_raw,
            'filename_result': filename_result,
            'waste_type': waste_type,
            'class_name': class_name,
            'confidence': conf
        })
    else:
        return jsonify({'error': 'Không chụp được ảnh'}), 500

if __name__ == '__main__':
    import threading
    t = threading.Thread(target=monitor_distance, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)
