import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'  # Đổi lại nếu Arduino của bạn ở cổng khác
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def send_servo_command(cmd):
    if cmd == 'LEFT':
        ser.write(b'LEFT\n')
        print('Đã gửi lệnh quay trái (LEFT)')
    elif cmd == 'RIGHT':
        ser.write(b'RIGHT\n')
        print('Đã gửi lệnh quay phải (RIGHT)')
    else:
        print('Lệnh không hợp lệ!')

if __name__ == '__main__':
    while True:
        cmd = input("Nhập lệnh (LEFT/RIGHT/EXIT): ").strip().upper()
        if cmd == 'EXIT':
            break
        send_servo_command(cmd)
        time.sleep(1)

ser.close()
