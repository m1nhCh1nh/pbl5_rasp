import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    MODEL_PATH: str = "app/models/best_float32.tflite"
    PREDICTION_INTERVAL: float = 1.0  # Khoảng thời gian giữa các lần dự đoán (giây)
    CONF_THRESHOLD: float = 0.5  # Ngưỡng độ tin cậy cho detection - khớp với Raspberry Pi
    NMS_IOU_THRESHOLD: float = 0.5  # Ngưỡng IoU cho Non-Maximum Suppression
    MAX_DETECTIONS_PER_CLASS: int = 5  # Số lượng phát hiện tối đa cho mỗi lớp

    # Cấu hình ghi log
    LOG_LEVEL: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    SAVE_DEBUG_IMAGES: bool = True  # Lưu hình ảnh debug
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()