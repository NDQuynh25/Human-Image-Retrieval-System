import cv2
import numpy as np
from PIL import Image

def extract_clothing_color_embedding(image_file):
    """Trích xuất đặc trưng từ ảnh (vector histogram màu 48 chiều)"""
    try:

        # 1. Đọc ảnh
        image_np = np.array(image_file)
        
        # 2. Phát hiện quần áo (loại bỏ da)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        clothing_mask = cv2.bitwise_not(skin_mask)  # Đảo ngược mask

        # Áp dụng bộ lọc closing để tách các phần quần áo
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
        clothing_area = cv2.bitwise_and(image_np, image_np, mask=clothing_mask)

        # 3. Trích xuất đặc trưng histogram màu (48D)
        chans = cv2.split(clothing_area)
        features = []
        bins_per_channel = 16
        for chan in chans:
            hist = cv2.calcHist([chan], [0], None, [bins_per_channel], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # Ép về float chuẩn + làm tròn
        return [float(f) for f in np.round(np.array(features, dtype=np.float64), decimals=8)]
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")
        return None

