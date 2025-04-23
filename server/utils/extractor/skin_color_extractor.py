import cv2
import numpy as np

def extract_skin_color_embedding(image_file, bins=16):
    """
    Trích xuất vector histogram màu (48D) từ vùng da trong ảnh RGB.
    Đầu vào: PIL Image hoặc NumPy RGB image
    Đầu ra: np.array với shape (48,)
    """
    try:
        # Convert PIL Image to NumPy (nếu cần)
        if not isinstance(image_file, np.ndarray):
            image = np.array(image_file.convert("RGB"))
        else:
            image = image_file

        # Chuyển đổi không gian màu
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # Tạo mask da
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        cr_mask = cv2.inRange(ycrcb[:, :, 1], 133, 173)
        skin_mask = cv2.bitwise_and(hsv_mask, cr_mask)

        # Lọc nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # Áp dụng mask
        masked_img = cv2.bitwise_and(image, image, mask=skin_mask)
        pixels = masked_img.reshape(-1, 3)
        pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
        if len(pixels) == 0:
            return None

        # Tính histogram (48 chiều)
        features = []
        for i in range(3):  # R, G, B
            hist = cv2.calcHist([pixels[:, i]], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # chuyển float64 sang float8 và làm tròn
        return [float(f) for f in np.round(np.array(features, dtype=np.float64), decimals=8)]

    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng da: {e}")
        return None
