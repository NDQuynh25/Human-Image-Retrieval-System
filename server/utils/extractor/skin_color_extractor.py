import cv2
import numpy as np
from PIL import Image
from typing import Union, List
import matplotlib.pyplot as plt

def extract_skin_color_embedding(
    image_file: Union[np.ndarray, Image.Image],
    bins: int = 16,
    debug: bool = False
) -> List[float]:
    """
    Trích xuất vector histogram màu da (48D) từ ảnh RGB.

    Args:
        image_file: PIL.Image hoặc NumPy RGB image
        bins: số bin mỗi kênh màu (16 → 48D tổng)
        debug: nếu True sẽ hiển thị mask da

    Returns:
        list[float]: vector đặc trưng da hoặc fallback [0.0]*48
    """
    try:
        # Bước 1: Đọc ảnh RGB
        if isinstance(image_file, Image.Image):
            image = np.array(image_file.convert("RGB"))
        elif isinstance(image_file, np.ndarray):
            image = image_file
        else:
            raise ValueError("image_file phải là PIL.Image hoặc np.ndarray")

        # Bước 2: Chuyển sang HSV & YCrCb
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        # Bước 3: Tạo skin mask (range mở rộng cho da sáng/tối)
        lower_hsv = np.array([0, 5, 30], dtype=np.uint8)
        upper_hsv = np.array([40, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        lower_cr = 120
        upper_cr = 185
        cr_mask = cv2.inRange(ycrcb[:, :, 1], lower_cr, upper_cr)

        # Kết hợp mask HSV và Cr
        skin_mask = cv2.bitwise_and(hsv_mask, cr_mask)

        # Lọc nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # Debug mask
        if debug:
            plt.imshow(skin_mask, cmap='gray')
            plt.title("Skin Mask")
            plt.axis("off")
            plt.show()

        # Bước 4: Lấy pixel vùng da
        masked_img = cv2.bitwise_and(image, image, mask=skin_mask)
        pixels = masked_img.reshape(-1, 3)
        pixels = pixels[np.any(pixels > 0, axis=1)]  # bỏ pixel đen

        # Bước 5: Nếu không có vùng da → fallback
        if len(pixels) == 0:
            print("⚠️ Không phát hiện được vùng da – trả về vector 0.")
            return [0.0] * (bins * 3)

        # Bước 6: Tính histogram R-G-B
        features = []
        for i in range(3):  # R, G, B
            hist = cv2.calcHist([pixels[:, i]], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        return [float(f) for f in np.round(features, 8)]

    except Exception as e:
        print(f"❌ Lỗi khi trích xuất đặc trưng da: {e}")
        return [0.0] * (bins * 3)
