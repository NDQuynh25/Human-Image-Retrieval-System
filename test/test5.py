import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def process_image(image_input, target_ratio=(1, 1)):
    """
    Xử lý ảnh với tỉ lệ tùy chỉnh (mặc định 2:3)
    
    Parameters:
        image_input: Path ảnh hoặc URL hoặc numpy array
        target_ratio: Tỉ lệ (width, height) mong muốn
        mode: 'padding' (thêm viền đen) hoặc 'crop' (cắt bớt)
    
    Returns:
        Ản đã xử lý dạng numpy array (RGB)
    """
    # Đọc ảnh từ nhiều nguồn
    if isinstance(image_input, str):
        if image_input.startswith(('http://', 'https://')):
            response = requests.get(image_input)
            img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
        else:
            img = cv2.imread(image_input)
            if img is None:
                img = np.array(Image.open(image_input).convert('RGB'))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()

    height, width = img.shape[:2]
    target_width = int(height * target_ratio[0] / target_ratio[1])
    
    
    if width < target_width:
        # Thiếu chiều rộng -> thêm padding trái/phải
        delta = target_width - width
        left = delta // 2
        right = delta - left
        img = cv2.copyMakeBorder(img, 0, 0, left, right, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif width > target_width:
        # Thừa chiều rộng -> thêm padding trên/dưới
        delta = int((width * target_ratio[1] / target_ratio[0]) - height)
        top = delta // 2
        bottom = delta - top
        img = cv2.copyMakeBorder(img, top, bottom, 0, 0,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    
    return img

# Sử dụng
if __name__ == "__main__":
    # Xử lý từ file local
    local_img = process_image("https://i.pinimg.com/736x/d2/2a/7b/d22a7bfa0d7f27208ad505c258b27b16.jpg")
    
    # Xử lý từ URL
    url_img = process_image("https://i.pinimg.com/736x/d2/2a/7b/d22a7bfa0d7f27208ad505c258b27b16.jpg")
    
    # Hiển thị kết quả
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(local_img)
    plt.title("Padding Mode")
    
    plt.subplot(1, 2, 2)
    plt.imshow(url_img)
    plt.title("Crop Mode")
    
    plt.show()