import sys
import os
import datetime
import asyncio

# Use relative imports with dots
from ..models.image_model import ImageModel
from ..utils.extractor.feature_extractor import feature_extractor
from ..utils.read_image import read_image_from_file_url

def save_image_data(image_name, path, height, width, hog, rgb, hsv, pose):
    # Tạo đối tượng ImageModel với UUID
    image = ImageModel(
       
        image_name=image_name,
        path=path,
        height=height,
        width=width,
        hog=hog,
        rgb=rgb,
        hsv=hsv,
        pose=pose,
        created_at=datetime.datetime.utcnow(),
        last_modified_at=datetime.datetime.utcnow()
    )
    
    # Lưu dữ liệu vào MongoDB
    image.save()

def search_image(image_url):
    try:
        image_file = read_image_from_file_url(image_url)
    except Exception as e:
        print(f"Không thể đọc ảnh từ URL: {e}")
        return None

    try:
        result = feature_extractor(image_file)
        return result
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ ảnh: {e}")
        return None

def test_feature_extraction():
    # Đường dẫn đến ảnh test
    test_image_path = "C:\\Users\\Admin\\Downloads\\anh-son-tung-mtp-thumb.jpg"
    print(f"Đang test với ảnh: {test_image_path}")
    
    result = search_image(test_image_path)
    if result is not None:
        print("Kết quả trích xuất đặc trưng:")
        print(result)
    else:
        print("Không thể trích xuất đặc trưng từ ảnh.")

# Chạy test nếu file được chạy trực tiếp
if __name__ == "__main__":
    test_feature_extraction()  # Removed asyncio.run as it's not needed unless test_feature_extraction is async