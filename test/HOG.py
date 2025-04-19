import requests
import numpy as np
import cv2
from io import BytesIO
from skimage import color, feature, io
import requests
import numpy as np
import cv2
from io import BytesIO
from skimage import color, feature, io
import matplotlib.pyplot as plt



def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = io.imread(BytesIO(response.content))  # Đọc ảnh từ bytes
        return image
    else:
        raise ValueError("Could not download image")

def convert_image_rgb_to_gray(img_rgb, resize="no"):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # Chuyển ảnh sang grayscale
    print(f"Shape ảnh grayscale: {img_gray.shape}, Min: {img_gray.min()}, Max: {img_gray.max()}")
    plt.imshow(img_gray, cmap='gray')
    plt.title("Ảnh Grayscale")
    plt.show()

    # if resize != "no":
    #     img_gray = cv2.resize(img_gray, (496, 496))  # Resize nếu cần
    return img_gray

def hog_feature(image_url):
    img_rgb = download_image(image_url)  # Tải ảnh từ URL
    
    print(f"Shape ảnh RGB: {img_rgb.shape}, Min: {img_rgb.min()}, Max: {img_rgb.max()}")

    img_gray = convert_image_rgb_to_gray(img_rgb)  # Chuyển sang grayscale

    # Tính HOG feature
    hog_feats, hog_image = feature.hog(
        img_gray, orientations=9, pixels_per_cell=(4, 4),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
        visualize=True)
    
    return hog_feats

# Gọi hàm với URL ảnh hợp lệ (Không dùng Google Drive vì cần xác thực)
print(hog_feature("https://drive.google.com/uc?id=1ajHjvMqBfATurG1yuEV7fFhv7GQ3ZeIc"))  # Thay URL ảnh thực tế

import test4

test4.test()