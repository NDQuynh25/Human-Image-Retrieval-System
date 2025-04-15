import cv2
import numpy as np
import requests
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.utils.background_removal import background_removal

def load_image_from_url(url):
    response = requests.get(url)

    img = Image.open(BytesIO(response.content)).convert("RGBA")  
    
    img = background_removal(img).convert("L")
    return np.array(img)

def extract_hog_features(image):
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(2, 2),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features, hog_image

# 🔹 Link ảnh
url1 = "https://drive.usercontent.google.com/download?id=1ajHjvMqBfATurG1yuEV7fFhv7GQ3ZeIc&authuser=0"
url2 = "https://drive.usercontent.google.com/download?id=1ajHjvMqBfATurG1yuEV7fFhv7GQ3ZeIc&authuser=0"

# 🔹 Tải ảnh & trích xuất HOG
img1 = load_image_from_url(url1)
img2 = load_image_from_url(url2)

hog1, hog_vis1 = extract_hog_features(img1)
hog2, hog_vis2 = extract_hog_features(img2)

# 🔹 Tính độ tương đồng
similarity = cosine_similarity([hog1], [hog2])[0][0]
print(f"📏 Độ tương đồng HOG giữa 2 ảnh: {similarity:.4f}")

# 🔹 Hiển thị ảnh gốc và HOG visualization
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title("Ảnh 1 (Grayscale)")
plt.imshow(img1, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("HOG ảnh 1")
plt.imshow(hog_vis1, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Ảnh 2 (Grayscale)")
plt.imshow(img2, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("HOG ảnh 2")
plt.imshow(hog_vis2, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
