import cv2
import numpy as np
import requests
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from rembg import remove  # Add this import for background removal

# Then update the load_image_from_url function to use background removal if needed
def load_image_from_url(url):
    response = requests.get(url)
    
    # First convert the binary content to an image
    img = Image.open(BytesIO(response.content))
    
    # Enable background removal
    img = remove(img)  # This works directly with PIL Image
    
    # For HOG features, we need grayscale
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((128, 192))  
    return np.array(img)

def extract_hog_features(image):
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features, hog_image

# 🔹 Link 2 ảnh
url1 = "https://i.pinimg.com/736x/e6/a8/e4/e6a8e437646c3287138e7fbd29c26eb5.jpg"
url2 = "https://i.pinimg.com/474x/71/97/8d/71978d5d7116ba1805816b44e49c7ee9.jpg"
def remove_background(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    removed_bg = remove(image_pil)  # Returns PIL.Image
    return cv2.cvtColor(np.array(removed_bg), cv2.COLOR_RGBA2BGR)

# 🔹 Tải ảnh & trích xuất đặc trưng
img1 = load_image_from_url(url1)
img2 = load_image_from_url(url2)

hog1, hog_img1 = extract_hog_features(img1)
hog2, hog_img2 = extract_hog_features(img2)


print(hog1)
np.savetxt("hog1.txt", hog1, fmt="%.6f")
np.savetxt("hog1.shape.txt", hog1.shape, fmt="%.6f")
print("✅ Đã lưu HOG vector vào hog1.txt")
# 🔹 So sánh HOG embedding bằng cosine similarity
similarity = cosine_similarity([hog1], [hog2])[0][0]
print(f"📏 Độ tương đồng HOG giữa 2 ảnh: {similarity:.4f}")

# 🔹 Hiển thị ảnh và HOG visualization
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Ảnh 1 (grayscale)")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(hog_img1, cmap='inferno')
plt.title("HOG ảnh 1")
plt.axis('off')

plt.tight_layout()
plt.show()

