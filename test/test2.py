import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# Tải mô hình PoseNet từ TensorFlow Hub
posenet_model_url = "https://tfhub.dev/tensorflow/posenet/mobilenet/4"  # PoseNet URL


# Kiểm tra mô hình đã được tải thành công
print("PoseNet model loaded successfully!")

# Tải ảnh từ Google Drive
def load_image_from_drive(share_id):
    url = f"https://drive.google.com/uc?id={share_id}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

# Chuyển ảnh PIL sang numpy array và đổi từ RGB sang BGR (OpenCV yêu cầu BGR)
def convert_pil_to_cv2(image_pil):
    return np.array(image_pil)[:, :, ::-1].astype(np.uint8)  # Convert RGB to BGR

# Dự đoán Pose Estimation với PoseNet
def estimate_pose(image_cv2):
    # Chuyển ảnh sang RGB và chuẩn bị TensorFlow input
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    input_image = tf.convert_to_tensor(image_rgb, dtype=tf.float32)

    # Tải mô hình PoseNet từ TensorFlow Hub
    model = hub.load(posenet_model_url)
    outputs = model(input_image)

    keypoints = outputs['keypoints'].numpy()  # Các điểm khớp (landmarks) trong không gian 2D

    # Vẽ các điểm khớp lên ảnh
    for point in keypoints:
        x, y = point[0], point[1]
        cv2.circle(image_cv2, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    return image_cv2, keypoints

# Chạy toàn bộ quy trình
share_id1 = "1iCiuD9eH0TD-5z6YBlAkX6C5vATFPous"  # ID ảnh từ Google Drive
share_id2 = "1lLGWXj4HfOgcthigwSH-xf6gNQ7ayxvx"  # ID ảnh thứ hai từ Google Drive

# Tải và xử lý hai ảnh
image_pil1 = load_image_from_drive(share_id1)
image_pil2 = load_image_from_drive(share_id2)

# Chuyển đổi ảnh PIL thành CV2 (numpy array)
image_cv21 = convert_pil_to_cv2(image_pil1)
image_cv22 = convert_pil_to_cv2(image_pil2)

# Ước lượng tư thế (Pose Estimation) cho ảnh 1 và ảnh 2
image_with_pose1, keypoints1 = estimate_pose(image_cv21)
image_with_pose2, keypoints2 = estimate_pose(image_cv22)

# Hiển thị hai ảnh với các điểm khớp đã phát hiện
plt.figure(figsize=(10, 10))

# Hiển thị ảnh 1
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_with_pose1, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.title("Tư thế trong ảnh 1")
plt.axis("off")

# Hiển thị ảnh 2
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_pose2, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.title("Tư thế trong ảnh 2")
plt.axis("off")

plt.show()
