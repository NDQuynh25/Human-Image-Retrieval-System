import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.utils.background_removal import background_removal

# Tải ảnh từ Google Drive
def load_image_from_drive(share_id):
    url = f"https://drive.google.com/uc?id={share_id}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    background_removal(image)
    return image

# Chuyển ảnh PIL sang numpy array và đổi từ RGB sang BGR (OpenCV yêu cầu BGR)
def convert_pil_to_cv2(image_pil):
    # Convert ảnh PIL sang numpy array và đổi màu từ RGB sang BGR
    return np.array(image_pil)[:, :, ::-1].astype(np.uint8)  # Convert RGB to BGR và đảm bảo kiểu uint8


# Phát hiện khuôn mặt và vẽ hình chữ nhật quanh khuôn mặt
def detect_and_draw_faces(image_cv2):
    # Mô hình Haar Cascade để nhận diện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image_cv2, faces  # Trả về ảnh đã vẽ hình chữ nhật và thông tin khuôn mặt

# Trích xuất đặc trưng khuôn mặt
def extract_face_embedding(image_cv2, faces):
    # Trích xuất đặc trưng khuôn mặt từ ảnh
    embeddings = []
    for (x, y, w, h) in faces:
        # Cắt phần khuôn mặt ra khỏi ảnh
        face = image_cv2[y:y+h, x:x+w]
        # Trích xuất đặc trưng khuôn mặt
        result = DeepFace.represent(face, model_name='VGG-Face', enforce_detection=False)
        embeddings.append(result)
    return embeddings

# Lưu đặc trưng khuôn mặt vào file
def save_embeddings_to_file(embeddings, filename="embeddings.txt"):
    with open(filename, "w") as f:
        for emb in embeddings:
            f.write(str(emb) + "\n")
    print(f"Đặc trưng đã được lưu vào {filename}")

# Chạy toàn bộ
#1qzKmY0JmSDFt9ue4yKp4XmKuQjYi9w98
share_id = "1j_NelCC1oddffMdeg6xHgSuGPmo-fEg8"  
#1j_NelCC1oddffMdeg6xHgSuGPmo-fEg8 
image_pil = load_image_from_drive(share_id)
image_cv2 = convert_pil_to_cv2(image_pil)

# Phát hiện và vẽ khuôn mặt
image_with_faces, faces = detect_and_draw_faces(image_cv2)

# Trích xuất đặc trưng của khuôn mặt
if len(faces) > 0:
    embeddings = extract_face_embedding(image_cv2, faces)
    print("Đặc trưng khuôn mặt được trích xuất:", embeddings)
    
    # Lưu đặc trưng vào file
    save_embeddings_to_file(embeddings)
else:
    print("Không phát hiện khuôn mặt.")

# Hiển thị kết quả
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
plt.title("Khuôn mặt được phát hiện và trích xuất đặc trưng")
plt.axis("off")
plt.show()
