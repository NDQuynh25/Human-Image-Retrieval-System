import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1

# Thiết bị tính toán
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Khởi tạo mô hình MTCNN và FaceNet
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Tải ảnh từ URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# Dò khuôn mặt và trích xuất đặc trưng
def extract_face_embedding_from_url(image_url):
    image = load_image_from_url(image_url)
    face_tensor = mtcnn(image)
    
    if face_tensor is None:
        raise ValueError("Không phát hiện được khuôn mặt.")
    
    face_tensor = face_tensor.unsqueeze(0).to(device)
    embedding = resnet(face_tensor).detach().cpu().numpy()[0]
    
    return embedding, face_tensor[0].cpu()

# Tính độ tương đồng cosine sử dụng scipy
def calculate_cosine_similarity(emb1, emb2):
    # Tính khoảng cách cosine
    distance = cosine(emb1, emb2)
    similarity = 1 - distance  # Chuyển khoảng cách cosine thành độ tương đồng
    return similarity

# --- Ví dụ sử dụng ---
url1 = "https://drive.google.com/uc?id=1G3guTt0JopxRyPxCM_aqTLuo2rZ5953i"
url2 = "https://drive.google.com/uc?id=1nN3otEL3EhMa8xCFu_phPoi4udZe8hjX"

embedding1, face1 = extract_face_embedding_from_url(url1)
embedding2, face2 = extract_face_embedding_from_url(url2)

# Lưu embedding1 vào file text
with open('embedding1.txt', 'w') as f:
    for value in embedding1:
        f.write(f"{value}\n")  # mỗi dòng là một số float

# Tính độ tương đồng cosine và xác suất
similarity_score = calculate_cosine_similarity(embedding1, embedding2)
accuracy = similarity_score * 100

print(f"Độ tương đồng (cosine similarity): {similarity_score:.4f}")
print(f"Xác suất hai khuôn mặt là cùng một người: {accuracy:.2f}%")

if similarity_score > 0.7:
    print("=> Hai ảnh này có thể là cùng một người.")
else:
    print("=> Có thể là người khác nhau.")

# Hiển thị ảnh khuôn mặt
plt.subplot(1, 2, 1)
plt.imshow(face1.permute(1, 2, 0).numpy())
plt.title("Face 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(face2.permute(1, 2, 0).numpy())
plt.title("Face 2")
plt.axis("off")

plt.show()
