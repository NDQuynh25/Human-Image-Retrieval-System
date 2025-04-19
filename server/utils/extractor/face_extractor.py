import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Thiết bị tính toán
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Khởi tạo mô hình MTCNN và FaceNet
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



# Dò khuôn mặt và trích xuất đặc trưng

def extract_face_embedding(image_file):
    
    face_tensor = mtcnn(image_file)
    
    if face_tensor is None:
        raise ValueError("Không phát hiện được khuôn mặt.")
    
    face_tensor = face_tensor.unsqueeze(0).to(device)
    embedding = resnet(face_tensor).detach().cpu().numpy()[0]
    
    return embedding
