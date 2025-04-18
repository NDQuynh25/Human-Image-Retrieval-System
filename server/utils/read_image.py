import urllib.parse
import cv2
import os

def read_image_from_file_url(file_path: str):
    try:
        # Kiểm tra xem file_path có phải là URL không
        if file_path.startswith(('http://', 'https://', 'file://')):
            parsed_url = urllib.parse.urlparse(file_path)
            path = os.path.abspath(os.path.join(parsed_url.netloc, parsed_url.path))
        else:
            # Nếu không phải URL, sử dụng trực tiếp đường dẫn
            path = file_path
            
        print(f"Đang đọc ảnh từ đường dẫn: {path}")
        image = cv2.imread(path)
        
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh từ đường dẫn: {path}")

        print("Đọc ảnh thành công.")
        return image
    except Exception as e:
        print(f"Lỗi khi đọc file: {file_path} - {e}")
        return None
