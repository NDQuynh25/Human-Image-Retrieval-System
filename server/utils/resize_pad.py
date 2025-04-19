import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def calculate_padding_color(image, border_size=20):
    """
    Tính toán màu padding từ các vùng viền xung quanh ảnh
    """
    h, w = image.shape[:2]
    
    # Lấy các vùng viền (trên, dưới, trái, phải)
    top_border = image[:border_size, :]
    bottom_border = image[-border_size:, :]
    left_border = image[:, :border_size]
    right_border = image[:, -border_size:]
    
    # Kết hợp tất cả các vùng viền
    all_borders = np.concatenate([
        top_border.reshape(-1, 3),
        bottom_border.reshape(-1, 3),
        left_border.reshape(-1, 3),
        right_border.reshape(-1, 3)
    ])
    
    # Tính màu trung bình của các viền
    avg_color = np.mean(all_borders, axis=0)
    return tuple(np.round(avg_color).astype(int))

def resize_with_padding(image, target_size):
    """
    Resize ảnh và thêm padding dựa trên màu viền
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Tính tỷ lệ scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize ảnh
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tính màu padding từ viền ảnh gốc
    padding_color = calculate_padding_color(image)
    
    # Tạo ảnh nền với màu padding
    padded_image = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    
    # Đặt ảnh đã resize vào giữa
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded_image

def process_image(index_filename_tuple, input_folder, output_folder, target_size):
    index, filename = index_filename_tuple
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"person_{index + 1}.jpg")

    try:
        # Đọc ảnh
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            return

        # Xử lý ảnh
        processed_image = resize_with_padding(image, target_size)
        
        # Lưu ảnh
        cv2.imwrite(output_path, processed_image)
        print(f"✅ Đã xử lý: {filename} -> person_{index + 1}.jpg")
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {filename}: {str(e)}")

def process_folder_parallel(input_folder, output_folder, target_size=(600, 900), max_workers=4):
    """Xử lý song song thư mục ảnh"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Lọc file ảnh
    valid_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print("⚠️ Không tìm thấy file ảnh nào trong thư mục nguồn")
        return
    
    # Xử lý song song
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            partial(process_image, 
                   input_folder=input_folder, 
                   output_folder=output_folder, 
                   target_size=target_size),
            enumerate(files)
        )

if __name__ == '__main__':
    # Cấu hình đường dẫn
    input_dir = r"C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\cut_img"
    output_dir = r"C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\images"
    
    # Kiểm tra và chạy chương trình
    if not os.path.exists(input_dir):
        print(f"❌ Thư mục nguồn '{input_dir}' không tồn tại")
    else:
        print("🔄 Đang xử lý ảnh (tự động tính màu padding từ viền)...")
        process_folder_parallel(input_dir, output_dir)
        print("🎉 Hoàn thành xử lý ảnh!")