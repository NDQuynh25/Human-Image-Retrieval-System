import cv2
import os

# Đường dẫn đến thư mục chứa ảnh
input_folder = r'C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\raw_images'

# Đường dẫn đến thư mục lưu ảnh đã chỉnh sửa
output_folder = r'C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\cut_img'

# Kiểm tra xem thư mục đầu ra có tồn tại không, nếu không thì tạo mới
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lặp qua tất cả các file trong thư mục đầu vào
for filename in os.listdir(input_folder):
    # Kiểm tra nếu là file ảnh (ở đây giả sử định dạng jpg)
    if filename.endswith('.jpg'):
        # Đọc ảnh từ thư mục đầu vào
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Lấy kích thước ảnh
        height, width = image.shape[:2]

        # Tính toán chiều dài tối đa theo tỷ lệ 2/3
        height_t = int(width * 3 / 2)

        if height > height_t:
            continue

        else:
            width_t = int(height * 2 / 3)
            x_offset = (width - width_t) // 2
            image = image[:, x_offset:x_offset + width_t]  # Sử dụng width_t ở đây

        # Lưu ảnh đã cắt vào thư mục đầu ra
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

        print(f"Đã lưu ảnh vào {output_path}")
