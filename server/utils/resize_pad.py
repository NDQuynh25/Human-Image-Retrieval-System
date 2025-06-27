import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def calculate_padding_color(image, border_size=20):
    h, w = image.shape[:2]
    top = image[:border_size, :]
    bottom = image[-border_size:, :]
    left = image[:, :border_size]
    right = image[:, -border_size:]

    borders = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])
    return tuple(np.round(np.mean(borders, axis=0)).astype(int))

def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_color = calculate_padding_color(image)
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    result_pil = remove(pil_image).convert("RGBA")  # giữ nền trong suốt
    return np.array(result_pil)

def process_image(index_filename_tuple, input_folder, output_folder, target_size):
    index, filename = index_filename_tuple
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"person_{index + 1}.png")  # PNG để giữ alpha

    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            return

        padded = resize_with_padding(image, target_size)
        final_rgba = remove_background(padded)
        Image.fromarray(final_rgba).save(output_path)
        print(f"✅ {filename} ➜ person_{index + 1}.png")
    except Exception as e:
        print(f"❌ Lỗi {filename}: {e}")

def process_folder_parallel(input_folder, output_folder, target_size=(600, 900), max_workers=4):
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

    if not files:
        print("⚠️ Không có ảnh nào trong thư mục.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            partial(process_image, input_folder=input_folder, output_folder=output_folder, target_size=target_size),
            enumerate(files)
        )

if __name__ == '__main__':
    input_dir = r"C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\raw_images"
    output_dir = r"C:\Users\Admin\Documents\Human-Image-Retrieval-System\server\dataset\images"

    if not os.path.exists(input_dir):
        print(f"❌ Không tồn tại thư mục: {input_dir}")
    else:
        print("🔄 Đang xử lý ảnh (xóa phông và resize)...")
        process_folder_parallel(input_dir, output_dir)
        print("🎉 Hoàn tất!")
