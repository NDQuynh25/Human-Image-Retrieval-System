import datetime
import os


# Use relative imports with dots
from ..models.image_model import ImageModel
from ..utils.extractor.feature_extractor import feature_extractor
from ..utils.read_image import read_image_from_file_url

def save_image_data(image_url):
    
    try:
        image_file = read_image_from_file_url(image_url)
    except Exception as e:
        print(f"Không thể đọc ảnh từ URL: {e}")
        return None

    try:
        result = feature_extractor(image_file)
        if result is None:
            print("Không thể trích xuất đặc trưng từ ảnh.")
            return None
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ ảnh: {e}")
        return None

    
    
    image_name = os.path.basename(image_url)
    path = image_url
    heigh = image_file.shape[0]
    width = image_file.shape[1]
    created_at = datetime.datetime.utcnow()
    last_modified_at = datetime.datetime.utcnow()

    image_data = ImageModel(
        image_name=image_name,
        path=path,
        height=heigh,
        width=width,
        created_at=created_at,
        last_modified_at=last_modified_at,
        body_ratios=result["body_ratios"],
        face=result["face"],
        shape=result["shape"],
        color=result["color"],  
    )
    image_data.save()
    return image_data

def search_image(image_url):
    try:
        image_file = read_image_from_file_url(image_url)
    except Exception as e:
        print(f"Không thể đọc ảnh từ URL: {e}")
        return None

    try:
        result = feature_extractor(image_file)
        return result
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ ảnh: {e}")
        return None







def search_engine(features):
    
    body_ratios = features["body_ratios"]
    face = features["face"]
    shape = features["shape"]
    color = features["color"]

    final_vector = np.concatenate([
        np.array(body_ratios),
        np.array(face),
        np.array(shape),
        np.array(color)
    ])

    all_images = ImageModel.objects.all()
    # Tìm kiếm ảnh dựa trên các đặc trưng
    
















def test_feature_extraction():
    # Đường dẫn đến ảnh test
    test_image_path = "C:\\Users\\Admin\\Downloads\\anh-son-tung-mtp-thumb.jpg"
    print(f"Đang test với ảnh: {test_image_path}")
    
    result = search_image(test_image_path)
    if result is not None:
        print("Kết quả trích xuất đặc trưng:")
        print(result)
    else:
        print("Không thể trích xuất đặc trưng từ ảnh.")

# Chạy test nếu file được chạy trực tiếp
if __name__ == "__main__":
    test_feature_extraction()  # Removed asyncio.run as it's not needed unless test_feature_extraction is async