import datetime
import os
import traceback
import numpy as np
from skimage import feature

from server.utils.extractor import clothing_color_extractor
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

from ..models.image_model import ImageModel
from ..utils.extractor.feature_extractor import feature_extractor
from ..utils.read_image import read_image_from_file_url
from ..services.cloudinary_service import upload_to_cloudinary

def compute_zscore_params():
    all_images = ImageModel.objects.all()
    feature_lists = {
        "body_ratios": [],
        "face": [],
        "shape": [],
        "clothing_color": [],
        "skin_color": []
    }

    for img in all_images:
        for key in feature_lists:
            if img.features.get(key):
                feature_lists[key].append(np.array(img.features[key], dtype=np.float32))

    scalers = {}
    for key, vectors in feature_lists.items():
        if vectors:
            X = np.stack(vectors)
            scaler = StandardScaler()
            scaler.fit(X)
            scalers[key] = scaler

    return scalers

def normalize_features_zscore(features: dict, scalers: dict) -> np.ndarray:
    def to_np(v):
        return np.array(v, dtype=np.float32).reshape(1, -1)

    normalized_parts = []
    for key in ["body_ratios", "face", "shape", "clothing_color", "skin_color"]:
        if key in scalers:
            part = scalers[key].transform(to_np(features[key]))[0]
        else:
            part = to_np(features[key])[0]  # fallback nếu không có scaler
        normalized_parts.append(part)

    full_vector = np.concatenate(normalized_parts)
    return normalize(full_vector.reshape(1, -1), norm='l2')[0]

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
        print(traceback.format_exc())
        print(f"Lỗi khi trích xuất đặc trưng từ ảnh: {e}")
        return None

    image_name = os.path.basename(image_url)
    path = upload_to_cloudinary(image_url)
    heigh = image_file.shape[0]
    width = image_file.shape[1]
    created_at = datetime.datetime.utcnow()
    last_modified_at = datetime.datetime.utcnow()
    features = {
        "body_ratios": result["body_ratios"] if result["body_ratios"] is not None else [],
        "face": result["face"] if result["face"] is not None else [],
        "shape": result["shape"] if result["shape"] is not None else [],
        "clothing_color": result["clothing_color"] if result["clothing_color"] is not None else [],
        "skin_color": result["skin_color"] if result["skin_color"] is not None else [],
    }
    normalized_features = normalize_features_zscore(features, compute_zscore_params())

    image_data = ImageModel(
        image_name=image_name,
        path=path,
        height=heigh,
        width=width,
        created_at=created_at,
        last_modified_at=last_modified_at,
        features=features,
        normalized_features=normalized_features.tolist()
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
        scalers = compute_zscore_params()
        return search_engine(result, scalers)
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ ảnh: {e}")
        return None

def search_engine(features, scalers):
    vector_query = normalize_features_zscore(features, scalers)
    all_images = ImageModel.objects.all()

    results = []
    for image in all_images:
        try:
            image_vector = normalize_features_zscore(image.features, scalers)
            sim = cosine_similarity(
                vector_query.reshape(1, -1),
                image_vector.reshape(1, -1)
            )[0][0]

            results.append({
                "image_url": image.path,
                "similarity": float(sim)  # Convert np.float32 -> float for JSON
            })
        except Exception as e:
            print(f"Lỗi so sánh ảnh {image.image_name}: {e}")

    top_results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:3]
    return top_results
