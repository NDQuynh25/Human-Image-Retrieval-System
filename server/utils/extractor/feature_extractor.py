import cv2
import numpy as np
from PIL import Image
from rembg import remove

# Use relative imports
from .body_ratios_extractor import extract_body_embedding
from .face_extractor import extract_face_embedding
from .shape_extractor import extract_shape_embedding
from .clothing_color_extractor import extract_clothing_color_embedding
from .skin_color_extractor import extract_skin_color_embedding


def read_image(image_file):
    if isinstance(image_file, str):
        image = cv2.imread(image_file)
    elif isinstance(image_file, np.ndarray):
        image = image_file
    else:
        raise ValueError("Unsupported image_file type")

    if image is None:
        raise ValueError("Image could not be read or is invalid.")

    return image


def remove_background(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    removed_bg = remove(image_pil)  # Returns PIL.Image
    return cv2.cvtColor(np.array(removed_bg), cv2.COLOR_RGBA2BGR)


def extract_features(image):
    features = {}

    # Extract body ratios
    try:
        features["body_ratios"] = extract_body_embedding(image)
    except Exception as e:
        print(f"[ERROR] Body extraction failed: {e}")
        features["body_ratios"] = None

    # Extract face vector
    try:
        features["face"] = extract_face_embedding(image)

    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        features["face"] = None

    # Extract shape vector (grayscale)
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features["shape"] = extract_shape_embedding(gray_image)
       
        np.savetxt("shape.txt", features["shape"], fmt="%s")
    except Exception as e:
        print(f"[ERROR] Shape extraction failed: {e}")
        features["shape"] = None

    

    # Extract clothing color vector
    try:
        features["clothing_color"] = extract_clothing_color_embedding(image)
        print(features["clothing_color"])
        np.savetxt("clothing_color.txt", features["clothing_color"], fmt="%s")
    except Exception as e:
        print(f"[ERROR] Clothing color extraction failed: {e}")
        features["clothing_color"] = None
    


    # Extract skin color vector
    try:
        features["skin_color"] = extract_skin_color_embedding(image)
    except Exception as e:
        print(f"[ERROR] Skin color extraction failed: {e}")
        features["skin_color"] = None

    return features


   
    
    


def feature_extractor(image_file):
    image = read_image(image_file)
    print(f"[INFO] Original image shape: {image.shape}")

    image_no_bg = remove_background(image)
    print(f"[INFO] Background removed. Shape: {image_no_bg.shape}")

    features = extract_features(image_no_bg)

    print("[INFO] Features extracted:")
    for k, v in features.items():
        print(f"  - {k}: {'OK' if v is not None else 'Failed'}")

    
    return features


