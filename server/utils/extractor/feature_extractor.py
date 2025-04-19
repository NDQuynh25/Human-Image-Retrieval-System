import sys
import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

# Use relative imports
from .body_ratios_extractor import extract_body_embedding
from .face_extractor import extract_face_embedding
from .shape_extractor import extract_shape_embedding


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
        # Change np.save to np.savetxt since you want to save as text with fmt parameter
        np.savetxt("shape.txt", features["shape"], fmt="%s")
    except Exception as e:
        print(f"[ERROR] Shape extraction failed: {e}")
        features["shape"] = None

    return features


   
    try:
        features["color"] = extract_color_embedding(image)
        np.savetxt("color.txt", features["color"], fmt="%s")
    except Exception as e:
        print(f"[ERROR] Color extraction failed: {e}")
        features["color"] = None


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


