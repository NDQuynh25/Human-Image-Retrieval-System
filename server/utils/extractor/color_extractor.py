import cv2
import numpy as np

def extract_color_embedding(image, bins=(8, 8, 8)):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.flatten().tolist()
    
    except Exception as e:
        print(f"[ERROR] Color extraction failed: {e}")
        return None

    
