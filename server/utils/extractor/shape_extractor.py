
from skimage.feature import hog
import traceback
import cv2
import numpy as np





def extract_shape_embedding(image_file):
    try:
        features, hog_image = hog(cv2.resize(image_file, (128, 192))  , orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    except Exception as e:
        print(traceback.format_exc())
        print(f"[ERROR] Shape extraction failed: {e}")
        return None
    # Ép về float chuẩn + làm tròn
    return [float(f) for f in np.round(np.array(features, dtype=np.float64), decimals=8)]
