
from skimage.feature import hog




def extract_shape_embedding(image_file):
    features, hog_image = hog(image_file, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

