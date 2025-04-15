from PIL import Image
from rembg import remove
import matplotlib.pyplot as plt
from PIL import Image

def background_removal(input_image):

    # Tách nền bằng U^2-Net
    output_image = remove(input_image)

    output_image.save("output.png")

    return output_image
