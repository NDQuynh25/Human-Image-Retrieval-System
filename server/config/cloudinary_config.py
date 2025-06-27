import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os

load_dotenv()

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("API_KEY")
CLOUDINARY_API_SECRET = os.getenv("API_SECRET")
def config_cloudinary():
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    print(f"✅ Kết nối Cloudinary thành công: {CLOUDINARY_CLOUD_NAME}")
