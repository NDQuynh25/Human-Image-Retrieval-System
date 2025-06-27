import cloudinary.uploader

def upload_to_cloudinary(image_path: str) -> str:
    """
    Upload ảnh lên Cloudinary và trả về URL ảnh
    """
    try:
        result = cloudinary.uploader.upload(image_path)
        return result.get("secure_url")
    except Exception as e:
        print(f"❌ Upload Cloudinary thất bại: {e}")
        return None
