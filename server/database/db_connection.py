
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import json
from datetime import datetime

uri = "mongodb+srv://NDQ25:NDQ250903@cluster0.0xdfw.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))



def getConnection():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")

        db = client["image_search"]
        collection = db["image_features"]
        return collection
    except Exception as e:
        print(e)

getConnection()
def save_feature(image_name, image_path, height, width, features):
    """
    Lưu đặc trưng ảnh vào MongoDB
    :param image_name: Tên file ảnh
    :param image_path: Đường dẫn ảnh
    :param height: Chiều cao ảnh
    :param width: Chiều rộng ảnh
    :param features: Dict chứa các đặc trưng ảnh (hog, rgb, hsv, pose)
    """
    image_data = {
       
        "image_name": image_name,
        "path": image_path,
        "height": height,
        "width": width,
        "created_at": datetime.utcnow().isoformat(),
        "last_modified_at": datetime.utcnow().isoformat(),
        "features": features
    }

    # Lưu vào MongoDB
    try:
        getConnection().insert_one(image_data)
        print(f"✅ Đã lưu ảnh {image_name} vào MongoDB!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu ảnh {image_name}: {e}")

