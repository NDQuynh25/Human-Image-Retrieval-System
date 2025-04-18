import os
from dotenv import load_dotenv
from mongoengine import connect

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

def init_db():
    connect(
        db=MONGO_DB_NAME,
        host=MONGO_URI,
        alias="default"
    )
    print(f"✅ Kết nối MongoDB Atlas thành công: {MONGO_DB_NAME}")
