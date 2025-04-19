from mongoengine import Document, StringField, DateTimeField, UUIDField, IntField, DictField, ListField, FloatField
import datetime

class ImageModel(Document):
    """Mô hình lưu trữ ảnh và các đặc trưng của ảnh"""

    meta = {
        'collection': 'image_features',  # Tên collection (tên bảng) trong MongoDB
        'indexes': ['image_name', 'path'],  # Tạo chỉ mục cho các trường image_name và path
        'ordering': ['-created_at'],  # Sắp xếp giảm dần theo trường created_at
        'strict': False  # Cho phép lưu các trường không khai báo trong mô hình
    }

    
    image_name = StringField(required=True)
    path = StringField(required=True)
    height = IntField(required=True)
    width = IntField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    last_modified_at = DateTimeField(default=datetime.datetime.utcnow)

    # Các đặc trưng của ảnh
    features = DictField()  # Các đặc trưng tổng hợp (ví dụ: HOG, RGB, HSV, Pose)
    body_ratios = ListField(FloatField())  # Các đặc trưng HOG
    face = ListField(FloatField())  # Các đặc trưng RGB
    shape = ListField(FloatField())  # Các đặc trưng HSV
    color = ListField(FloatField())  # Các đặc trưng Pose
