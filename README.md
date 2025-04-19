📂 ImageSearchProject
│── 📂 data                         # Thư mục chứa dữ liệu ảnh
│   ├── 📂 images                   # Ảnh gốc
│   ├── 📂 features                  # Lưu vector đặc trưng của ảnh
│   ├── images_list.json             # Metadata danh sách ảnh
│
│── 📂 models                        # Chứa các model & thuật toán tìm kiếm
│   ├── feature_extraction.py        # Trích xuất đặc trưng HOG, HSV, RGB
│   ├── kd_tree.py                   # Cấu trúc KD-Tree để tìm kiếm ảnh
│   ├── search_engine.py             # Hàm tìm kiếm ảnh gần nhất
│
│── 📂 database                      # Lưu trữ dữ liệu NoSQL
│   ├── save_features.py             # Lưu vector đặc trưng vào MongoDB
│   ├── load_features.py             # Truy vấn và tải dữ liệu
│
│── 📂 server                        # Backend xử lý API
│   ├── app.py                        # Flask/FastAPI để nhận yêu cầu tìm kiếm
│   ├── search_api.py                 # API nhận ảnh, xử lý & trả kết quả
│
│── 📂 frontend                      # Giao diện hiển thị kết quả
│   ├── app.js                        # React/Ant Design giao diện người dùng
│
│── requirements.txt                  # Danh sách thư viện cần cài đặt
│── README.md                          # Hướng dẫn cài đặt và sử dụng
408x612





✅ | ❌ | Loại đặc trưng | Mô hình/Method | Chiều vector | Ứng dụng
☐ | ☐ | Đặc trưng CNN | ResNet50/ResNet101 | 2048 | Nhận diện tổng quan
☐ | ☐ | Đặc trưng màu | Color Histogram (LAB) | 256 | Phân biệt qua trang phục
☐ | ☐ | Đặc trưng hình dáng | HOG | 3780 | Nhận diện dáng người
☐ | ☐ | Đặc trưng khuôn mặt | FaceNet | 512 | Nhận diện cá nhân
☐ | ☐ | Tỉ lệ cơ thể | MediaPipe Pose | 5-10 | Phân biệt hình thể







python -m server.app