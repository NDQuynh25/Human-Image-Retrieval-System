import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def load_image_from_url(url):
    """Tải ảnh từ URL"""
    response = requests.get(url)
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calculate_body_ratios(image):
    """Tính toán các tỉ lệ cơ thể quan trọng"""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image)
        if not results.pose_landmarks:
            print("Không phát hiện được tư thế người trong ảnh!")
            return None, None
        
        landmarks = results.pose_landmarks.landmark
        
        def get_point(idx):
            return landmarks[idx]

        left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)
        nose = get_point(mp_pose.PoseLandmark.NOSE)
        
        def calc_dist(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        shoulder_width = calc_dist(left_shoulder, right_shoulder)
        hip_width = calc_dist(left_hip, right_hip)
        torso_height = calc_dist(nose, left_hip)
        leg_length = (calc_dist(left_hip, left_knee) + calc_dist(left_knee, left_ankle) +
                      calc_dist(right_hip, right_knee) + calc_dist(right_knee, right_ankle)) / 2
        
        full_height = calc_dist(nose, left_ankle)

        ratios = {}
        ratios['SHR (Shoulder/Hip)'] = shoulder_width / hip_width if hip_width > 0 else 0
        ratios['TOR (Torso/Height)'] = torso_height / full_height if full_height > 0 else 0
        ratios['LLR (Leg/Height)'] = leg_length / full_height if full_height > 0 else 0
        ratios['SHRvsHeight (Shoulder/Height)'] = shoulder_width / full_height if full_height > 0 else 0
        
        return ratios, results.pose_landmarks

def process_and_compare(image_url):
    try:
        image = load_image_from_url(image_url)
        if image is None:
            raise ValueError("Không tải được ảnh từ URL")
            
        small_img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        large_img = cv2.resize(image, (0,0), fx=2.0, fy=2.0)
        
        ratios_orig, landmarks_orig = calculate_body_ratios(image)
        ratios_small, landmarks_small = calculate_body_ratios(small_img)
        ratios_large, landmarks_large = calculate_body_ratios(large_img)
        
        def draw_landmarks(img, landmarks):
            annotated = img.copy()
            if landmarks:
                mp_drawing.draw_landmarks(
                    annotated, landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
            return annotated
        
        def format_ratios_text(ratios):
            return "\n".join([f"{k}: {v:.2f}" for k,v in ratios.items()])
        
        plt.figure(figsize=(18, 6))
        
        # Ảnh gốc (không hiển thị tỉ lệ bên dưới)
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis('off')
        
        # Ảnh nhỏ và hiển thị tỉ lệ bên dưới ảnh
        plt.subplot(1, 3, 2)
        plt.imshow(draw_landmarks(small_img, landmarks_small))
        plt.title("Small (50%)")
        plt.axis('off')
        
        # Ảnh lớn và hiển thị tỉ lệ bên dưới ảnh
        plt.subplot(1, 3, 3)
        plt.imshow(draw_landmarks(large_img, landmarks_large))
        plt.title("Large (200%)")
        plt.axis('off')
        
        plt.suptitle("Body Ratios Comparison (SHR=Shoulder/Hip, TOR=Torso/Height, LLR=Leg/Height, SHRvsHeight=Shoulder/Height)", fontsize=14)
        
        # Hiển thị tỉ lệ dưới từng ảnh nhỏ và lớn ở dưới cùng, căn giữa từng phần
        plt.figtext(0.41, 0.05, format_ratios_text(ratios_small), ha='center', fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        plt.figtext(0.79, 0.05, format_ratios_text(ratios_large), ha='center', fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        print("Gợi ý sửa lỗi:")
        print("- Kiểm tra URL ảnh có hợp lệ không")
        print("- Đảm bảo ảnh có chứa người rõ ràng")
        print("- Cài đặt đủ thư viện: pip install opencv-python mediapipe requests numpy matplotlib")

# Chạy thử với ảnh mẫu
process_and_compare("https://i.pinimg.com/736x/00/49/88/004988fef28db10ef2fb4242c9abb2d2.jpg")
