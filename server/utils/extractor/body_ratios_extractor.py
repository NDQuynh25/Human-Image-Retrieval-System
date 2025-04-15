import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from matplotlib import pyplot as plt

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



def calculate_all_body_ratios(image):
    """Tính toán tất cả các tỉ lệ cơ thể quan trọng"""
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Chuyển đổi màu BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None, None
        
        # Vẽ các điểm pose lên ảnh
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66)),  # Màu các điểm
            mp_drawing.DrawingSpec(color=(245,66,230)))  # Màu đường nối
        
        landmarks = results.pose_landmarks.landmark
        
        # Lấy tọa độ các điểm quan trọng
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        # Tính toán các khoảng cách
        def calculate_distance(point1, point2):
            return np.linalg.norm([point1.x - point2.x, point1.y - point2.y])
        
        # 1. Tỉ lệ vai/hông
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        hip_width = calculate_distance(left_hip, right_hip)
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
        
        # 2. Tỉ lệ đầu/thân
        head_height = abs(nose.y - left_shoulder.y)
        torso_height = abs(left_shoulder.y - left_hip.y)
        head_torso_ratio = head_height / torso_height if torso_height > 0 else 0
        
        # 3. Tỉ lệ tay/chân
        arm_length = (calculate_distance(left_shoulder, left_elbow) + 
                    calculate_distance(left_elbow, left_wrist))
        leg_length = (calculate_distance(left_hip, left_knee) + 
                    calculate_distance(left_knee, left_ankle))
        arm_leg_ratio = arm_length / leg_length if leg_length > 0 else 0
        
        # 4. Tỉ lệ eo/đùi
        waist_width = calculate_distance(left_hip, right_hip)
        thigh_length = calculate_distance(left_hip, left_knee)
        waist_thigh_ratio = waist_width / thigh_length if thigh_length > 0 else 0
        
        # 5. Tỉ lệ chiều cao/cánh tay
        height = abs(nose.y - left_ankle.y)
        arm_span = calculate_distance(left_shoulder, left_wrist)
        height_arm_ratio = height / arm_span if arm_span > 0 else 0
        
        body_ratios = {
            "shoulder_hip_ratio": shoulder_hip_ratio,
            "head_torso_ratio": head_torso_ratio,
            "arm_leg_ratio": arm_leg_ratio,
            "waist_thigh_ratio": waist_thigh_ratio,
            "height_arm_ratio": height_arm_ratio,
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "arm_length": arm_length,
            "leg_length": leg_length
        }
        
        return body_ratios, annotated_image






def extract_body_embedding_vector(image):
    
    ratios, annotated_image = calculate_all_body_ratios(image)
    
    if ratios is None:
        print("Không phát hiện được người trong ảnh")
        return
    
    return np.array([
        ratios.get("shoulder_hip_ratio", 0),
        ratios.get("head_torso_ratio", 0),
        ratios.get("arm_leg_ratio", 0),
        ratios.get("waist_thigh_ratio", 0),
        ratios.get("height_arm_ratio", 0),
      
    ], dtype=np.float32)


