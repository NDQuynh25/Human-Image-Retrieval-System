import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
gender_categories = ['Nam', 'Nữ']
gender_counts = [25, 20]

age_categories = ['Trẻ em (0-12)', 'Thanh thiếu niên (13-19)', 'Người lớn (20+)', 'Người già']
age_counts = [10, 15, 30, 0]  # Giả sử người già = 0 (bạn có thể thay số liệu thực)

# Tạo figure với 2 subplot (1 hàng, 2 cột)
plt.figure(figsize=(12, 5))

# Subplot 1: Phân loại theo giới tính (Bar Horizontal)
plt.subplot(1, 2, 1)
plt.barh(gender_categories, gender_counts, color=['blue', 'pink'])
plt.title('Phân loại theo giới tính')
plt.xlabel('Số lượng ảnh')
plt.xlim(0, 30)
for i, count in enumerate(gender_counts):
    plt.text(count + 0.5, i, str(count), ha='left', va='center')

# Subplot 2: Phân loại theo độ tuổi (Bar Vertical)
plt.subplot(1, 2, 2)
bars = plt.bar(age_categories, age_counts, color=['lightgreen', 'orange', 'red', 'gray'])
plt.title('Phân loại theo độ tuổi')
plt.ylabel('Số lượng ảnh')
plt.ylim(0, 35)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, str(int(height)), ha='center', va='bottom')

# Hiển thị đồ thị
plt.tight_layout()
plt.show()