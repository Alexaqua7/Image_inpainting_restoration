import os
from PIL import Image

# 기존 폴더 경로와 새 폴더 경로
data_dir = "C:/Users/User/Desktop/DACON_1/data"
train_gt_dir = os.path.join(data_dir, "train_gt")
grayscale_dir = os.path.join(data_dir, "train_gt_grayscale")

# 새 폴더 생성
os.makedirs(grayscale_dir, exist_ok=True)

# 폴더 내 이미지 처리
for file_name in os.listdir(train_gt_dir):
    file_path = os.path.join(train_gt_dir, file_name)

    # 이미지 파일만 처리
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        with Image.open(file_path) as img:
            # 흑백 변환
            grayscale_img = img.convert("L")
            # 새 폴더에 저장
            grayscale_img.save(os.path.join(grayscale_dir, file_name))

print(f"이미지를 흑백으로 변환하여 {grayscale_dir} 폴더에 저장 완료.")