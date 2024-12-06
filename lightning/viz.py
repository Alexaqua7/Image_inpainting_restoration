import numpy as np
from PIL import Image
import os
valid_dir = '../../data/valid_input'
output_dir = '../../data/valid_image_gray_masked'  # 저장할 디렉토리 경로 설정

# 저장할 디렉토리 없으면 생성
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(valid_dir):
    # npy 파일만 처리
    if file.endswith('.npy'):
        file_path = os.path.join(valid_dir, file)
        
        # npy 파일 로드
        data = np.load(file_path, allow_pickle=True).item()
        
        # 'image_gray_masked' 키만 가져오기
        if 'image_gray_masked' in data:
            image_gray_masked = data['image_gray_masked']
            
            # 이미지 배열을 PIL 이미지로 변환 (0~255 스케일 가정)
            # image_pil = Image.fromarray(image_gray_masked.astype(np.uint8))
            
            # 저장 경로 생성
            save_path = os.path.join(output_dir, file.replace('.npy', '_gray_masked.png'))
            
            # 이미지 저장
            image_gray_masked.save(save_path)
            print(f"Saved: {save_path}")