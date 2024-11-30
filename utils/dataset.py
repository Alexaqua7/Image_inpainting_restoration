import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from utils.utils import get_input_image
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, grayscale_dir, mask_gt_dir, color_gt_dir, transform=None):
        """
        Args:
            grayscale_dir (str): 흑백 이미지 디렉터리 경로
            mask_gt_dir (str): 마스크 정답 이미지 디렉터리 경로
            color_gt_dir (str): 색상 정답 이미지 디렉터리 경로
            transform (callable, optional): 이미지 변환 함수
        """
        self.grayscale_dir = grayscale_dir
        self.mask_gt_dir = mask_gt_dir
        self.color_gt_dir = color_gt_dir
        self.transform = transform
        
        self.grayscale_files = sorted(os.listdir(grayscale_dir))
        self.mask_gt_files = sorted(os.listdir(mask_gt_dir))
        self.color_gt_files = sorted(os.listdir(color_gt_dir))

        assert len(self.grayscale_files) == len(self.mask_gt_files) == len(self.color_gt_files), \
            "데이터셋 크기가 일치하지 않습니다."

    def __len__(self):
        return len(self.grayscale_files)

    def __getitem__(self, idx):
        # 파일 이름 로드
        grayscale_img_name = self.grayscale_files[idx]
        mask_gt_name = self.mask_gt_files[idx]
        color_gt_name = self.color_gt_files[idx]

        # 파일 경로 생성
        grayscale_img_path = os.path.join(self.grayscale_dir, grayscale_img_name)
        mask_gt_path = os.path.join(self.mask_gt_dir, mask_gt_name)
        color_gt_path = os.path.join(self.color_gt_dir, color_gt_name)

        # 이미지 로드
        grayscale_img = Image.open(grayscale_img_path).convert("L")
        mask_gt = Image.open(mask_gt_path).convert("L")
        color_gt = Image.open(color_gt_path).convert("RGB")

        # 변환 적용
        if self.transform:
            grayscale_img = self.transform(grayscale_img)
            mask_gt = self.transform(mask_gt)
            color_gt = self.transform(color_gt)

        return {
            'grayscale': grayscale_img,  # 흑백 이미지 (U-Net1 입력)
            'mask_gt': mask_gt,          # 마스크 복원 정답 (U-Net1 출력)
            'color_gt': color_gt         # 색상 복원 정답 (U-Net2 출력)
        }
    

class CustomImageDataset(Dataset):
    def __init__(self, df, data_dir='./data/train_gt', mode='train', min_polygon_bbox_size=50, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.min_polygon_bbox_size = min_polygon_bbox_size
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(transforms.ToTensor(),
        transforms.Normalize([0.5], [0.225]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = os.path.basename(self.df.iloc[idx, 0])  # Assuming first column is the path
        img_path = os.path.join(self.data_dir, img_path)
        
        # Apply augmentation if in training mode
        if self.mode == 'train':
            image = Image.open(img_path)
            image_input = get_input_image(image, self.min_polygon_bbox_size)
            return image_input

        elif self.mode == 'valid':
            image_input = self.load_input_image(img_path)
            print(image_input)
            print(type(image_input))
            return image_input
        elif self.mode == 'test':
            image = Image.open(img_path)
            return {
                'image_gray_masked':image
            }

    def load_input_image(self, img_input_path):
        image_input = Image.open(img_input_path)
        return image_input