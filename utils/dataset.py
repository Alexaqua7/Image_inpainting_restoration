import os
from PIL import Image
from torch.utils.data import Dataset

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