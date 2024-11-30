import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
from utils.dataset import CustomDataset, CustomImageDataset
from torch.utils.data import DataLoader
import argparse
from model.UNetGenerator import UNetGenerator, PatchGANDiscriminator
from utils.trainer import train
from utils.utils import seed_everything
import wandb
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    # 경로 설정
    parser.add_argument('--origin_dir', type=str, default='../data/train_gt', help='원본 색상 이미지 폴더 경로')
    parser.add_argument('--mask_gt_dir', type=str, default='../data/train_gt_grayscale', help='원본 흑백 이미지 폴더 경로')
    parser.add_argument('--gray_scale_dir', type=str, default='../data/train_input', help='손상된 이미지 폴더 경로')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate 설정')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size 설정')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Model이 저장될 경로')
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch 수 설정')
    parser.add_argument('--val_every', type=int, default=1, help='Validation을 수행할 에폭 주기를 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='Valid Try', help='WandB Run name 설정')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)
    train_df = pd.read_csv('../data/train.csv')
    # 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.225])
    ])
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )
    # 데이터셋 및 DataLoader 생성
    train_dataset = CustomDataset(grayscale_dir = args.gray_scale_dir, 
                                  mask_gt_dir = args.mask_gt_dir, 
                                  color_gt_dir = args.origin_dir,
                                  transform=transform)

    # train_dataset = CustomImageDataset(train_df, args.origin_dir, mode='valid', min_polygon_bbox_size=50, transform=transform)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=4,
                                  pin_memory = True)
    
    val_dataloader = DataLoader(train_dataset, 
                                  batch_size=1, 
                                  shuffle=True, 
                                  num_workers=4,
                                  pin_memory = True)

    # 모델 저장을 위한 디렉토리 생성
    os.makedirs(args.model_save_dir, exist_ok=True)

    # 손실 함수 및 옵티마이저 설정
    unet1 = smp.UPerNet(encoder_name="tu-hrnet_w64", encoder_weights="imagenet", in_channels=1, classes=1)  # 흑백 -> 마스크 복원
    unet2 = smp.UPerNet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=3)  # 흑백+복원 -> 색상 복원

    # 손실 함수
    criterion_pixelwise = nn.L1Loss()

    # 옵티마이저 및 스케줄러
    optimizer_U1 = optim.AdamW(unet1.parameters(), lr=args.lr)
    optimizer_U2 = optim.AdamW(unet2.parameters(), lr=args.lr)
    scheduler_G = CosineAnnealingLR(optimizer_U1, T_max=args.num_epoch * len(train_dataloader), eta_min=0)
    scheduler_D = CosineAnnealingLR(optimizer_U2, T_max=args.num_epoch * len(train_dataloader), eta_min=0)

    # 학습 실행
    train(
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        epochs=50,
        unet1=unet1,
        unet2=unet2,
        optimizer_U1=optimizer_U1,
        optimizer_U2=optimizer_U2,
        scheduler_U1=scheduler_G,
        scheduler_U2=scheduler_D,
        criterion_pixelwise=criterion_pixelwise,
        val_every = args.val_every,
        model_save_dir=args.model_save_dir,
    )
if __name__ == '__main__':
    main()