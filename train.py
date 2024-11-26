import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
import argparse
from model.UNetGenerator import UNetGenerator, PatchGANDiscriminator
from utils.trainer import train
from utils.utils import seed_everything
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    # 경로 설정
    parser.add_argument('--origin_dir', type=str, default='../data/train_gt', help='원본 이미지 폴더 경로')
    parser.add_argument('--damage_dir', type=str, default='../data/train_input', help='손상된 이미지 폴더 경로')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning Rate 설정')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size 설정')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Model이 저장될 경로')
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='Baseline_with_Scheduler', help='WandB Run name 설정')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)
    # 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )
    # 데이터셋 및 DataLoader 생성
    train_dataset = CustomDataset(damage_dir=args.damage_dir, 
                                  origin_dir=args.origin_dir, 
                                  transform=transform)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  num_workers=4)

    # 모델 저장을 위한 디렉토리 생성
    os.makedirs(args.model_save_dir, exist_ok=True)

    lambda_pixel = 100  # 픽셀 손실에 대한 가중치
            
    # Generator와 Discriminator 초기화
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()

    # 손실 함수 및 옵티마이저 설정
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr = args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr = args.lr) 

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=args.num_epoch * len(train_dataloader), eta_min=0)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=args.num_epoch * len(train_dataloader), eta_min=0)

    train(train_dataloader=train_dataloader, epochs=args.num_epoch, generator=generator, discriminator=discriminator, optimizer_G=optimizer_G, 
          optimizer_D=optimizer_D, scheduler_G=scheduler_G, scheduler_D=scheduler_D, criterion_GAN=criterion_GAN, criterion_pixelwise=criterion_pixelwise, 
          lambda_pixel=lambda_pixel, model_save_dir = args.model_save_dir)
if __name__ == '__main__':
    main()