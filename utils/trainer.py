import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import segmentation_models_pytorch as smp
import wandb
import numpy as np


def train(train_dataloader, epochs, unet1, unet2, optimizer_U1, optimizer_U2, scheduler_U1, scheduler_U2, criterion_pixelwise, model_save_dir='./checkpoints'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # 모델 초기화
    unet1.to(device)
    unet2.to(device)

    best_loss_sum = float("inf")  # 손실 합 최소화 기준

    for epoch in range(1, epochs + 1):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training: Epoch [{epoch}/{epochs}]")

        total_loss_mask = 0
        total_loss_color = 0
        count = 0

        for i, batch in progress_bar:
            unet1.train()
            unet2.train()
            optimizer_U1.zero_grad()
            optimizer_U2.zero_grad()

            # 데이터 불러오기
            grayscale_img = batch['grayscale'].to(device)  # 흑백 이미지 (input to U-Net1)
            color_gt = batch['color_gt'].to(device)        # 색상 이미지 (ground truth for U-Net2)
            mask_gt = batch['mask_gt'].to(device)          # 복원할 마스크 이미지 (ground truth for U-Net1)

            # U-Net1: 마스크 복원
            with torch.cuda.amp.autocast():
                mask_pred = unet1(grayscale_img)
                loss_mask = criterion_pixelwise(mask_pred, mask_gt)
            
            scaler.scale(loss_mask).backward()  # Mask 손실만 backward
            scaler.step(optimizer_U1)
            scaler.update()
            scheduler_U1.step()

            # U-Net2: 색상 복원
            with torch.cuda.amp.autocast():
                color_pred = unet2(mask_pred.detach())  # detach로 gradient 차단
                loss_color = criterion_pixelwise(color_pred, color_gt)
            
            scaler.scale(loss_color).backward()
            scaler.step(optimizer_U2)
            scaler.update()
            scheduler_U2.step()

            # 손실 누적
            total_loss_mask += loss_mask.item()
            total_loss_color += loss_color.item()
            count += 1

            # wandb 로그 기록 (배치 단위)
            wandb.log({
                "train/epoch": epoch,
                "train/batch": i,
                "train/loss_mask": loss_mask.item(),
                "train/loss_color": loss_color.item(),
                "train/lr_U1": optimizer_U1.param_groups[0]['lr'],
                "train/lr_U2": optimizer_U2.param_groups[0]['lr'],
            })

            # 진행 상황 출력
            progress_bar.set_postfix(loss_mask=f"{loss_mask.item():.4f}", loss_color=f"{loss_color.item():.4f}")

        # 에폭 단위 평균 손실 계산
        avg_loss_mask = total_loss_mask / count
        avg_loss_color = total_loss_color / count
        loss_sum = avg_loss_mask + avg_loss_color  # 두 손실의 합

        # wandb 로그 기록 (에폭 단위)
        wandb.log({
            "epoch": epoch,
            "metrics/avg_loss_mask": avg_loss_mask,
            "metrics/avg_loss_color": avg_loss_color,
            "metrics/loss_sum": loss_sum,
        })

        print(f"[Epoch {epoch}/{epochs}] Loss Sum: {loss_sum:.4f} (Mask: {avg_loss_mask:.4f}, Color: {avg_loss_color:.4f})")

        # 최적 모델 저장 (손실 합 최소화 기준)
        if loss_sum < best_loss_sum:
            best_loss_sum = loss_sum
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(unet1.state_dict(), os.path.join(model_save_dir, "best_unet1.pth"))
            torch.save(unet2.state_dict(), os.path.join(model_save_dir, "best_unet2.pth"))