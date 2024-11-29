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
            generator, discriminator = generator.to(device), discriminator.to(device)
            generator.apply(weights_init_normal).to(device)
            discriminator.apply(weights_init_normal).to(device)
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Generator 훈련
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
            scheduler_G.step()

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
            progress_bar.set_postfix(loss_D=f"{loss_D.item():.4f}", loss_G=f"{loss_G.item():.4f}", time=datetime.datetime.now().strftime("%H:%M:%S"))
            # print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_dataloader)}] [D loss: {loss_D.item()}] [G loss: {loss_G.item()}]")

            try:
                current_lr_G = optimizer_G.param_groups[0]['lr']
                current_lr_D = optimizer_D.param_groups[0]['lr']
                wandb.log({
                    "train/loss_D": loss_D.item(),
                    "train/loss_G": loss_G.item(),
                    "train/step": epoch * len(train_dataloader) + i,
                    "train/epoch": epoch,
                    "train/lr_D":current_lr_D,
                    "train/lr_G":current_lr_G,
                })
            except:
                pass
            # 현재 에포크에서의 손실이 best_loss보다 작으면 모델 저장
            if loss_G.item() < best_loss:
                best_loss = loss_G.item()
                torch.save(generator.state_dict(), os.path.join(model_save_dir, "best_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join(model_save_dir, "best_discriminator.pth"))
                print(f"Best model saved at epoch {epoch}, batch {i} with G loss: {loss_G.item()} and D loss: {loss_D.item()}")