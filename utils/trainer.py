import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import ssim_score, histogram_similarity, masked_ssim_score
import segmentation_models_pytorch as smp
import wandb
import numpy as np


def train(train_dataloader, valid_dataloader, epochs, unet1, unet2, optimizer_U1, optimizer_U2, scheduler_U1, scheduler_U2, criterion_pixelwise, val_every = 1, model_save_dir='./checkpoints'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    # 모델 초기화
    unet1.to(device)
    unet2.to(device)

    best_loss_sum = float("inf")  # 손실 합 최소화 기준

    for epoch in range(1, epochs + 1):
        # progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training: Epoch [{epoch}/{epochs}]")

        # total_loss_mask = 0
        # total_loss_color = 0
        # count = 0

        # for i, batch in progress_bar:
        #     unet1.train()
        #     unet2.train()
        #     optimizer_U1.zero_grad()
        #     optimizer_U2.zero_grad()

        #     # 데이터 불러오기
        #     grayscale_img = batch['grayscale'].to(device)  # 흑백 이미지 (input to U-Net1)
        #     color_gt = batch['color_gt'].to(device)        # 색상 이미지 (ground truth for U-Net2)
        #     mask_gt = batch['mask_gt'].to(device)          # 복원할 마스크 이미지 (ground truth for U-Net1)

        #     # U-Net1: 마스크 복원
        #     with torch.cuda.amp.autocast():
        #         mask_pred = unet1(grayscale_img)
        #         loss_mask = criterion_pixelwise(mask_pred, mask_gt)
            
        #     scaler.scale(loss_mask).backward()  # Mask 손실만 backward
        #     scaler.step(optimizer_U1)
        #     scaler.update()
        #     scheduler_U1.step()

        #     # U-Net2: 색상 복원
        #     with torch.cuda.amp.autocast():
        #         color_pred = unet2(mask_pred.detach())  # detach로 gradient 차단
        #         loss_color = criterion_pixelwise(color_pred, color_gt)
            
        #     scaler.scale(loss_color).backward()
        #     scaler.step(optimizer_U2)
        #     scaler.update()
        #     scheduler_U2.step()

        #     # 손실 누적
        #     total_loss_mask += loss_mask.item()
        #     total_loss_color += loss_color.item()
        #     count += 1

        #     # wandb 로그 기록 (배치 단위)
        #     try:
        #         wandb.log({
        #             "train/epoch": epoch,
        #             "train/batch": epoch * len(train_dataloader) + i,
        #             "train/loss_mask": loss_mask.item(),
        #             "train/loss_color": loss_color.item(),
        #             "train/lr_U1": optimizer_U1.param_groups[0]['lr'],
        #             "train/lr_U2": optimizer_U2.param_groups[0]['lr'],
        #         })
        #     except:
        #         pass

        #     # 진행 상황 출력
        #     progress_bar.set_postfix(loss_mask=f"{loss_mask.item():.4f}", loss_color=f"{loss_color.item():.4f}")

        # # 에폭 단위 평균 손실 계산
        # avg_loss_mask = total_loss_mask / count
        # avg_loss_color = total_loss_color / count
        # loss_sum = avg_loss_mask + avg_loss_color  # 두 손실의 합

        # # wandb 로그 기록 (에폭 단위)
        # try:
        #     wandb.log({
        #         "epoch": epoch,
        #         "metrics/mean_loss_mask": avg_loss_mask,
        #         "metrics/mean_loss_color": avg_loss_color,
        #         "metrics/loss_sum": loss_sum,
        #     })
        # except:
        #     pass

        # print(f"[Epoch {epoch}/{epochs}] Loss Sum: {loss_sum:.4f} (Mask: {avg_loss_mask:.4f}, Color: {avg_loss_color:.4f})")
        if (epoch + 1) % val_every == 0:
            validation(valid_dataloader, epoch, unet1, unet2, criterion_pixelwise, model_save_dir)

        # # 최적 모델 저장 (손실 합 최소화 기준)
        # if loss_sum < best_loss_sum:
        #     best_loss_sum = loss_sum
        #     os.makedirs(model_save_dir, exist_ok=True)
        #     torch.save(unet1.state_dict(), os.path.join(model_save_dir, "best_unet1.pth"))
        #     torch.save(unet2.state_dict(), os.path.join(model_save_dir, "best_unet2.pth"))

def validation(valid_dataloader, epoch, unet1, unet2, criterion_pixelwise, model_save_dir='./checkpoints'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet1.eval()
    unet2.eval()
    progress_bar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc=f"Validation: Epoch [{epoch}]")
    total_ssim = 0
    total_ssim_masked = 0
    total_ssim_color = 0
    count = 0

    for i, batch in progress_bar:
        grayscale_img = batch['grayscale'].to(device)  # 흑백 이미지 (input to U-Net1)
        color_gt = batch['color_gt'].to(device)        # 색상 이미지 (ground truth for U-Net2)
        mask_gt = batch['mask_gt'].to(device)          # 복원할 마스크 이미지 (ground truth for U-Net1)

        with torch.cuda.amp.autocast():
            mask_pred = unet1(grayscale_img)
            color_pred = unet2(mask_pred.detach())  # detach로 gradient 차단

        # SSIM 계산
        # 마스크 차원 확장
        # mask_gt_np = mask_gt.detach().cpu().numpy()[:, np.newaxis, :, :]  # (batch_size, 1, H, W)로 변환
        # mask_gt_np = np.repeat(mask_gt_np, repeats=color_gt.shape[1], axis=1)  # (batch_size, C, H, W)로 확장
        mask_gt_np = mask_gt.detach().squeeze().cpu().numpy().astype(np.float32)

        # SSIM 계산
        color_pred_np = color_pred.detach().squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        color_gt_np = color_gt.detach().squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
        total_ssim += ssim_score(color_pred_np, color_gt_np, channel_axis=-1) # 전체 ssim 계산
        total_ssim_masked += masked_ssim_score(color_pred_np, color_gt_np, mask_gt_np)
        total_ssim_color += histogram_similarity(color_pred_np, color_gt_np) # 색상 ssim 계산

        # 손실 누적
        count += 1
        if i == 10: break

    avg_ssim = total_ssim / count
    avg_ssim_masked = total_ssim_masked / count
    avg_ssim_color = total_ssim_color / count
    score = 0.2 * avg_ssim + 0.4 * avg_ssim_masked + 0.4 * avg_ssim_color
    print(f"Total SSIM: {avg_ssim}")
    print(f"Total Masked SSIM: {avg_ssim_masked}")
    print(f"Total Color SSIM: {avg_ssim_color}")
    print(f"Score: {score}")

    try:
        wandb.log({
            "valid/score": score,
            "valid/mean_total_ssim": avg_ssim,
            "valid/mean_masked_ssim": avg_ssim_masked,
            "valid/mean_ssim_color": avg_ssim_color,
        })
    except:
        pass
