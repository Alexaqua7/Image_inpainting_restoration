import torch
from utils.metric import get_ssim_score, get_masked_ssim_score, get_histogram_similarity
import cv2
import torch.nn.functional as F
import wandb
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR

class LitIRModel(L.LightningModule):
    def __init__(self, model_1, model_2, image_mean=0.5, image_std=0.225):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.image_mean=image_mean
        self.image_std=image_std

        # Validation metrics storage
        self.val_metrics = {"val_score": 0, "val_total_ssim_score": 0, "val_masked_ssim_score": 0, "val_hist_sim_score": 0}
        self.val_count = 0  # Batch count for averaging

    def forward(self, images_gray_masked):
        images_gray_restored = self.model_1(images_gray_masked)+images_gray_masked
        images_restored = self.model_2(images_gray_restored)
        return images_gray_restored, images_restored
        
    def unnormalize(self, output, round=False):
        image_restored = ((output*self.image_std+self.image_mean)*255).clamp(0,255)
        if round:
            image_restored = torch.round(image_restored)
        return image_restored
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        total_epochs = 30  # 기본값 100

        # 2. CosineAnnealing 스케줄러 정의
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_score',  # (Optional) 성능 지표를 모니터링하려면 추가
            'interval': 'epoch',    # 학습률 업데이트 주기 (에포크마다)
        }
    }

    def training_step(self, batch, batch_idx):
        # 데이터셋 업데이트 로직 (현재 epoch과 총 epoch 전달)
        if hasattr(self.trainer.datamodule, 'update_bbox_size'):
            self.trainer.datamodule.update_bbox_size(self.current_epoch)
        masks, images_gray_masked, images_gray, images_gt = batch['masks'], batch['images_gray_masked'], batch['images_gray'], batch['images_gt']
        images_gray_restored, images_restored = self(images_gray_masked)
        
        loss_pixel_gray = F.l1_loss(images_gray, images_gray_restored, reduction='mean') * 0.5 + F.mse_loss(images_gray, images_gray_restored, reduction='mean') * 0.5
        loss_pixel = F.l1_loss(images_gt, images_restored, reduction='mean') * 0.5 + F.mse_loss(images_gt, images_restored, reduction='mean') * 0.5
        loss = loss_pixel_gray * 0.5 + loss_pixel * 0.5

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_loss_pixel_gray", loss_pixel_gray, on_step=True, on_epoch=False)
        self.log("train_loss_pixel", loss_pixel, on_step=True, on_epoch=False)
        try:
            current_lr = self.optimizers().param_groups[0]['lr']
            wandb.log({
                        "train/loss": loss,
                        "train/loss_pixel_gray": loss_pixel_gray,
                        "train/loss_pixel": loss_pixel,
                        "train/step":self.global_step,
                        "train/epoch": self.current_epoch,
                        "train/lr":current_lr,
                    })
        except:
            pass
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images_gray_masked, images_gt = batch['masks'], batch['images_gray_masked'], batch['images_gt']
        images_gray_restored, images_restored = self(images_gray_masked)
        images_gt, images_restored = self.unnormalize(images_gt, round=True), self.unnormalize(images_restored, round=True)
        masks_np = masks.detach().cpu().numpy()
        images_gt_np = images_gt.detach().cpu().permute(0,2,3,1).float().numpy().astype(np.uint8)
        images_restored_np = images_restored.detach().cpu().permute(0,2,3,1).float().numpy().astype(np.uint8)
        total_ssim_score = 0
        masked_ssim_score = 0
        hist_sim_score = 0
        for image_gt_np, image_restored_np, mask_np in zip(images_gt_np, images_restored_np, masks_np):
            total_ssim_score += get_ssim_score(image_gt_np, image_restored_np) / len(images_gt)
            masked_ssim_score += get_masked_ssim_score(image_gt_np, image_restored_np, mask_np)/ len(images_gt)
            hist_sim_score += get_histogram_similarity(image_gt_np, image_restored_np, cv2.COLOR_RGB2HSV)/ len(images_gt)
        score = total_ssim_score * 0.2 + masked_ssim_score * 0.4 + hist_sim_score * 0.4
        self.log(f"val_score", score, on_step=False, on_epoch=True)
        self.log(f"val_total_ssim_score", total_ssim_score, on_step=False, on_epoch=True)
        self.log(f"val_masked_ssim_score", masked_ssim_score, on_step=False, on_epoch=True)
        self.log(f"val_hist_sim_score", hist_sim_score, on_step=False, on_epoch=True)

        # Batch-level 결과를 저장
        self.val_metrics["val_score"] += score
        self.val_metrics["val_total_ssim_score"] += total_ssim_score
        self.val_metrics["val_masked_ssim_score"] += masked_ssim_score
        self.val_metrics["val_hist_sim_score"] += hist_sim_score
        self.val_count += 1  # 배치 수 증가

        return score

    def predict_step(self, batch, batch_idx):
        images_gray_masked = batch['images_gray_masked']
        images_gray_restored, images_restored = self(images_gray_masked)
        images_restored = self.unnormalize(images_restored, round=True)
        images_restored_np = images_restored.detach().cpu().permute(0,2,3,1).float().numpy().astype(np.uint8)
        return images_restored_np
    
    def on_validation_epoch_end(self):
        # Validation 결과를 평균 내고 wandb에 기록
        avg_val_metrics = {key: value / self.val_count for key, value in self.val_metrics.items()}

        self.log("val_score", avg_val_metrics["val_score"], on_epoch=True)
        self.log("val_total_ssim_score", avg_val_metrics["val_total_ssim_score"], on_epoch=True)
        self.log("val_masked_ssim_score", avg_val_metrics["val_masked_ssim_score"], on_epoch=True)
        self.log("val_hist_sim_score", avg_val_metrics["val_hist_sim_score"], on_epoch=True)

        try:
            wandb.log({
                "valid/score": avg_val_metrics["val_score"],
                "valid/total_ssim_score": avg_val_metrics["val_total_ssim_score"],
                "valid/masked_ssim_score": avg_val_metrics["val_masked_ssim_score"],
                "valid/hist_sim_score": avg_val_metrics["val_hist_sim_score"],
                "valid/epoch": self.current_epoch,
            })
        except Exception as e:
            print(f"Error logging to wandb: {e}")

        # Metrics 초기화
        self.val_metrics = {key: 0 for key in self.val_metrics}
        self.val_count = 0