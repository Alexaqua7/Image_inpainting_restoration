import os
import torch
from tqdm import tqdm
from model.UNetGenerator import weights_init_normal
import datetime
import wandb

def train(train_dataloader, epochs, generator, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, criterion_GAN, criterion_pixelwise, lambda_pixel, model_save_dir = './checkpoints'):
    # 학습
    best_loss = float("inf")
    generator, discriminator = generator.to(device), discriminator.to(device)
    generator.apply(weights_init_normal).to(device)
    discriminator.apply(weights_init_normal).to(device)
    for epoch in range(1, epochs + 1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training: Epoch [{epoch}/{epochs}]")
        for i, batch in progress_bar:
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

            # Discriminator 훈련
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()
            scheduler_D.step()

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