import argparse
import lightning as L
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from utils.dataset import CustomImageDataset
from utils.model import LitIRModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.split import split_dataset
import wandb
import pandas as pd
from utils.dataset import CollateFn

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--n_split', type=int, default=5, help='kFold를 할 Fold 수 설정')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size 설정')
    parser.add_argument('--min_polygon_bbox_size', type=int, default=50, help='min_polygon_bbox_size를 설정')
    parser.add_argument('--train_data_dir', type=str, default='../../data/train_gt', help='Train data 경로 설정')
    parser.add_argument('--train_df', type=str, default='../../data/train.csv', help='Train data 경로 설정')
    parser.add_argument('--val_data_dir', type=str, default=f'../../data/valid_input', help='Validation data 경로 설정')
    parser.add_argument('--image_mean', type=float, default=0.5, help='Image pixel의 mean value')
    parser.add_argument('--image_std', type=float, default=0.225, help='Image pixel의 std value')
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='UPerNet_resnet34', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main():
    args = parse_args()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = split_dataset(args.seed, args.n_split, train_df)
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args)
    )

    train_dataset = CustomImageDataset(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size)
    valid_dataset = CustomImageDataset(valid_fold_df, data_dir=args.val_data_dir, mode='valid', min_polygon_bbox_size=args.min_polygon_bbox_size)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=CollateFn(mean=args.image_mean, std=args.image_std, mode='train'), num_workers=8, pin_memory=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size*2, shuffle=False, collate_fn=CollateFn(mean=args.image_mean, std=args.image_std, mode='valid'), num_workers=8, pin_memory=True, persistent_workers=True)

    model_1 = smp.UPerNet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=1,                  
    classes=1,                      
)

    # gray -> color
    model_2 = smp.UPerNet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=3,                      
    )

    lit_ir_model = LitIRModel(model_1=model_1, model_2=model_2)

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'smp-unet-resnet34'+'-{epoch:02d}-{val_score:.4f}',
    save_top_k=1,
    save_weights_only=True,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=3)

    trainer = L.Trainer(max_epochs=100, precision='16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)
    wandb.finish()
if __name__ == '__main__':
    main()