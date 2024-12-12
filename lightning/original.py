import argparse
import lightning as L
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from utils.dataset import CustomImageDataset, StratifiedImageDataset
from utils.model import LitIRModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.split import stratified_split_dataset, split_dataset
import wandb
import pandas as pd
from utils.dataset import CollateFn
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--n_split', type=int, default=5, help='kFold를 할 Fold 수 설정')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size 설정')
    parser.add_argument('--min_polygon_bbox_size', type=int, default=50, help='min_polygon_bbox_size를 설정')
    parser.add_argument('--max_polygon_bbox_size', type=int, default=250, help='max_polygon_bbox_size를 설정')
    parser.add_argument('--train_data_dir', type=str, default='../../data/train_gt', help='Train data 경로 설정')
    parser.add_argument('--train_df', type=str, default='../../data/train.csv', help='Train data 경로 설정')
    parser.add_argument('--val_data_dir', type=str, default=f'../../data/valid_input', help='Validation data 경로 설정')
    parser.add_argument('--image_mean', type=float, default=0.5, help='Image pixel의 mean value')
    parser.add_argument('--image_std', type=float, default=0.225, help='Image pixel의 std value')
    parser.add_argument('--num_epoch', type=int, default=20, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--resume', type=str, default='./checkpoint/smp-unetpp-maxvit_base_tf_512-efficientnetb7-epoch=05-val_score=0.6426.ckpt', help='모델 학습을 Resume하려면 Model 주소를 입력하세요')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='TEST(Last quarter_50_250)', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main():
    args = parse_args()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = stratified_split_dataset(args.seed, args.n_split, train_df, args.train_data_dir, args.val_data_dir, args.min_polygon_bbox_size, 250)
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        # resume='allow',
        # id='aoh5o9l9'
    )

    train_dataset = StratifiedImageDataset(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size, max_polygon_bbox_size=args.max_polygon_bbox_size, pct=0.1, max_points=16,)
    valid_dataset = StratifiedImageDataset(valid_fold_df, data_dir=args.val_data_dir, mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,collate_fn=CollateFn(mode='train'))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=4, pin_memory=True,collate_fn=CollateFn(mode='valid'), persistent_workers=True)

    # gray masked -> gray restoration
    model_1 = smp.UnetPlusPlus(
    encoder_name="tu-maxvit_base_tf_512",        
    encoder_weights="imagenet",     
    in_channels=1,                  
    classes=1,                      
)

    # gray -> color
    model_2 = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=3,                      
    )
    if args.resume == "":
        lit_ir_model = LitIRModel(model_1=model_1, model_2=model_2)
    else:
        lit_ir_model = LitIRModel.load_from_checkpoint(
            args.resume,
            model_1 = model_1,
            model_2 = model_2
        )

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'smp-unetpp-maxvit_base_tf_512-efficientnetb7'+'-{epoch:02d}-{val_score:.4f}',
    save_top_k=1,
    save_weights_only=False,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=20)

    trainer = L.Trainer(max_epochs=args.num_epoch, precision='bf16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)#, ckpt_path= args.resume if args.resume is not None else None)
    wandb.finish()
if __name__ == '__main__':
    main()
