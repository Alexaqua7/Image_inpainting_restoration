import argparse
import lightning as L
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.split import stratified_split_dataset, split_dataset
import wandb
import pandas as pd
from utils.dataset import CollateFn
from torchvision import transforms

# 1분기 데이터셋
from utils.dataset_1Q import StratifiedImageDataset as StratifiedImageDataset1Q
from utils.model_1Q import LitIRModel as LitIRModel1Q

# 2분기 데이터셋
from utils.dataset_2Q import StratifiedImageDataset as StratifiedImageDataset2Q
from utils.model_2Q import LitIRModel as LitIRModel2Q

# 3분기 데이터셋
from utils.dataset_3Q import StratifiedImageDataset as StratifiedImageDataset3Q
from utils.model_3Q import LitIRModel as LitIRModel3Q

# 4분기 데이터셋
from utils.dataset_4Q import StratifiedImageDataset as StratifiedImageDataset4Q
from utils.model_4Q import LitIRModel as LitIRModel4Q

def parse_args1():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--n_split', type=int, default=5, help='kFold를 할 Fold 수 설정')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size 설정')
    parser.add_argument('--min_polygon_bbox_size', type=int, default=50, help='min_polygon_bbox_size를 설정')
    parser.add_argument('--max_polygon_bbox_size', type=int, default=300, help='max_polygon_bbox_size를 설정')
    parser.add_argument('--train_data_dir', type=str, default='../../data/train_gt', help='Train data 경로 설정')
    parser.add_argument('--train_df', type=str, default='../../data/train.csv', help='Train data 경로 설정')
    parser.add_argument('--val_data_dir', type=str, default=f'../../data/valid_input', help='Validation data 경로 설정')
    parser.add_argument('--image_mean', type=float, default=0.5, help='Image pixel의 mean value')
    parser.add_argument('--image_std', type=float, default=0.225, help='Image pixel의 std value')
    parser.add_argument('--num_epoch', type=int, default=8, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--resume', type=str, default='', help='모델 학습을 Resume하려면 Model 주소를 입력하세요')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='CURRICULUM_LOCAL', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main1():
    args = parse_args1()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = stratified_split_dataset(args.seed, args.n_split, train_df, args.train_data_dir, args.val_data_dir, 50, 300)
    # Wandb initalize
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
    )

    train_dataset = StratifiedImageDataset1Q(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size, max_polygon_bbox_size=args.max_polygon_bbox_size, max_points=16,)
    valid_dataset = StratifiedImageDataset1Q(valid_fold_df, data_dir=args.val_data_dir, mode='valid')

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
        lit_ir_model = LitIRModel1Q(model_1=model_1, model_2=model_2)
    else:
        lit_ir_model = LitIRModel1Q.load_from_checkpoint(
            args.resume,
            model_1 = model_1,
            model_2 = model_2
        )

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'curriculum1Q',
    save_top_k=1,
    save_weights_only=False,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=20)

    trainer = L.Trainer(max_epochs=args.num_epoch, precision='bf16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)#, ckpt_path= args.resume if args.resume is not None else None)

def parse_args2():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--n_split', type=int, default=5, help='kFold를 할 Fold 수 설정')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size 설정')
    parser.add_argument('--min_polygon_bbox_size', type=int, default=150, help='min_polygon_bbox_size를 설정')
    parser.add_argument('--max_polygon_bbox_size', type=int, default=250, help='max_polygon_bbox_size를 설정')
    parser.add_argument('--train_data_dir', type=str, default='../../data/train_gt', help='Train data 경로 설정')
    parser.add_argument('--train_df', type=str, default='../../data/train.csv', help='Train data 경로 설정')
    parser.add_argument('--val_data_dir', type=str, default=f'../../data/valid_input', help='Validation data 경로 설정')
    parser.add_argument('--image_mean', type=float, default=0.5, help='Image pixel의 mean value')
    parser.add_argument('--image_std', type=float, default=0.225, help='Image pixel의 std value')
    parser.add_argument('--num_epoch', type=int, default=7, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--resume', type=str, default='./checkpoint/curriculum1Q.ckpt', help='모델 학습을 Resume하려면 Model 주소를 입력하세요')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='CURRICULUM', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main2():
    args = parse_args2()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = stratified_split_dataset(args.seed, args.n_split, train_df, args.train_data_dir, args.val_data_dir, 50, 300)

    train_dataset = StratifiedImageDataset2Q(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size, max_polygon_bbox_size=args.max_polygon_bbox_size, max_points=16,)
    valid_dataset = StratifiedImageDataset2Q(valid_fold_df, data_dir=args.val_data_dir, mode='valid')

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
        lit_ir_model = LitIRModel2Q(model_1=model_1, model_2=model_2)
    else:
        lit_ir_model = LitIRModel2Q.load_from_checkpoint(
            args.resume,
            model_1 = model_1,
            model_2 = model_2
        )

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'curriculum2Q',
    save_top_k=1,
    save_weights_only=False,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=20)

    trainer = L.Trainer(max_epochs=args.num_epoch, precision='bf16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)#, ckpt_path= args.resume if args.resume is not None else None)

def parse_args3():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--n_split', type=int, default=5, help='kFold를 할 Fold 수 설정')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size 설정')
    parser.add_argument('--min_polygon_bbox_size', type=int, default=150, help='min_polygon_bbox_size를 설정')
    parser.add_argument('--max_polygon_bbox_size', type=int, default=250, help='max_polygon_bbox_size를 설정')
    parser.add_argument('--train_data_dir', type=str, default='../../data/train_gt', help='Train data 경로 설정')
    parser.add_argument('--train_df', type=str, default='../../data/train.csv', help='Train data 경로 설정')
    parser.add_argument('--val_data_dir', type=str, default=f'../../data/valid_input', help='Validation data 경로 설정')
    parser.add_argument('--image_mean', type=float, default=0.5, help='Image pixel의 mean value')
    parser.add_argument('--image_std', type=float, default=0.225, help='Image pixel의 std value')
    parser.add_argument('--num_epoch', type=int, default=6, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--resume', type=str, default='./checkpoint/curriculum2Q.ckpt', help='모델 학습을 Resume하려면 Model 주소를 입력하세요')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='CURRICULUM', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main3():
    args = parse_args3()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = stratified_split_dataset(args.seed, args.n_split, train_df, args.train_data_dir, args.val_data_dir, 50, 300)

    train_dataset = StratifiedImageDataset3Q(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size, max_polygon_bbox_size=args.max_polygon_bbox_size, max_points=16,)
    valid_dataset = StratifiedImageDataset3Q(valid_fold_df, data_dir=args.val_data_dir, mode='valid')

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
        lit_ir_model = LitIRModel3Q(model_1=model_1, model_2=model_2)
    else:
        lit_ir_model = LitIRModel3Q.load_from_checkpoint(
            args.resume,
            model_1 = model_1,
            model_2 = model_2
        )

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'curriculum3Q',
    save_top_k=1,
    save_weights_only=False,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=20)

    trainer = L.Trainer(max_epochs=args.num_epoch, precision='bf16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)#, ckpt_path= args.resume if args.resume is not None else None)

def parse_args4():
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
    parser.add_argument('--num_epoch', type=int, default=15, help='Epoch 수 설정')
    parser.add_argument('--seed', type=int, default=42, help='Seed 설정')
    parser.add_argument('--resume', type=str, default='./checkpoint/curriculum3Q.ckpt', help='모델 학습을 Resume하려면 Model 주소를 입력하세요')
    parser.add_argument('--wandb_project', type=str, default='Image-Inpainting', help='WandB Project 이름')
    parser.add_argument('--wandb_entity', type=str, default='alexseo-inha-university', help='WandB Entity 이름')
    parser.add_argument('--wandb_run_name', type=str, default='CURRICULUM', help='WandB Run name 설정')
    parser = parser.parse_args()
    
    return parser

def main4():
    args = parse_args4()
    L.seed_everything(args.seed)
    train_df = pd.read_csv(args.train_df)
    train_fold_df, valid_fold_df = stratified_split_dataset(args.seed, args.n_split, train_df, args.train_data_dir, args.val_data_dir, 50, 300)

    train_dataset = StratifiedImageDataset4Q(train_fold_df, data_dir=args.train_data_dir, mode='train',min_polygon_bbox_size=args.min_polygon_bbox_size, max_polygon_bbox_size=args.max_polygon_bbox_size, max_points=16,)
    valid_dataset = StratifiedImageDataset4Q(valid_fold_df, data_dir=args.val_data_dir, mode='valid')

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
        lit_ir_model = LitIRModel4Q(model_1=model_1, model_2=model_2)
    else:
        lit_ir_model = LitIRModel4Q.load_from_checkpoint(
            args.resume,
            model_1 = model_1,
            model_2 = model_2
        )

    checkpoint_callback = ModelCheckpoint(
    monitor='val_score',
    mode='max',
    dirpath='./checkpoint/',
    filename=f'curriculum4Q',
    save_top_k=1,
    save_weights_only=False,
    verbose=True
)
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=20)

    trainer = L.Trainer(max_epochs=args.num_epoch, precision='bf16-mixed', callbacks=[checkpoint_callback, earlystopping_callback], detect_anomaly=False)

    trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)#, ckpt_path= args.resume if args.resume is not None else None)
    wandb.finish()
if __name__ == '__main__':
    main1()
    main2()
    main3()
    main4()