from utils.model import LitIRModel
import zipfile
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from glob import glob
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.dataset import CustomImageDataset, CollateFn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import segmentation_models_pytorch as smp
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration')
    parser.add_argument('--model', type=str, default='./checkpoint/smp-unet-resnet34-epoch=00-val_score=0.4098.ckpt', help='Path for model')
    parser.add_argument('--test_data_dir', type=str, default='../../data/test_input', help='Test Data Directory 경로를 설정')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size 설정')
    parser.add_argument('--submission_dir', type=str, default='./submission', help='Submission directory 설정')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    test_df = pd.read_csv('../../data/test.csv')
    test_dataset = CustomImageDataset(test_df, data_dir=args.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=CollateFn(mode='test'))

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

    lit_ir_model = LitIRModel.load_from_checkpoint(
        args.model,
        model_1 = model_1,
        model_2 = model_2
    )

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
    predictions = trainer.predict(lit_ir_model, test_dataloader)
    predictions = np.concatenate(predictions)
    current_time = time.strftime("%m-%d_%H-%M-%S")
    submission_dir = os.path.join(args.submission_dir, current_time)
    submission_file = f'{args.submission_dir}/{current_time}.zip'
    os.makedirs(submission_dir, exist_ok=True)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # row['input_image_path']에서 경로 중복 방지
        input_image_name = os.path.basename(row['input_image_path'])  # 파일명만 추출
        save_path = os.path.join(args.test_data_dir, input_image_name)  # 올바른 경로 생성
        image_pred = Image.fromarray(predictions[idx])
        image_pred.save(save_path, "PNG")
        print(f"Image saved to {save_path}")

    # Step 3: Compress the directory into a ZIP file using glob
    with zipfile.ZipFile(submission_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in glob(f"{submission_dir}/*.png"):
            arcname = os.path.relpath(file_path, submission_dir)
            zipf.write(file_path, arcname)
    print('Submission saved successfully!')

if __name__ == '__main__':
    main()