import os
from PIL import Image
import argparse
from model.UNetGenerator import UNetGenerator
import torch
import cv2
import zipfile
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# 이미지 로드 및 전처리
def load_image(image_path, transform):
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원을 추가합니다.
    return image

def test(model1, model2, transform, test_dir, submission_dir):
    device = 'cuda'
    # 모델 로드 및 설정
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()
    # 파일 리스트 불러오기
    test_images = sorted(os.listdir(test_dir))

    # 모든 테스트 이미지에 대해 추론 수행
    for image_name in test_images:
        test_image_path = os.path.join(test_dir, image_name)

        # 손상된 테스트 이미지 로드 및 전처리
        test_image = load_image(test_image_path, transform).to(device)

        with torch.no_grad():
            # 모델로 예측
            pred_image = model1(test_image)
            pred_image = model2(pred_image)
            pred_image = pred_image.cpu().squeeze(0)  # 배치 차원 제거
            pred_image = pred_image * 0.5 + 0.5  # 역정규화
            pred_image = pred_image.numpy().transpose(1, 2, 0)  # HWC로 변경
            pred_image = (pred_image * 255).astype('uint8')  # 0-255 범위로 변환
            
            # 예측된 이미지를 실제 이미지와 같은 512x512로 리사이즈
            pred_image_resized = cv2.resize(pred_image, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 결과 이미지 저장
        output_path = os.path.join(submission_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(pred_image_resized, cv2.COLOR_RGB2BGR))    
        
    print(f"Saved all images")

    zip_filename = "submission.zip"
    with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
        for image_name in test_images:
            image_path = os.path.join(submission_dir, image_name)
            submission_zip.write(image_path, arcname=image_name)

    print(f"All images saved in {zip_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description='Image_Inpainting_Restoration_Inference')
    parser.add_argument('--test_dir', type=str, default='../data/test_input', help='Test Data의 경로')
    parser.add_argument('--submission_dir', type=str, default='./submission', help='Submission file이 저장될 경로')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Model이 저장될 경로')
    args = parser.parse_args()
    return args

def main():
    # 모델 경로 설정
    args = parse_args()
    generator_path = os.path.join(args.model_save_dir, "best_generator.pth")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    model1 = smp.UnetPlusPlus(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=1,                  
    classes=1,                      
    )

# gray -> color
    model2 = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=3,                      
    )
    model1.load_state_dict(torch.load('./saved_models/best_unet1.pth'))
    model2.load_state_dict(torch.load('./saved_models/best_unet2.pth'))
    os.makedirs(args.submission_dir, exist_ok=True)
    test(model1, model2, transform, args.test_dir, args.submission_dir)

if __name__ == '__main__':
    main()