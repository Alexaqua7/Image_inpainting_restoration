import random
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import skimage
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from skimage.draw import disk
import torch
import skimage.draw

# 원을 그리는 함수
def random_circle(radius, center=None):
    """
    원을 생성하는 함수입니다.
    :param radius: 원의 반지름
    :param center: 원의 중심 좌표 (없으면 이미지 중앙에 원을 생성)
    :return: 원형 mask (numpy array)
    """
    height, width = 256, 256  # 예시로 256x256 크기의 이미지를 사용
    if center is None:
        center = (width // 2, height // 2)  # 이미지의 중앙에 원을 생성
    
    # 원을 그리기
    rr, cc = disk(center, radius, shape=(height, width))
    
    # 원형 mask 생성
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    
    return mask

class CustomImageDataset(Dataset):
    def __init__(self, df, data_dir='../../data/train_gt', mode='train', min_polygon_bbox_size=50):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.min_polygon_bbox_size = min_polygon_bbox_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.df.iloc[idx, 0].split('/')[-1]  # Assuming first column is the path
        img_path = os.path.join(self.data_dir, img_path)
        
        # Apply augmentation if in training mode
        if self.mode == 'train':
            image = Image.open(img_path)
            image_input = get_input_image(image, self.min_polygon_bbox_size)
            return image_input

        elif self.mode == 'valid':
            image_input = self.load_input_image(img_path)
            return image_input
        elif self.mode == 'test':
            image = Image.open(img_path).convert('L')
            return {
                'image_gray_masked':image
            }

    def load_input_image(self, img_input_path):
        image_input = np.load(img_input_path, allow_pickle=True)
        return image_input.item()

def get_input_image(image, min_polygon_bbox_size=50):
    '''
    이미지를 로드하는 함수입니다
    Return
    image_gt: 컬러로 복원된 ground truth
    mask: mask가 있는 컬러 이미지
    image_gray: 흑백의 ground truth 이미지
    image_gray_masked: mask가 있는 흑백 이미지
    '''
    width, height = image.size
    while True:
        bbox_x1 = random.randint(0, width-min_polygon_bbox_size)
        bbox_y1 = random.randint(0, height-min_polygon_bbox_size)
        bbox_x2 = random.randint(bbox_x1, width)  # Ensure width > 10
        bbox_y2 = random.randint(bbox_y1, height)  # Ensure height > 10
        if (bbox_x2-bbox_x1)<min_polygon_bbox_size or (bbox_y2-bbox_y1)<min_polygon_bbox_size:
            continue
        
        mask_bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        mask_width = bbox_x2-bbox_x1
        mask_height = bbox_y2-bbox_y1
    
        num_points = random.randint(3,20)
        polygon_func = random.choice([
            random_polygon,
            random_star_shaped_polygon,
            random_convex_polygon
        ])
        polygon = polygon_func(num_points=num_points) #scaled 0~1
        polygon = [(round(r*mask_width), round(c*mask_height)) for r,c in polygon]
        polygon_mask = skimage.draw.polygon2mask((mask_width, mask_height), polygon)
        if np.sum(polygon_mask)>(min_polygon_bbox_size//2)**2:
            break
    full_image_mask = np.zeros((width, height), dtype=np.uint8)
    full_image_mask[bbox_x1:bbox_x2, bbox_y1:bbox_y2] = polygon_mask
    
    image_gray = image.convert('L')
    image_gray_array = np.array(image_gray)  # Convert to numpy array for manipulation
    random_color = random.randint(0, 100)  # Random grayscale color
    image_gray_array[full_image_mask == 1] = random_color
    image_gray_masked = Image.fromarray(image_gray_array)

    return {
        'image_gt':image,
        'mask':full_image_mask,
        'image_gray':image_gray,
        'image_gray_masked':image_gray_masked
    }

class StratifiedImageDataset(Dataset):
    def __init__(self, df, data_dir='../../data/train_gt', mode='train', min_polygon_bbox_size=50, max_polygon_bbox_size=100, pct=0.1, max_points=20, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.min_polygon_bbox_size = min_polygon_bbox_size
        self.max_polygon_bbox_size = max_polygon_bbox_size
        self.updated_min_size = min_polygon_bbox_size
        self.updated_max_size = min_polygon_bbox_size
        self.pct = pct
        self.max_points = max_points
        self.updated_points = max_points
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.df.iloc[idx, 0].split('/')[-1]  # Assuming first column is the path
        img_path = os.path.join(self.data_dir, img_path)
        
        # Apply augmentation if in training mode
        if self.mode == 'train':
            image = Image.open(img_path)
            image_input = get_stratified_input_image(
                image, 
                min_polygon_bbox_size=self.updated_min_size, 
                max_polygon_bbox_size=self.updated_max_size,
                max_points=self.updated_points
            )

            # Apply transform to image_gray_masked if available
            if self.transform:
                image_gray_masked = image_input['image_gray_masked']
                image_gray_masked = np.array(image_gray_masked)
                image_gray_masked = np.expand_dims(image_gray_masked, axis=-1)  # Add channel dimension
                transformed = self.transform(image=image_gray_masked)
                image_gray_masked = transformed["image"]
                image_gray_masked = np.squeeze(image_gray_masked, axis=-1)  # Remove channel dimension

                # # Save the transformed image to check if augmentation is working
                # save_path = 'augmented_images'
                # os.makedirs(save_path, exist_ok=True)
                # save_name = f"augmented_{idx}.png"
                # transformed_image = Image.fromarray(image_gray_masked.astype(np.uint8))
                # transformed_image.save(os.path.join(save_path, save_name))
            
                return {
                    'image_gt': image_input['image_gt'],
                    'mask': image_input['mask'],
                    'image_gray': image_input['image_gray'],
                    'image_gray_masked': Image.fromarray(image_gray_masked),
                }

            return image_input

        elif self.mode == 'valid':
            image_input = self.load_input_image(img_path)
            return image_input
        elif self.mode == 'test':
            image = Image.open(img_path).convert('L')
            return {
                'image_gray_masked':image
            }

    def load_input_image(self, img_input_path):
        image_input = np.load(img_input_path, allow_pickle=True)
        return image_input.item()
    
    def update_bbox_size(self, epoch, num_epoch):
    # pct에 따른 에포크 간격 계산
        update_interval = int(num_epoch * self.pct)  # 예: pct=0.1, num_epoch=50 -> update_interval=5

        # update_interval 간격에 해당하는 에포크에서만 업데이트 수행
        if epoch % update_interval != 0:
            return  # 조건에 맞지 않으면 업데이트 건너뜀

        progress = epoch / num_epoch  # 전체 진행률 (0.0 ~ 1.0)
        # range_diff = self.max_polygon_bbox_size - self.min_polygon_bbox_size

        # 새로운 min/max 크기 계산
        # new_min_size = self.min_polygon_bbox_size + int(range_diff * progress / 2)
        new_max_size = self.updated_max_size + 50 #int(range_diff * progress)
        
        # 크기 업데이트
        # self.updated_min_size = max(self.min_polygon_bbox_size, min(new_min_size, self.max_polygon_bbox_size))
        self.updated_max_size = min(new_max_size, self.max_polygon_bbox_size)
        self.updated_points = min(self.max_points, max(3, int(self.max_points*progress)))


def get_stratified_input_image(image, min_polygon_bbox_size=50, max_polygon_bbox_size=100, min_points=3, max_points=20):
    '''
    이미지를 로드하는 함수입니다
    Return
    image_gt: 컬러로 복원된 ground truth
    mask: mask가 있는 컬러 이미지
    image_gray: 흑백의 ground truth 이미지
    image_gray_masked: mask가 있는 흑백 이미지
    '''
    width, height = image.size
    while True:
        bbox_x1 = random.randint(0, width-min_polygon_bbox_size) # x1의 좌표가 0 미만이 되지 않도록 조정
        bbox_y1 = random.randint(0, height-min_polygon_bbox_size) # y1의 좌표가 0 미만이 되지 않도록 조정
        bbox_x2 = random.randint(bbox_x1, width)  # Ensure width > 10
        bbox_y2 = random.randint(bbox_y1, height)  # Ensure height > 10
        if (bbox_x2-bbox_x1)<min_polygon_bbox_size or (bbox_y2-bbox_y1)<min_polygon_bbox_size or (bbox_x2-bbox_x1)>max_polygon_bbox_size or (bbox_y2-bbox_y1)>max_polygon_bbox_size:
            continue
        
        mask_bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        mask_width = bbox_x2-bbox_x1
        mask_height = bbox_y2-bbox_y1
    
        num_points = random.randint(min_points, max_points)
        if num_points < 5:
            polygon_func = random.choice([
            random_polygon])
        else:
            polygon_func = random.choice([
                random_polygon,
                random_star_shaped_polygon,
                random_convex_polygon,
            ])
        polygon = polygon_func(num_points=num_points) #scaled 0~1
        polygon = [(round(r*mask_width), round(c*mask_height)) for r,c in polygon]
        polygon_mask = skimage.draw.polygon2mask((mask_width, mask_height), polygon)
        if np.sum(polygon_mask)>(min_polygon_bbox_size//2)**2:
            break
    full_image_mask = np.zeros((width, height), dtype=np.uint8)
    full_image_mask[bbox_x1:bbox_x2, bbox_y1:bbox_y2] = polygon_mask
    
    image_gray = image.convert('L')
    image_gray_array = np.array(image_gray)  # Convert to numpy array for manipulation
    random_color = random.randint(0, 100)  # Random grayscale color
    image_gray_array[full_image_mask == 1] = random_color
    image_gray_masked = Image.fromarray(image_gray_array)

    return {
        'image_gt':image,
        'mask':full_image_mask,
        'image_gray':image_gray,
        'image_gray_masked':image_gray_masked
    }


class CollateFn:
    def __init__(self, mean=0.5, std=0.225, mode='train'):
        self.mode = mode
        self.mean = mean
        self.std = std

    def __call__(self, examples):
        if self.mode =='train' or self.mode=='valid':
            # Initialize lists to store each component of the batch
            masks= []
            images_gray = []
            images_gray_masked = []
            images_gt = []
    
            for example in examples:
                # Assuming each example is a dictionary with keys 'mask', 'image_gray', 'image_gray_masked', 'image_gt'
                masks.append(example['mask'])
                images_gray.append(self.normalize_image(example['image_gray']))
                images_gray_masked.append(self.normalize_image(example['image_gray_masked']))
                images_gt.append(self.normalize_image(np.array(example['image_gt'])))

            return {
                'masks': torch.from_numpy(np.stack(masks)).long(),
                'images_gray': torch.from_numpy(np.stack(images_gray)).unsqueeze(1).float(),
                'images_gray_masked': torch.from_numpy(np.stack(images_gray_masked)).unsqueeze(1).float(),
                'images_gt': torch.from_numpy(np.stack(images_gt)).permute(0,3,1,2).float()
            }

        elif self.mode == 'test':
            images_gray_masked = []
            for example in examples:
                images_gray_masked.append(self.normalize_image(example['image_gray_masked']))
            return {
                'images_gray_masked': torch.from_numpy(np.stack(images_gray_masked)).unsqueeze(1).float(),

            }
        
    def normalize_image(self, image):
        return (np.array(image)/255-self.mean)/self.std