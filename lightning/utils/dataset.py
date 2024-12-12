import random
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import skimage
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import skimage.draw

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
        img_path = self.df.iloc[idx, 0].split('/')[-1]
        img_path = os.path.join(self.data_dir, img_path)
        
        if self.mode == 'train':
            image = Image.open(img_path)
            image_input = get_stratified_input_image(
                image, 
                min_polygon_bbox_size=self.updated_min_size, 
                max_polygon_bbox_size=self.updated_max_size,
                max_points=self.updated_points
            )

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
        update_interval = int(num_epoch * self.pct)

        if epoch % update_interval != 0:
            return

        progress = epoch / num_epoch

        new_max_size = self.updated_max_size + 50
        
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
        bbox_x1 = random.randint(0, width-min_polygon_bbox_size)
        bbox_y1 = random.randint(0, height-min_polygon_bbox_size)
        bbox_x2 = random.randint(bbox_x1, width)
        bbox_y2 = random.randint(bbox_y1, height)
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
        polygon = polygon_func(num_points=num_points)
        polygon = [(round(r*mask_width), round(c*mask_height)) for r,c in polygon]
        polygon_mask = skimage.draw.polygon2mask((mask_width, mask_height), polygon)
        if np.sum(polygon_mask)>(min_polygon_bbox_size//2)**2:
            break
    full_image_mask = np.zeros((width, height), dtype=np.uint8)
    full_image_mask[bbox_x1:bbox_x2, bbox_y1:bbox_y2] = polygon_mask
    
    image_gray = image.convert('L')
    image_gray_array = np.array(image_gray)
    random_color = random.randint(0, 100)
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
            masks= []
            images_gray = []
            images_gray_masked = []
            images_gt = []
    
            for example in examples:
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