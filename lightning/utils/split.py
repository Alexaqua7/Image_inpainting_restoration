from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from utils.dataset import get_input_image, get_stratified_input_image
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def split_dataset(seed, n_split, train_df, TRAIN_DATA_DIR, VALID_DATA_DIR, MIN_POLYGON_BBOX_SIZE):
    os.makedirs(VALID_DATA_DIR, exist_ok=True)
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_path = train_df.iloc[idx, 0]
        img_path = os.path.join(TRAIN_DATA_DIR, os.path.basename(img_path))
        save_image_name = os.path.basename(img_path).replace('TRAIN', 'VALID').replace('png','npy')
        save_image_path = f'{VALID_DATA_DIR}/{save_image_name}'
        if os.path.exists(save_image_path):
            continue
        image = Image.open(img_path)
        valid_input_image = get_input_image(image, MIN_POLYGON_BBOX_SIZE)
        np.save(save_image_path, valid_input_image)
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    for fold_idx, (train_indices, valid_indices) in enumerate(kf.split(train_df['input_image_path'])):
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        valid_fold_df = train_df.iloc[valid_indices].reset_index(drop=True)
        valid_fold_df['input_image_path'] = valid_fold_df['input_image_path'].apply(lambda x: x.replace('TRAIN', 'VALID').replace('png', 'npy'))
        # valid_fold_df = valid_fold_df.drop_duplicates('label') # for fast validation
        # train_fold_df = pd.concat([train_fold_df,train_df_outlier],axis=0).reset_index(drop=True)
        break
    return train_fold_df, valid_fold_df

def stratified_split_dataset(seed, n_split, train_df, TRAIN_DATA_DIR, VALID_DATA_DIR, MIN_POLYGON_BBOX_SIZE, MAX_POLYGON_BBOX_SIZE):
    os.makedirs(VALID_DATA_DIR, exist_ok=True)
    groups = np.linspace(MIN_POLYGON_BBOX_SIZE, MAX_POLYGON_BBOX_SIZE, num=15, dtype=int)
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img_path = train_df.iloc[idx, 0]
        img_path = os.path.join(TRAIN_DATA_DIR, os.path.basename(img_path))
        
        save_image_name = os.path.basename(img_path).replace('TRAIN', 'VALID').replace('png', 'npy')
        save_image_path = f'{VALID_DATA_DIR}/{save_image_name}'
        
        # 이미 저장된 경우 건너뜀
        if os.path.exists(save_image_path):
            continue

        # 그룹별 고정된 mask 크기 적용
        image = Image.open(img_path)
        group_idx = idx % len(groups)  # 현재 그룹 인덱스
        group_min_size = groups[group_idx]
        group_max_size = groups[group_idx]
        
        valid_input_image = get_stratified_input_image(
            image,
            min_polygon_bbox_size=group_min_size,
            max_polygon_bbox_size=group_max_size
        )
        
        # 생성된 데이터를 저장
        np.save(save_image_path, valid_input_image)
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    for fold_idx, (train_indices, valid_indices) in enumerate(kf.split(train_df['input_image_path'])):
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        valid_fold_df = train_df.iloc[valid_indices].reset_index(drop=True)
        valid_fold_df['input_image_path'] = valid_fold_df['input_image_path'].apply(lambda x: x.replace('TRAIN', 'VALID').replace('png', 'npy'))
        # valid_fold_df = valid_fold_df.drop_duplicates('label') # for fast validation
        # train_fold_df = pd.concat([train_fold_df,train_df_outlier],axis=0).reset_index(drop=True)
        break
    return train_fold_df, valid_fold_df