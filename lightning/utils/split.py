from sklearn.model_selection import KFold

def split_dataset(seed, n_split, train_df):
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    for fold_idx, (train_indices, valid_indices) in enumerate(kf.split(train_df['input_image_path'])):
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        valid_fold_df = train_df.iloc[valid_indices].reset_index(drop=True)
        valid_fold_df['input_image_path'] = valid_fold_df['input_image_path'].apply(lambda x: x.replace('TRAIN', 'VALID').replace('png', 'npy'))
        # valid_fold_df = valid_fold_df.drop_duplicates('label') # for fast validation
        # train_fold_df = pd.concat([train_fold_df,train_df_outlier],axis=0).reset_index(drop=True)
        break
    return train_fold_df, valid_fold_df