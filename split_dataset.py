import os, argparse, pandas as pd, shutil
from sklearn.model_selection import train_test_split

def split(metadata_csv, images_dir, out_dir, test_size=0.15, val_size=0.15, seed=42):
    df = pd.read_csv(metadata_csv)
    # expected columns: image_id, dx (diagnosis)
    if 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError("metadata csv must contain 'image_id' and 'dx' columns")
    df['filename'] = df['image_id'].astype(str) + '.jpg'
    # train/test split
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['dx'], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), stratify=train_df['dx'], random_state=seed)
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in split_df.iterrows():
            label = row['dx']
            src = os.path.join(images_dir, row['filename'])
            dest_dir = os.path.join(out_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, row['filename']))
    print("Split done. Check", out_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--out', default='dataset')
    args = p.parse_args()
    split(args.metadata, args.images, args.out)
