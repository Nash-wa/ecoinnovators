import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SolarDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train', quality_filter=None, task='classification'):
        """
        Args:
            csv_file (str): Path to master_metadata.csv
            root_dir (str): Project root (to combine with relative paths in CSV)
            mode (str): 'train' or 'val'. Applies augmentations only in 'train'.
            quality_filter (str): 'high' (for Segmentation) or None (for Classification).
            task (str): 'classification' returns (img, label). 'segmentation' returns (img, mask).
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode
        self.task = task
        
        # --- FILTERING LOGIC ---
        # 1. Filter by Split (Train vs Test/Val)
        # Assuming your CSV has a 'split' column. If not, we use everything or split manually.
        if 'split' in self.df.columns:
            if mode == 'train':
                self.df = self.df[self.df['split'] == 'train']
            else:
                self.df = self.df[self.df['split'] == 'test'] # or 'val'

        # 2. Filter by Quality (Crucial for U-Net)
        if quality_filter == 'high':
            self.df = self.df[self.df['quality'] == 'high']
            
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Define Augmentations
        self.transforms = self.get_transforms()


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img_path = row['image_path'].replace("\\", "/") 

        # Combine with root_dir if needed
        full_img_path = os.path.join(self.root_dir, img_path) if not os.path.isabs(img_path) else img_path
        
        image = cv2.imread(full_img_path)
        if image is None:
            # Print the path that failed so we can debug easier
            raise FileNotFoundError(f"Image not found: {full_img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load Mask 
        mask_path = row['mask_path'].replace("\\", "/")
        
        full_mask_path = os.path.join(self.root_dir, mask_path) if not os.path.isabs(mask_path) else mask_path
        
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Create dummy mask if missing 
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Binarize mask (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Aumentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Return based on Task
        if self.task == 'classification':
            label = int(row['has_solar'])
            return image, torch.tensor(label, dtype=torch.float32)
        
        elif self.task == 'segmentation':
            # U-Net expects mask shape [1, H, W]
            return image, mask.unsqueeze(0)

    def get_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                A.Resize(256, 256), # Resize for standard training speed
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

# TEST BLOCK
if __name__ == "__main__":
    # Point this to where your master_metadata.csv actually is
    csv_path = "data/processed/master_metadata.csv" 
    
    if os.path.exists(csv_path):
        print("Testing Classification Dataset...")
        ds = SolarDataset(csv_path, root_dir=".", mode='train', task='classification')
        img, label = ds[0]
        print(f"Image Shape: {img.shape}, Label: {label}")

        print("\nTesting Segmentation Dataset (High Quality Only)...")
        ds_seg = SolarDataset(csv_path, root_dir=".", mode='train', quality_filter='high', task='segmentation')
        if len(ds_seg) > 0:
            img, mask = ds_seg[0]
            print(f"Image Shape: {img.shape}, Mask Shape: {mask.shape}")
        else:
            print("No high quality data found.")
    else:
        print("CSV not found. Run preprocessor.py first.")