import json
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# paths defined assuming we run from root
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MASKS_DIR = os.path.join(PROCESSED_DIR, "masks")

# defining datasets and their types
# NOTE: Dataset 3 is "box" type (low quality), 1 & 2 are "polygon" (high quality)
DATASETS = [
    {"name": "dataset1", "type": "polygon", "quality": "high"},
    {"name": "dataset2", "type": "polygon", "quality": "high"},
    {"name": "dataset3", "type": "box", "quality": "low"},
]

os.makedirs(MASKS_DIR, exist_ok=True)


def create_mask_from_polygon(img_shape, annotations):
    """Generates a binary mask from COCO polygon points."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for ann in annotations:
        for seg in ann['segmentation']:
            # COCO format is flat [x, y, x, y...], reshape to Nx2
            poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
    return mask


def create_mask_from_box(img_shape, annotations):
    """Generates a binary mask from simple bounding boxes (Dataset 3)."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for ann in annotations:
        # Check if 'bbox' key exists (COCO standard) or if it's custom
        bbox = ann.get('bbox') 
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    return mask


def process_folder(dataset_name, split_name, ds_type, quality):
    """
    Processes a specific folder (e.g., data/raw/dataset1/train)
    """
    folder_path = os.path.join(RAW_DATA_DIR, dataset_name, split_name)
    json_files = glob(os.path.join(folder_path, "*_annotations.coco.json"))
    
    # Sometimes the JSON is named differently, check generic name too
    if not json_files:
        json_files = glob(os.path.join(folder_path, "annotations.json"))
        
    if not json_files:
        print(f"Skipping {dataset_name}/{split_name} (No JSON found)")
        return []

    json_path = json_files[0]
    print(f"Processing {dataset_name} [{split_name}]...")
    
    with open(json_path) as f:
        coco = json.load(f)

    # Index images and annotations
    images_info = {img['id']: img for img in coco['images']}
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns: img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    processed_records = []
    
    for img_id, info in tqdm(images_info.items()):
        file_name = info['file_name']
        img_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(img_path):
            continue

        # Read Image (to get shape)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Create Mask
        anns = img_to_anns.get(img_id, [])
        if ds_type == 'polygon':
            mask = create_mask_from_polygon(img.shape, anns)
        else:
            mask = create_mask_from_box(img.shape, anns)
            
        # Save Mask
        # Naming convention: dataset_split_filename.png to avoid collisions
        safe_name = f"{dataset_name}_{split_name}_{file_name}".replace("/", "_").replace(".jpg", ".png")
        mask_path = os.path.join(MASKS_DIR, safe_name)
        cv2.imwrite(mask_path, mask)
        
        processed_records.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "has_solar": 1 if len(anns) > 0 else 0,
            "dataset": dataset_name,
            "split": split_name,   # train or test
            "quality": quality     # high or low
        })
        
    return processed_records


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_data = []
    
    for ds in DATASETS:
        # Process 'train' folder
        all_data.extend(process_folder(ds['name'], "train", ds['type'], ds['quality']))
        
        # Process 'test' folder (Optional: usually for validation)
        # Only process if it exists
        if os.path.exists(os.path.join(RAW_DATA_DIR, ds['name'], "test")):
            all_data.extend(process_folder(ds['name'], "test", ds['type'], ds['quality']))

    # Save Master CSV
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(PROCESSED_DIR, "master_metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Done! Processed {len(df)} images.")
    print(f"📝 Metadata saved to: {csv_path}")
    print(f"🖼️ Masks saved to: {MASKS_DIR}")