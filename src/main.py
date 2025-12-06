import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
import random
import timm
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A

# --- CONFIGURATION ---
# Paths
TEST_CSV = "data/test_rooftop_data.csv"
OUTPUT_JSON = "outputs/results.json"
AUDIT_DIR = "outputs/audit_images"
MODEL_DIR = "models"
DUMMY_IMAGE_SOURCE = "data/raw/dataset1/train" # Source for mock images

# Constants
ZOOM_LEVEL = 20
THRESHOLD = 0.63
API_KEY = os.environ.get("MAPBOX_API_KEY", "") 

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directories exist
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


# --- 1. LOAD MODELS ---
def load_models():
    print(f"Loading models on {device}...")
    
    # A. Classifier (EfficientNet-B0)
    classifier = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    
    classifier_path = f"{MODEL_DIR}/classifier.pth"
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}")
        
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # B. Segmenter (U-Net ResNet34)
    segmenter = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1, activation='sigmoid')
    
    segmenter_path = f"{MODEL_DIR}/segmenter.pth"
    if not os.path.exists(segmenter_path):
        raise FileNotFoundError(f"Segmenter model not found at {segmenter_path}")
        
    segmenter.load_state_dict(torch.load(segmenter_path, map_location=device))
    segmenter.to(device)
    segmenter.eval()
    
    print("✅ Models loaded successfully.")
    return classifier, segmenter


# --- 2. THE "FETCHER" (Mock Mode) ---
def fetch_satellite_image(lat, lon):
    """
    MOCK API: Loads a random local image instead of calling Mapbox.
    """
    if not os.path.exists(DUMMY_IMAGE_SOURCE):
        return None, "dummy_source_missing"
        
    valid_images = [f for f in os.listdir(DUMMY_IMAGE_SOURCE) if f.lower().endswith(('.jpg', '.png'))]
    
    if not valid_images:
        return None, "no_local_images"
    
    # Pick random image to simulate new location
    random_filename = random.choice(valid_images)
    img_path = os.path.join(DUMMY_IMAGE_SOURCE, random_filename)
    
    img = cv2.imread(img_path)
    if img is None:
        return None, "read_error"
        
    # CRITICAL: Convert BGR (OpenCV) to RGB (Model Expectation)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "success_mock"


# --- 3. THE "PHYSICS" (Math) ---
def calculate_area(mask_pixels, lat):
    """
    Converts pixel count to sq meters based on Web Mercator projection.
    Formula: 156543.03 * cos(lat_rad) / 2^zoom
    """
    meters_per_pixel = 156543.03 * np.cos(np.radians(lat)) / (2 ** ZOOM_LEVEL)
    area_per_pixel = meters_per_pixel ** 2
    return mask_pixels * area_per_pixel


# --- 4. PRE-PROCESSING ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# --- MAIN PIPELINE ---
def run_inference():
    if not os.path.exists(TEST_CSV):
        print(f"❌ Test CSV not found at {TEST_CSV}. Run the dummy generator script first!")
        return

    df = pd.read_csv(TEST_CSV)
    classifier, segmenter = load_models()
    results = []

    print(f"🚀 Starting Inference on {len(df)} sites...")

    for idx, row in df.iterrows():
        site_id = row.get('id', idx)
        lat = float(row['lat'])
        lon = float(row['long'])
        
        # --- STEP A: FETCH ---
        image, status = fetch_satellite_image(lat, lon)
        
        if image is None:
            results.append({
                "sample_id": site_id,
                "lat": lat, "lon": lon,
                "has_solar": False,
                "qc_status": "NOT_VERIFIABLE",
                "qc_notes": [f"Fetch failed: {status}"]
            })
            continue

        # --- STEP B: PREPARE INPUT ---
        orig_h, orig_w = image.shape[:2]
        aug = transform(image=image)["image"].unsqueeze(0).to(device)

        # --- STEP C: CLASSIFY (The Gatekeeper) ---
        with torch.no_grad():
            logit1 = classifier(aug)
            logit2 = classifier(torch.flip(aug, [3]))
            prob = (torch.sigmoid(logit1) + torch.sigmoid(logit2)) / 2
        
        has_solar = prob > THRESHOLD
        
        record = {
            "sample_id": site_id,
            "lat": lat, "lon": lon,
            "has_solar": bool(has_solar),
            "confidence": round(prob, 4),
            "qc_status": "VERIFIABLE",
            "image_metadata": {"source": "MockData", "zoom": ZOOM_LEVEL}
        }

        # --- STEP D: SEGMENT (The Surveyor) ---
        if has_solar:
            with torch.no_grad():
                mask_logits = segmenter(aug)
                mask_prob = mask_logits.sigmoid().cpu().numpy()[0, 0]
            
            # Threshold to Binary (0 or 1)
            binary_mask_small = (mask_prob > THRESHOLD).astype(np.uint8)
            
            # Resize mask back to original image size
            binary_mask = cv2.resize(binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # --- FIND BIGGEST PANEL ---
            # connectedComponents returns stats: [x, y, w, h, area]
            # labels is a matrix of the same size as the image, where pixels have value 1, 2, 3... corresponding to their blob
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            max_pixel_count = 0
            total_pixel_count = 0
            largest_label_idx = -1

            if num_labels > 1:
                # Label 0 is background. stats[1:, 4] extracts areas of labels 1, 2, 3...
                all_areas = stats[1:, 4]
                total_pixel_count = np.sum(all_areas)
                
                # Find the index of the largest blob (argmax gives index in all_areas, so add 1 for real label)
                largest_label_idx = np.argmax(all_areas) + 1
                max_pixel_count = all_areas[largest_label_idx - 1]
            
            # --- CALCULATE REAL METRICS ---
            scale_factor = (orig_h * orig_w) / (256 * 256) 
            real_max_pixels = max_pixel_count * scale_factor
            real_total_pixels = total_pixel_count * scale_factor
            
            max_area_sqm = calculate_area(real_max_pixels, lat)
            total_area_sqm = calculate_area(real_total_pixels, lat)
            
            # Update Record
            record.update({
                "panel_count_Est": int(total_area_sqm / 1.6),
                "max_single_panel_area_sqm": round(max_area_sqm, 2),
                "pv_area_sqm_est": round(total_area_sqm, 2),
                "capacity_kw_est": round(total_area_sqm * 0.200, 2),
                "bbox_or_mask": "See audit image",
                "qc_notes": ["Solar Detected", "Max Area Calculated"]
            })
            
            # --- VISUALIZATION START ---
            
            # 1. Create Red Overlay (ALL Panels)
            overlay = image.copy()
            overlay[binary_mask == 1] = [255, 0, 0] # Red (RGB)
            
            # 2. Blend to create the tint
            audit_img = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            
            # 3. Draw Green Outline (BIGGEST Panel Only)
            if largest_label_idx > 0:
                # Create a mask that ONLY contains the biggest label
                biggest_blob_mask = (labels == largest_label_idx).astype(np.uint8)
                
                # Find the polygon/contour of this specific blob
                contours, _ = cv2.findContours(biggest_blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the contour (Green: 0, 255, 0) with thickness 3
                cv2.drawContours(audit_img, contours, -1, (0, 255, 0), 3)

            # 4. Add Text (Confidence + Max Area)
            label_text = f"Conf: {prob:.2f} | Max: {max_area_sqm:.1f}m2"
            
            # Black Outline (for readability)
            cv2.putText(audit_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            # White Text
            cv2.putText(audit_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 5. Save (Convert RGB -> BGR)
            save_path = os.path.join(AUDIT_DIR, f"{site_id}_audit.png")
            cv2.imwrite(save_path, cv2.cvtColor(audit_img, cv2.COLOR_RGB2BGR))
            
        else:
            record["qc_notes"] = ["No Solar Detected"]

        results.append(record)
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(df)} sites...")

    # --- 5. SAVE JSON ---
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Pipeline Complete!")
    print(f"📄 Results: {OUTPUT_JSON}")
    print(f"🖼️  Audit Images: {AUDIT_DIR}")

if __name__ == "__main__":
    run_inference()