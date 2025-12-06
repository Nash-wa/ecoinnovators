import json
import cv2
import random
import matplotlib.pyplot as plt
import os

# Load results
with open("outputs/results.json") as f:
    results = json.load(f)

# Filter for only positives (since negatives have no audit image)
positives = [r for r in results if r['has_solar']]

if positives:
    # Pick random
    entry = random.choice(positives)
    site_id = entry['sample_id']
    
    print(f"--- Checking Site ID: {site_id} ---")
    print(f"Confidence: {entry['confidence']}")
    print(f"Est. Area: {entry['pv_area_sqm_est']} m2")
    
    # Load Image
    img_path = f"outputs/audit_images/{site_id}_audit.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("Audit image not found!")
else:
    print("No solar panels detected in the entire result set.")