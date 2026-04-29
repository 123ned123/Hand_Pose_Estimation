#
# This script will run exactly once before you start training.
# It will use YOLO to automatically crop your raw training images 
# and save them into the processed folder alongside a new, clean JSON file.


import os
import json
import torch
from PIL import Image
from ultralytics import YOLO

def prepare_dataset(raw_img_dir, raw_annotations_file, processed_img_dir, processed_annotations_file):
    print("Loading YOLO model for data preparation...")
    # Load YOLO to find the hands in your raw images
    yolo_model = YOLO("yolov8n.pt") 
    
    # Ensure processed directory exists
    os.makedirs(processed_img_dir, exist_ok=True)
    
    # Load raw annotations
    with open(raw_annotations_file, 'r') as f:
        raw_annotations = json.load(f)
        
    processed_annotations = []
    
    print(f"Processing {len(raw_annotations)} images...")
    
    for item in raw_annotations:
        img_name = item['image_name']
        raw_img_path = os.path.join(raw_img_dir, img_name)
        
        try:
            image = Image.open(raw_img_path).convert("RGB")
        except Exception as e:
            print(f"Could not read {img_name}: {e}")
            continue

        # Run YOLO detection
        results = yolo_model(image, verbose=False)
        
        # If YOLO fails to find a hand, we skip this image for training the 3D model
        if len(results[0].boxes) == 0:
            print(f"Skipping {img_name} - No hand detected by YOLO.")
            continue
            
        # Get the bounding box of the first detected hand/person
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        
        # Add a small padding to the bounding box so we don't cut off fingertips
        padding = 20
        width, height = image.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # Crop the image
        cropped_hand = image.crop((x1, y1, x2, y2))
        
        # Save the cropped image to the processed folder
        processed_img_path = os.path.join(processed_img_dir, img_name)
        cropped_hand.save(processed_img_path)
        
        # Append to our new processed annotations list
        # We keep the 3D keypoints exactly the same, assuming they are 
        # relative to the hand itself (e.g., origin is the wrist).
        processed_annotations.append({
            "image_name": img_name,
            "keypoints_3d": item['keypoints_3d']
        })
        
    # Save the new JSON file pointing to the cropped images
    with open(processed_annotations_file, 'w') as f:
        json.dump(processed_annotations, f, indent=4)
        
    print(f"Done! Successfully processed {len(processed_annotations)} images.")

if __name__ == "__main__":
    # Define your paths
    RAW_IMG_DIR = "data/raw/images/"
    RAW_JSON = "data/raw/annotations.json"
    
    PROCESSED_IMG_DIR = "data/processed/images/"
    PROCESSED_JSON = "data/processed/train_annotations.json"
    
    prepare_dataset(RAW_IMG_DIR, RAW_JSON, PROCESSED_IMG_DIR, PROCESSED_JSON)