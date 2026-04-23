# data/prepare_data.py

import os
import shutil
import random
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Configuration
SOURCE_DIR = r"C:\Users\A\OneDrive\Desktop\Thesis_folder\steel-defect-thesis\data\NEU"  # original dataset folder
TARGET_DIR = r"C:\Users\A\OneDrive\Desktop\Thesis_folder\steel-defect-thesis\data\NEU_prepared"  # save processed data
SEED = 42
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15  # from the remaining training data
IMG_SIZE = (224, 224)

random.seed(SEED)
np.random.seed(SEED)    

def prepare_dataset():
    print("=" * 50)
    print("Preparing NEU dataset for training")
    print("=" * 50)
    
    # Get class names (folder names in SOURCE_DIR)
    classes = [d for d in os.listdir(SOURCE_DIR) 
               if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    classes.sort()
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)
    
    # Collect all image paths
    all_images = []
    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        # Add class label
        images = [(img, cls) for img in images]
        all_images.extend(images)
    
    print(f"Total images: {len(all_images)}")
    
    # Shuffle
    random.shuffle(all_images)
    
    # Split indices
    n_total = len(all_images)
    n_test = int(n_total * TEST_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_test - n_val
    
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Assign splits
    test_images = all_images[:n_test]
    val_images = all_images[n_test:n_test + n_val]
    train_images = all_images[n_test + n_val:]
    
    # Copy images to target directories with preprocessing
    splits = [
        ('train', train_images),
        ('val', val_images),
        ('test', test_images)
    ]
    
    for split_name, images in splits:
        print(f"\nProcessing {split_name} split...")
        for img_path, cls in tqdm(images):
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, IMG_SIZE)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Save to target
            filename = os.path.basename(img_path)
            target_path = os.path.join(TARGET_DIR, split_name, cls, filename)
            cv2.imwrite(target_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print(f"Prepared data saved in: {TARGET_DIR}")
    print("=" * 50)
    
    # Print class distribution
    print("\nClass distribution:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split}:")
        for cls in classes:
            count = len(os.listdir(os.path.join(TARGET_DIR, split, cls)))
            print(f"  {cls}: {count} images")

if __name__ == "__main__":
    prepare_dataset()
    