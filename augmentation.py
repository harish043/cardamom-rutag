import os
import cv2
import albumentations as A
from tqdm import tqdm

# Paths
RAW_DIR = "images/raw_cardamom"
OUTPUT_DIR = "images/synthetic_cardamom"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Updated Classes and corresponding augmentations
AUGMENTATIONS = {
    "raw": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(0.0, 0.1), contrast_limit=(-0.1, 0.1), p=0.3)
    ]),
    "partially_dried": A.Compose([
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=-20, val_shift_limit=10, p=1),
        A.RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(-0.1, 0.0), p=1),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3)
    ]),
    "mostly_dried": A.Compose([
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=-30, val_shift_limit=20, p=1),
        A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3), contrast_limit=(-0.2, 0.0), p=1),
        A.MotionBlur(blur_limit=3, p=0.3)
    ]),
    "fully_dried": A.Compose([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=-40, val_shift_limit=30, p=1),
        A.RandomBrightnessContrast(brightness_limit=(0.3, 0.4), contrast_limit=(-0.3, -0.1), p=1),
        A.ImageCompression(quality_lower=40, quality_upper=60, p=0.5),
        A.GaussianBlur(p=0.3)
    ])
}


# Process images
for class_name, transform in AUGMENTATIONS.items():
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(RAW_DIR), desc=f"Creating {class_name}"):
        img_path = os.path.join(RAW_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        augmented = transform(image=image)['image']
        out_path = os.path.join(class_dir, img_name)
        cv2.imwrite(out_path, augmented)
