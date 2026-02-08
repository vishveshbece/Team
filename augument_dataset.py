import os
import cv2
import random
import shutil
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = r'E:\IESA'
OUTPUT_DIR = r'E:\IESA\Balanced_Dataset'

TARGET_CLASSES = [
    'bridge', 'Clean', 'Crack', 'malformed_via', 'Open_circuit', 
    'Other', 'Scratch_CMP', 'viapinholedefect', 'viavoiddefect'
]

VAL_SPLIT = 0.3
MAX_COUNT = 1500

# ------------------ AUGMENTATION FUNCTIONS ------------------

def random_rotate_90(img):
    k = random.choice([0, 1, 2, 3])
    return np.rot90(img, k).copy()

def random_flip(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() < 0.5:
        img = cv2.flip(img, 0)  # vertical
    return img

def random_brightness_contrast(img):
    alpha = random.uniform(0.75, 1.25)  # contrast
    beta = random.randint(-40, 40)      # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_gamma(img):
    gamma = random.uniform(0.8, 1.2)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_gaussian_noise(img):
    if random.random() < 0.3:
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    return img

def augment_image(img):
    img = random_rotate_90(img)
    img = random_flip(img)
    img = random_brightness_contrast(img)
    img = random_gamma(img)
    img = add_gaussian_noise(img)
    return img

# ------------------ MAIN PIPELINE ------------------

def balance_and_split():
    for cls in TARGET_CLASSES:
        print(f"\nProcessing class: {cls}")
        cls_source_path = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(cls_source_path):
            continue

        images = [f for f in os.listdir(cls_source_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images:
            continue

        random.shuffle(images)

        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_list = images[:split_idx]
        val_list = images[split_idx:]

        for subset in ['train', 'val']:
            os.makedirs(os.path.join(OUTPUT_DIR, subset, cls), exist_ok=True)

        # Copy validation
        for img in val_list:
            shutil.copy(
                os.path.join(cls_source_path, img),
                os.path.join(OUTPUT_DIR, 'val', cls, img)
            )

        # Copy training originals
        for img in train_list:
            shutil.copy(
                os.path.join(cls_source_path, img),
                os.path.join(OUTPUT_DIR, 'train', cls, img)
            )

        # Augment
        needed = MAX_COUNT - len(train_list)
        if needed > 0:
            print(f"Generating {needed} augmented images for {cls}...")
            for i in tqdm(range(needed)):
                src_name = random.choice(train_list)
                img = cv2.imread(os.path.join(cls_source_path, src_name))

                aug = augment_image(img)
                new_name = f"aug_{i}_{src_name}"
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, 'train', cls, new_name),
                    aug
                )

    print("\nFinished! Balanced dataset ready.")

if __name__ == "__main__":
    balance_and_split()
