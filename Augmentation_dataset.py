import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
])

input_folder = "dataset/students"
output_base = "dataset/students/facenet_dataset"

train_folder = os.path.join(output_base, "train")
val_folder = os.path.join(output_base, "val")
test_folder = os.path.join(output_base, "test")

for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

def read_image_unicode(path):
    try:
        with open(path, "rb") as f:
            img_data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot read image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Cannot read image: {path} ({e})")
        return None

def save_image(img, path):
    img_pil = Image.fromarray(img)
    img_pil.save(path, quality=95)

for student_id in tqdm(os.listdir(input_folder), desc="Processing Students"):
    student_path = os.path.join(input_folder, student_id)
    if not os.path.isdir(student_path):
        continue    

    images = [os.path.join(student_path, f) for f in os.listdir(student_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(images) < 50:
        print(f"{student_id} has less than 50 images!")
        continue

    all_images = []
    
    for img_path in images:
        img = read_image_unicode(img_path)
        if img is None:
            continue

        all_images.append(img)

        aug_img = augmenter(image=img)["image"]
        all_images.append(aug_img)

    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    def save_images(img_list, folder):
        save_path = os.path.join(folder, student_id)
        os.makedirs(save_path, exist_ok=True)
        for i, img in enumerate(img_list):
            save_image(img, os.path.join(save_path, f"img_{i}.jpg"))

    save_images(train_imgs, train_folder)
    save_images(val_imgs, val_folder)
    save_images(test_imgs, test_folder)

    print(f"{student_id}: Train {len(train_imgs)}, Val {len(val_imgs)}, Test {len(test_imgs)}")

print("Augmentation & Dataset Splitting successfully")
