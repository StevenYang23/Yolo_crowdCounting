import os
import scipy.io
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# Set dataset paths
img_dir = "/home/stevenyang/Desktop/cmput469/YOLO/NWPU_60/img"
mat_dir = "/home/stevenyang/Desktop/cmput469/YOLO/NWPU_60/mats"
yolo_dataset_dir = "/home/stevenyang/Desktop/cmput469/YOLO/dataset_60"
labels_dir = os.path.join(yolo_dataset_dir, "labels")
images_dir = os.path.join(yolo_dataset_dir, "images")

# Ensure dataset directories exist
for d in [yolo_dataset_dir, labels_dir, images_dir]:
    os.makedirs(d, exist_ok=True)

# Convert dataset to YOLO format
print("Converting dataset to YOLO format...")
for mat_file in tqdm(os.listdir(mat_dir)):
    if not mat_file.endswith(".mat"):
        continue
    
    mat_path = os.path.join(mat_dir, mat_file)
    img_file = mat_file.replace(".mat", ".jpg")
    img_path = os.path.join(img_dir, img_file)
    
    if not os.path.exists(img_path):
        continue
    
    # Load annotations
    mat_data = scipy.io.loadmat(mat_path)
    ann_boxes = mat_data['annBoxes']  # Shape (num_people, 4)
    
    # Load image for normalization
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Create YOLO annotation file
    label_file = os.path.join(labels_dir, mat_file.replace(".mat", ".txt"))
    with open(label_file, "w") as f:
        for box in ann_boxes:
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
    
    # Copy image to dataset
    shutil.copy(img_path, os.path.join(images_dir, img_file))

print("Dataset conversion complete.")

# Create YOLO dataset YAML file
dataset_yaml = os.path.join(yolo_dataset_dir, "dataset.yaml")
with open(dataset_yaml, "w") as f:
    f.write(f"""
path: {yolo_dataset_dir}
train: images
val: images
nc: 1
names: ['human']
""")

print(f"Dataset YAML saved at {dataset_yaml}.")
print("You can now train YOLOv8 using 'train_yolo.py'.")
