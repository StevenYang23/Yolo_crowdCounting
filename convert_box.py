import os

# Define paths
dataset_path = "/home/stevenyang/Desktop/cmput469/YOLO/yolo_Box"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
classes_file = os.path.join(dataset_path, "classes.txt")
yaml_file = os.path.join(dataset_path, "dataset.yaml")

# Read class names
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Write dataset.yaml
yaml_content = f"""train: {images_path}
val: {images_path}
nc: {len(class_names)}
names: {class_names}
"""

with open(yaml_file, "w") as f:
    f.write(yaml_content)

print(f"dataset.yaml created at {yaml_file}")
