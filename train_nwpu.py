import torch
from ultralytics import YOLO

# Ensure GPU is available
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available! Check your GPU installation.")
    exit(1)

print("CUDA is available. Training on GPU.")

# Set dataset YAML path
dataset_yaml = "/home/stevenyang/Desktop/cmput469/YOLO/yolo_Box/dataset.yaml"


model = YOLO("yolo11s.pt")
# model = YOLO("/home/stevenyang/Desktop/cmput469/YOLO/runs/detect/crowd_counting_v11/weights/last.pt")


model.train(
    data=dataset_yaml,
    epochs=5000,
    batch=1,
    imgsz=1280,
    device="cuda",
    save=True,
    save_period=10,
    project="runs/detect",
    name="crowd_counting_v11",
    amp=True,
    # resume=True,
    optimizer = "auto",
    momentum = 0.937,
    weight_decay = 0.0005,
    cls = 0.5,
    box = 7.5,
    plots = True,
    val = True
)
