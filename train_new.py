import torch
from ultralytics import YOLO

# Ensure GPU is available
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available! Check your GPU installation.")
    exit(1)

print("CUDA is available. Training on GPU.")

# Set dataset YAML path
dataset_yaml = "/home/stevenyang/Desktop/cmput469/YOLO/yolo_new/data.yaml"


# model = YOLO("yolo11s.pt")
model = YOLO("/home/stevenyang/Desktop/cmput469/YOLO/runs/detect/crowd_counting_v11_new/weights/best.pt")


model.train(
    data=dataset_yaml,
    epochs=5000,
    batch=1,
    imgsz=1280,
    device="cuda",
    save=True,
    save_period=1,
    project="runs/detect",
    name="crowd_counting_v11_new",
    amp=True,
    resume = True
    
)

print("Training complete. Model saved in 'runs/detect/crowd_counting/weights/best.pt'")
