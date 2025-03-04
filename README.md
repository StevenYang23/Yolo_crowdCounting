# YOLO-Based Crowd Counting

This repository contains a project for crowd counting using YOLO (You Only Look Once). The goal is to detect and count people in images using a deep learning model trained on a dataset of crowded scenes. This approach provides a real-time estimation of the number of individuals in a given image, making it useful for applications such as crowd monitoring, safety management, and public space analytics.

## Task Overview
Crowd counting is a computer vision problem that involves estimating the number of people present in an image. Traditional methods rely on density estimation, segmentation, or detection-based approaches. This project leverages YOLO, a popular object detection model, to directly detect and count individuals in an image. The model is trained on a dataset of crowded scenes and can generalize to different environments.

## Project Structure
- `best.pt`: The trained YOLO model weights.
- `dataset.yaml`: The dataset configuration file.
- `convert_box.py`, `convert_data.py`: Scripts for preprocessing and converting data into the required format.
- `train_new.py`, `train_nwpu.py`: Scripts to train the YOLO model on different datasets.
- `filterData.ipynb`, `plot_dist.ipynb`, `plot_sample.ipynb`, `visualizing.ipynb`: Jupyter notebooks for data analysis and visualization.
- `face.png`, `body1.png`, `body2.png`, `face2.png`: Sample images used for training and testing.

## Demo
Below are some example images showcasing the model's performance:

### Input Image 1
![Demo 1](demo/demo1.png)

### Input Image 2
![Demo 2](demo/demo2.png)

### Detection Output 1
![Demo 3](demo/demo3.png)

### Detection Output 2
![Demo 4](demo/demo4.png)

## Usage
### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/StevenYang23/demo.git
   cd demo
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the YOLO model weights (`best.pt`) and place them in the appropriate directory.

### Running Inference
To perform crowd counting on a new image, run:
```sh
python detect.py --weights best.pt --source path/to/image.jpg
```

### Training the Model
To train the model from scratch, use:
```sh
python train_new.py --data dataset.yaml --weights yolov5s.pt --epochs 50
```

## Acknowledgments
This project is inspired by recent advancements in object detection and crowd counting research. Special thanks to the datasets and open-source tools that made this work possible.

---
For any questions or contributions, feel free to open an issue or submit a pull request.

