# Dental Tooth Detection Using YOLOv8

## Project Overview
This project detects and classifies teeth in dental panoramic images using the YOLOv8 object detection model. The dataset uses the FDI tooth numbering system for accurate identification of each tooth.

## Environment Setup
- Python 3.8 or higher
- Install dependencies using pip:


## Dataset
- Around 500 dental panoramic images with YOLO-format annotations.
- Annotations use the FDI numbering system (32 classes).
- Dataset split: 80% train, 10% validation, 10% test.

## Training

### Configuration
- Training data paths and class names are defined in `data.yaml`.

### Training Command
Train the model with:
yolo detect train model=yolov8s.pt data=data.yaml imgsz=640 epochs=10 batch=8 name=dental_teeth_model 
Or using Python script (`train.py`):


