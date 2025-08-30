from ultralytics import YOLO
import os

model_path = "runs/detect/dental_teeth_model7/weights/best.pt"
model = YOLO(model_path)

image_folder = "C:/Users/bvspranav999/OneDrive/Desktop/Environment Project/Dental_Teeth_Project/Val/images"

output_folder = "runs/detect/dental_teeth_model7/predictions"
os.makedirs(output_folder, exist_ok=True)

# Run inference on a subset or all images to save results with bounding boxes + labels
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, img_name)
        results = model.predict(source=img_path, save=True, save_dir=output_folder, save_conf=False, save_txt=False, line_thickness=2)

