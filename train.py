from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO('yolov8s.pt')  

# Train the model
model.train(
    data='C:/Users/bvspranav999/OneDrive/Desktop/Environment Project/Dental_Teeth_Project/dental_data.yaml',   # update path to the actual path
    imgsz=640,         
    epochs=10,         
    batch=8,            
    name='dental_teeth_model'
)

print("Model training is finished")
