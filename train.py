import os
from ultralytics import YOLO

# Set up paths
data_yaml_path = '/Users/garvagarwal/Desktop/SPORTS-AI-TRAINER/Tracer---basketball-3/data.yaml'

# Initialize the model
model = YOLO('yolov8s.pt')  # You can choose other models like 'yolov5m.pt', 'yolov5l.pt', etc.

# Train the model
model.train(data=data_yaml_path, epochs=100,imgsz=640,patience=10)

# Save the trained model
model_path = 'best.pt'  # Adjust the path if necessary
print(f"Model trained and saved at {model_path}")
