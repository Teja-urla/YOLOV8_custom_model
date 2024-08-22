from ultralytics import YOLO
import os

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Specify the source image
source = "https://ultralytics.com/images/bus.jpg"

# Make predictions
results = model.predict(source, save=True, imgsz=320, conf=0.5)

# Extract bounding box dimensions
boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")

# Download a dataset from Roboflow
os.system('curl -L "https://public.roboflow.com/ds/5EYxQJiUTb?key=Lr1mQYl8CA" -o roboflow.zip')

# Extract the dataset
os.system('tar -xf roboflow.zip')
os.remove("roboflow.zip")

# Train the YOLO model using the Python interface
model.train(data="data.yaml", epochs=10, lr0=0.01)
