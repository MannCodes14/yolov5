import torch
import cv2
from PIL import Image

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/41_classes_model27/weights/best.pt', source='local')

# Path to your test image
image_path = "C:\\Users\\manns\\Downloads\\car1.jpg"

# Perform inference
results = model(image_path)

# Print results in the console
print(results.pandas().xyxy[0])  # Outputs prediction as a DataFrame

# Show the image with detections
results.show()
