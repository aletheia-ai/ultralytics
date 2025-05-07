from ultralytics import YOLO
from pathlib import Path
# Load a model
model = YOLO("/app/home/omair/Downloads/raza-06052025.pt")  # load a pretrained model (recommended for training)
model.export(format="onnx", dynamic=True, half=False, imgsz=[384, 640])
