from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/app/home/omair/Desktop/muzzle/annotated_data/dataset.yaml",
    batch=48,
    epochs=50, 
    imgsz=640,
    augment=True,
    bgr=0.1,
    project="./runs/detect/", name="muzzle-annotated-data-v11n-22042025",
    )