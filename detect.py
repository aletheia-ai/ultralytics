from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("runs/detect/muzzle-annotated-data-v11n/weights/best.pt")

# Perform object detection on an image
results = model("/app/home/omair/Desktop/muzzle/annotated_data/images/test/")

# Display the results
# results[0].show()