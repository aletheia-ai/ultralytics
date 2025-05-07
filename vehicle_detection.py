# from ultralytics import YOLO
# model = YOLO(f"yolov8x.pt")
# result = model.predict(source="video.mp4", tracker="custom_tracker.yaml", stream=True,  conf=0.7, show=True, save=True, show_conf=False, classes=[2,5,7])
# for r in result:
#     print(r)


# import cv2
# import supervision as sv
# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")

# image = cv2.imread(<PATH TO IMAGE>)
# results = model(image)[0]
# detections = sv.Detections.from_ultralytics(results)


import numpy as np
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolo11x.pt")

coco_names = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "commercial_vehicle",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # results = model(frame)[0]
    results = model.predict(source=frame, tracker="custom_tracker.yaml",  conf=0.7, classes=[2,5,7])[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        coco_names[class_id+1]
        # model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image
    # return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="/app/home/omair/Downloads/big_trucks.mp4",
    target_path="result_big_trucks.mp4",
    callback=callback
)
