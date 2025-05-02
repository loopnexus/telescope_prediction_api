import os
import glob
from ultralytics import YOLO
import base64
import cv2
import numpy as np

# Load all YOLO models from weights/
MODEL_DIR = "weights"
models = {
    os.path.splitext(os.path.basename(w))[0]: YOLO(w)
    for w in glob.glob(os.path.join(MODEL_DIR, "*.pt"))
}

def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def handler(event):
    input_data = event["input"]
    img_b64 = input_data.get("image_base64")

    if not img_b64:
        return {"error": "Missing image_base64 input."}

    image = decode_image(img_b64)
    results = {}

    for name, model in models.items():
        prediction = model.predict(image, task="segment")[0]
        detections = []

        for mask, box, cls in zip(prediction.masks.data, prediction.boxes.xyxy, prediction.boxes.cls):
            detections.append({
                "class": int(cls),
                "bbox": box.tolist(),
                "mask": mask.cpu().numpy().tolist()
            })

        results[name] = detections

    return {"predictions": results}
