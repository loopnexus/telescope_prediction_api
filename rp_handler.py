import os
import glob
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import runpod
import runpod.serverless

# ─── Load every .pt in weights/ ───────────────────────────────────────────────
models = {}
for w in glob.glob(os.path.join("weights", "*.pt")):
    name = os.path.basename(w)
    models[name] = YOLO(w)

def decode_image(b64: str):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def handler(event):
    # Extract base64 payload
    img_b64 = event.get("input", {}).get("image_base64", "")
    img = decode_image(img_b64)

    all_preds = {}
    for name, model in models.items():
        # run segmentation at conf=0.5
        res = model.predict(source=img, task="segment", conf=0.5)[0]
        boxes  = res.boxes.xyxy.tolist() if hasattr(res, "boxes") else []
        classes= res.boxes.cls.tolist()  if hasattr(res, "boxes") else []
        masks  = res.masks.xy         if getattr(res, "masks", None) else []

        dets = []
        for box, cls, mask in zip(boxes, classes, masks):
            dets.append({
                "class": int(cls),
                "bbox": [float(x) for x in box],
                "mask": mask.tolist()
            })
        all_preds[name] = dets

    return {"predictions": all_preds}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
