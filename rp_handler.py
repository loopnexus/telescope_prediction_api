import os
import sys
import io
import uuid
import base64
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# ─── locate your weights ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"

# gather all .pt files in weights/
weight_files = sorted(WEIGHTS_DIR.glob("*.pt"))
if not weight_files:
    raise RuntimeError(f"No .pt weights found in {WEIGHTS_DIR}")

# Load a YOLO model for each weight file
each_model = []
for wfile in weight_files:
    model_name = wfile.stem
    print(f"🔔 Loading YOLO weights from: {wfile}", file=sys.stderr, flush=True)
    each_model.append((model_name, YOLO(str(wfile))))

# Optionally allow external override of structure type
def _get_structure_type():
    try:
        return int(os.environ.get("STRUCTURE_TYPE", 1))
    except ValueError:
        return 1


def handler(event: dict):
    """
    RunPod entrypoint.
    In:  {"input": {"image_base64": "...", "image_name": "..."}}
    Out: JSON array matching schema
    """
    inp = event.get("input", {})
    b64 = inp.get("image_base64")
    image_name = inp.get("image_name", "")
    if not b64:
        return {"error": "No image_base64 provided"}

    # decode and load image
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil)

    records = []
    struct_type = _get_structure_type()

    # run each model
    for model_name, model in each_model:
        results = model.predict(source=arr, conf=0.4, save=False, verbose=False)
        res = results[0]

        # iterate detections
        for i, box in enumerate(res.boxes.xyxy):
            # parse bbox
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            bbox = [x1, y1, x2, y2]

            # extract class and confidence
            cls_id = int(res.boxes.cls[i].cpu().numpy())
            class_name = model.names[cls_id]
            conf = float(res.boxes.conf[i].cpu().numpy())

            # split into type/orientation/mod
            parts = class_name.split("_")
            eq_type = parts[0] if len(parts) > 0 else ""
            orientation = parts[1] if len(parts) > 1 else ""
            try:
                eq_modification = int(parts[2]) if len(parts) > 2 else None
            except ValueError:
                eq_modification = None

            # build polygon via contour of mask if available
            polygon = []
            if res.masks and i < len(res.masks.data):
                mask_np = (res.masks.data[i].cpu().numpy() * 255).astype(np.uint8)
                # threshold
                _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # take largest
                    contour = max(contours, key=lambda c: cv2.contourArea(c))
                    polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in contour]

            record = {
                "prediction_id": str(uuid.uuid4()),
                "ml_model_name": model_name,
                "class_name": class_name,
                "confidence": conf,
                "eq_type": eq_type,
                "orientation": orientation,
                "eq_modification": eq_modification,
                "bounding_box": bbox,
                "polygon": polygon
            }
            records.append(record)

    # wrap in top-level object
    return [{
        "image_name": image_name,
        "predictions": records,
        "structure_type": struct_type
    }]
