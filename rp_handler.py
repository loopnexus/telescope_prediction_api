# rp_handler.py

import base64
import io
import os

from PIL import Image
import numpy as np
from ultralytics import YOLO

# on cold start this will download the model if not already cached
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n-seg.pt")
model = YOLO(MODEL_PATH)

def handler(event: dict) -> dict:
    """
    RunPod entrypoint.
    In:  {"input": {"image_base64": "..."}}
    Out: {"output": {"mask_base64": "..."}}
    """
    inp = event.get("input", {})
    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "No image_base64 provided"}

    # 1) decode upload
    img_bytes = base64.b64decode(b64)

    # 2) load as PIL, convert to RGB
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 3) turn into numpy array (H x W x C)
    arr = np.array(pil)

    # 4) run segmentation on the numpy array
    results = model.predict(
        source=arr,
        conf=0.4,
        save=False,
        verbose=False
    )

    # 5) extract the first mask, scale to 0–255, and build a PIL image
    mask_arr = (results[0].masks.data[0].cpu().numpy() * 255).astype("uint8")
    mask_img = Image.fromarray(mask_arr)

    # 6) encode mask to PNG base64
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"mask_base64": out_b64}}
