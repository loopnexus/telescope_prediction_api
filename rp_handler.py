import base64
import io
import os
from PIL import Image
from ultralytics import YOLO

# load once at cold start
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n-seg.pt")
model = YOLO(MODEL_PATH)

def handler(event: dict) -> dict:
    """
    RunPod serverless entry point.
    Expects JSON: { "input": { "image_base64": "…" } }
    Returns    JSON: { "output": { "mask_base64": "…" } }
    """
    inp = event.get("input", {})
    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "missing image_base64 in input"}

    # decode image
    img_data = base64.b64decode(b64)
    # predict
    results = model.predict(source=io.BytesIO(img_data), conf=0.4, save=False)
    # grab first mask, first image
    mask_arr = results[0].masks.data[0].numpy().astype("uint8") * 255
    mask_img = Image.fromarray(mask_arr)

    # encode mask back to PNG → base64
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"mask_base64": out_b64}}
