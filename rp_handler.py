import base64
import io
import os
from PIL import Image
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

    # decode upload
    img_bytes = base64.b64decode(b64)
    # run segmentation
    results = model.predict(
        source=io.BytesIO(img_bytes),
        conf=0.4,
        save=False,
        verbose=False
    )
    # take first mask
    mask_arr = (results[0].masks.data[0].cpu().numpy() * 255).astype("uint8")
    mask_img = Image.fromarray(mask_arr)

    # encode mask to PNG base64
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"mask_base64": out_b64}}
