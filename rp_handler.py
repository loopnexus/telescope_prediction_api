# rp_handler.py

import base64
import io
import os
import glob
from pathlib import Path

from PIL import Image
import numpy as np
from ultralytics import YOLO

# ─── locate your weights ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"

# gather all .pt files in weights/
weight_files = sorted(WEIGHTS_DIR.glob("*.pt"))
if not weight_files:
    raise RuntimeError(f"No .pt weights found in {WEIGHTS_DIR}")

# allow overriding via env var, else pick the last one
env = os.environ.get("MODEL_PATH")
if env:
    candidate = Path(env)
    # if it's not an absolute path, look in weights/
    if not candidate.is_file():
        candidate = WEIGHTS_DIR / env
    if not candidate.is_file():
        raise RuntimeError(f"MODEL_PATH env var set to '{env}', but file not found")
    MODEL_PATH = candidate
else:
    MODEL_PATH = weight_files[-1]


# ←—————— ADD THESE TWO LINES RIGHT HERE ——————→
print(f"🔔 MODEL_PATH resolved to: {MODEL_PATH}", file=sys.stderr, flush=True)
print(f"🔔 Contents of weights/: {list(WEIGHTS_DIR.iterdir())}", file=sys.stderr, flush=True)
# ←———————————————————————————————————————————————→


print(f"🔔 Loading YOLO weights from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
# ─────────────────────────────────────────────────────────────────────────────

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

    # 3) turn into numpy array (H × W × C)
    arr = np.array(pil)

    # 4) run segmentation on the numpy array
    results = model.predict(
        source=arr,
        conf=0.4,
        save=False,
        verbose=False
    )

    # 5) if no masks, produce blank mask; else take first mask
    res = results[0]
    if res.masks is None or len(res.masks.data) == 0:
        h, w, _ = arr.shape
        mask_arr = np.zeros((h, w), dtype="uint8")
    else:
        mask_arr = (res.masks.data[0].cpu().numpy() * 255).astype("uint8")

    # 6) build PIL image from mask array
    mask_img = Image.fromarray(mask_arr)

    # 7) encode mask to PNG base64
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"mask_base64": out_b64}}
