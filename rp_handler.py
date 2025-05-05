import base64
import io
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# ─── locate your weights ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"

# gather all .pt files in weights/
weight_files = sorted(WEIGHTS_DIR.glob("*.pt"))
if not weight_files:
    raise RuntimeError(f"No .pt weights found in {WEIGHTS_DIR}")

# allow overriding via env var, else pick the last one
env = os.environ.get("MODEL_PATH")
if env:
    candidate = Path(env)
    if not candidate.is_file():
        candidate = WEIGHTS_DIR / env
    if not candidate.is_file():
        raise RuntimeError(f"MODEL_PATH env var set to '{env}', but file not found")
    MODEL_PATH = candidate
else:
    MODEL_PATH = weight_files[-1]

# debug logs
print(f"🔔 MODEL_PATH resolved to: {MODEL_PATH}", file=sys.stderr, flush=True)
print(f"🔔 Contents of weights/: {list(WEIGHTS_DIR.iterdir())}", file=sys.stderr, flush=True)
print(f"🔔 Loading YOLO weights from: {MODEL_PATH}", file=sys.stderr, flush=True)

# load model
model = YOLO(str(MODEL_PATH))

# load font if available
try:
    FONT = ImageFont.load_default()
except Exception:
    FONT = None


def handler(event: dict) -> dict:
    """
    RunPod entrypoint.
    In:  {"input": {"image_base64": "..."}}
    Out: {"output": {"annotated_base64": "..."}}
    """
    inp = event.get("input", {})
    b64 = inp.get("image_base64")
    if not b64:
        return {"error": "No image_base64 provided"}

    # decode and open as PIL
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil)

    # run inference
    results = model.predict(
        source=arr,
        conf=0.4,
        save=False,
        verbose=False
    )
    res = results[0]

    # prepare mask overlay
    mask_overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    for mask in (res.masks.data if res.masks else []):
        mask_np = (mask.cpu().numpy() * 255).astype("uint8")
        # ensure mask matches image size
        mask_img = Image.fromarray(mask_np, mode="L").resize(pil.size)

        colored_mask = Image.new("RGBA", pil.size, (255, 0, 0, 100))
        mask_overlay = Image.composite(colored_mask, mask_overlay, mask_img)

    # composite masks onto original
    annotated = pil.convert("RGBA")
    annotated = Image.alpha_composite(annotated, mask_overlay)

    # draw boxes and labels
    draw = ImageDraw.Draw(annotated)
    for i, bbox in enumerate(res.boxes.xyxy if res.boxes else []):
        x1, y1, x2, y2 = map(int, bbox.cpu().numpy())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        cls = int(res.boxes.cls[i].cpu().numpy())
        conf = float(res.boxes.conf[i].cpu().numpy())
        label = f"{model.names[cls]} {conf:.2f}"
        # measure text size
        if FONT:
            text_size = draw.textsize(label, font=FONT)
        else:
            text_size = (len(label) * 6, 11)
        # background for text
        draw.rectangle(
            [x1, y1 - text_size[1] - 4, x1 + text_size[0] + 4, y1],
            fill="red"
        )
        draw.text((x1 + 2, y1 - text_size[1] - 2), label, fill="white", font=FONT)

    # encode annotated image
    buf = io.BytesIO()
    annotated.convert("RGB").save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"annotated_base64": out_b64}}
