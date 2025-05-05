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

# Load a YOLO model for each weight file, assign a unique color
MODELS = []
# A simple palette of RGBA colors (semi-transparent)
PALETTE = [
    (255, 0, 0, 100),    # red
    (0, 255, 0, 100),    # green
    (0, 0, 255, 100),    # blue
    (255, 255, 0, 100),  # yellow
    (255, 0, 255, 100),  # magenta
    (0, 255, 255, 100),  # cyan
]
for idx, wfile in enumerate(weight_files):
    color = PALETTE[idx % len(PALETTE)]
    print(f"🔔 Loading YOLO weights from: {wfile}", file=sys.stderr, flush=True)
    MODELS.append((wfile.stem, YOLO(str(wfile)), color))

# load default font
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

    # Prepare overlay canvas
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Iterate over each model and its color
    for name, model, color in MODELS:
        results = model.predict(
            source=arr,
            conf=0.4,
            save=False,
            verbose=False
        )
        res = results[0]

        # masks
        if res.masks:
            for mask in res.masks.data:
                mask_np = (mask.cpu().numpy() * 255).astype("uint8")
                mask_img = Image.fromarray(mask_np, mode="L").resize(pil.size)
                colored = Image.new("RGBA", pil.size, color)
                overlay = Image.composite(colored, overlay, mask_img)

        # boxes + labels
        if res.boxes:
            for i, bbox in enumerate(res.boxes.xyxy):
                x1, y1, x2, y2 = map(int, bbox.cpu().numpy())
                # draw box
                draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
                cls = int(res.boxes.cls[i].cpu().numpy())
                conf = float(res.boxes.conf[i].cpu().numpy())
                label = f"{name}:{model.names[cls]} {conf:.2f}"

                # measure text size
                if FONT:
                    try:
                        text_size = FONT.getsize(label)
                    except Exception:
                        bbox0 = draw.textbbox((0, 0), label, font=FONT)
                        text_size = (bbox0[2] - bbox0[0], bbox0[3] - bbox0[1])
                else:
                    text_size = (len(label) * 6, 11)

                # text background
                bg = [x1, y1 - text_size[1] - 4, x1 + text_size[0] + 4, y1]
                draw.rectangle(bg, fill=color[:3] + (200,))
                draw.text((x1 + 2, y1 - text_size[1] - 2), label, fill=(255,255,255,255), font=FONT)

    # Composite overlay onto original
    annotated = Image.alpha_composite(pil.convert("RGBA"), overlay)

    # encode annotated image
    buf = io.BytesIO()
    annotated.convert("RGB").save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf8")

    return {"output": {"annotated_base64": out_b64}}
