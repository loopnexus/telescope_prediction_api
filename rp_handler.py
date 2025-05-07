import os
import sys
import io
import uuid
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import runpod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ─── locate your weights ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"

def load_models() -> List[tuple]:
    """Load all YOLO models from weights directory."""
    weight_files = sorted(WEIGHTS_DIR.glob("*.pt"))
    if not weight_files:
        raise RuntimeError(f"No .pt weights found in {WEIGHTS_DIR}")
    
    models = []
    for wfile in weight_files:
        model_name = wfile.stem
        logger.info(f"Loading YOLO weights from: {wfile}")
        try:
            model = YOLO(str(wfile))
            models.append((model_name, model))
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    return models

# Load models at startup
try:
    each_model = load_models()
    logger.info(f"Successfully loaded {len(each_model)} models")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

def _get_structure_type() -> int:
    """Get structure type from environment variable."""
    try:
        return int(os.environ.get("STRUCTURE_TYPE", 1))
    except ValueError:
        logger.warning("Invalid STRUCTURE_TYPE, defaulting to 1")
        return 1

def process_image(image_data: bytes) -> np.ndarray:
    """Process image data into numpy array."""
    try:
        pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(pil)
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise

def process_detection(model_name: str, model: YOLO, arr: np.ndarray) -> List[Dict[str, Any]]:
    """Process detections for a single model."""
    records = []
    try:
        results = model.predict(source=arr, conf=0.4, save=False, verbose=False)
        res = results[0]

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
                _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
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
    except Exception as e:
        logger.error(f"Error processing detections for model {model_name}: {str(e)}")
        raise
    return records

def handler(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    RunPod entrypoint.
    In:  {"input": {"image_base64": "...", "image_name": "..."}}
    Out: JSON array matching schema, including predictions_count
    """
    try:
        inp = event.get("input", {})
        b64 = inp.get("image_base64")
        image_name = inp.get("image_name", "")
        
        if not b64:
            return {"error": "No image_base64 provided"}

        # decode and load image
        img_bytes = base64.b64decode(b64)
        arr = process_image(img_bytes)

        records = []
        struct_type = _get_structure_type()

        # run each model
        for model_name, model in each_model:
            model_records = process_detection(model_name, model, arr)
            records.extend(model_records)

        # wrap in top-level object and add predictions_count
        output = {
            "image_name": image_name,
            "predictions_count": len(records),
            "predictions": records,
            "structure_type": struct_type
        }
        return [output]
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})