import os
import uuid
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()

# Load all YOLO segmentation models from the weights folder
models = {}
weights_dir = os.path.join(os.path.dirname(__file__), "weights")
for fname in os.listdir(weights_dir):
    if fname.endswith(".pt"):
        path = os.path.join(weights_dir, fname)
        model_key = os.path.splitext(fname)[0]
        models[model_key] = YOLO(path)

@app.get("/health")
async def health():
    # Return status and list of loaded model names
    return {"status": "ok", "models": list(models.keys())}

@app.post("/process")
async def process(images: List[UploadFile] = File(...)):
    results = []
    for image in images:
        content = await image.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Invalid image: {image.filename}")

        record = {"image_name": image.filename, "predictions": []}

        for name, model in models.items():
            try:
                result = model.predict(source=img, conf=0.5)[0]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{name} predict error: {e}")

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else []

            for box, score, cls_id, mask in zip(boxes, scores, classes, masks):
                x1, y1, x2, y2 = box.tolist()
                # Extract polygon as list of [x, y] pairs
                mask_bin = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pts = contours[0].reshape(-1, 2).tolist() if contours else []

                prediction = {
                    "prediction_id": str(uuid.uuid4()),
                    "ml_model_name": name,
                    "class_name": model.names[cls_id],
                    "confidence": float(score),
                    "bounding_box": [x1, y1, x2, y2],
                    "polygon": pts,
                }
                record["predictions"].append(prediction)

        results.append(record)

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
