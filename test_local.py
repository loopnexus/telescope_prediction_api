# test_local.py
import base64
import json
from pathlib import Path
from rp_handler_image import handler

# 1. load your image
img_path = Path(r"C:\Dev\telescope_prediction_api\test_data\TUA2618_2.JPG")
with open(img_path, "rb") as f:
    raw = f.read()

# 2. wrap & call your handler
event = {
    "input": {
        "image_base64": base64.b64encode(raw).decode("utf8"),
        "image_name": img_path.name
    }
}
resp = handler(event)

# 3. pretty-print the JSON output
print(json.dumps(resp, indent=2))
