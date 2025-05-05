# test_local.py
import base64
from rp_handler import handler

# 1. load your image (updated path)
img_path = r"C:\Dev\telescope_prediction_api\test_data\TUA2618_2.JPG"
with open(img_path, "rb") as f:
    raw = f.read()

# 2. wrap & call your handler
event = {"input": {"image_base64": base64.b64encode(raw).decode()}}
resp = handler(event)

# 3. decode & save mask to disk
mask_b64 = resp["output"]["mask_base64"]
with open("mask.png", "wb") as out:
    out.write(base64.b64decode(mask_b64))

print("✅ mask.png written")
