import base64
from rp_handler import handler

# load your image
with open("test_data/TUA2618_2.JPG","rb") as f:
    raw = f.read()

# call handler
event = {"input":{"image_base64": base64.b64encode(raw).decode()}}
resp = handler(event)

# save annotated output
ann_b64 = resp["output"]["annotated_base64"]
with open("annotated.png","wb") as out:
    out.write(base64.b64decode(ann_b64))
print("✅ annotated.png written")
