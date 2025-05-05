# test_local.py

import base64
import json
from pathlib import Path
from rp_handler import handler  # your new handler module

def main():
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

    # 3. save the JSON output to file in 'output' directory
    output_dir = Path(r"C:\Dev\telescope_prediction_api\output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.json"
    with open(out_path, "w", encoding="utf8") as out_file:
        json.dump(resp, out_file, indent=2)

    print(f"✅ Predictions written to {out_path}")

if __name__ == "__main__":
    main()