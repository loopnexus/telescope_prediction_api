FROM python:3.10-slim

WORKDIR /app

# install only what we need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# drop in handler + model
COPY rp_handler.py .
# make sure you have downloaded yolov8n-seg.pt into this folder
COPY yolov8n-seg.pt .

# tell RunPod to invoke rp_handler.handler
CMD ["rp_handler.handler"]
