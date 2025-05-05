# 1) Base Python image
FROM python:3.10-slim

# 2) Set working dir
WORKDIR /app

# 3) Install dependencies + RunPod SDK
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

# 4) Copy your handler + weights
COPY rp_handler.py .
COPY weights/ ./weights/

# 5) (Optional) debug contents of weights/
RUN echo "---- weights folder contents ----" \
  && ls -R /app/weights \
  && echo "---------------------------------"

# 6) Entrypoint: start the RunPod serverless worker
CMD ["python", "-u", "-c", "import rp_handler, runpod; runpod.serverless.start({'handler': rp_handler.handler})"]
