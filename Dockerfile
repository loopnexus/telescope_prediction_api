# 1. Base image
FROM python:3.10-slim

# 2. Set working dir
WORKDIR /app

# 3. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your handler + weights
COPY rp_handler.py .
COPY weights/ ./weights/

# 5. Tell RunPod which function to invoke
#    (RunPod will call rp_handler.handler(event))
CMD ["rp_handler.handler"]
