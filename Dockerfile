# 1. use slim Python
FROM python:3.10-slim

WORKDIR /app

# 2. install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. copy handler
COPY rp_handler.py .

# 4. tell RunPod which function to invoke
CMD ["rp_handler.handler"]
