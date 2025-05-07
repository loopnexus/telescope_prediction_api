# Use lightweight Python base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY rp_handler.py .

# Launch the RunPod handler
CMD ["python", "-u", "-c", "import rp_handler, runpod; runpod.serverless.start({'handler': rp_handler.handler})"]
