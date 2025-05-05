# Use RunPod's Python function runtime so CMD ["rp_handler.handler"] works out of the box
FROM runpod/runner:python3.10-slim

# 1. Set working dir
WORKDIR /app

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your handler + weights
COPY rp_handler.py .
COPY weights/ ./weights/

# 4. (Temporary) debug the weights folder at build time
RUN echo "---- weights folder contents ----" \
  && ls -R /app/weights \
  && echo "---------------------------------"

# 5. Entry point — RunPod will invoke your handler function
CMD ["rp_handler.handler"]
