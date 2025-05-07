# Use RunPod's base image with CUDA support
FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Copy source code and weights
COPY rp_handler.py /handler.py
COPY weights/ /weights/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STRUCTURE_TYPE=1

# Run the handler
CMD python -u /handler.py
