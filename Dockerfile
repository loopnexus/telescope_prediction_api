FROM python:3.10-slim

WORKDIR /app

# Install dependencies and RunPod CLI
COPY requirements.txt .
RUN pip install --no-cache-dir runpod && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python code
COPY rp_handler.py .
COPY rp_handler_wrapper.py .
COPY weights/ ./weights/

# Debug weights folder
RUN echo "---- weights folder contents ----" && ls -R /app/weights && echo "---------------------"

# Correct entrypoint
ENV RUNPOD_ENTRYPOINT=rp_handler_wrapper
CMD ["runpod", "start"]
