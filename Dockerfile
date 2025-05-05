# ─── 1) Use RunPod's function runner base ─────────────────────────────────────
# This image already bundles the RunPod runtime, so CMD ["rp_handler.handler"] works.
FROM ghcr.io/runpod/runner:latest

# ─── 2) Set working dir ────────────────────────────────────────────────────────
WORKDIR /app

# ─── 3) Install your Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── 4) Copy handler + weights ─────────────────────────────────────────────────
COPY rp_handler.py .
COPY weights/ ./weights/

# ─── 5) Debug: verify weights made it into the image ───────────────────────────
RUN echo "---- weights folder contents ----" \
  && ls -R /app/weights \
  && echo "---------------------------------"

# ─── 6) Tell RunPod which function to invoke ───────────────────────────────────
# The runner will import your module and call `handler(event)` on every job.
CMD ["rp_handler.handler"]
