FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler_wrapper.py .
COPY rp_handler.py .
COPY weights/ ./weights/

RUN echo "---- weights folder contents ----" \
  && ls -R /app/weights \
  && echo "---------------------------------"

ENV RUNPOD_ENTRYPOINT=rp_handler_wrapper

CMD ["runpod", "start"]
