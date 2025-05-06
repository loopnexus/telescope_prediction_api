FROM runpod/base:0.6.3-cuda11.8.0

RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

COPY rp_handler.py /app/rp_handler.py
COPY rp_handler_wrapper.py /app/handler.py      
COPY weights/ /app/weights/

WORKDIR /app

RUN echo "---- weights folder contents ----" && ls -R /app/weights && echo "----------------"

CMD ["python", "-u", "handler.py"]
