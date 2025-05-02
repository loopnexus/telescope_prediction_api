FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


COPY . .

CMD ["python3", "-u", "rp_handler.py"]
