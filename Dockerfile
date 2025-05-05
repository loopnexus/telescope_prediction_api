FROM python:3.10-slim

WORKDIR /app
COPY . .

# install everything (see below for requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# launch the serverless handler
CMD ["python", "-u", "rp_handler.py"]
