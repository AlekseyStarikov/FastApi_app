FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install  -r /app/requirements.txt

COPY . .
ENTRYPOINT ["python3","/app/main.py"]
