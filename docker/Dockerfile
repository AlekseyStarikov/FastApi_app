FROM python:3.10-slim

# Установка зависимостей PostgreSQL
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev

# Установка MLflow и зависимостей
RUN pip install mlflow boto3 psycopg2

# Точка входа
CMD ["mlflow", "server", "--host", "0.0.0.0"]