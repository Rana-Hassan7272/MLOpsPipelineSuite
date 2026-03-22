FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the full project so Docker can resolve imports and load MLflow artifacts.
# (In docker-compose we also mount the repo, which keeps `mlflow.db` and artifacts in sync.)
COPY . /app

EXPOSE 8000

RUN mkdir -p /tmp/matplotlib /app/experiments /app/data_pipeline/processed_data \
    && chmod -R 777 /tmp/matplotlib \
    && chown -R 1000:1000 /app
    
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
