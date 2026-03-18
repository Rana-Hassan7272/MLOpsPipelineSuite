# FastAPI Model Serving API

Production-ready API for serving all three ML models.

## Features

- **Logistic Regression**: Spam detection endpoint
- **K-Means**: Customer clustering endpoint
- **Isolation Forest**: Fraud detection endpoint
- **Model Registry Integration**: Automatically loads models from MLflow
- **Batch Predictions**: Support for batch processing
- **Health Checks**: Monitor API and model status
- **Auto Documentation**: Swagger UI at `/docs`

## Endpoints

### Health Check
```
GET /health
```

### Spam Detection
```
POST /predict/spam
Body: {"features": [0.0, 1.0, ...]}  # 5000 features
Response: {"prediction": 0, "probability": 0.23, "model_version": "1"}
```

### Customer Clustering
```
POST /predict/cluster
Body: {"features": [25.0, 50000.0, 2.0, 1.0]}  # Customer features
Response: {"cluster": 2, "distance_to_centroid": 0.45, "model_version": "1"}
```

### Fraud Detection
```
POST /predict/fraud
Body: {"features": [0.0, 1.0, ...]}  # 30 transaction features
Response: {"is_fraud": 0, "anomaly_score": 0.12, "probability": 0.15, "model_version": "1"}
```

### Batch Spam Detection
```
POST /predict/batch/spam
Body: [{"features": [...]}, {"features": [...]}, ...]
```

## Running the API

### Development Mode
```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Accessing the API

- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

## Model Loading

The API automatically loads models from MLflow Model Registry on startup:
- Looks for models in "Staging" stage first
- Falls back to "Production" stage
- Models must be registered in MLflow before serving

## Testing

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Spam prediction
curl -X POST http://localhost:8000/predict/spam \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0] * 5000}'
```

### Using Python
```python
import requests

# Spam detection
response = requests.post(
    "http://localhost:8000/predict/spam",
    json={"features": [0.0] * 5000}
)
print(response.json())
```

## Error Handling

- **503 Service Unavailable**: Model not loaded
- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Prediction error

## Notes

- Models are loaded from MLflow Model Registry on startup
- If models are not found, endpoints return 503 errors
- Input features should match the training data format
- For fraud detection, features are automatically scaled if scaler is available
