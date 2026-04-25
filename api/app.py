"""
FastAPI Application for ML Model Serving
=========================================
Production-ready API for serving all three ML models:
- Logistic Regression (Spam Detection)
- K-Means (Customer Clustering)
- Isolation Forest (Fraud Detection)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import joblib
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import custom algorithms
from algorithms.logistic_regression.logistic_regression import Hassan
from algorithms.isolation_forest.scratch_isolation_forest import HassanIsolationForest
from api.monitoring import MonitoringManager, REQUEST_COUNT, REQUEST_LATENCY

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Infrastructure API",
    description="Production API for Logistic Regression, K-Means, and Isolation Forest models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set MLflow tracking URI (env override lets Kubernetes point to shared volume path).
default_mlflow_uri = f"sqlite:///{project_root / 'mlflow.db'}"
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_mlflow_uri)
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Global model storage
models: Dict[str, Any] = {}
scalers: Dict[str, Any] = {}
monitor = MonitoringManager(project_root)


# ============================================================================
# Request/Response Models
# ============================================================================

class SpamPredictionRequest(BaseModel):
    """Request model for spam detection."""
    features: List[float] = Field(..., description="Feature vector (sparse format supported)")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 5000  # Example: 5000 features for spam dataset
            }
        }


class SpamPredictionResponse(BaseModel):
    """Response model for spam detection."""
    prediction: int = Field(..., description="Predicted class (0=Not Spam, 1=Spam)")
    probability: float = Field(..., description="Probability of being spam")
    model_version: Optional[str] = Field(None, description="Model version used")


class ClusterPredictionRequest(BaseModel):
    """Request model for customer clustering."""
    features: List[float] = Field(..., description="Feature vector for customer")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [25.0, 50000.0, 2.0, 1.0]  # Age, Income, Spending Score, etc.
            }
        }


class ClusterPredictionResponse(BaseModel):
    """Response model for customer clustering."""
    cluster: int = Field(..., description="Assigned cluster ID")
    distance_to_centroid: float = Field(..., description="Distance to cluster centroid")
    model_version: Optional[str] = Field(None, description="Model version used")


class FraudPredictionRequest(BaseModel):
    """Request model for fraud detection."""
    features: List[float] = Field(..., description="Transaction features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 30  # Example: 30 features for credit card dataset
            }
        }


class FraudPredictionResponse(BaseModel):
    """Response model for fraud detection."""
    is_fraud: int = Field(..., description="Prediction (0=Normal, 1=Fraud)")
    anomaly_score: float = Field(..., description="Anomaly score (lower = more anomalous)")
    probability: float = Field(..., description="Fraud probability")
    model_version: Optional[str] = Field(None, description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ready: bool
    models_loaded: Dict[str, bool]
    mlflow_connected: bool
    missing_models: List[str]


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model_from_registry(model_name: str, stage: str = "Staging") -> Optional[Any]:
    """Load model from MLflow Model Registry."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get latest version in specified stage
        model_version = client.get_latest_versions(model_name, stages=[stage])
        if not model_version:
            # Try Production stage
            model_version = client.get_latest_versions(model_name, stages=["Production"])
        if not model_version:
            # Try any stage
            model_version = client.get_latest_versions(model_name)
        
        if not model_version:
            return None, None
        
        model_uri = f"models:/{model_name}/{model_version[0].version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model, model_version[0].version
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None


def load_model_from_experiment(experiment_name: str, run_name: str) -> Optional[Any]:
    """Load model from a specific experiment run (fallback if not in registry)."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None, None
        
        # Search for the run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
            order_by=["start_time DESC"]
        )
        
        if len(runs) == 0:
            return None, None
        
        run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            return model, run_id
        except Exception:
            # Model might not be logged in this run
            return None, None
    except Exception as e:
        print(f"Error loading from experiment: {e}")
        return None, None


def load_models():
    """Load all models from MLflow Model Registry, with fallback to experiment runs."""
    global models, scalers
    
    # Load Logistic Regression
    try:
        # Try registry first
        result = load_model_from_registry("LogisticRegression_sklearn", "Staging")
        if result[0] is None:
            # Fallback to experiment run
            print("  Trying to load from experiment run...")
            result = load_model_from_experiment("Logistic_Regression_Comparison", "Logistic_Regression_sklearn")
        
        if result[0] is not None:
            models["logistic_regression"] = result[0]
            models["logistic_regression_version"] = result[1]
            print(f"✓ Loaded LogisticRegression_sklearn (version: {result[1]})")
        else:
            print("⚠ LogisticRegression_sklearn not found (register model or run experiment first)")
    except Exception as e:
        print(f"⚠ Could not load LogisticRegression: {e}")
    
    # Load K-Means
    try:
        # Try registry first
        result = load_model_from_registry("KMeans_sklearn", "Staging")
        if result[0] is None:
            # Fallback to experiment run
            print("  Trying to load from experiment run...")
            result = load_model_from_experiment("KMeans_Comparison", "KMeans_sklearn")
        
        if result[0] is not None:
            models["kmeans"] = result[0]
            models["kmeans_version"] = result[1]
            print(f"✓ Loaded KMeans_sklearn (version: {result[1]})")
        else:
            print("⚠ KMeans_sklearn not found (register model or run experiment first)")
    except Exception as e:
        print(f"⚠ Could not load KMeans: {e}")
    
    # Load Isolation Forest
    try:
        # Try registry first
        result = load_model_from_registry("IsolationForest_sklearn", "Staging")
        if result[0] is None:
            # Fallback to experiment run
            print("  Trying to load from experiment run...")
            result = load_model_from_experiment("Isolation_Forest_Comparison", "Isolation_Forest_sklearn")
        
        if result[0] is not None:
            models["isolation_forest"] = result[0]
            models["isolation_forest_version"] = result[1]
            print(f"✓ Loaded IsolationForest_sklearn (version: {result[1]})")
        else:
            print("⚠ IsolationForest_sklearn not found (register model or run experiment first)")
    except Exception as e:
        print(f"⚠ Could not load IsolationForest: {e}")
    
    # Load scalers if available
    scaler_path = project_root / "data_pipeline" / "processed_data" / "scaler.pkl"
    if scaler_path.exists():
        try:
            scalers["default"] = joblib.load(scaler_path)
            print("✓ Loaded default scaler")
        except Exception as e:
            print(f"⚠ Could not load scaler: {e}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("=" * 60)
    print("Loading models from MLflow Model Registry...")
    print("=" * 60)
    load_models()
    baseline_status = monitor.load_baselines()
    for model_name in ["logistic_regression", "kmeans", "isolation_forest"]:
        monitor.set_model_loaded(model_name, model_name in models)
    print(f"Baseline stats loaded: {baseline_status}")
    print("=" * 60)
    print("API ready to serve predictions!")
    print("=" * 60)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "MLOps Infrastructure API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "spam_detection": "/predict/spam",
            "customer_clustering": "/predict/cluster",
            "fraud_detection": "/predict/fraud",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Readiness endpoint.

    Returns 200 only when all required models are loaded and MLflow is reachable.
    Returns 503 when serving dependencies are not ready.
    """
    required_models = ["logistic_regression", "kmeans", "isolation_forest"]
    loaded_map = {name: name in models for name in required_models}
    missing_models = [name for name, loaded in loaded_map.items() if not loaded]

    mlflow_connected = False
    try:
        from mlflow.tracking import MlflowClient
        # Lightweight connectivity check against tracking backend.
        MlflowClient().search_experiments(max_results=1)
        mlflow_connected = True
    except Exception:
        mlflow_connected = False

    ready = (len(missing_models) == 0) and mlflow_connected
    payload = HealthResponse(
        status="healthy" if ready else "unhealthy",
        ready=ready,
        models_loaded=loaded_map,
        mlflow_connected=mlflow_connected,
        missing_models=missing_models,
    )

    if ready:
        return payload
    serialized = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return JSONResponse(status_code=503, content=serialized)


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
    return response


@app.post("/predict/spam", response_model=SpamPredictionResponse, tags=["Predictions"])
async def predict_spam(request: SpamPredictionRequest):
    """
    Predict if an email is spam using Logistic Regression.
    
    - **features**: Feature vector (sparse format, typically 5000 features for spam dataset)
    - Returns: Prediction (0=Not Spam, 1=Spam) and probability
    """
    if "logistic_regression" not in models:
        raise HTTPException(
            status_code=503,
            detail="Logistic Regression model not loaded. Please train and register the model first."
        )
    
    try:
        # Convert to numpy array - sklearn expects float64
        features = np.array(request.features, dtype=np.float64)
        
        # Handle sparse format (if features are sparse, convert to dense)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get prediction
        model = models["logistic_regression"]
        probability = model.predict_proba(features)[0, 1]  # Probability of spam
        prediction = int(probability >= 0.5)
        monitor.observe_prediction("logistic_regression", prediction, float(probability), "probability")
        monitor.compute_and_record_drift("logistic_regression", features[0])
        
        return SpamPredictionResponse(
            prediction=prediction,
            probability=float(probability),
            model_version=models.get("logistic_regression_version")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/cluster", response_model=ClusterPredictionResponse, tags=["Predictions"])
async def predict_cluster(request: ClusterPredictionRequest):
    """
    Predict customer cluster using K-Means.
    
    - **features**: Feature vector for customer (e.g., Age, Income, Spending Score)
    - Returns: Assigned cluster ID and distance to centroid
    """
    if "kmeans" not in models:
        raise HTTPException(
            status_code=503,
            detail="K-Means model not loaded. Please train and register the model first."
        )
    
    try:
        # Convert to numpy array - sklearn expects float64
        features = np.array(request.features, dtype=np.float64).reshape(1, -1)
        
        # Get prediction
        model = models["kmeans"]
        cluster = int(model.predict(features)[0])
        monitor.observe_prediction("kmeans", cluster)
        monitor.compute_and_record_drift("kmeans", features[0])
        
        # Calculate distance to centroid
        centroid = model.cluster_centers_[cluster]
        distance = float(np.linalg.norm(features[0] - centroid))
        
        return ClusterPredictionResponse(
            cluster=cluster,
            distance_to_centroid=distance,
            model_version=models.get("kmeans_version")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/fraud", response_model=FraudPredictionResponse, tags=["Predictions"])
async def predict_fraud(request: FraudPredictionRequest):
    """
    Predict if a transaction is fraudulent using Isolation Forest.
    
    - **features**: Transaction features (typically 30 features for credit card dataset)
    - Returns: Fraud prediction, anomaly score, and probability
    """
    if "isolation_forest" not in models:
        raise HTTPException(
            status_code=503,
            detail="Isolation Forest model not loaded. Please train and register the model first."
        )
    
    try:
        # Convert to numpy array - sklearn expects float64
        features = np.array(request.features, dtype=np.float64).reshape(1, -1)
        
        # Scale features if scaler available
        if "default" in scalers:
            features = scalers["default"].transform(features)
        
        # Get prediction
        model = models["isolation_forest"]
        prediction = model.predict(features)[0]
        is_fraud = int(prediction == -1)  # -1 = anomaly (fraud), 1 = normal
        
        # Get anomaly score (lower = more anomalous)
        score = model.score_samples(features)[0]
        anomaly_score = float(-score)  # Negative because lower scores = more anomalous
        
        # Convert to probability (normalize score to [0, 1])
        # Lower anomaly score = higher fraud probability
        score_min, score_max = -0.5, 0.5  # Typical range
        probability = float(np.clip((anomaly_score - score_min) / (score_max - score_min), 0, 1))
        monitor.observe_prediction("isolation_forest", is_fraud, probability, "fraud_probability")
        monitor.compute_and_record_drift("isolation_forest", features[0])
        
        return FraudPredictionResponse(
            is_fraud=is_fraud,
            anomaly_score=anomaly_score,
            probability=probability,
            model_version=models.get("isolation_forest_version")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch/spam", tags=["Batch Predictions"])
async def predict_spam_batch(requests: List[SpamPredictionRequest]):
    """Batch prediction for spam detection."""
    if "logistic_regression" not in models:
        raise HTTPException(status_code=503, detail="Logistic Regression model not loaded")
    
    try:
        features_list = [np.array(req.features, dtype=np.float64) for req in requests]
        features = np.vstack(features_list)
        
        model = models["logistic_regression"]
        probabilities = model.predict_proba(features)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        for pred, prob, feat in zip(predictions, probabilities, features):
            monitor.observe_prediction("logistic_regression", int(pred), float(prob), "probability")
            monitor.compute_and_record_drift("logistic_regression", feat)
        
        return [
            {
                "prediction": int(pred),
                "probability": float(prob),
                "model_version": models.get("logistic_regression_version")
            }
            for pred, prob in zip(predictions, probabilities)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
