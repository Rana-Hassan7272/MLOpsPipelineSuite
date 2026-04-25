def test_health_returns_200_when_ready(client_ready):
    response = client_ready.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["ready"] is True
    assert payload["missing_models"] == []


def test_health_returns_503_when_not_ready(client_not_ready):
    response = client_not_ready.get("/health")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["ready"] is False
    assert set(payload["missing_models"]) == {"logistic_regression", "kmeans", "isolation_forest"}


def test_predict_spam_contract(client_ready):
    response = client_ready.post("/predict/spam", json={"features": [0.0] * 5000})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"prediction", "probability", "model_version"}
    assert isinstance(payload["prediction"], int)
    assert isinstance(payload["probability"], float)


def test_predict_spam_validation_error(client_ready):
    # Missing required key should fail contract validation.
    response = client_ready.post("/predict/spam", json={"invalid": []})
    assert response.status_code == 422


def test_predict_cluster_contract(client_ready):
    response = client_ready.post("/predict/cluster", json={"features": [25.0, 50000.0, 2.0, 1.0]})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"cluster", "distance_to_centroid", "model_version"}
    assert isinstance(payload["cluster"], int)
    assert isinstance(payload["distance_to_centroid"], float)


def test_predict_fraud_contract(client_ready):
    response = client_ready.post("/predict/fraud", json={"features": [0.0] * 30})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"is_fraud", "anomaly_score", "probability", "model_version"}
    assert payload["is_fraud"] in (0, 1)
