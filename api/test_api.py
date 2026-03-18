"""
Simple test script for the FastAPI endpoints.
Run this after starting the API server.
"""

import requests
import numpy as np

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_spam_prediction():
    """Test spam detection endpoint."""
    print("Testing /predict/spam endpoint...")
    # Create dummy features (5000 features for spam dataset)
    features = [0.0] * 5000
    # Add some non-zero values to simulate an email
    features[100] = 1.0
    features[500] = 0.5
    
    response = requests.post(
        f"{BASE_URL}/predict/spam",
        json={"features": features}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']} (0=Not Spam, 1=Spam)")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Model Version: {result.get('model_version', 'N/A')}\n")
    else:
        print(f"Error: {response.text}\n")
    return response.status_code == 200

def test_cluster_prediction():
    """Test customer clustering endpoint."""
    print("Testing /predict/cluster endpoint...")
    # Example customer features (Age, Income, Spending Score, etc.)
    features = [25.0, 50000.0, 2.0, 1.0]
    
    response = requests.post(
        f"{BASE_URL}/predict/cluster",
        json={"features": features}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Cluster: {result['cluster']}")
        print(f"Distance to Centroid: {result['distance_to_centroid']:.4f}")
        print(f"Model Version: {result.get('model_version', 'N/A')}\n")
    else:
        print(f"Error: {response.text}\n")
    return response.status_code == 200

def test_fraud_prediction():
    """Test fraud detection endpoint."""
    print("Testing /predict/fraud endpoint...")
    # Create dummy transaction features (30 features for credit card dataset)
    features = np.random.randn(30).tolist()
    
    response = requests.post(
        f"{BASE_URL}/predict/fraud",
        json={"features": features}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Is Fraud: {result['is_fraud']} (0=Normal, 1=Fraud)")
        print(f"Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"Fraud Probability: {result['probability']:.4f}")
        print(f"Model Version: {result.get('model_version', 'N/A')}\n")
    else:
        print(f"Error: {response.text}\n")
    return response.status_code == 200

def test_batch_spam():
    """Test batch spam prediction."""
    print("Testing /predict/batch/spam endpoint...")
    batch = [
        {"features": [0.0] * 5000},
        {"features": [1.0] * 5000},
        {"features": [0.5] * 5000}
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch/spam",
        json=batch
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results)} predictions")
        for i, result in enumerate(results):
            print(f"  Sample {i+1}: Prediction={result['prediction']}, "
                  f"Probability={result['probability']:.4f}")
        print()
    else:
        print(f"Error: {response.text}\n")
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 60)
    print("FastAPI Endpoint Testing")
    print("=" * 60)
    print("Make sure the API server is running: uvicorn app:app --reload\n")
    
    try:
        # Test all endpoints
        test_health()
        test_spam_prediction()
        test_cluster_prediction()
        test_fraud_prediction()
        test_batch_spam()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Please start the server first:")
        print("  cd api")
        print("  uvicorn app:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")
