# --------------------------
# test-api.py
# Unit tests for the FastAPI endpoints.
# --------------------------
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

# -------------------------
# Test the health endpoint to ensure the API is running.
# -------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# -------------------------
# Test the prediction endpoint with a sample payload.
# -------------------------
def test_predict_endpoint():
    sample_payload = {
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving_accounts": "moderate",
        "Checking_account": "little",
        "Credit_amount": 5000,
        "Duration": 24,
        "Purpose": "car"
    }

    response = client.post("/predict", json=sample_payload)
    print(response.json())

    assert response.status_code == 200

    data = response.json()

    # Check expected keys exist
    assert "prediction" in data
    assert "probability" in data

    # Probability should be between 0 and 1
    assert 0 <= data["probability"] <= 1
    # Prediction should be either "High Risk" or "Low Risk"
    assert data["prediction"] in ["High Risk", "Low Risk"]