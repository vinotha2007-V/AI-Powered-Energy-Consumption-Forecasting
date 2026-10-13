"""
Unit tests for the Flask Prediction API endpoints and request validation.
"""

import pytest
import json
from api import app


@pytest.fixture
def client():
    """Provides Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_endpoint(client):
    """Verifies root endpoint returns service catalog."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert "service" in data
    assert "endpoints" in data
    assert "GET /health" in data["endpoints"]


def test_health_endpoint(client):
    """Verifies health check endpoint returns 200 or 503 with structured payload."""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_invalid_history_length(client):
    """Verifies /predict rejects arrays with fewer than 168 readings."""
    short_payload = {
        "history_kwh": [1.2, 1.4, 1.1]  # Only 3 values
    }
    response = client.post("/predict", json=short_payload)
    assert response.status_code in [400, 503]
    data = response.get_json()
    assert "error" in data


def test_predict_invalid_non_numeric_history(client):
    """Verifies /predict rejects non-numeric entries."""
    bad_payload = {
        "history_kwh": ["not_a_number"] * 170
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code in [400, 503]
    data = response.get_json()
    assert "error" in data


def test_forecast_invalid_horizon_limits(client):
    """Verifies /forecast rejects horizons out of 1-168 range."""
    # Negative horizon
    resp_neg = client.post("/forecast", json={"horizon_hours": -5})
    assert resp_neg.status_code in [400, 503]

    # Excessive horizon > 168
    resp_large = client.post("/forecast", json={"horizon_hours": 500})
    assert resp_large.status_code in [400, 503]
