"""
API tests for HomeValue.

Run:  pytest
The tests use FastAPI's TestClient (in-process, no server needed).
They expect the model to be trained first (python train.py) — the
/predict test needs model/model.pkl to exist.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Import the app from the project root (tests run from the repo root).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app  # noqa: E402

client = TestClient(app)

# A realistic sample: a typical California block group.
SAMPLE = {
    "MedInc": 3.85,
    "HouseAge": 37.0,
    "AveRooms": 6.3,
    "AveBedrms": 1.1,
    "Population": 1500.0,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_positive_dollar_value():
    res = client.post("/predict", json=SAMPLE)
    assert res.status_code == 200
    body = res.json()
    assert "predicted_value" in body
    assert "predicted_value_dollars" in body
    # A California median house value should be a plausible dollar amount.
    assert body["predicted_value_dollars"] > 0
    assert 50_000 < body["predicted_value_dollars"] < 1_000_000
    # Dollars = $100k units x 100,000 (within rounding: predicted_value
    # is rounded to 4 decimals, so allow a few dollars of slack).
    assert body["predicted_value_dollars"] == pytest.approx(
        body["predicted_value"] * 100_000, abs=10.0
    )


def test_features_lists_eight():
    res = client.get("/features")
    assert res.status_code == 200
    features = res.json()["features"]
    assert len(features) == 8
    names = {f["name"] for f in features}
    assert names == {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    }
    # Each feature documents itself.
    for f in features:
        assert f["description"]
        assert len(f["typical_range"]) == 2


def test_invalid_input_returns_422():
    # Missing required fields.
    res = client.post("/predict", json={"MedInc": 3.85})
    assert res.status_code == 422
    # Wrong type for a field.
    bad = dict(SAMPLE, MedInc="not-a-number")
    res = client.post("/predict", json=bad)
    assert res.status_code == 422


def test_index_serves_html_form():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "<form" in res.text
    assert "/predict" in res.text
