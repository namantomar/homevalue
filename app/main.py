"""
HomeValue — house price prediction API.

A FastAPI service that predicts a California house's median value from
8 census-block features, using a scikit-learn model trained by train.py
(saved at model/model.pkl).

The dataset target is the median house value in units of $100,000, so
every prediction is converted to dollars (x 100,000) before returning it.

Endpoints:
    GET  /          - small HTML form to try a prediction in the browser
    GET  /health    - liveness check + model status
    GET  /features  - the 8 input features, with descriptions and ranges
    POST /predict   - predict the median house value for one block

Run:
    uvicorn app.main:app --reload
"""

import os
from functools import lru_cache

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

app = FastAPI(
    title="HomeValue — California House Price Prediction",
    description=(
        "Predicts a California house's median value from 8 census-block "
        "features, using a scikit-learn model trained on the California "
        "Housing dataset."
    ),
    version="1.0.0",
)

# The 8 features, in the order the model was trained on.
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

FEATURE_INFO = {
    "MedInc": {
        "description": "Median income in the block group, in tens of "
                       "thousands of dollars (e.g. 3.5 = $35,000).",
        "typical_range": [0.5, 15.0],
    },
    "HouseAge": {
        "description": "Median age of the houses in the block group, "
                       "in years.",
        "typical_range": [1, 52],
    },
    "AveRooms": {
        "description": "Average number of rooms per household.",
        "typical_range": [1, 10],
    },
    "AveBedrms": {
        "description": "Average number of bedrooms per household.",
        "typical_range": [0.5, 2.5],
    },
    "Population": {
        "description": "Population of the block group.",
        "typical_range": [3, 35_685],
    },
    "AveOccup": {
        "description": "Average number of household members.",
        "typical_range": [0.6, 10],
    },
    "Latitude": {
        "description": "Latitude of the block group (California: "
                       "32.5–42.0).",
        "typical_range": [32.5, 42.0],
    },
    "Longitude": {
        "description": "Longitude of the block group (California: "
                       "-124.3 to -114.3).",
        "typical_range": [-124.3, -114.3],
    },
}


class PredictRequest(BaseModel):
    """The 8 housing features for one census block group."""

    MedInc: float = Field(..., gt=0, description="Median income ($10k units)")
    HouseAge: float = Field(..., ge=0, description="Median house age (years)")
    AveRooms: float = Field(..., gt=0, description="Avg rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Avg bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Avg household members")
    Latitude: float = Field(..., ge=32.0, le=42.5, description="Latitude")
    Longitude: float = Field(..., ge=-125.0, le=-114.0, description="Longitude")


class PredictResponse(BaseModel):
    predicted_value: float  # in $100k units, as the model outputs
    predicted_value_dollars: float


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and cache it (lazy, so /health works
    even before the model file exists — it just reports not loaded)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    model = load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": "scikit-learn regressor (see model/metrics.json)",
    }


@app.get("/features")
def features():
    """List the 8 input features with descriptions and typical ranges."""
    return {
        "features": [
            {
                "name": name,
                "description": FEATURE_INFO[name]["description"],
                "typical_range": FEATURE_INFO[name]["typical_range"],
            }
            for name in FEATURES
        ]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = load_model()
    if model is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Model not trained yet — run `python train.py` first.",
        )
    # Features must be in the same order the model was trained on.
    X = [[getattr(req, name) for name in FEATURES]]
    value_100k = float(model.predict(X)[0])
    return PredictResponse(
        predicted_value=round(value_100k, 4),
        # Dataset target is in $100k units, so convert to dollars.
        predicted_value_dollars=round(value_100k * 100_000, 2),
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeValue — California House Price Prediction</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.6rem; }
  p { color: #555; }
  form { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
  label { display: flex; flex-direction: column; font-size: 0.9rem; gap: 0.25rem; }
  input { padding: 0.5rem; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; }
  button { grid-column: 1 / -1; padding: 0.7rem; font-size: 1rem; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  #result { margin-top: 1.5rem; padding: 1rem; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; font-size: 1.25rem; display: none; }
  #error { margin-top: 1.5rem; padding: 1rem; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; color: #991b1b; display: none; }
  code { background: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 4px; }
  .hint { font-size: 0.85rem; color: #777; }
</style>
</head>
<body>
<h1>🏠 HomeValue</h1>
<p>Predict a California house's median value from 8 census-block features.
Model: scikit-learn regressor trained on the California Housing dataset.
Try the defaults, or tweak the numbers — the API is <code>POST /predict</code>.</p>
<form id="form">
  <label>Median income ($10k) <span class="hint">e.g. 3.85</span><input name="MedInc" type="number" step="any" value="3.85" required></label>
  <label>Median house age (years)<input name="HouseAge" type="number" step="any" value="37" required></label>
  <label>Avg rooms per household<input name="AveRooms" type="number" step="any" value="6.3" required></label>
  <label>Avg bedrooms per household<input name="AveBedrms" type="number" step="any" value="1.1" required></label>
  <label>Population<input name="Population" type="number" step="any" value="1500" required></label>
  <label>Avg household members<input name="AveOccup" type="number" step="any" value="3.0" required></label>
  <label>Latitude<input name="Latitude" type="number" step="any" value="34.05" required></label>
  <label>Longitude<input name="Longitude" type="number" step="any" value="-118.25" required></label>
  <button type="submit">Predict price</button>
</form>
<div id="result"></div>
<div id="error"></div>
<script>
  const form = document.getElementById('form');
  const result = document.getElementById('result');
  const error = document.getElementById('error');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    result.style.display = 'none';
    error.style.display = 'none';
    const data = {};
    new FormData(form).forEach((v, k) => { data[k] = parseFloat(v); });
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Prediction failed');
      result.textContent = 'Predicted median value: $' +
        json.predicted_value_dollars.toLocaleString('en-US', { maximumFractionDigits: 0 });
      result.style.display = 'block';
    } catch (err) {
      error.textContent = 'Error: ' + err.message;
      error.style.display = 'block';
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
