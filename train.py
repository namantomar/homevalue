"""
Train a house-price prediction model on the California Housing dataset.

Dataset: sklearn.datasets.fetch_california_housing
  - 20,640 samples of California census-block groups (1990 census).
  - Target: median house value for the block, in units of $100,000.
  - 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population,
    AveOccup, Latitude, Longitude.

We train two candidate regressors (RandomForest and GradientBoosting),
compare them on a held-out test set, and save the better one to
model/model.pkl. The saved model is kept under 10 MB so it can be
committed to a public GitHub repo.

Usage:
    python train.py
"""

import json
import os
import time

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
MAX_MODEL_MB = 10.0

# Feature names, in the order the dataset returns them.
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


def build_candidates():
    """Return the two candidate pipelines to compare."""
    return {
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=40,
                        max_depth=12,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate(pipeline, X_test, y_test):
    """Return (rmse, r2) for a fitted pipeline on the test set."""
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return rmse, r2_score(y_test, preds)


def main():
    print("Loading California Housing dataset ...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  target: median house value, in $100k units")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  train: {len(X_train)}, test: {len(X_test)}")

    results = {}
    for name, pipeline in build_candidates().items():
        print(f"\nTraining {name} ...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        rmse, r2 = evaluate(pipeline, X_test, y_test)
        results[name] = (pipeline, rmse, r2)
        print(f"  {name}: RMSE = {rmse:.4f} ($ {rmse * 100_000:,.0f}), "
              f"R² = {r2:.4f}  ({time.time() - start:.1f}s)")

    # Pick the model with the lowest test RMSE.
    best_name = min(results, key=lambda n: results[n][1])
    best_pipeline, best_rmse, best_r2 = results[best_name]
    print(f"\nBest model: {best_name} "
          f"(RMSE {best_rmse:.4f}, R² {best_r2:.4f})")

    # Feature importances of the winning model.
    importances = best_pipeline.named_steps["model"].feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda t: t[1], reverse=True)
    print("\nTop feature importances:")
    for feature, importance in ranked:
        print(f"  {feature:>11}: {importance:.4f}")

    # Save the model, shrinking it if it is over the size budget.
    # compress=3 (zlib) shrinks tree ensembles several-fold.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\nSaved {best_name} to {MODEL_PATH} ({size_mb:.2f} MB)")
    if size_mb > MAX_MODEL_MB:
        print(f"  WARNING: over the {MAX_MODEL_MB} MB budget — "
              "reduce n_estimators / max_depth and retrain.")

    metrics = {
        "model": best_name,
        "rmse_100k_units": round(best_rmse, 4),
        "rmse_dollars": round(best_rmse * 100_000, 2),
        "r2": round(best_r2, 4),
        "feature_importances": {f: round(float(i), 4) for f, i in ranked},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_size_mb": round(size_mb, 2),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
