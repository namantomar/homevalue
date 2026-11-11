# 🏠 HomeValue — California House Price Prediction

A FastAPI service that predicts a California house's median value from 8
census-block features, using a scikit-learn model trained on the
[California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

## How it works

- **Dataset:** California Housing — 20,640 census block groups from the
  1990 California census. Target is the median house value for the block,
  in units of **$100,000** (so a target of `2.0` means $200,000).
- **Model:** `StandardScaler` + `GradientBoostingRegressor`
  (200 trees, max depth 3, learning rate 0.1). Two candidates were trained
  and compared on a held-out test set — GradientBoosting won with the
  better RMSE once the model was kept under the 10 MB size budget
  (a RandomForest scored a marginally better RMSE of 0.5079 but its
  saved model was ~34 MB).
- **Results on the test set** (4,128 samples, 20% hold-out):
  - **RMSE: 0.5114 → $51,138**
  - **R²: 0.8004**
- **Top feature importances:** MedInc 0.59, AveOccup 0.12,
  Longitude 0.12, Latitude 0.10, HouseAge 0.04.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/` | HTML form — try a prediction in the browser |
| GET | `/health` | Liveness check + whether the model is loaded |
| GET | `/features` | The 8 input features, with descriptions and typical ranges |
| POST | `/predict` | Predict the median house value for one block group |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.85, "HouseAge": 37.0, "AveRooms": 6.3,
    "AveBedrms": 1.1, "Population": 1500.0, "AveOccup": 3.0,
    "Latitude": 34.05, "Longitude": -118.25
  }'
```

```json
{"predicted_value": 2.226, "predicted_value_dollars": 222598.14}
```

`predicted_value` is in the dataset's $100k units; `predicted_value_dollars`
is the same prediction converted to dollars.

### The 8 features

| Feature | Meaning |
| ------- | ------- |
| `MedInc` | Median income in the block group, in $10k units |
| `HouseAge` | Median age of the houses, in years |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Population of the block group |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude of the block group |
| `Longitude` | Longitude of the block group |

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py        # trains the model, saves it to model/model.pkl
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 — or try the API docs at
http://127.0.0.1:8000/docs.

## Tests

```bash
pytest
```

5 tests: `/health`, `/predict` returns a positive dollar value,
`/features` lists all 8 features, invalid input returns 422, and the
index page serves the HTML form. The model must be trained first
(`python train.py`), since the prediction test loads `model/model.pkl`.

## Project structure

```
homevalue/
├── train.py            # trains both candidate models, saves the best
├── app/
│   └── main.py         # FastAPI app + built-in HTML UI
├── model/
│   ├── model.pkl       # trained model (created by train.py, gitignored)
│   └── metrics.json    # test metrics from the last training run
├── tests/
│   └── test_api.py     # pytest API tests
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
