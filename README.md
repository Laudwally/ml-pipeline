# End-to-End ML Pipeline

A churn-prediction pipeline that takes synthetic customer data all the way from ingestion to a
served model, with a data-quality gate and a drift/performance report along the way.

```
generate_data.py → validate.py → prepare_features.py → feast apply → train.py → serve.py
                                                                            │
                                                                        monitor.py
```

## Stack

- **Ingestion**: synthetic customer-churn CSV (`src/generate_data.py`) — swap for a real
  S3/CSV/JSON source without touching anything downstream of `data/raw/customers.csv`.
- **Validation**: [Great Expectations](https://greatexpectations.io/) (`src/validate.py`) — 9
  expectations on nullability, ranges, and categorical sets. Exits non-zero on failure, so it
  gates the pipeline in CI.
- **Feature store**: [Feast](https://feast.dev/) (`feature_repo/`) — local file offline store +
  SQLite online store. `train.py` pulls a point-in-time-correct training set via
  `get_historical_features`, so the same feature definitions could later back a low-latency
  `get_online_features` call for live serving.
- **Training/tracking**: scikit-learn `RandomForestClassifier` in a `ColumnTransformer` +
  `Pipeline`, logged to [MLflow](https://mlflow.org/) (params, metrics, model artifact) and
  registered in the MLflow Model Registry as `churn-classifier`.
- **Serving**: [FastAPI](https://fastapi.tiangolo.com/) (`src/serve.py`) loads
  `models:/churn-classifier/latest` from the registry on startup and exposes `POST /predict`.
- **Monitoring**: [Evidently](https://www.evidentlyai.com/) (`src/monitor.py`) splits the dataset
  into a reference/current pair, scores both with the registered model, and writes an HTML
  data-drift + classification-performance report.

## Running the pipeline

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python src/generate_data.py        # -> data/raw/customers.csv
python src/validate.py             # Great Expectations gate; exits 1 on failure
python src/prepare_features.py     # -> feature_repo/data/customer_features.parquet
cd feature_repo && feast apply && cd ..
python src/train.py                # trains, logs + registers model in MLflow
python src/monitor.py              # -> reports/monitoring_report.html

uvicorn src.serve:app --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months": 2, "monthly_charges": 95.5, "total_charges": 190.0, "support_tickets": 4, "plan_type": "basic"}'
```

Inspect experiment runs and the model registry with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Project layout

```
src/
  generate_data.py      synthetic data source
  validate.py            Great Expectations gate
  prepare_features.py    raw CSV -> Feast's parquet source
  train.py                Feast -> sklearn -> MLflow
  serve.py                FastAPI inference service
  monitor.py               Evidently drift/performance report
feature_repo/
  feature_store.yaml       local Feast project config
  features.py               entity + feature view definitions
tests/
  test_generate_data.py    unit tests for the synthetic data generator
```

## CI

`.github/workflows/ci.yml` runs the full pipeline on every push: install deps, unit tests,
generate data, validate, build features, `feast apply`, train, and generate the monitoring
report (uploaded as a build artifact).

## Known limitations

- `generate_data.py` is a stand-in for a real ingestion step (S3/warehouse pull) — the pipeline
  has no raw-data versioning beyond what's on disk at run time.
- Feast's online store is populated by `feast apply`/materialize but nothing currently calls
  `get_online_features` — `serve.py` takes features directly in the request body rather than
  looking them up by `customer_id`. Wiring that up is the natural next step toward a
  "real-time features" demo.
- MLflow and Feast both use local SQLite/file-based backends; there's no artifact store or
  registry server, so none of this is shared across machines as-is.
