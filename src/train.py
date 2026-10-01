"""Build a point-in-time-correct training set from Feast, train a model, and
log/register it with MLflow.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "customers.csv"
FEATURE_REPO = ROOT / "feature_repo"

FEATURES = [
    "customer_features:tenure_months",
    "customer_features:monthly_charges",
    "customer_features:total_charges",
    "customer_features:support_tickets",
    "customer_features:plan_type",
]
NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
CATEGORICAL_COLS = ["plan_type"]
MODEL_NAME = "churn-classifier"


def build_training_frame() -> pd.DataFrame:
    labels = pd.read_csv(RAW_CSV, parse_dates=["event_timestamp"])[["customer_id", "event_timestamp", "churned"]]
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"], utc=True)

    store = FeatureStore(repo_path=str(FEATURE_REPO))
    training_df = store.get_historical_features(entity_df=labels, features=FEATURES).to_df()
    return training_df


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("churn")

    df = build_training_frame()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[("plan_type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_params({"n_estimators": 200, "max_depth": 8, "train_rows": len(X_train)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        print(f"Run {run.info.run_id} metrics: {metrics}")


if __name__ == "__main__":
    main()
