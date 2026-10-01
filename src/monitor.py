"""Generate a data-drift + classification-performance report with Evidently.

Splits the dataset in two to simulate a "reference" batch (what the model was
trained on) vs a "current" batch (freshly arrived data), scores both with the
registered model, and writes an HTML report.
"""

from __future__ import annotations

from pathlib import Path

import mlflow.sklearn
import pandas as pd
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report
from evidently import ColumnMapping

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "customers.csv"
REPORTS_DIR = ROOT / "reports"
MODEL_NAME = "churn-classifier"

FEATURE_COLS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "plan_type"]


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

    df = pd.read_csv(RAW_CSV).sort_values("event_timestamp").reset_index(drop=True)
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    for frame in (reference, current):
        frame["prediction"] = model.predict(frame[FEATURE_COLS])

    column_mapping = ColumnMapping(
        target="churned",
        prediction="prediction",
        numerical_features=["tenure_months", "monthly_charges", "total_charges", "support_tickets"],
        categorical_features=["plan_type"],
    )

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "monitoring_report.html"
    report.save_html(str(out_path))
    print(f"Wrote monitoring report to {out_path}")


if __name__ == "__main__":
    main()
