"""Convert validated raw CSV data into the parquet source Feast reads from."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "customers.csv"
FEATURE_PARQUET = ROOT / "feature_repo" / "data" / "customer_features.parquet"


def main() -> None:
    df = pd.read_csv(RAW_CSV, parse_dates=["event_timestamp"])
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    feature_cols = [
        "customer_id",
        "event_timestamp",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets",
        "plan_type",
    ]
    FEATURE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df[feature_cols].to_parquet(FEATURE_PARQUET, index=False)
    print(f"Wrote {len(df)} rows to {FEATURE_PARQUET}")


if __name__ == "__main__":
    main()
