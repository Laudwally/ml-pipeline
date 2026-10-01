"""Generate a synthetic customer-churn dataset for the pipeline to ingest."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
N_ROWS = 4000
SEED = 42


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 72, size=n_rows)
    monthly_charges = np.round(rng.uniform(20, 150, size=n_rows), 2)
    support_tickets = rng.poisson(1.2, size=n_rows)
    plan_type = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.5, 0.35, 0.15])
    total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) * rng.uniform(0.9, 1.0, size=n_rows), 2)

    plan_risk = {"basic": 0.15, "standard": 0.0, "premium": -0.2}
    churn_logit = (
        -1.5
        - 0.03 * tenure_months
        + 0.01 * monthly_charges
        + 0.35 * support_tickets
        + np.array([plan_risk[p] for p in plan_type])
        + rng.normal(0, 0.5, size=n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (rng.uniform(size=n_rows) < churn_prob).astype(int)

    now = dt.datetime.now(dt.timezone.utc)
    event_timestamp = [now - dt.timedelta(days=int(d)) for d in rng.integers(0, 30, size=n_rows)]

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "event_timestamp": event_timestamp,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "plan_type": plan_type,
            "churned": churned,
        }
    )
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    csv_path = RAW_DIR / "customers.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
