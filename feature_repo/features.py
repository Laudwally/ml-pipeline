from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

customer = Entity(name="customer_id", join_keys=["customer_id"])

# Feast resolves FileSource paths relative to the process's working directory,
# not the feature repo, so use an absolute path to make `feast apply` and
# `get_historical_features` work regardless of where they're invoked from.
DATA_PATH = Path(__file__).resolve().parent / "data" / "customer_features.parquet"

customer_source = FileSource(
    name="customer_features_source",
    path=str(DATA_PATH),
    timestamp_field="event_timestamp",
)

customer_features_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="support_tickets", dtype=Int64),
        Field(name="plan_type", dtype=String),
    ],
    online=True,
    source=customer_source,
)
