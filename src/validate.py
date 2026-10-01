"""Validate the raw ingested data with Great Expectations before it's used
for feature building or training. Exits non-zero (failing the pipeline/CI)
if any expectation is violated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import great_expectations as gx

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "customers.csv"


def main() -> None:
    context = gx.get_context()
    validator = context.sources.pandas_default.read_csv(str(RAW_CSV))

    validator.expect_column_values_to_not_be_null("customer_id")
    validator.expect_column_values_to_be_unique("customer_id")
    validator.expect_column_values_to_not_be_null("event_timestamp")

    validator.expect_column_values_to_be_between("tenure_months", min_value=0, max_value=120)
    validator.expect_column_values_to_be_between("monthly_charges", min_value=0, max_value=1000)
    validator.expect_column_values_to_be_between("total_charges", min_value=0)
    validator.expect_column_values_to_be_between("support_tickets", min_value=0, max_value=50)
    validator.expect_column_values_to_be_in_set("plan_type", ["basic", "standard", "premium"])
    validator.expect_column_values_to_be_in_set("churned", [0, 1])

    result = validator.validate()
    print(f"Great Expectations: {result.statistics['successful_expectations']}/"
          f"{result.statistics['evaluated_expectations']} expectations passed")

    if not result.success:
        print("Validation FAILED:")
        for res in result.results:
            if not res.success:
                print(f"  - {res.expectation_config.expectation_type}: {res.expectation_config.kwargs}")
        sys.exit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
