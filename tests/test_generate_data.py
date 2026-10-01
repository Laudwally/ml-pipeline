import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import generate  # noqa: E402


def test_generate_shape_and_columns():
    df = generate(n_rows=200, seed=1)
    assert len(df) == 200
    assert set(df.columns) == {
        "customer_id",
        "event_timestamp",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets",
        "plan_type",
        "churned",
    }


def test_generate_value_ranges():
    df = generate(n_rows=500, seed=2)
    assert df["tenure_months"].between(0, 71).all()
    assert (df["monthly_charges"] > 0).all()
    assert df["plan_type"].isin(["basic", "standard", "premium"]).all()
    assert df["churned"].isin([0, 1]).all()


def test_generate_is_deterministic_for_a_given_seed():
    # event_timestamp is anchored to wall-clock "now" and is expected to differ
    # between calls; everything else should be exactly reproducible per seed.
    cols = ["customer_id", "tenure_months", "monthly_charges", "total_charges", "support_tickets", "plan_type", "churned"]
    a = generate(n_rows=100, seed=7)
    b = generate(n_rows=100, seed=7)
    assert a[cols].equals(b[cols])
