import sys
import os
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_processing import (  # noqa: E402
    build_pipeline,
    AggregateFeatureExtractor
)


def test_pipeline_creation():
    """
    Test that the pipeline is created successfully.
    """
    pipeline = build_pipeline()

    assert pipeline is not None


def test_aggregate_feature_extractor():
    """
    Test aggregate feature generation.
    """

    sample_data = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100, 200, 300],
        }
    )

    transformer = AggregateFeatureExtractor()

    result = transformer.transform(sample_data)

    expected_columns = [
        "CustomerId",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_number_of_customers():
    """
    Test aggregation produces one row per customer.
    """

    sample_data = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100, 200, 300],
        }
    )

    transformer = AggregateFeatureExtractor()

    result = transformer.transform(sample_data)

    assert len(result) == 2
