import sys
import os
import pandas as pd
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_processing import (  # noqa: E402
    build_pipeline,
    AggregateFeatureExtractor,
    DateTimeFeatureExtractor,
    WoEEncoder
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
    result = transformer.fit_transform(sample_data)

    expected_columns = [
        "CustomerId",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_number_of_transactions():
    """
    Test aggregation keeps row count matching the input transaction count.
    """
    sample_data = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100, 200, 300],
        }
    )

    transformer = AggregateFeatureExtractor()
    result = transformer.fit_transform(sample_data)

    assert len(result) == 3


def test_datetime_feature_extractor():
    """
    Test extraction of date time features.
    """
    sample_data = pd.DataFrame(
        {
            "TransactionStartTime": [
                "2018-11-15T02:18:49Z",
                "2018-11-16T15:30:00Z"
            ]
        }
    )
    extractor = DateTimeFeatureExtractor()
    result = extractor.fit_transform(sample_data)

    assert "TransactionHour" in result.columns
    assert "TransactionDay" in result.columns
    assert "TransactionMonth" in result.columns
    assert "TransactionYear" in result.columns

    assert result["TransactionHour"].iloc[0] == 2
    assert result["TransactionHour"].iloc[1] == 15
    assert result["TransactionDay"].iloc[0] == 15
    assert result["TransactionDay"].iloc[1] == 16
    assert result["TransactionMonth"].iloc[0] == 11
    assert result["TransactionYear"].iloc[0] == 2018


def test_woe_encoder():
    """
    Test Weight of Evidence encoding.
    """
    X = pd.DataFrame(
        {
            "ProviderId": ["P1", "P1", "P2", "P2", "P2"],
        }
    )
    y = np.array([0, 0, 1, 1, 0])

    encoder = WoEEncoder(cols=["ProviderId"])
    encoder.fit(X, y)
    result = encoder.transform(X)

    # Values should be floats representing log ratios
    assert isinstance(result["ProviderId"].iloc[0], float)
    # Check that different categories got different encoded values
    assert result["ProviderId"].iloc[0] != result["ProviderId"].iloc[2]
