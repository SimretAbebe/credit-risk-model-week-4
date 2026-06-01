import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Create customer-level aggregate features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        customer_features = (
            X.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                average_transaction_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_transaction_amount=("Amount", "std"),
            )
            .reset_index()
        )

        customer_features["std_transaction_amount"] = (
            customer_features["std_transaction_amount"]
            .fillna(0)
        )

        return customer_features


def build_pipeline():
    """
    Build preprocessing pipeline for customer-level credit risk modeling.
    """

    numerical_features = [
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            )
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("aggregate_features", AggregateFeatureExtractor()),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


if __name__ == "__main__":
    pipeline = build_pipeline()
    print("Data processing pipeline created successfully.")