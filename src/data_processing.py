import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate features."""

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
        # Fill missing std with 0
        customer_features["std_transaction_amount"] = (
            customer_features["std_transaction_amount"].fillna(0)
        )
        return customer_features


def create_rfm_target(df):
    """Create an RFM-based proxy target variable."""

    rfm_df = df.copy()
    # Convert transaction time to datetime
    rfm_df["TransactionStartTime"] = pd.to_datetime(
        rfm_df["TransactionStartTime"]
    )
    # Define snapshot date as day after the latest transaction
    snapshot_date = (
        rfm_df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    )
    rfm = (
        rfm_df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    # Scale numeric RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    # Cluster customers into risk groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)
    # Identify high‑risk cluster (lowest frequency)
    cluster_summary = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()
    )
    high_risk_cluster = cluster_summary["Frequency"].idxmin()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm[["CustomerId", "is_high_risk"]]


def build_pipeline():
    """Build preprocessing pipeline for credit‑risk modeling."""

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
            ("num", numerical_pipeline, numerical_features)
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
    raw_data = pd.read_csv("data/raw/data.csv")
    rfm_target = create_rfm_target(raw_data)
    processed_data = raw_data.merge(
        rfm_target,
        on="CustomerId",
        how="left",
    )
    processed_data.to_csv(
        "data/processed/processed_data.csv", index=False
    )
    print("Processed dataset created successfully.")
# End of data processing script
