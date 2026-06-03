import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, and year from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TransactionStartTime" in X.columns:
            dt_col = pd.to_datetime(X["TransactionStartTime"])
            X["TransactionHour"] = dt_col.dt.hour
            X["TransactionDay"] = dt_col.dt.day
            X["TransactionMonth"] = dt_col.dt.month
            X["TransactionYear"] = dt_col.dt.year
        else:
            # Fallback if column is missing (e.g. at inference if not passed)
            X["TransactionHour"] = 12
            X["TransactionDay"] = 15
            X["TransactionMonth"] = 6
            X["TransactionYear"] = 2026
        return X


class AggregateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Create and map customer-level aggregate features."""

    def __init__(self):
        self.customer_stats_ = {}
        self.global_medians_ = {}

    def fit(self, X, y=None):
        # Calculate stats for each customer based on training data
        stats = X.groupby("CustomerId").agg(
            total_transaction_amount=("Amount", "sum"),
            average_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),
        )
        stats["std_transaction_amount"] = stats["std_transaction_amount"].fillna(0.0)
        self.customer_stats_ = stats.to_dict(orient="index")

        # Calculate global medians for unseen customers at transform time
        if len(stats) > 0:
            self.global_medians_ = {
                "total_transaction_amount": float(stats["total_transaction_amount"].median()),
                "average_transaction_amount": float(stats["average_transaction_amount"].median()),
                "transaction_count": float(stats["transaction_count"].median()),
                "std_transaction_amount": float(stats["std_transaction_amount"].median()),
            }
        else:
            self.global_medians_ = {
                "total_transaction_amount": 0.0,
                "average_transaction_amount": 0.0,
                "transaction_count": 1.0,
                "std_transaction_amount": 0.0,
            }
        return self

    def transform(self, X):
        X = X.copy()

        total_vals = []
        avg_vals = []
        count_vals = []
        std_vals = []

        for cust_id in X["CustomerId"]:
            if cust_id in self.customer_stats_:
                stats = self.customer_stats_[cust_id]
            else:
                stats = self.global_medians_
            total_vals.append(stats["total_transaction_amount"])
            avg_vals.append(stats["average_transaction_amount"])
            count_vals.append(stats["transaction_count"])
            std_vals.append(stats["std_transaction_amount"])

        X["total_transaction_amount"] = total_vals
        X["average_transaction_amount"] = avg_vals
        X["transaction_count"] = count_vals
        X["std_transaction_amount"] = std_vals

        return X


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns using Weight of Evidence (WoE)."""

    def __init__(self, cols=None, smoothing=0.5):
        self.cols = cols
        self.smoothing = smoothing
        self.woe_maps_ = {}

    def fit(self, X, y=None):
        if y is None:
            return self

        y_arr = np.array(y)
        n_good_total = np.sum(y_arr == 0)
        n_bad_total = np.sum(y_arr == 1)

        # Avoid division by zero
        if n_good_total == 0:
            n_good_total = 1
        if n_bad_total == 0:
            n_bad_total = 1

        cols_to_encode = (
            self.cols
            if self.cols is not None
            else X.select_dtypes(include=["object", "category"]).columns
        )

        self.woe_maps_ = {}
        for col in cols_to_encode:
            col_data = X[col].astype(str)
            unique_cats = col_data.unique()
            woe_map = {}
            for cat in unique_cats:
                mask = col_data == cat
                n_good_cat = np.sum((y_arr == 0) & mask)
                n_bad_cat = np.sum((y_arr == 1) & mask)

                # Formula with smoothing
                p_good = (n_good_cat + self.smoothing) / (n_good_total + 2 * self.smoothing)
                p_bad = (n_bad_cat + self.smoothing) / (n_bad_total + 2 * self.smoothing)

                woe_map[cat] = float(np.log(p_good / p_bad))
            self.woe_maps_[col] = woe_map

        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if col in X.columns:
                col_str = X[col].astype(str)
                X[col] = col_str.map(woe_map).fillna(0.0)
            else:
                X[col] = 0.0
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from the dataframe."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                X_out[col] = X[col]
            else:
                X_out[col] = 0.0
        return X_out


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

    features_to_keep = [
        "Amount",
        "Value",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "TransactionHour",
        "TransactionDay",
        "TransactionMonth",
        "TransactionYear",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
        "FraudResult"
    ]

    categorical_cols = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    pipeline = Pipeline(
        steps=[
            ("datetime", DateTimeFeatureExtractor()),
            ("customer_agg", AggregateFeatureExtractor()),
            ("woe_encode", WoEEncoder(cols=categorical_cols)),
            ("select", FeatureSelector(features_to_keep)),
            ("scaler", StandardScaler()),
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
