import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data_processing import build_pipeline


def calculate_iv(X, y, col, smoothing=0.5):
    """
    Calculate the Information Value (IV) for a categorical feature.
    """
    y_arr = np.array(y)
    n_good_total = np.sum(y_arr == 0)
    n_bad_total = np.sum(y_arr == 1)
    if n_good_total == 0:
        n_good_total = 1
    if n_bad_total == 0:
        n_bad_total = 1

    col_data = X[col].astype(str)
    unique_cats = col_data.unique()
    iv = 0.0
    for cat in unique_cats:
        mask = col_data == cat
        n_good_cat = np.sum((y_arr == 0) & mask)
        n_bad_cat = np.sum((y_arr == 1) & mask)

        p_good = (n_good_cat + smoothing) / (n_good_total + 2 * smoothing)
        p_bad = (n_bad_cat + smoothing) / (n_bad_total + 2 * smoothing)

        woe = np.log(p_good / p_bad)
        iv += (p_good - p_bad) * woe
    return iv


def evaluate_model(model, X_test, y_test):
    """
    Calculate evaluation metrics.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def load_data():
    """
    Load processed dataset with required raw columns.
    """
    df = pd.read_csv("data/processed/processed_data.csv")

    # Drop rows where target variable is missing
    df = df.dropna(subset=["is_high_risk"])

    # Keep the columns needed by the preprocessing pipeline
    required_cols = [
        "CustomerId",
        "TransactionStartTime",
        "Amount",
        "Value",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
        "FraudResult"
    ]

    X = df[required_cols]
    y = df["is_high_risk"].astype(int)

    return X, y


def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train and log Logistic Regression.
    """
    with mlflow.start_run(
        run_name="LogisticRegression"
    ) as run:
        pipeline = Pipeline([
            ("preprocessor", build_pipeline()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        for key, value in metrics.items():
            mlflow.log_metric(
                key,
                value
            )

        mlflow.sklearn.log_model(
            pipeline,
            "logistic_regression_model"
        )

        print("\nLogistic Regression Results:")
        print(metrics)

        return pipeline, metrics, run.info.run_id


def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train and tune Random Forest.
    """
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
    }

    base_pipeline = Pipeline([
        ("preprocessor", build_pipeline()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_pipeline = grid_search.best_estimator_

    with mlflow.start_run(
        run_name="RandomForest"
    ) as run:
        metrics = evaluate_model(
            best_pipeline,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            "RandomForest"
        )

        mlflow.log_param(
            "n_estimators",
            grid_search.best_params_["classifier__n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            grid_search.best_params_["classifier__max_depth"]
        )

        for key, value in metrics.items():
            mlflow.log_metric(
                key,
                value
            )

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="random_forest_model"
        )

        print("\nRandom Forest Results:")
        print(metrics)

        print("\nBest Parameters:")
        print(grid_search.best_params_)

        return best_pipeline, metrics, run.info.run_id


def main():
    # Set tracking URI to local SQLite database so model registry works
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(
        "Credit Risk Modeling"
    )

    X, y = load_data()

    # Log Information Values (IV) for categorical features
    print("\nInformation Values (IV) of key categorical features:")
    categorical_cols = ["ProviderId", "ProductId", "ProductCategory", "ChannelId", "PricingStrategy"]

    with mlflow.start_run(run_name="InformationValueCalculation"):
        for col in categorical_cols:
            iv = calculate_iv(X, y, col)
            print(f" - {col}: {iv:.4f}")
            mlflow.log_metric(f"iv_{col}", iv)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    lr_pipeline, lr_metrics, lr_run_id = train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test
    )

    rf_pipeline, rf_metrics, rf_run_id = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Register the best model based on F1-score
    if rf_metrics["f1_score"] >= lr_metrics["f1_score"]:
        best_run_id = rf_run_id
        best_path = "random_forest_model"
        best_f1 = rf_metrics["f1_score"]
        print(f"\nRandom Forest is the best model (F1: {best_f1:.4f}).")
    else:
        best_run_id = lr_run_id
        best_path = "logistic_regression_model"
        best_f1 = lr_metrics["f1_score"]
        print(f"\nLogistic Regression is the best model (F1: {best_f1:.4f}).")

    model_uri = f"runs:/{best_run_id}/{best_path}"
    print(f"Registering model {model_uri} as 'CreditRiskModel' in MLflow Registry...")
    mlflow.register_model(model_uri, "CreditRiskModel")

    print(
        "\nTraining workflow completed successfully."
    )


if __name__ == "__main__":
    main()
