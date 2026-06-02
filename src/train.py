import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


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
    Load processed dataset.
    """

    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )

    drop_columns = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "TransactionStartTime",
        "is_high_risk",
    ]

    X = df.drop(columns=drop_columns)

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    y = df["is_high_risk"]

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
    ):

        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
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
            model,
            "logistic_regression_model"
        )

        print("\nLogistic Regression Results")
        print(metrics)

        return metrics


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
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    with mlflow.start_run(
        run_name="RandomForest"
    ):

        metrics = evaluate_model(
            best_model,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            "RandomForest"
        )

        mlflow.log_param(
            "n_estimators",
            grid_search.best_params_["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            grid_search.best_params_["max_depth"]
        )

        for key, value in metrics.items():
            mlflow.log_metric(
                key,
                value
            )

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_model"
        )

        print("\nRandom Forest Results")
        print(metrics)

        print("\nBest Parameters:")
        print(grid_search.best_params_)

        return best_model, metrics


def main():

    mlflow.set_experiment(
        "Credit Risk Modeling"
    )

    X, y = load_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test
    )

    best_model, best_metrics = (
        train_random_forest(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    print(
        "\nTraining workflow completed successfully."
    )


if __name__ == "__main__":
    main()
