import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
)

# Set tracking URI to sqlite db
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Credit Risk API"
)


class DummyPredictor:
    """Fallback predictor if MLflow registry is empty/fails."""
    def predict_proba(self, X):
        import numpy as np
        # Return mock probabilities
        return np.array([[0.9, 0.1]] * len(X))

    def predict(self, X):
        import numpy as np
        return np.array([0] * len(X))


# Load model with fallback
try:
    try:
        model = mlflow.sklearn.load_model(
            "models:/CreditRiskModel/latest"
        )
        print("Loaded scikit-learn pipeline from MLflow model registry.")
    except Exception:
        model = mlflow.pyfunc.load_model(
            "models:/CreditRiskModel/latest"
        )
        print("Loaded pyfunc model from MLflow model registry.")
except Exception as e:
    print("Warning: Could not load model from MLflow registry.")
    print("Using DummyPredictor fallback.")
    print("Error details:", e)
    model = DummyPredictor()


@app.get("/")
def home():
    return {"message": "Credit Risk API"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    # Convert Pydantic request to DataFrame
    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    try:
        # Since model is the complete pipeline, we pass raw DataFrame directly
        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(input_data)[0][1]
            )
        else:
            probability = float(
                model.predict(input_data)[0]
            )
    except Exception as e:
        print("Prediction ERROR:", e)
        # Fallback to default/mean value
        probability = 0.5

    return PredictionResponse(
        risk_probability=round(
            probability,
            4
        )
    )
