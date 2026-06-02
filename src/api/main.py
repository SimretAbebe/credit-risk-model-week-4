import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Credit Risk API"
)

# Load model with fallback
try:
    model = mlflow.pyfunc.load_model(
        "models:/CreditRiskModel/latest"
    )
except Exception:
    class DummyModel:
        def predict(self, df):
            import numpy as np
            return np.array([0.5] * len(df))
    model = DummyModel()


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
    input_data = pd.DataFrame(
        [request.model_dump()]
    )
    print(input_data)

    """
    Temporary prediction endpoint.
    """

    prediction = model.predict(input_data)

    probability = float(prediction[0])

    return PredictionResponse(
        risk_probability=probability,
    )
