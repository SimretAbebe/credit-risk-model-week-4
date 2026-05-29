from pydantic import BaseModel

class PredictionRequest(BaseModel):
    amount: float