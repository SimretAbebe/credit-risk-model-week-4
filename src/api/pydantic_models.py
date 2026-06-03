from pydantic import BaseModel


class PredictionRequest(BaseModel):
    CustomerId: str
    TransactionStartTime: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int


class PredictionResponse(BaseModel):
    risk_probability: float