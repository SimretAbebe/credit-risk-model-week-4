import sys
import os
from fastapi.testclient import TestClient

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.api.main import app  # noqa: E402

client = TestClient(app)


def test_read_main():
    """
    Test the home endpoint of the API.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API"}


def test_predict_endpoint():
    """
    Test the predict endpoint with a typical payload.
    """
    payload = {
        "CustomerId": "CustomerId_1",
        "TransactionStartTime": "2018-11-15T02:18:49Z",
        "CountryCode": 256,
        "ProviderId": "ProviderId_6",
        "ProductId": "ProductId_10",
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_3",
        "Amount": 1000.0,
        "Value": 1000.0,
        "PricingStrategy": 2,
        "FraudResult": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert isinstance(data["risk_probability"], float)
    assert 0.0 <= data["risk_probability"] <= 1.0
