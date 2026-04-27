from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

# Ensure we can import from src/serving
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # inference logic

app = FastAPI(title="Telco Churn Prediction API")

# ─── Health Check ─────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}

# ─── Request Schema ───────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

# ─── Prediction Endpoint ──────────────────────────────
@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        result = predict(data.dict())

        # Normalize output (important for UI)
        if isinstance(result, (int, float)):
            label = "Likely to churn" if result == 1 else "Not likely to churn"
        else:
            label = str(result)

        return {
            "prediction": label,
            "raw": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }