from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model


app = FastAPI(
    title="Customer Churn Prediction API"
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_ann.h5"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"


model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_columns = joblib.load(FEATURE_PATH)


class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():

    return {
        "message": "Customer Churn Deep Learning API"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.model_dump()])

    input_data = pd.get_dummies(
        input_data
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(
        input_data
    )

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    prediction = int(
        probability >= 0.5
    )

    return {
        "prediction": prediction,
        "churn_probability": float(probability),
        "result": (
            "Customer will churn"
            if prediction == 1
            else "Customer will not churn"
        )
    }