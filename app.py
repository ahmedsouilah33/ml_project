from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from model_pipeline import (
    preprocess_data,
    load_model,
    predict
)

# --------------------------------------
# Load model + scaler + trained columns
# --------------------------------------
MODEL_PATH = "knn_loan_model.pkl"
model, scaler, trained_columns = load_model(MODEL_PATH)

app = FastAPI(
    title="Loan Approval API",
    description="Predict if a loan application will be approved.",
    version="2.0"
)

# --------------------------------------
# Request Body Schema
# --------------------------------------
class LoanInput(BaseModel):
    Loan_ID: str
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float | None = None
    Credit_History: float
    Property_Area: str


# --------------------------------------
# Prediction endpoint
# --------------------------------------
@app.post("/predict")
def predict_loan(data: LoanInput):

    # Convert JSON → pandas DataFrame
    df = pd.DataFrame([data.dict()])

    try:
        # Preprocess using model_pipeline
        processed = preprocess_data(df, is_train=False)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # If trained_columns exist, align columns
    if trained_columns is not None:
        for col in trained_columns:
            if col not in processed.columns:
                processed[col] = 0

        # Remove extra columns not used for training
        processed = processed[trained_columns]

    # Final check
    if processed.isnull().any().any():
        raise HTTPException(status_code=400, detail="Missing values remain after preprocessing.")

    # Make prediction
    try:
        raw_pred = predict(model, scaler, processed)[0]
        result = "Approved" if raw_pred == 1 else "Not Approved"

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# --------------------------------------
# Health check
# --------------------------------------
@app.get("/")
def root():
    return {"message": "Loan Prediction API is running."}
