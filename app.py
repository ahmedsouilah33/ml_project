from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from model_pipeline import (
    preprocess_data,
    load_model,
    prepare_data,
    train_model, 
    evaluate_model,
    save_model, 
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


class RetrainParams(BaseModel):
    n_neighbors: int = 15
# ----------------------------
# ENDPOINT : retrain du modèle
# ----------------------------
@app.post("/retrain")
def retrain_model(params: RetrainParams):

    global model, scaler, trained_columns

    try:
        # 1. Charger et préparer les données
        data = prepare_data("data/train.csv", "data/test.csv")

        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # 2. Entraîner le nouveau modèle
        model, scaler = train_model(X_train, y_train, n_neighbors=params.n_neighbors)

        # 3. Évaluer
        eval_result = evaluate_model(model, scaler, X_test, y_test)

        # 4. Sauvegarder
        save_model(
            model=model,
            scaler=scaler,
            file_path=MODEL_PATH,
            trained_columns=X_train.columns
        )

        # 5. Recharger dans l’API
        model, scaler, trained_columns = load_model(MODEL_PATH)

        return {
            "message": "Model retrained successfully.",
            "accuracy": eval_result["accuracy"],
            "n_neighbors": params.n_neighbors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------
# Health check
# --------------------------------------
@app.get("/")
def root():
    return {"message": "Loan Prediction API is running."}
