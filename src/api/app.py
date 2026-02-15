# ---------------------------------
# app.py
# Contains the FastAPI application to serve the model and make predictions.
# ---------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.predict import CreditRiskPredictor

app = FastAPI(title="Credit Risk API")

predictor = CreditRiskPredictor()

# -------------------------
# Input Schema
# -------------------------
class CreditInput(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: Optional[str] = None
    Checking_account: Optional[str] = None
    Credit_amount: float
    Duration: int
    Purpose: str

# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(input: CreditInput):
    input_dict = input.dict()

    # Rename fields back to original dataset column names
    formatted_input = {
        "Age": input_dict["Age"],
        "Sex": input_dict["Sex"],
        "Job": input_dict["Job"],
        "Housing": input_dict["Housing"],
        "Saving accounts": input_dict["Saving_accounts"],
        "Checking account": input_dict["Checking_account"],
        "Credit amount": input_dict["Credit_amount"],
        "Duration": input_dict["Duration"],
        "Purpose": input_dict["Purpose"],
    }

    result = predictor.predict(formatted_input)

    return result
