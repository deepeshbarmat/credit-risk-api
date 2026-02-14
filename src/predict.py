# ---------------------------------
# predict.py
# Contains the CreditRiskPredictor class to load the trained model and make predictions.
# ---------------------------------
import mlflow
import pandas as pd

MODEL_URI = "models:/CreditRiskModel/latest"

# --------------------------
# CreditRiskPredictor - Class to load the trained model and make predictions.
# --------------------------
class CreditRiskPredictor:
    def __init__(self):
        self.model = mlflow.sklearn.load_model(MODEL_URI)

    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        probs = self.model.predict(df)
        proba = self.model.predict_proba(df)[:, 1]
        prob = round(float(proba[0]), 4) # if proba[0] >= 0.5 else 1 - proba[0]
        label = "High Risk" if proba[0] > 0.5 else "Low Risk"
        return {
            "probability": prob,
            "prediction": label
        }