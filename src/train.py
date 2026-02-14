# ---------------------------------
# train.py
# Contains the function to train the model.
# ---------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
import datetime

from features.build_features import build_feature_pipeline
from models.model import get_model
from utils import load_data
import warnings
warnings.filterwarnings("ignore")

# Set up MLflow experiment and run name
mlflow.set_experiment("credit-risk")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"XGBoost_{timestamp}"

# --------------------------
# train_model - Main function to train the model, evaluate it, and save the model and preprocessor.
# --------------------------
def train_model():
    # Load and prepare data
    credit_risk_df = load_data()
    credit_risk_df = credit_risk_df.drop(credit_risk_df.columns[0], axis=1)
    X = credit_risk_df.drop("Credit Risk", axis=1)
    y = credit_risk_df["Credit Risk"].map({1: 0, 2: 1})

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build preprocessing pipeline and transform data
    preprocessor = build_feature_pipeline(X_train)
    model = get_model("xgb")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Accuracy Score:", accuracy_score(y_test, preds))
    print("ROC AUC Score:", roc_auc_score(y_test, probs))
    print("F1 Score:", f1_score(y_test, preds))
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))

    # Save the pipeline to MLflow
    mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="credit-risk-pipeline",
            input_example=X_train.head(5),
        )
    
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/credit-risk-pipeline",
        "CreditRiskModel"
    )


if __name__ == "__main__":
    with mlflow.start_run(run_name=run_name):
        train_model()

