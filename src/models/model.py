# ---------------------------------
# model.py
# Contains the function to define and return the model.
# ---------------------------------
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_name="xgb"):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000)

    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )

    raise ValueError("Unknown model type")