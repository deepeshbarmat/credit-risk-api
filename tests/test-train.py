# --------------------------
# test_train.py
# Unit tests for the training process.
# --------------------------

from src.train import train_model

# Test to ensure that the training process runs without errors.
def test_training_runs():
    try:
        train_model("xgb")
    except Exception as e:
        assert False, f"Training failed with error: {e}"