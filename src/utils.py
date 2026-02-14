# ---------------------------------
# utils.py
# Utility functions for data loading
# ---------------------------------
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# load_data - Loads the ratings data from the CSV file and removes duplicates.
# ---------------------------
def load_data():
    DATA_DIR = Path(__file__).parent.parent / "data"
    credit_risk_df = pd.read_csv(DATA_DIR / "german_credit_data.csv")
    return credit_risk_df

if __name__ == "__main__":
    df = load_data()