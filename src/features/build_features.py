# ---------------------------------
# build_features.py - 
# Contains the function to create a preprocessing pipeline.
# ---------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ------------------------------
# build_feature_pipeline - Creates a preprocessing pipeline 
# for both numeric and categorical features.
# ------------------------------
def build_feature_pipeline(credit_risk_df):

    # Identify categorical and numeric columns
    categorical_cols = credit_risk_df.select_dtypes(include=['object']).columns
    numeric_cols = credit_risk_df.select_dtypes(include=["int64", "float64"]).columns

    # Define pipelines for numeric and categorical features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor