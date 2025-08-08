# trained_model.py

import pandas as pd
import numpy as np
import cloudpickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load data
df_loans = pd.read_csv("loans.csv")
df_customers = pd.read_csv("customers.csv")
df_bureau = pd.read_csv("bureau_data.csv")
df = df_loans.merge(df_customers, on="cust_id").merge(df_bureau, on="cust_id")

# Drop extreme values
df = df[df["processing_fee"] < 0.02 * df["sanction_amount"]]

# Feature engineering
df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
df["delinquent_months_ratio"] = df["delinquent_months"] / df["total_loan_months"].replace(0, np.nan)

# Map target
df["default"] = df["default"].map({True: 1, False: 0})

# Drop leakage columns
leakage_cols = [
    "loan_id", "cust_id", "disbursal_date", "installment_start_dt", "city", "state",
    "delinquent_months", "total_dpd", "loan_amount", "sanction_amount", "net_disbursement",
    "zipcode", "processing_fee", "gst", "principal_outstanding", "number_of_closed_accounts"
]
df.drop(columns=[col for col in leakage_cols if col in df.columns], inplace=True)

# Features/target
X = df.drop(columns="default")
y = df["default"]

# Define columns
numeric_features = [
    "loan_to_income",
    "delinquent_months_ratio",
    "loan_tenure_months",
    "age",
    "bank_balance_at_application",
    "number_of_dependants",
    "credit_utilization_ratio"
]

categorical_features = [
    "employment_status",
    "marital_status",
    "loan_purpose",
    "loan_type"
]

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# Final full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X, y)

# Save model with cloudpickle
with open("credit_default_pipeline.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)

print("✅ Model saved to 'credit_default_pipeline.pkl'")
