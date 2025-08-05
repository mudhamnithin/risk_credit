import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load and merge datasets
df_loans = pd.read_csv("loans.csv")
df_customers = pd.read_csv("customers.csv")
df_bureau = pd.read_csv("bureau_data.csv")
df = df_loans.merge(df_customers, on="cust_id").merge(df_bureau, on="cust_id")

# Drop rows with extreme processing fees
df = df[df["processing_fee"] < 0.02 * df["sanction_amount"]]

# Feature engineering
df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
df["delinquent_months_ratio"] = df["delinquent_months"] / df["total_loan_months"].replace(0, np.nan)

# Label encoding
df["default"] = df["default"].map({True: 1, False: 0})

# Remove leakage and irrelevant columns
leakage_cols = [
    "loan_id", "cust_id", "disbursal_date", "installment_start_dt", "city", "state",
    "delinquent_months", "total_dpd", "loan_amount", "sanction_amount", "net_disbursement",
    "zipcode", "processing_fee", "gst", "principal_outstanding", "number_of_closed_accounts"
]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# Features and target
X = df.drop(columns="default")
y = df["default"]

# Define numeric and categorical columns
numeric_features = [
    "income", "enquiry_count", "number_of_open_accounts", "years_at_current_address",
    "total_loan_months", "age", "bank_balance_at_application", "number_of_dependants",
    "credit_utilization_ratio", "loan_to_income", "delinquent_months_ratio"
]
categorical_features = ["employment_status", "gender", "marital_status", "residence_type", "loan_purpose", "loan_type"]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Model pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "credit_default_pipeline.joblib")
print("✅ Model saved to 'credit_default_pipeline.joblib'")
