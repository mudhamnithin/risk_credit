# code.py

import streamlit as st
import numpy as np
import pandas as pd

import cloudpickle
with open("credit_default_pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

st.title("Credit Risk Prediction App")

employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal"])
loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, value=12)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bank_balance_at_application = st.number_input("Bank Balance at Application", value=50000.0)
number_of_dependants = st.number_input("Number of Dependants", min_value=0, value=1)
credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3)
loan_to_income = st.number_input("Loan to Income Ratio", min_value=0.0, value=0.25)
delinquent_months_ratio = st.number_input("Delinquent Months Ratio", min_value=0.0, value=0.0)
if st.button("Predict Default Risk"):
    input_data = pd.DataFrame([{
        "employment_status": employment_status,
        "marital_status": marital_status,
        "loan_purpose": loan_purpose,
        "loan_type": loan_type,
        "loan_to_income": loan_to_income,
        "delinquent_months_ratio": delinquent_months_ratio,
        "loan_tenure_months": loan_tenure_months,
        "age": age,
        "bank_balance_at_application": bank_balance_at_application,
        "number_of_dependants": number_of_dependants,
        "credit_utilization_ratio": credit_utilization_ratio
    }])

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of default! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of default. Probability: {probability:.2%}")
