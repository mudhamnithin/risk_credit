import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Credit Risk Prediction")

# Load model
pipeline = joblib.load("credit_default_pipeline.joblib")

# Input features
input_data = {
    "income": st.number_input("Income", min_value=0.0),
    "employment_status": st.selectbox("Employment Status", ["Salaried", "Self-Employed"]),
    "gender": st.selectbox("Gender", ["M", "F"]),
    "marital_status": st.selectbox("Marital Status", ["Single", "Married"]),
    "residence_type": st.selectbox("Residence Type", ["Owned", "Mortgage", "Rented"]),
    "age": st.number_input("Age", min_value=18, max_value=100),
    "bank_balance_at_application": st.number_input("Bank Balance at Application", min_value=0.0),
    "number_of_dependants": st.number_input("Number of Dependents", min_value=0),
    "credit_utilization_ratio": st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0),
    "enquiry_count": st.number_input("Enquiry Count (last 6 months)", min_value=0),
    "number_of_open_accounts": st.number_input("Number of Open Accounts", min_value=0),
    "years_at_current_address": st.number_input("Years at Current Address", min_value=0),
    "total_loan_months": st.number_input("Total Loan Months", min_value=1),
    "loan_purpose": st.selectbox("Loan Purpose", ["Education", "Home", "Personal"]),
    "loan_type": st.selectbox("Loan Type", ["Secured", "Unsecured"]),
    "loan_to_income": st.number_input("Loan Amount / Income", min_value=0.0),
    "delinquent_months_ratio": st.number_input("Delinquent Months Ratio", min_value=0.0, max_value=1.0)
}

# Predict button
if st.button("Predict Default Risk"):
    try:
        input_df = pd.DataFrame([input_data])
        proba = pipeline.predict_proba(input_df)[0][1]
        st.subheader(f"Default Probability: {proba * 100:.2f}%")

        if proba < 0.5:
            st.success("✅ Low risk of default — loan can be sanctioned.")
        else:
            st.error("⚠️ High risk of default — reconsider sanctioning.")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
