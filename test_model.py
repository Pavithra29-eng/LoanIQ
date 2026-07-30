"""
test_model.py
Runs the trained LoanIQ model against a set of hand-crafted applicant
profiles (strong, weak, borderline, and edge cases) to sanity-check
its behavior before deployment.
"""

import pandas as pd
import joblib

model = joblib.load("model/loan_model.pkl")

profiles = {
    "Strong applicant (high income, great credit, no debt)": {
        "Applicant_Income": 120000, "Coapplicant_Income": 40000, "Age": 34,
        "Dependents": 1, "Credit_Score": 820, "Existing_Loans": 0,
        "DTI_Ratio": 12.0, "Savings": 800000, "Collateral_Value": 1500000,
        "Loan_Amount": 500000, "Loan_Term": 180, "Employment_Status": "Salaried",
        "Marital_Status": "Married", "Loan_Purpose": "Home", "Property_Area": "Urban",
        "Education_Level": "Postgraduate", "Gender": "Female", "Employer_Category": "Govt",
    },
    "Weak applicant (low income, poor credit, high debt)": {
        "Applicant_Income": 15000, "Coapplicant_Income": 0, "Age": 45,
        "Dependents": 3, "Credit_Score": 480, "Existing_Loans": 3,
        "DTI_Ratio": 70.0, "Savings": 5000, "Collateral_Value": 0,
        "Loan_Amount": 600000, "Loan_Term": 60, "Employment_Status": "Self-Employed",
        "Marital_Status": "Married", "Loan_Purpose": "Personal", "Property_Area": "Rural",
        "Education_Level": "Undergraduate", "Gender": "Male", "Employer_Category": "Self",
    },
    "Borderline (mid income, mid credit score)": {
        "Applicant_Income": 45000, "Coapplicant_Income": 10000, "Age": 30,
        "Dependents": 1, "Credit_Score": 650, "Existing_Loans": 1,
        "DTI_Ratio": 38.0, "Savings": 80000, "Collateral_Value": 100000,
        "Loan_Amount": 400000, "Loan_Term": 120, "Employment_Status": "Salaried",
        "Marital_Status": "Single", "Loan_Purpose": "Personal", "Property_Area": "Semi-Urban",
        "Education_Level": "Graduate", "Gender": "Male", "Employer_Category": "Private",
    },
    "High income but bad credit score": {
        "Applicant_Income": 150000, "Coapplicant_Income": 0, "Age": 40,
        "Dependents": 2, "Credit_Score": 500, "Existing_Loans": 2,
        "DTI_Ratio": 45.0, "Savings": 50000, "Collateral_Value": 0,
        "Loan_Amount": 1000000, "Loan_Term": 240, "Employment_Status": "Business",
        "Marital_Status": "Married", "Loan_Purpose": "Business", "Property_Area": "Urban",
        "Education_Level": "Graduate", "Gender": "Male", "Employer_Category": "Self",
    },
    "Low income but excellent credit + high savings": {
        "Applicant_Income": 25000, "Coapplicant_Income": 0, "Age": 28,
        "Dependents": 0, "Credit_Score": 800, "Existing_Loans": 0,
        "DTI_Ratio": 20.0, "Savings": 600000, "Collateral_Value": 0,
        "Loan_Amount": 200000, "Loan_Term": 60, "Employment_Status": "Salaried",
        "Marital_Status": "Single", "Loan_Purpose": "Education", "Property_Area": "Semi-Urban",
        "Education_Level": "Postgraduate", "Gender": "Female", "Employer_Category": "Private",
    },
    "High collateral, self-employed, modest reported income": {
        "Applicant_Income": 30000, "Coapplicant_Income": 0, "Age": 50,
        "Dependents": 2, "Credit_Score": 680, "Existing_Loans": 1,
        "DTI_Ratio": 40.0, "Savings": 150000, "Collateral_Value": 2000000,
        "Loan_Amount": 700000, "Loan_Term": 180, "Employment_Status": "Self-Employed",
        "Marital_Status": "Married", "Loan_Purpose": "Home", "Property_Area": "Rural",
        "Education_Level": "Graduate", "Gender": "Male", "Employer_Category": "Self",
    },
    "High DTI ratio but strong credit score": {
        "Applicant_Income": 60000, "Coapplicant_Income": 20000, "Age": 36,
        "Dependents": 2, "Credit_Score": 780, "Existing_Loans": 2,
        "DTI_Ratio": 65.0, "Savings": 100000, "Collateral_Value": 200000,
        "Loan_Amount": 900000, "Loan_Term": 240, "Employment_Status": "Salaried",
        "Marital_Status": "Married", "Loan_Purpose": "Business", "Property_Area": "Urban",
        "Education_Level": "Graduate", "Gender": "Female", "Employer_Category": "Private",
    },
    "Young applicant, no dependents, first loan": {
        "Applicant_Income": 35000, "Coapplicant_Income": 0, "Age": 23,
        "Dependents": 0, "Credit_Score": 700, "Existing_Loans": 0,
        "DTI_Ratio": 25.0, "Savings": 40000, "Collateral_Value": 0,
        "Loan_Amount": 150000, "Loan_Term": 36, "Employment_Status": "Salaried",
        "Marital_Status": "Single", "Loan_Purpose": "Education", "Property_Area": "Urban",
        "Education_Level": "Graduate", "Gender": "Male", "Employer_Category": "Private",
    },
}

rows = []
for name, profile in profiles.items():
    df = pd.DataFrame([profile])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    rows.append({
        "Profile": name,
        "Prediction": "APPROVED" if pred == 1 else "REJECTED",
        "Approval Probability": f"{prob*100:.1f}%"
    })

results = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", None)
print(results.to_string(index=False))
