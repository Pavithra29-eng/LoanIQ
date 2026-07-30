"""
train_model.py
Trains the loan approval model and saves a single deployable pipeline
(preprocessing + model bundled together) to model/loan_model.pkl
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "model/loan_model.pkl"

CATEGORICAL_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                    "Property_Area", "Education_Level", "Gender", "Employer_Category"]

NUMERIC_COLS = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents",
                "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings",
                "Collateral_Value", "Loan_Amount", "Loan_Term"]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Applicant_ID"])
    return df


def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():
    df = load_data()
    X = df[FEATURE_COLS]
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("=== Model Evaluation on Test Set ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall   : {recall_score(y_test, preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
