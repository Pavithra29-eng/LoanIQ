"""
LoanIQ — Intelligent Loan Approval Dashboard
Streamlit application: loan officers enter applicant details and get an
instant Approve/Reject prediction with confidence score.
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="LoanIQ — Loan Approval Dashboard",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = "model/loan_model.pkl"

# ============================================================== THEME / CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #2DD4BF;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0;
    line-height: 1.1;
}
.hero-caption {
    font-family: 'Inter', sans-serif;
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #E2E8F0;
    border-left: 3px solid #2DD4BF;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
}

div[data-testid="stForm"] {
    background-color: #141B2E;
    border: 1px solid #223049;
    border-radius: 14px;
    padding: 1.6rem 1.6rem 1rem 1.6rem;
}

div[data-baseweb="input"], div[data-baseweb="select"] > div {
    border-radius: 8px !important;
}

div[data-testid="stFormSubmitButton"] button {
    background-color: #2DD4BF;
    color: #0B1220;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 0;
    letter-spacing: 0.02em;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #5EEAD4;
    color: #0B1220;
}

.result-card {
    border-radius: 14px;
    padding: 1.6rem;
    text-align: center;
    border: 1px solid #223049;
    background-color: #141B2E;
}
.pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
    margin-bottom: 1rem;
}
.pill-approved { background-color: rgba(45, 212, 191, 0.15); color: #2DD4BF; border: 1px solid #2DD4BF; }
.pill-rejected { background-color: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid #F87171; }

.gauge-label {
    font-family: 'Inter', sans-serif;
    color: #94A3B8;
    font-size: 0.85rem;
    margin-top: 0.6rem;
}
.disclaimer {
    font-family: 'Inter', sans-serif;
    color: #64748B;
    font-size: 0.8rem;
}

.placeholder-box {
    border: 1px dashed #334155;
    border-radius: 14px;
    padding: 2rem 1.2rem;
    text-align: center;
    color: #94A3B8;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


def confidence_gauge(prob: float, approved: bool) -> str:
    """Returns an inline SVG donut gauge showing the confidence percentage."""
    pct = prob if approved else (1 - prob)
    color = "#2DD4BF" if approved else "#F87171"
    radius = 54
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct)
    label = f"{pct*100:.1f}%"

    return (
        f'<svg width="140" height="140" viewBox="0 0 140 140">'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="#223049" stroke-width="12"/>'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.1f}" '
        f'stroke-dashoffset="{offset:.1f}" transform="rotate(-90 70 70)"/>'
        f'<text x="70" y="76" text-anchor="middle" font-family="Space Grotesk, sans-serif" '
        f'font-size="26" font-weight="700" fill="{color}">{label}</text>'
        f'</svg>'
    )


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------------------------------------------------- Header
st.markdown('<div class="eyebrow">SecureTrust Bank</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">💳 LoanIQ</h1>', unsafe_allow_html=True)
st.markdown('<div class="hero-caption">ML-powered pre-screening for loan applications</div>', unsafe_allow_html=True)
st.write("")

if model is None:
    st.error(
        "Model file not found at `model/loan_model.pkl`. "
        "Run `python train_model.py` first to train and save the model."
    )
    st.stop()

st.write("")

# ---------------------------------------------------------------- Layout
col_form, col_result = st.columns([1.3, 1])

with col_form:
    st.markdown('<div class="section-label">Applicant Details</div>', unsafe_allow_html=True)

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=45000, step=1000)
            coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, value=0, step=1000)
            age = st.number_input("Age", min_value=18, max_value=75, value=35)
            dependents = st.selectbox("Dependents", [0, 1, 2, 3], index=0)
            credit_score = st.slider("Credit Score", 300, 900, 700)

        with c2:
            existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3, 4], index=0)
            dti_ratio = st.slider("DTI Ratio (%)", 0.0, 90.0, 30.0, step=0.5)
            savings = st.number_input("Savings Balance (₹)", min_value=0, value=100000, step=5000)
            collateral_value = st.number_input("Collateral Value (₹)", min_value=0, value=0, step=10000)
            loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=1000, value=500000, step=10000)

        with c3:
            loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 120, 180, 240, 360], index=4)
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
            property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

        c4, c5 = st.columns(2)
        with c4:
            education_level = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c5:
            employer_category = st.selectbox("Employer Category", ["Govt", "Private", "Self"])

        submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

with col_result:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    if not submitted:
        st.markdown(
            '<div class="placeholder-box">'
            'Fill in the applicant details and click <b>Predict Loan Approval</b><br>to see the result here.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        input_df = pd.DataFrame([{
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age": age,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Employment_Status": employment_status,
            "Marital_Status": marital_status,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        approved = pred == 1

        pill_class = "pill-approved" if approved else "pill-rejected"
        pill_text = "✅ APPROVED" if approved else "❌ REJECTED"
        gauge_svg = confidence_gauge(prob, approved)

        result_html = (
            f'<div class="result-card">'
            f'<div class="pill {pill_class}">{pill_text}</div><br>'
            f'{gauge_svg}'
            f'<div class="gauge-label">Model confidence</div>'
            f'</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        st.write("")
        st.markdown(
            '<div class="disclaimer">⚠️ This is an automated pre-screening estimate, not a final lending '
            'decision. Final approval is subject to human verification.</div>',
            unsafe_allow_html=True
        )

st.write("")
st.write("")

# ---------------------------------------------------------------- Footer / About
with st.expander("ℹ️ About this model"):
    st.markdown("""
    - **Model**: Random Forest Classifier (tuned, class-balanced)
    - **Trained on**: Historical loan application data (income, credit score, DTI ratio, collateral, employment info, etc.)
    - **Test set performance**: ~86.5% accuracy, 0.937 ROC-AUC
    - This dashboard is a decision-support tool for loan officers, replacing slow and inconsistent manual pre-screening — not a replacement for final human review.
    """)