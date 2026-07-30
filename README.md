# LoanIQ — Intelligent Loan Approval System 💳

An end-to-end machine learning project that predicts whether a loan application
should be **Approved** or **Rejected**, built for a fictional bank (SecureTrust
Bank) to replace a slow, inconsistent manual verification process.

**[Live Demo →](#)** *(add your deployed Streamlit link here once deployed)*

---

## 🎯 Problem Statement

SecureTrust Bank manually verifies loan applications by checking income,
employment, and credit history — a process that is slow, inconsistent, and
biased. This project builds an ML model that pre-screens applications,
flagging likely approvals/rejections before final human review.

## 📊 Project Structure

```
├── app.py                  # Streamlit dashboard (the running application)
├── train_model.py          # Script to train & save the model pipeline
├── requirements.txt        # Python dependencies
├── data/
│   └── loan_data.csv   # Training dataset
├── model/
│   └── loan_model.pkl       # Saved trained pipeline (preprocessing + model)
└── notebooks/
    └── LoanIQ_Analysis.ipynb   # Full EDA + modeling notebook
```

## 🧠 Approach

1. **EDA** — distribution analysis, boxplots by approval outcome, correlation heatmap
2. **Preprocessing** — median/mode imputation, one-hot encoding, standard scaling
   (all bundled into a single `sklearn.Pipeline`)
3. **Modeling** — compared Logistic Regression, Decision Tree, KNN, Naive Bayes,
   and Random Forest; selected a tuned, class-balanced **Random Forest** as the
   final model
4. **Deployment** — wrapped the trained pipeline in a Streamlit dashboard for
   interactive predictions

## 📈 Model Performance (test set)

| Metric | Score |
|---|---|
| Accuracy | 0.865 |
| Precision | 0.870 |
| Recall | 0.862 |
| F1 Score | 0.866 |
| ROC-AUC | 0.937 |

## 🚀 Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/LoanIQ.git
cd LoanIQ

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain the model
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🌐 Deploying (free, no server needed)

This app is ready to deploy on **[Streamlit Community Cloud](https://streamlit.io/cloud)**:

1. Push this repo to GitHub (see below).
2. Go to share.streamlit.io, sign in with GitHub.
3. Click **New app**, select this repo, set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL to share.

## 📦 Dataset

The dataset is synthetically generated to match a realistic loan application
schema (income, credit score, DTI ratio, collateral, employment info, etc.),
with intentional noise and a few missing values to mimic real-world data.

## ⚠️ Disclaimer

This is an educational project. The dataset is synthetic and the model is a
decision-support prototype, not a certified credit-risk model.
