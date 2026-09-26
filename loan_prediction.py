"""
Loan Approval Prediction System
--------------------------------
Trains and compares three classifiers (Decision Tree, Random Forest,
Logistic Regression) to predict whether a loan application will be
approved, based on applicant income, credit score, loan amount, and
employment status.

Usage:
    python loan_prediction.py

Requires:
    data/loan_prediction.csv  (see data/README.md for the expected format)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/loan_prediction.csv"

# Numeric columns that get median-imputed for missing values.
NUMERIC_COLUMNS = ["Applicant_Income", "Credit_Score", "Loan_Amount"]


def load_and_prepare_data(path: str):
    """Load the CSV, impute missing values, and encode categorical columns.

    - Numeric columns are median-imputed. This is cross-sectional tabular
      data (each row is an independent applicant), so there's no ordering
      for a forward-fill to exploit; the median is a neutral, outlier-robust
      fill that doesn't leak information from an arbitrary neighboring row.
    - Employment_Status is one-hot encoded (drop_first=True) rather than
      label-encoded, since "Employed" / "Self-Employed" / "Unemployed" have
      no natural order and LabelEncoder would impose a false one. This also
      produces the same Is_Self_Employed / Is_Unemployed columns that
      app.js's client-side scoring expects.
    - Loan_Approval_Status is mapped explicitly (Approved -> 1, Rejected ->
      0) instead of LabelEncoder, so the positive class is guaranteed rather
      than left to alphabetical assignment.
    """
    data = pd.read_csv(path)

    for col in NUMERIC_COLUMNS:
        data[col] = data[col].fillna(data[col].median())

    data = pd.get_dummies(
        data,
        columns=["Employment_Status"],
        prefix="Is",
        drop_first=True,
    )

    data["Loan_Approval_Status"] = data["Loan_Approval_Status"].map(
        {"Approved": 1, "Rejected": 0}
    )

    return data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model and print accuracy, recall, and ROC-AUC."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} results")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    return model


def main():
    data = load_and_prepare_data(DATA_PATH)

    employment_dummy_cols = [c for c in data.columns if c.startswith("Is_")]
    features = ["Applicant_Income", "Credit_Score", "Loan_Amount"] + employment_dummy_cols
    X = data[features]
    y = data["Loan_Approval_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    evaluate_model("Decision Tree", DecisionTreeClassifier(), X_train, X_test, y_train, y_test)
    evaluate_model("Random Forest", RandomForestClassifier(), X_train, X_test, y_train, y_test)
    evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000),
        X_train, X_test, y_train, y_test,
    )


if __name__ == "__main__":
    main()
