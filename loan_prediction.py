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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/loan_prediction.csv"


def load_and_prepare_data(path: str):
    """Load the CSV, fill missing values, and encode categorical columns."""
    data = pd.read_csv(path)
    data.ffill(inplace=True)

    le = LabelEncoder()
    data["Employment_Status"] = le.fit_transform(data["Employment_Status"])
    data["Loan_Approval_Status"] = le.fit_transform(data["Loan_Approval_Status"])

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

    features = ["Applicant_Income", "Credit_Score", "Loan_Amount", "Employment_Status"]
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
