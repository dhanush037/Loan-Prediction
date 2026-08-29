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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/loan_prediction.csv"
RANDOM_STATE = 42

# Explicit mapping so the target's meaning (0 = Rejected, 1 = Approved) is
# never left to LabelEncoder's alphabetical ordering.
TARGET_MAP = {"Rejected": 0, "Approved": 1}


def load_and_prepare_data(path: str):
    """Load the CSV, impute missing values, and encode categorical columns."""
    data = pd.read_csv(path)

    # Median imputation: this is cross-sectional applicant data, not a time
    # series, so forward-filling would just borrow an unrelated applicant's
    # value. The median is a robust, order-independent fill for numeric gaps.
    data["Credit_Score"] = data["Credit_Score"].fillna(data["Credit_Score"].median())

    # One-hot encode Employment_Status instead of label-encoding it, since
    # Employed/Self-Employed/Unemployed have no natural order and integer
    # codes would falsely imply one.
    data = pd.get_dummies(data, columns=["Employment_Status"], drop_first=True)

    # Explicit target mapping avoids relying on LabelEncoder's alphabetical
    # assignment, which would silently flip which class is "1".
    data["Loan_Approval_Status"] = data["Loan_Approval_Status"].map(TARGET_MAP)

    return data


def evaluate_with_cv(name, model, X, y, cv):
    """Run stratified k-fold CV and print mean +/- std for each metric."""
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print(f"\n{name} results (5-fold stratified CV)")
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        print(f"{metric.capitalize()}: {vals.mean():.3f} +/- {vals.std():.3f}")

    return scores


def evaluate_on_holdout(name, model, X_train, X_test, y_train, y_test):
    """Fit on the train split and report a single holdout evaluation."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} holdout results")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    return model


def main():
    data = load_and_prepare_data(DATA_PATH)

    employment_cols = [c for c in data.columns if c.startswith("Employment_Status_")]
    features = ["Applicant_Income", "Credit_Score", "Loan_Amount"] + employment_cols
    X = data[features]
    y = data["Loan_Approval_Status"]

    # Stratified split keeps the Approved/Rejected ratio consistent between
    # train and test, which matters on a dataset this small (~300 rows).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 5-fold cross-validation gives a more reliable performance estimate
    # than a single holdout split at this sample size.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        # Logistic Regression needs standardized inputs since Applicant_Income
        # and Loan_Amount are on a much larger scale than Credit_Score or the
        # one-hot employment flags; without scaling, its coefficients would be
        # dominated by the largest-magnitude feature rather than reflecting
        # true importance.
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
    }

    for name, model in models.items():
        evaluate_with_cv(name, model, X, y, cv)

    print("\n" + "=" * 50)
    print("Holdout set evaluation (single 80/20 split, for reference)")
    print("=" * 50)
    for name, model in models.items():
        evaluate_on_holdout(name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
