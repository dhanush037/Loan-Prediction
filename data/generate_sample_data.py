"""
Generates a small synthetic loan-application dataset for testing the
pipeline end-to-end, without needing a real dataset.

Usage:
    python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 300

employment_options = ["Employed", "Self-Employed", "Unemployed"]

income = np.random.randint(15000, 150000, n_rows)
credit_score = np.random.randint(300, 850, n_rows)
loan_amount = np.random.randint(5000, 500000, n_rows)
employment_status = np.random.choice(employment_options, n_rows)

# Rough rule so the target isn't pure noise: higher credit score and
# income relative to loan amount raises approval odds.
approval_score = (
    (credit_score - 300) / 550 * 0.5
    + (income / loan_amount).clip(0, 2) / 2 * 0.3
    + (employment_status != "Unemployed") * 0.2
)
loan_approval_status = np.where(
    approval_score + np.random.normal(0, 0.1, n_rows) > 0.5, "Approved", "Rejected"
)

df = pd.DataFrame(
    {
        "Applicant_Income": income,
        "Credit_Score": credit_score,
        "Loan_Amount": loan_amount,
        "Employment_Status": employment_status,
        "Loan_Approval_Status": loan_approval_status,
    }
)

# Introduce a few missing values to exercise the ffill step in the pipeline
missing_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[missing_idx, "Credit_Score"] = np.nan

df.to_csv("data/loan_prediction.csv", index=False)
print(f"Wrote {len(df)} rows to data/loan_prediction.csv")
