# Data

This project expects a CSV file at `data/loan_prediction.csv` with the
following columns:

| Column                 | Type        | Description                                  |
|-------------------------|-------------|-----------------------------------------------|
| Applicant_Income         | numeric     | Applicant's monthly/annual income             |
| Credit_Score             | numeric     | Applicant's credit score                      |
| Loan_Amount              | numeric     | Requested loan amount                         |
| Employment_Status        | categorical | e.g. "Employed", "Self-Employed", "Unemployed"|
| Loan_Approval_Status     | categorical | Target column: "Approved" / "Rejected"        |

## Getting a dataset

The original dataset used for this project is not included in this repo.
You can either:

1. Use your own loan-application data with matching column names, or
2. Use a public dataset such as the "Loan Prediction Problem Dataset" on
   Kaggle, renaming columns to match the table above.

Place the CSV at `data/loan_prediction.csv` before running
`loan_prediction.py`.
