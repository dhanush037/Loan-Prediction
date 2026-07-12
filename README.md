# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be
approved or rejected, based on applicant income, credit score, loan amount,
and employment status. Three classifiers — Decision Tree, Random Forest, and
Logistic Regression — are trained and compared using accuracy, recall, and
ROC-AUC.

## Project structure

```
Loan-Prediction/
├── loan_prediction.py     # main training and evaluation script
├── requirements.txt       # Python dependencies
├── data/
│   ├── README.md          # dataset format and where to get one
│   └── loan_prediction.csv  # (not included — add your own, see data/README.md)
└── README.md
```

## How it works

1. **Load and clean data** — reads the CSV and forward-fills missing values.
2. **Encode categorical columns** — `Employment_Status` and
   `Loan_Approval_Status` are label-encoded into numbers the models can use.
3. **Train/test split** — 80% of rows for training, 20% held out for testing.
4. **Train three classifiers** — Decision Tree, Random Forest, and Logistic
   Regression are each trained on the same split.
5. **Evaluate** — each model reports accuracy, recall, and ROC-AUC on the
   test set, so you can compare which performs best.

## Setup

```bash
git clone https://github.com/dhanush037/Loan-Prediction.git
cd Loan-Prediction
pip install -r requirements.txt
```

Add your dataset at `data/loan_prediction.csv` (see `data/README.md` for the
expected columns), then run:

```bash
python loan_prediction.py
```

## Results

Add your model's accuracy/recall/ROC-AUC numbers here once you've run it
against your dataset, e.g.:

| Model               | Accuracy | Recall | ROC-AUC |
|---------------------|----------|--------|---------|
| Decision Tree        | —        | —      | —       |
| Random Forest         | —        | —      | —       |
| Logistic Regression   | —        | —      | —       |

## Tech stack

- Python
- pandas
- scikit-learn
