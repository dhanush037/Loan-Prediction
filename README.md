# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be
approved or rejected, based on applicant income, credit score, loan amount,
and employment status. Three classifiers — Decision Tree, Random Forest, and
Logistic Regression — are trained and compared using accuracy, recall, and
ROC-AUC, evaluated with stratified 5-fold cross-validation.

## Project structure

```
Loan-Prediction/
├── loan_prediction.py     # main training and evaluation script
├── requirements.txt       # Python dependencies
├── data/
│   ├── README.md          # dataset format and where to get one
│   └── loan_prediction.csv  # Loan approval dataset CSV file containing applicant details
└── README.md
```

## How it works

1. **Load and clean data** — reads the CSV and fills missing `Credit_Score`
   values with the column median (this is cross-sectional applicant data,
   not a time series, so forward-filling isn't appropriate here).
2. **Encode categorical columns** — `Employment_Status` is one-hot encoded
   (via `pd.get_dummies`) since it has no natural order, and
   `Loan_Approval_Status` is mapped explicitly to 0/1 (`Rejected` = 0,
   `Approved` = 1) rather than relying on alphabetical label encoding.
3. **Cross-validate** — each model is evaluated with stratified 5-fold
   cross-validation, reporting the mean and standard deviation of accuracy,
   recall, and ROC-AUC. A single stratified 80/20 holdout split is also
   reported for reference.
4. **Train three classifiers** — Decision Tree and Random Forest are seeded
   with `random_state=42` for reproducibility; Logistic Regression is
   wrapped in a `Pipeline` with `StandardScaler`, since its coefficients are
   sensitive to the large scale differences between features like
   `Applicant_Income` and `Credit_Score`.
5. **Evaluate** — each model reports accuracy, recall, and ROC-AUC, so you
   can compare which performs best.

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

### Quick start with sample data

Don't have a dataset handy? Generate a small synthetic one to try the
pipeline end-to-end:

```bash
python data/generate_sample_data.py
python loan_prediction.py
```

## Results

Sample output from running the pipeline against the included dataset
(~300 rows), using stratified 5-fold cross-validation:

```
Decision Tree results (5-fold stratified CV)
Accuracy: 0.803 +/- 0.037
Recall: 0.741 +/- 0.090
Roc_auc: 0.794 +/- 0.041

Random Forest results (5-fold stratified CV)
Accuracy: 0.867 +/- 0.028
Recall: 0.805 +/- 0.082
Roc_auc: 0.945 +/- 0.003

Logistic Regression results (5-fold stratified CV)
Accuracy: 0.897 +/- 0.016
Recall: 0.862 +/- 0.086
Roc_auc: 0.960 +/- 0.003
```

| Model               | Accuracy | Recall | ROC-AUC |
| ------------------- | -------- | ------ | ------- |
| Decision Tree       | 0.803    | 0.741  | 0.794   |
| Random Forest       | 0.867    | 0.805  | 0.945   |
| Logistic Regression | 0.897    | 0.862  | 0.960   |

Logistic Regression performs best on this dataset. Cross-validation is used
instead of a single train/test split because the dataset is small (~300
rows), so a single split's metrics can vary noticeably depending on which
rows happen to land in the test set. The script also prints a single-split
holdout evaluation for reference — comparing it against the CV mean shows
how much noisier a one-off split can be at this sample size.

## Web demo

`index.html` and `app.js` provide a live prediction demo in the browser. The
scoring logic in `app.js` mirrors the trained Logistic Regression model
directly: it uses the same standardization (mean/scale) and learned
coefficients produced by `loan_prediction.py`, passed through a sigmoid, so
the demo reflects the actual trained model rather than a hand-tuned formula.

## Tech stack

- Python
- pandas
- scikit-learn
- Tailwind CSS (web demo)
