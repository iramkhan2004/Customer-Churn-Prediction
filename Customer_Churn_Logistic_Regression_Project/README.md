# Customer Churn Risk — Logistic Regression

## Problem Statement
Predict whether a subscription customer will churn in the next billing cycle.

## Dataset
`dataset_01_customer_churn_risk.csv`

- Rows: 1,000
- Features: 6
- Target: `target` (0 = no churn, 1 = churn)
- Missing values: 0
- Duplicate rows: 0
- Class balance: 50% / 50%

### Features
| Feature | Meaning |
|---|---|
| tenure_months | Customer tenure in months |
| monthly_charges | Monthly subscription charges |
| support_tickets | Number of support tickets |
| avg_session_minutes | Average session duration |
| late_payments | Number of late payments |
| contract_months | Contract duration |
| target | Churn label |

## Methodology
1. Load and inspect the data.
2. Check missing values, duplicates, target balance, and data quality.
3. Separate features and target.
4. Use an 80/20 stratified train/test split (`random_state=42`).
5. Standardize numeric features inside a `Pipeline`.
6. Train Logistic Regression with `max_iter=1000`.
7. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
8. Interpret standardized coefficients.

The scaler is inside the pipeline, so it is fitted only on the training set and does not leak test-set information.

## Test Results
Using the fixed split and configuration:

- Accuracy: **68.00%**
- Precision: **68.37%**
- Recall: **67.00%**
- F1-score: **67.68%**
- ROC-AUC: **74.95%**
- Confusion matrix: `[[69, 31], [33, 67]]`

## Coefficient Interpretation
Because features are standardized, coefficient magnitudes are directly comparable.

- `monthly_charges` (+0.647): higher charges are associated with higher predicted churn probability.
- `contract_months` (+0.536): longer contract duration is associated with higher predicted churn in this synthetic dataset.
- `avg_session_minutes` (+0.508): longer average sessions are associated with higher predicted churn.
- `support_tickets` (-0.448): more support tickets are associated with lower predicted churn in this synthetic dataset.
- `late_payments` (-0.428): more late payments are associated with lower predicted churn in this synthetic dataset.
- `tenure_months` (-0.351): longer tenure is associated with lower predicted churn.

These relationships describe this synthetic educational dataset and should not be treated as real-world business causality.

## Limitations
- The dataset is synthetic and created for educational practice.
- Logistic Regression assumes a linear relationship between predictors and log-odds.
- The model may miss nonlinear interactions.
- A single train/test split gives only one estimate of performance.
- Real churn prediction would require time-aware validation and real customer behavior data.

## Improvements
- Use cross-validation and hyperparameter tuning.
- Compare Logistic Regression with tree-based models.
- Add real behavioral and engagement features.
- Tune the probability threshold based on the business cost of false positives vs false negatives.
- Use calibration and monitoring after deployment.

## Run
```bash
pip install -r requirements.txt
python customer_churn_logistic_regression.py
```
