# Final Project Report — Customer Churn Risk

## 1. Introduction
The objective is to predict whether a subscription customer will churn in the next billing cycle using Logistic Regression.

## 2. Dataset Inspection
The dataset contains 1,000 observations and 6 predictor variables plus a binary target. There are no missing values and no duplicate rows. The target is perfectly balanced: 500 non-churn and 500 churn records.

## 3. Preprocessing
All six predictors are numeric, so no categorical encoding is required. StandardScaler is used because Logistic Regression coefficients and optimization can be affected by feature scale. Scaling is performed inside a scikit-learn Pipeline to prevent data leakage.

## 4. Train/Test Strategy
An 80/20 split is used with `random_state=42` and `stratify=y`. The stratification keeps the 50/50 target balance approximately unchanged in both subsets.

## 5. Model
Model: Logistic Regression  
Configuration: `max_iter=1000`, `random_state=42`  
Preprocessing: StandardScaler

## 6. Results
| Metric | Score |
|---|---:|
| Accuracy | 68.00% |
| Precision | 68.37% |
| Recall | 67.00% |
| F1-score | 67.68% |
| ROC-AUC | 74.95% |

Confusion matrix:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 69 | 31 |
| Actual Churn | 33 | 67 |

The model correctly identifies 67 of 100 churn cases and 69 of 100 non-churn cases in the held-out test set. ROC-AUC of 0.7495 indicates useful ranking ability, although the model is not highly accurate.

## 7. Feature Interpretation
With standardized predictors, positive coefficients increase the model's log-odds of churn and negative coefficients decrease them.

The strongest positive coefficient is monthly charges, while tenure has a negative coefficient. The remaining relationships should be interpreted cautiously because this is synthetic data and coefficient direction does not establish causation.

## 8. Limitations
The main limitation is that the dataset is synthetic. A real deployment would need representative historical customer data, time-based validation, drift monitoring, fairness checks, probability calibration, and a business-specific decision threshold.

## 9. Conclusion
The Logistic Regression model provides a reproducible baseline for customer churn classification. Its 68% accuracy and 0.7495 ROC-AUC show that the selected variables contain predictive signal. It is suitable as an educational baseline, but additional real-world data, validation, feature engineering, and model comparison would be needed before using such a system for business decisions.
