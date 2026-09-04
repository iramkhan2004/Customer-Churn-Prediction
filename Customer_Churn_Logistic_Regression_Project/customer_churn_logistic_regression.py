"""
Customer Churn Risk Prediction using Logistic Regression
LearnDepth Academy - Applied Classification Project

Problem statement:
Predict whether a subscription customer will churn in the next billing cycle.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_01_customer_churn_risk.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)

print("\n=== DATASET OVERVIEW ===")
print(df.head())
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTarget balance:\n", df["target"].value_counts())
print("\nTarget proportions:\n", df["target"].value_counts(normalize=True))

# 2. Data-quality checks
assert df.isna().sum().sum() == 0, "Missing values found."
assert df.duplicated().sum() == 0, "Duplicate rows found."
assert set(df["target"].unique()).issubset({0, 1}), "Target must be binary."

# 3. Features and target
X = df.drop(columns="target")
y = df["target"]

# 4. Stratified train/test split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Pipeline: scaling is fitted only on training data, avoiding leakage
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

# 6. Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
}

print("\n=== TEST METRICS ===")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("=== CONFUSION MATRIX ===")
print(cm)

# 7. Confusion-matrix figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot()
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.close()

# 8. Coefficient interpretation
coefs = pd.Series(
    model.named_steps["logistic_regression"].coef_[0],
    index=X.columns
).sort_values(key=abs, ascending=False)

print("\n=== STANDARDIZED LOGISTIC REGRESSION COEFFICIENTS ===")
print(coefs)

coefs.to_csv(OUTPUT_DIR / "feature_coefficients.csv", header=["coefficient"])

print("\nProject completed. Outputs saved in:", OUTPUT_DIR)
