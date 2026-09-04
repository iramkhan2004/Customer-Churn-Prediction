import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Risk Prediction")
st.write(
    "Logistic Regression based Customer Churn Prediction System"
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("dataset_01_customer_churn_risk.csv")

features = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "avg_session_minutes",
    "late_payments",
    "contract_months"
]

target = "target"

X = df[features]
y = df[target]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic_regression",
        LogisticRegression(max_iter=1000)
    )
])

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)

col5.metric(
    "ROC-AUC",
    f"{roc_auc:.2%}"
)


st.divider()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📋 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Rows",
    df.shape[0]
)

col2.metric(
    "Total Features",
    len(features)
)

col3.metric(
    "Missing Values",
    int(df.isnull().sum().sum())
)


with st.expander("🔎 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


st.divider()


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("🎯 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(6, 4)
)

ax.imshow(cm)

ax.set_title("Confusion Matrix")

ax.set_xlabel("Predicted")

ax.set_ylabel("Actual")

ax.set_xticks([0, 1])

ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Not Churn", "Churn"]
)

ax.set_yticklabels(
    ["Not Churn", "Churn"]
)

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

plt.close(fig)


st.divider()


# ============================================================
# FEATURE COEFFICIENTS
# ============================================================

st.header("🔍 Feature Importance")

coefficients = (
    model
    .named_steps["logistic_regression"]
    .coef_[0]
)

coefficient_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
})

coefficient_df = coefficient_df.sort_values(
    "Coefficient",
    ascending=False
)

st.dataframe(
    coefficient_df,
    use_container_width=True
)


# Graph

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"]
)

ax.set_xlabel(
    "Logistic Regression Coefficient"
)

ax.set_title(
    "Feature Coefficients"
)

ax.axvline(
    0,
    linewidth=1
)

st.pyplot(fig)

plt.close(fig)


st.divider()


# ============================================================
# CUSTOMER CHURN PREDICTION
# ============================================================

st.header("👤 Predict Customer Churn")

st.write(
    "Enter customer information below and click "
    "**Predict Churn**."
)


col1, col2 = st.columns(2)


with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

    support_tickets = st.number_input(
        "Support Tickets",
        min_value=0,
        value=2
    )


with col2:

    avg_session_minutes = st.number_input(
        "Average Session Minutes",
        min_value=0.0,
        value=30.0
    )

    late_payments = st.number_input(
        "Late Payments",
        min_value=0,
        value=1
    )

    contract_months = st.number_input(
        "Contract Months",
        min_value=1,
        value=12
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Churn",
    type="primary"
):

    customer_data = pd.DataFrame(
        [[
            tenure,
            monthly_charges,
            support_tickets,
            avg_session_minutes,
            late_payments,
            contract_months
        ]],
        columns=features
    )

    prediction = model.predict(
        customer_data
    )[0]

    probability = model.predict_proba(
        customer_data
    )[0][1]


    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is predicted to CHURN"
        )

    else:

        st.success(
            "✅ Customer is predicted NOT to churn"
        )


    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Customer Churn Prediction | Logistic Regression "
    "| Educational Machine Learning Project"
)
