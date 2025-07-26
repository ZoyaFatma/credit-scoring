import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Creditworthiness Predictor", layout="centered")
st.title("💳 German Creditworthiness Predictor")
st.write("This app predicts whether a customer is **Good (likely to repay)** or **Bad (likely to default)** based on financial inputs.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("german.data-numeric", sep=r"\s+", header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({1: 0, 2: 1})  # 0 = Good, 1 = Bad
    return X, y

# Train Random Forest model
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model

X, y = load_data()
model = train_model(X, y)

st.sidebar.header("Enter Customer Information")
user_input = []

# Realistic dropdowns
checking_map = {
    "Less than 0 DM": 1,
    "0 to < 200 DM": 2,
    ">= 200 DM or salary assignment": 3,
    "No checking account": 4
}
user_input.append(checking_map[st.sidebar.selectbox("Checking Account Status", checking_map.keys())])

user_input.append(st.sidebar.slider("Duration in Months", 6, 72, 24))

credit_history_map = {
    "No credits taken": 0,
    "All credits paid (this bank)": 1,
    "Credits paid till now": 2,
    "Delays in past": 3,
    "Critical account": 4
}
user_input.append(credit_history_map[st.sidebar.selectbox("Credit History", credit_history_map.keys())])

purpose_map = {
    "Car (new)": 0,
    "Car (used)": 1,
    "Furniture": 2,
    "Radio/TV": 3
}
user_input.append(purpose_map[st.sidebar.selectbox("Purpose", purpose_map.keys())])

user_input.append(st.sidebar.slider("Credit Amount", 250, 20000, 2000))

savings_map = {
    "Less than 100 DM": 1,
    "100 to 500 DM": 2,
    "500 to 1000 DM": 3,
    "Greater than 1000 DM": 4,
    "Unknown / No Account": 5
}
user_input.append(savings_map[st.sidebar.selectbox("Savings Account", savings_map.keys())])

employment_map = {
    "Unemployed": 0,
    "Less than 1 year": 1,
    "1 to 4 years": 2,
    "4 to 7 years": 3,
    "More than 7 years": 4
}
user_input.append(employment_map[st.sidebar.selectbox("Employment Since", employment_map.keys())])

user_input.append(st.sidebar.slider("Installment Rate (% of Income)", 1, 4, 2))

personal_status_map = {
    "Male - Divorced/Separated": 0,
    "Female - Married/Separated": 1,
    "Male - Single": 2,
    "Male - Married/Widowed": 3,
    "Female - Single": 4
}
user_input.append(personal_status_map[st.sidebar.selectbox("Personal Status / Sex", personal_status_map.keys())])

debtors_map = {
    "None": 0,
    "Co-applicant": 1,
    "Guarantor": 2
}
user_input.append(debtors_map[st.sidebar.selectbox("Other Debtors/Guarantors", debtors_map.keys())])

user_input.append(st.sidebar.slider("Years at Current Residence", 1, 4, 2))

property_map = {
    "Real estate": 0,
    "Insurance or savings agreement": 1,
    "Car or other": 2,
    "No property": 3
}
user_input.append(property_map[st.sidebar.selectbox("Property", property_map.keys())])

user_input.append(st.sidebar.slider("Age", 18, 75, 30))

installment_plan_map = {
    "Bank": 0,
    "Stores": 1,
    "None": 2
}
user_input.append(installment_plan_map[st.sidebar.selectbox("Other Installment Plans", installment_plan_map.keys())])

housing_map = {
    "Rent": 0,
    "Own": 1,
    "Free": 2
}
user_input.append(housing_map[st.sidebar.selectbox("Housing", housing_map.keys())])

user_input.append(st.sidebar.slider("Existing Credits at Bank", 1, 4, 1))

job_map = {
    "Unemployed / Unskilled": 0,
    "Unskilled - Resident": 1,
    "Skilled Employee / Official": 2,
    "Management / Highly Qualified": 3
}
user_input.append(job_map[st.sidebar.selectbox("Job", job_map.keys())])

user_input.append(st.sidebar.slider("Number of Dependents", 1, 2, 1))

telephone_map = {
    "No": 0,
    "Yes (Registered)": 1
}
user_input.append(telephone_map[st.sidebar.selectbox("Telephone", telephone_map.keys())])

foreign_worker_map = {
    "Yes": 0,
    "No": 1
}
user_input.append(foreign_worker_map[st.sidebar.selectbox("Foreign Worker", foreign_worker_map.keys())])

# Dummy zeros for missing features (4 dummy binary flags)
user_input += [0, 0, 0, 0]

# --- Prediction ---
if st.sidebar.button("Predict Creditworthiness"):
    input_array = np.array(user_input).reshape(1, -1)
    proba = model.predict_proba(input_array)[0][1]  # Probability of class 1 = Bad

    result = "✅ Good (Likely to repay)" if proba < 0.5 else "❌ Bad (Likely to default)"
    
    st.subheader("Prediction Result")
    if proba < 0.5:
        st.success(result)
    else:
        st.error(result)

    st.write(f"**Probability of Default (Bad credit):** `{proba:.2f}`")
    
    acc = accuracy_score(y, model.predict(X))
    st.write(f"Model trained on 1000 samples — estimated accuracy: **{acc:.2f}**")
