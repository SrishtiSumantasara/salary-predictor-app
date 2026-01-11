import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained PIPELINE model
# --------------------------------------------------
model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on details.")

# --------------------------------------------------
# Sidebar Inputs (MUST match training columns)
# --------------------------------------------------
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)

workclass = st.sidebar.selectbox(
    "Workclass",
    [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov"
    ]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent"
    ]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv"
    ]
)

relationship = st.sidebar.selectbox(
    "Relationship",
    ["Husband", "Wife", "Own-child", "Not-in-family", "Other-relative"]
)

race = st.sidebar.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

native_country = st.sidebar.selectbox(
    "Native Country",
    ["United-States", "India", "Mexico", "Canada", "Philippines"]
)

educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)

# --------------------------------------------------
# Input DataFrame (COLUMN NAMES MUST MATCH EXACTLY)
# --------------------------------------------------
input_df = pd.DataFrame({
    "age": [age],
    "workclass": [workclass],
    "marital-status": [marital_status],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "gender": [gender],
    "native-country": [native_country],
    "educational-num": [educational_num],
    "hours-per-week": [hours_per_week],
    "capital-gain": [capital_gain],
    "capital-loss": [capital_loss]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)[0]

    if prediction in [">50K", 1]:
        st.success("✅ Prediction: Income **>50K**")
    else:
        st.info("ℹ️ Prediction: Income **≤50K**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Model trained on Adult Census Income Dataset")
